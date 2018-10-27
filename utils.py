
import logging
import warnings

from collections import Counter

import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
from sklearn.metrics import mean_squared_error, mean_absolute_error

from learner_configs import ConfigGB, ConfigLSTM
from data import Data2d, Data3d


LOGGER = logging.getLogger('main.utils')


class Result:

    def __init__(self, train_mse, test_mse, train_mae, test_mae,
                 train_mape, test_mape, yhat_is, yhat_oos,
                 config_vals, feature_scores=()):
        self.train_mse = train_mse
        self.test_mse = test_mse
        self.train_mae = train_mae
        self.test_mae = test_mae
        self.train_mape = train_mape
        self.test_mape = test_mape
        self.yhat_is = yhat_is
        self.yhat_oos = yhat_oos
        self.config_vals = tuple(config_vals.items())
        self.feature_scores = feature_scores

    def __str__(self):
        return "RMSE: %.3f MAE: %.3f MAPE: %.3f" % (self.test_mse,
                                                      self.test_mae,
                                                      self.test_mape)


def interpolate(df):
    # interpolate data for missing dates
    for x in df.columns:
        if x == "date":
            continue
        df[x] = df[x].interpolate(method='linear', axis=0).ffill().bfill()
    return df


def run_config(args):
    """
    :param c: learner config
    :param mode: 'test' or 'val'
    """
    data, c, mode = args

    if isinstance(c, (ConfigGB, ConfigLSTM)):
        model = c.fit(data.trainX, data.trainY, data.valX, data.valY)
    else:
        model = c.fit(data.trainX, data.trainY)

    # in-sample
    yhat_is = c.forecast(model, data.trainX)
    train_mse = get_mse(data, yhat_is, "train")
    train_mae = get_mae(data, yhat_is, "train")
    train_mape = get_mape(data, yhat_is, "train")

    # out-of-sample
    if mode == 'test':
        yhat_oos = c.forecast(model, data.testX)
    else:
        yhat_oos = c.forecast(model, data.valX)

    test_mse = get_mse(data, yhat_oos, mode)
    test_mae = get_mae(data, yhat_oos, mode)
    test_mape = get_mape(data, yhat_oos, mode)

    # feature importances
    if hasattr(model, 'feature_importances_'):
        assert len(data.feature_names) == len(model.feature_importances_)
        feature_scores = Counter(dict((x, y)
                                 for x, y in zip(data.feature_names,
                                                 model.feature_importances_))
                                ).most_common()
    else:
        feature_scores = []

    return Result(train_mse, test_mse, train_mae, test_mae,
                  train_mape, test_mape, yhat_is, yhat_oos, c.vals,
                  feature_scores)


def get_mse(data, yhat, mode="train"):
    """Root Mean Squared Error
    """
    y = getattr(data, mode + "Yref")
    mse = mean_squared_error(y, data.revert(yhat, mode))
    return np.sqrt(mse)


def get_mae(data, yhat, mode="train"):
    """Mean Absolute Error
    """
    y = getattr(data, mode + "Yref")
    mae = mean_absolute_error(y, data.revert(yhat, mode))
    return mae


def get_mape(data, yhat, mode="train"):
    """Mean Absolute Percentage Error
    """
    y = getattr(data, mode + "Yref")
    y_flat = y.values.reshape((-1,))
    yhat_flat = data.revert(yhat, mode).flatten()
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            return np.mean(np.abs((y_flat - yhat_flat) / y_flat)) * 100
        except Warning:
            return 0.0


def separate_exogs(data, lags):
    """Output two arrays, one with endogs and one with exogs
    """
    return data[:, :lags], data[:, lags:]


def load_df(filename, date_format='%Y-%m-%d %H:%M:%S'):
    return pd.read_csv(filename, index_col='date', parse_dates=['date'],
        date_parser=lambda x: pd.datetime.strptime(x, date_format))


def prepare_data(pc, dim=None):
    """
    :param config: preprocessing config
    """
    df = load_df(pc['data_file'], pc['date_format'])
    df = interpolate(df)
    d = Data3d(df, pc) if dim == '3d' else Data2d(df, pc)
    LOGGER.debug("Prepared data:\n%s" % d)
    return d


def mean_baseline(d, mode='test'):
    """Always predict the mean of train data
    """
    m = d.trainY.mean()
    y = getattr(d, mode + "Y")
    preds = np.array([m] * y.shape[0])
    return (get_mse(d, preds, mode), get_mae(d, preds, mode),
              get_mape(d, preds, mode))


def persistence_baseline(d, mode='test'):
    """Always predict the previous day's Y value
    """
    y = getattr(d, mode + "Y")
    preds = shift(y, 1, cval=0.0)
    return (get_mse(d, preds, mode), get_mae(d, preds, mode),
              get_mape(d, preds, mode))


def do_baseline(d):
    """Evaluate the baseline
    """
    train_mse, train_mae, train_mape = persistence_baseline(d, "train")
    val_mse, val_mae, val_mape = persistence_baseline(d, "val")
    test_mse, test_mae, test_mape = persistence_baseline(d, "test")
    LOGGER.info("%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f" % (
            train_mse, val_mse, test_mse, train_mae, val_mae, test_mae,
            train_mape, val_mape, test_mape))
    return


def get_best_config(learner_config_space, preproc_config, mse_scores, results):
    LOGGER.debug("Validation set results:")
    best_config = None
    best_result = None
    for k, v in list(reversed(mse_scores.most_common()))[:10]:
        LOGGER.debug("%s:\t%s" % (k, v))
        if best_config is None:
            best_config = learner_config_space.Config(dict(k), preproc_config)
            best_result = results[k]
    return best_config, best_result


def run_config_space(pc, learner_config_space, get_val_results, baseline=False):
    """Run experiments with all possible settings in the config space
    :param pc: preprocessing config
    :param get_val_results: a function to run cross-validation on the
        validattion set, e.g. see example in `run.py`
    """

    # load data
    data = prepare_data(pc, dim=learner_config_space.dim)

    if baseline:
        do_baseline(data)
        return

    # search for best parameters on the validation set
    mse_scores, val_results = get_val_results(data, learner_config_space, pc)

    # select the best config according to validation set results
    best_config, val_result = get_best_config(learner_config_space, pc,
                                               mse_scores, val_results)

    # apply the best parameter config to the test set
    test_result = run_config([data, best_config, 'test'])

    yhat_oos = data.revert(test_result.yhat_oos, "test").flatten().tolist()
    yhat_is = data.revert(test_result.yhat_is, "train").flatten().tolist()
    yhat_val = data.revert(val_result.yhat_oos, "val").flatten().tolist()

    LOGGER.info("Best config %s:\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s" % (
        best_config.vals, test_result.train_mse, val_result.test_mse,
        test_result.test_mse, test_result.train_mae, val_result.test_mae,
        test_result.test_mae, test_result.train_mape, val_result.test_mape,
        test_result.test_mape, yhat_oos))

    LOGGER.info("Informative features:")
    for k, v in test_result.feature_scores:
        LOGGER.info("%s: %.3f" % (k, v))

    return {
                'date': best_config.datetime,
                'preproc_config': pc,
                'learner': best_config.learner,
                'best_learner_config': best_config.vals,
                'mse': {
                        'train': test_result.train_mse,
                        'val': val_result.test_mse,
                        'test': test_result.test_mse
                        },
                'mae': {
                        'train': test_result.train_mae,
                        'val': val_result.test_mae,
                        'test': test_result.test_mae
                        },
                'mape': {
                        'train': test_result.train_mape,
                        'val': val_result.test_mape,
                        'test': test_result.test_mape
                        },
                'yhat_is': yhat_is,
                'yhat_oos': yhat_oos,
                'yhat_val': yhat_val,
                'validation_runs': [{'config': dict(x),
                                     'scores': {'mse': v.test_mse,
                                                'mae': v.test_mae,
                                                'mape': v.test_mape}}
                                        for x, v in val_results.items()],
                'test_feature_scores': test_result.feature_scores,
                'val_feature_scores': val_result.feature_scores
            }
