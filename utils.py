import logging
import warnings

from collections import Counter

import numpy as np
import pandas as pd
from scipy.ndimage.interpolation import shift
from sklearn.metrics import mean_squared_error, mean_absolute_error

from skater.core.explanations import Interpretation
from skater.model import InMemoryModel

from data import Data2d, Data3d


LOGGER = logging.getLogger('main.utils')


class Result:

    def __init__(self, config_vals):
        self.config_vals = tuple(config_vals.items())

        self.train_mse_list = []
        self.test_mse_list = []
        self.train_mae_list = []
        self.test_mae_list = []
        self.train_mape_list = []
        self.test_mape_list = []
        self.yhat_is_list = []
        self.yhat_oos_list = []
        self.feature_scores_list = []
        self.permuted_scores_list = []

    def get_mean(self, alist):
        a = np.array(alist)
        return a.mean(), a.std()

    def get_mean_counter(self, alist):
        c = Counter()
        for x in alist:
            c.update(Counter(dict(x)))
        num_counters = len(alist)
        return [(x, y/num_counters) for x, y in c.most_common()]

    def calc_means(self):
        self.train_mse, self.train_mse_std = self.get_mean(self.train_mse_list)
        self.test_mse, self.test_mse_std = self.get_mean(self.test_mse_list)
        self.train_mae, self.train_mae_std = self.get_mean(self.train_mae_list)
        self.test_mae, self.test_mae_std = self.get_mean(self.test_mae_list)
        self.train_mape, self.train_mape_std = self.get_mean(self.train_mape_list)
        self.test_mape, self.test_mape_std = self.get_mean(self.test_mape_list)
        self.yhat_is = np.array(self.yhat_is_list).mean(axis=0)
        self.yhat_oos = np.array(self.yhat_oos_list).mean(axis=0)
        self.feature_scores = self.get_mean_counter(self.feature_scores_list)
        self.permuted_scores = self.get_mean_counter(self.permuted_scores_list)

    def __str__(self):
        return "RMSE: %.3f MAE: %.3f MAPE: %.3f" % (self.test_mse,
                                                      self.test_mae,
                                                      self.test_mape)


def interpolate(df):
    """Interpolate data for missing dates
    """
    for x in df.columns:
        if x == "date":
            continue
        df[x] = df[x].interpolate(method='linear', axis=0).ffill().bfill()
    return df


def sort_feature_scores(data, scores):
    scores = np.array(scores).flatten()
    assert len(data.feature_names) == len(scores)
    return Counter(dict((x, y) for x, y in zip(data.feature_names, scores))
                            ).most_common()


def get_permuted_feature_scores(model, data):
    """Computed permuted feature importances, using skater
    """
    interpreter = Interpretation(data.testX, feature_names=data.feature_names)
    pyint_model = InMemoryModel(model.predict, examples=data.testX)
    feature_scores = list(interpreter.feature_importance.feature_importance(
        pyint_model, ascending=False, progressbar=False).items())
    return feature_scores


def get_feature_scores(model, data):
    """Get feature scores from either feature importances or rankings
    """
    def get_features_by_attr(model):
        for name in ['feature_importances_', 'coef_']:
            if hasattr(model, name):
                importances = getattr(model, name)
                if not isinstance(importances, list):
                    importances = importances.tolist()
                return importances
        return None

    features = get_features_by_attr(model)
    if features is None:
        if hasattr(model, 'rankings_'):
            features = sort_feature_scores(data, -model.rankings_)
        elif hasattr(model, 'estimator_'):
            features = get_features_by_attr(model.estimator_)
            # update remaining feature names
            data.feature_names = [data.feature_names_orig[x]
                                  for x in model.get_support(indices=True)]

    return sort_feature_scores(data, features) if features else []


def run_config(args):
    """
    :param c: learner config
    :param mode: 'test' or 'val'
    """
    data, c, mode = args

    result = Result(c.vals)

    # if num_random_seeds == 0 (i.e., the function is then used by
    # emp_intervals_viz.py), then use the passed random state, don't change it
    if c.pc['num_random_seeds'] == 0:
        iter_range = range(c.pc['random_state'], c.pc['random_state']+1)
    else:
        iter_range = range(c.pc['num_random_seeds'])

    for seed_number in iter_range:

        np.random.seed(seed_number)
        c.pc['random_state'] = seed_number
        model = c.train(data)

        # in-sample
        yhat_is = c.forecast(model, data.trainX)
        result.train_mse_list.append(get_mse(data, yhat_is, "train"))
        result.train_mae_list.append(get_mae(data, yhat_is, "train"))
        result.train_mape_list.append(get_mape(data, yhat_is, "train"))
        result.yhat_is_list.append(yhat_is)

        # out-of-sample
        yhat_oos = c.forecast(model, data.testX) if mode == 'test' \
            else c.forecast(model, data.valX)

        result.test_mse_list.append(get_mse(data, yhat_oos, mode))
        result.test_mae_list.append(get_mae(data, yhat_oos, mode))
        result.test_mape_list.append(get_mape(data, yhat_oos, mode))
        result.yhat_oos_list.append(yhat_oos)

        feature_scores = get_feature_scores(model, data)
        permuted_scores = []#get_permuted_feature_scores(model, data)
        result.feature_scores_list.append(feature_scores)
        result.permuted_scores_list.append(permuted_scores)

    result.calc_means()

    return result


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
    y_flat = y.reshape((-1,))
    yhat_flat = data.revert(yhat, mode).flatten()
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            return np.mean(np.abs((y_flat - yhat_flat) / y_flat)) * 100
        except Warning:
            return 0.0


def get_mda(data, yhat, mode="train"):
    """Mean Directional Accuracy, as per:
    https://www.wikiwand.com/en/Mean_Directional_Accuracy
    """
    a = np.sign(np.diff(getattr(data, mode + "Yref")))
    b = np.sign(np.diff(yhat))
    return np.sum(a == b)/a.shape[0]


def load_df(filename, date_format='%Y-%m-%d %H:%M:%S', eliminate_features=[]):
    df = pd.read_csv(filename, index_col='date', parse_dates=['date'],
        date_parser=lambda x: pd.datetime.strptime(x, date_format))
    for x in eliminate_features:
        del df[x]
    return df


def prepare_data(pc, dim=None, eliminate_features=[]):
    """
    :param config: preprocessing config
    """
    df = load_df(pc['data_file'], pc['date_format'],
        eliminate_features=eliminate_features)

    if pc.get('freq_threshold', 0) > 0:
        # remove features with overall frequency below the threshold
        cluster2freq = dict(zip(df.columns[:-1], df.sum(axis=0)[:-1]))
        for cl_id, count in cluster2freq.items():
            if count < pc['freq_threshold']:
                del df[cl_id]

    df = interpolate(df)
    d = Data3d(df, pc) if dim == '3d' else Data2d(df, pc)
    LOGGER.debug(f"Prepared data:\n{d}")
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
    LOGGER.info("%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f\t%.8f" % (
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

    yhat_oos = data.revert(test_result.yhat_oos, "test", True)
    yhat_is = data.revert(test_result.yhat_is, "train", True)
    yhat_val = data.revert(val_result.yhat_oos, "val", True)

    yhat_oos_list = [data.revert(x, "test", True) for x in test_result.yhat_oos_list]
    yhat_is_list = [data.revert(x, "train", True) for x in test_result.yhat_is_list]
    yhat_val_list = [data.revert(x, "val", True) for x in val_result.yhat_oos_list]

    LOGGER.info("Best config %s:\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s" % (
        best_config.vals, test_result.train_mse, val_result.test_mse,
        test_result.test_mse, test_result.train_mae, val_result.test_mae,
        test_result.test_mae, test_result.train_mape, val_result.test_mape,
        test_result.test_mape, yhat_oos))

    LOGGER.info("Informative features:")
    for k, v in test_result.feature_scores[:5]:
        LOGGER.info("%s: %.3f" % (k, v))

    return {
                'date': best_config.datetime,
                'preproc_config': pc,
                'learner': best_config.learner,
                'best_learner_config': best_config.vals,
                'mse': {
                        'train': {
                            'mean': test_result.train_mse,
                            'std': test_result.train_mse_std,
                            'obs': test_result.train_mse_list
                            },
                        'val': {
                            'mean': val_result.test_mse,
                            'std': val_result.test_mse_std,
                            'obs': val_result.test_mse_list,
                            },
                        'test': {
                            'mean': test_result.test_mse,
                            'std': test_result.test_mse_std,
                            'obs': test_result.test_mse_list,
                            }
                        },
                'mae': {
                        'train': {
                            'mean': test_result.train_mae,
                            'std': test_result.train_mae_std,
                            'obs': test_result.train_mae_list
                            },
                        'val': {
                            'mean': val_result.test_mae,
                            'std': val_result.test_mae_std,
                            'obs': val_result.test_mae_list,
                            },
                        'test': {
                            'mean': test_result.test_mae,
                            'std': test_result.test_mae_std,
                            'obs': test_result.test_mae_list,
                            }
                        },
                'mape': {
                        'train': {
                            'mean': test_result.train_mape,
                            'std': test_result.train_mape_std,
                            'obs': test_result.train_mape_list
                            },
                        'val': {
                            'mean': val_result.test_mape,
                            'std': val_result.test_mape_std,
                            'obs': val_result.test_mape_list,
                            },
                        'test': {
                            'mean': test_result.test_mape,
                            'std': test_result.test_mape_std,
                            'obs': test_result.test_mape_list,
                            }
                        },
                'yhat_is': {
                    'mean': yhat_is,
                    'obs': yhat_is_list
                    },
                'yhat_oos': {
                    'mean': yhat_oos,
                    'obs': yhat_oos_list
                    },
                'yhat_val': {
                    'mean': yhat_val,
                    'obs': yhat_val_list
                    },
                'validation_runs': [{'config': dict(x),
                                     'scores': {'mse': v.test_mse,
                                                'mae': v.test_mae,
                                                'mape': v.test_mape}}
                                        for x, v in val_results.items()],
                'feature_scores': test_result.feature_scores,
                'permuted_scores': test_result.permuted_scores
            }
