
import os
import itertools
import logging
import warnings
import numpy as np

from copy import deepcopy
from datetime import datetime

from sklearn.svm.classes import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE

from learner_wrappers import GBWrapper, XGBWrapper

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import regularizers
from tensorflow.keras import backend


logging.getLogger("tensorflow").disabled = True
warnings.simplefilter(action='ignore', category=FutureWarning)

LOGGER = logging.getLogger('main.learner_configs')


class ConfigSpace:

    def __init__(self, Config, adict, pc):
        """
        :param Config: learner config class
        :param adict: a dictionary with learner parameters and value ranges
        :param pc: preprocessing config
        """
        self.Config = Config
        self.dim = "3d" if Config is ConfigLSTM else "2d"
        self.num_configs = 1
        self.parameter_ranges = adict
        self.pc = pc
        for k, v in self.parameter_ranges.items():
            setattr(self, k, v)
            self.num_configs *= len(v)
        LOGGER.info("Will evaluate %d configs" % self.num_configs)

    def generate_config(self):
        """Yield one hyperparameter configuration
        """
        param_names, param_values = zip(*self.parameter_ranges.items())
        i = 0
        for x in itertools.product(*param_values):
            adict = dict(zip(param_names, x))
            yield self.Config(adict, self.pc)
            i += 1
            if i % 10 == 0:
                try:
                    LOGGER.info("%s: completed %d of %d" % (datetime.now(), i,
                                                            self.num_configs))
                except TypeError:
                    pass
        LOGGER.info("%s: completed %d of %d" % (datetime.now(),
                                        self.num_configs, self.num_configs))


class Config:

    def __init__(self, adict, pc):

        for k, v in adict.items():
            setattr(self, k, v)

        self.name = ", ".join(["%s: %s" % x for x in self.__dict__.items()])
        self.datetime = str(datetime.now())
        self.vals = adict
        self.learner = self.__class__.__name__.replace("Config", "")
        self.pc = pc

    def __str__(self):
        return self.name

    def train(self, data):
        if self.pc['rfe_step'] != 0 and self.pc['feature_selection'] > 0:
            model = self.rfe_fit(data)
        else:
            model = self.fit(data)
        return model

    def fit(self, data):
        if isinstance(self, (ConfigGB, ConfigXGBoost)) and self.early_stopping:
            model = self.init_model(early_stopping=self.early_stopping,
                                    num_train=data.trainX.shape[0])
            x = np.concatenate((data.trainX, data.valX))
            y = np.concatenate((data.trainY, data.valY))
            model.fit(x, y)
        else:
            model = self.init_model()
            model.fit(data.trainX, data.trainY)
        return model

    def rfe_fit(self, data):
        """Recursive feature elimination
        """
        if isinstance(self, (ConfigGB, ConfigXGBoost)):
            model = self.init_model(early_stopping=self.early_stopping,
                                    num_train=data.trainX.shape[0])
        else:
            model = self.init_model()
        num = int(data.trainX.shape[1] * self.pc['feature_selection'])
        if num < 1:
            raise Exception(f"There will be {num} after selection!, "+
                "change the feature_selection setting")
        LOGGER.debug("RFE will select %d features" % num)
        step = self.pc['rfe_step']
        selector = RFE(model, n_features_to_select=num, step=step)
        if isinstance(self, (ConfigGB, ConfigXGBoost)) and self.early_stopping:
            x = np.concatenate((data.trainX, data.valX))
            y = np.concatenate((data.trainY, data.valY))
            return selector.fit(x, y)
        else:
            return selector.fit(data.trainX, data.trainY)

    def forecast(self, model, testX):
        """Return forecasts for all horizons up to `horizon`
        """

        results = []
        horizon = self.pc['horizon']
        lags = self.pc['lags']
        LOGGER.debug("Forecasting test: %s" % testX)
        for i in range(len(testX) - horizon + 1):
            for j in range(horizon):
                instance = testX[i+j]
                if j == 0:
                    buf = []
                else:
                    # insert the buffer into instance
                    buf_len = len(buf)
                    if buf_len > lags:
                        buf = buf[-lags:]
                        buf_len = lags
                    start = lags - buf_len
                    end = start + buf_len
                    instance = np.concatenate((instance[:start], buf,
                                               instance[end:]))
                pred_val = model.predict(instance.reshape(1, -1))[0]
                if np.isnan(pred_val) or pred_val < -1.0 or pred_val > 1.0:
                    warnings.warn("Error forecasting %s, returning 0.5"
                                  % instance.tolist())
                    pred_val = 0.5
                buf.append(pred_val)
                LOGGER.debug("Predicting with instance %s, result %s" %
                      (instance, pred_val))
            results.append(pred_val)

        return np.array(results).reshape(-1, 1)


class ConfigSVR(Config):

    kernel = 'linear'
    degree = 3
    c = 1.0
    eps = 0.1

    def init_model(self):
        return SVR(kernel=self.kernel, degree=self.degree, C=self.c,
                    epsilon=self.eps)


class ConfigAdaBoost(Config):

    n_estimators = None
    learning_rate = 1.0
    loss = 'linear'

    def init_model(self):
        return AdaBoostRegressor(n_estimators=self.n_estimators,
            learning_rate=self.learning_rate, loss=self.loss,
            random_state=self.pc['random_state'])


class ConfigRFR(Config):

    n_estimators = 100
    max_features = 'auto'
    max_depth = None
    min_samples_split = 2
    min_samples_leaf = 1
    max_leaf_nodes = None

    def init_model(self):
        return RandomForestRegressor(n_estimators=self.n_estimators,
            max_features=self.max_features, max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=self.pc['random_state'])


class ConfigGB(Config):

    n_estimators = 100
    learning_rate = 0.1
    loss = 'ls'
    max_features = None
    max_depth = 3
    min_samples_split = 2
    min_samples_leaf = 1
    max_leaf_nodes = None
    min_weight_fraction_leaf = 0.0
    subsample = 1.0
    alpha = 0.9
    warm_start = False
    early_stopping = None

    def init_model(self, early_stopping=None, num_train=None):
        return GBWrapper(n_estimators=self.n_estimators,
            learning_rate=self.learning_rate, loss=self.loss,
            max_features=self.max_features, max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            subsample=self.subsample, alpha=self.alpha,
            warm_start=self.warm_start,
            random_state=self.pc['random_state'],
            early_stopping=early_stopping, num_train=num_train)


class ConfigXGBoost(Config):

    max_depth = 3
    learning_rate = 0.1
    n_estimators = 100
    objective = 'reg:linear'
    booster = 'gbtree'
    n_jobs = 1
    reg_alpha = 0
    early_stopping = None

    def init_model(self, early_stopping=None, num_train=None):
        return XGBWrapper(max_depth=self.max_depth,
            objective=self.objective, n_estimators=self.n_estimators,
            learning_rate=self.learning_rate, booster=self.booster,
            reg_alpha=self.reg_alpha, n_jobs=self.n_jobs, nthread=self.n_jobs,
            random_state=self.pc['random_state'],
            early_stopping=early_stopping, num_train=num_train)


class ConfigLSTM(Config):

    bidirectional = None
    topology = []
    epochs = None
    batch_size = None
    activation = None
    dropout_rate = None
    optimizer = None
    kernel_regularizer = (0.0, 0.0)
    bias_regularization = (0.0, 0.0)
    early_stopping = None
    stateful = None

    def fit(self, data):
        """Importing Keras to be able to use it with multiprocessing
        """

        backend.clear_session()

        LOGGER.debug("Pid: %s: training LSTM ..." % os.getpid())

        np.random.seed(self.pc['random_state'])

        model = models.Sequential()

        kernel_regularizer = regularizers.L1L2(l1=self.kernel_regularizer[0],
                           l2=self.kernel_regularizer[1])
        bias_regularizer = regularizers.L1L2(l1=self.bias_regularization[0],
                           l2=self.bias_regularization[1])

        return_sequences_on_input = False if len(self.topology) == 2 else True

        # first layer
        if self.bidirectional:
            model.add(layers.Bidirectional(layers.LSTM(
                      units=self.topology[0],
                      activation=self.activation,
                      return_sequences=return_sequences_on_input,
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer),
                      input_shape=(None, data.trainX.shape[2]),))
        else:
            model.add(layers.LSTM(input_shape=(None, data.trainX.shape[2]),
                    units=self.topology[0],
                    activation=self.activation,
                    return_sequences=return_sequences_on_input,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer))

        if self.dropout_rate:
            model.add(layers.Dropout(self.dropout_rate))

        # other layers
        for n_layer in self.topology[1:-1]:
            if self.bidirectional:
                model.add(layers.Bidirectional(layers.LSTM(
                                    n_layer, return_sequences=False,
                                    kernel_regularizer=kernel_regularizer,
                                    bias_regularizer=bias_regularizer)))
            else:
                model.add(layers.LSTM(n_layer, return_sequences=False,
                            kernel_regularizer=kernel_regularizer,
                            bias_regularizer=bias_regularizer))
            if self.dropout_rate:
                model.add(layers.Dropout(self.dropout_rate))

        model.add(layers.Dense(units=self.topology[-1]))

        model.compile(loss="mean_squared_error", optimizer=self.optimizer)

        # train
        if self.early_stopping:
            # early stopping: Keras to stop training when loss didn't improve
            # for `c.early_stopping` epochs
            early_stopping = callbacks.EarlyStopping(monitor='val_loss',
                patience=self.early_stopping, verbose=0)
            model_check_point = callbacks.ModelCheckpoint(
                filepath="logs/%s" % os.getpid() + \
                    "-weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                monitor='val_loss', save_best_only=True, verbose=0)
            model.fit(data.trainX, data.trainY, epochs=self.epochs,
                batch_size=self.batch_size, verbose=0,
                # validation_data=(valX, valY), callbacks=callbacks)
                validation_data=(data.trainX, data.trainY),
                                 callbacks=[early_stopping, model_check_point])
        else:
            model.fit(data.trainX, data.trainY, epochs=self.epochs,
                batch_size=self.batch_size, verbose=0,
                # validation_data=(valX, valY))
                validation_data=(data.trainX, data.trainY))

        LOGGER.debug("Pid: %s: trained LSTM." % os.getpid())

        return model

    def rfe_fit(self, data):
        """Cannot do Recursive Feature Elimination
        """
        raise Exception("Cannot use RFE with %s" % self.learner)

    def forecast(self, model, testX):
        """Return forecasts for all horizons up to `horizon`
        """
        results = []
        lags = self.pc['lags']
        horizon = self.pc['horizon']

        for i in range(len(testX) - horizon + 1):
            for j in range(horizon):
                if j == 0:
                    buf = []
                    instance = testX[i+j]
                else:
                    # insert the buffer into instance
                    buf_len = len(buf)
                    if buf_len > lags:
                        buf = buf[-lags:]
                        buf_len = lags
                    instance = deepcopy(testX[i+j])
                    instance[-buf_len:, -1] = np.array(buf)
                pred_val = model.predict(instance[np.newaxis, :, :])[0][0]
                buf.append(pred_val)
                #LOGGER.debug("Predicting with instance %s, result %s" %
                #             (instance, pred_val))
            results.append(pred_val)

        return np.array(results).reshape(-1, 1)
