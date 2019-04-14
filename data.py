import logging

from copy import deepcopy
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose


LOGGER = logging.getLogger('main.data')


def deseasonalize(train, test, n_periods):
    """Return deseasonalized train and test arrays
    """

    train_seasonal = seasonal_decompose(train, freq=n_periods).seasonal
    cycle = train_seasonal[:n_periods].tolist()

    train_tail = train.shape[0] % n_periods
    test_seasonal = cycle[train_tail:]

    n_test_cycles = int(((test.shape[0] - len(test_seasonal))/n_periods)) + 1
    test_seasonal = test_seasonal + cycle*n_test_cycles
    test_seasonal = np.array(test_seasonal[:test.shape[0]])

    return train - train_seasonal, test - test_seasonal


class Data:

    def __init__(self, df, config):
        """
        :param config: preprocessing config
        """

        # options
        # differenced values will not be reverted after testing
        if config['difference']:
            # both dependent and independent variables are differenced
            for x in df.columns:
                df[x] = self._difference(deepcopy(df[x])).ravel()

        dep_var_name = config['dep_var_name']

        self.use_exog = config['use_exog']
        self.lags = config['lags']
        self.test_split = config['test_split']
        self.horizon = config['horizon']
        self.y_scaler = None
        if config['scale_range'][0] != 0 or config['scale_range'][1] != 0:
            self.y_scaler = MinMaxScaler(feature_range=config['scale_range'])
        self.scale_range = config['scale_range']

        self.feature_selection = config['feature_selection']
        self.rfe_step = config['rfe_step']

        # train, validation and test positions in the dataset
        test_size = int((df.shape[0] + self.lags) * self.test_split)
        self.train_start = self.lags
        self.train_end = df.shape[0] - test_size*2
        self.val_start = self.train_end + self.lags
        self.val_end = self.val_start + test_size - self.lags
        self.test_start = self.val_end + self.lags
        self.test_end = df.shape[0]

        if config['deseason']:
            df[dep_var_name] = self._deseasonalize(df[dep_var_name],
                                                config['seasonal_period'])
        self.index = df.index

        # set original level Y's
        y_orig = deepcopy(df[dep_var_name])
        self.trainYref = y_orig[self.train_start+self.horizon-1:self.train_end].values
        self.valYref = y_orig[self.val_start+self.horizon-1:self.val_end].values
        self.testYref = y_orig[self.test_start+self.horizon-1:self.test_end].values

        # feature_names
        exog_names = deepcopy(df.columns).tolist()
        exog_names.remove(dep_var_name)
        self.exog_names = ["%s%s" % (x, i) for i in range(self.lags)
                           for x in exog_names]
        self.feature_names = []

        # data
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.valX = None
        self.valY = None

        self.endog_train = None
        self.exog_train = None
        self.endog_test = None
        self.exog_test = None
        self.endog_test = None
        self.exog_test = None

        self.preprocess(df)

    def __str__(self):
        return f"trainX {self.trainX.shape}, trainY {self.trainY.shape}, " +\
               f"valX {self.valX.shape}, valY {self.valY.shape}, " +\
               f"testX {self.testX.shape}, testY {self.testY.shape}"

    def scale(self):

        if self.y_scaler is None:
            return

        # y's
        self.trainY = self.y_scaler.fit_transform(
                self.trainY.reshape(-1, 1)).ravel()
        self.valY = self.y_scaler.transform(self.valY.reshape(-1, 1)).ravel()
        self.testY = self.y_scaler.transform(self.testY.reshape(-1, 1)).ravel()

        # x's
        trainxt = self.trainX.T
        valxt = self.valX.T
        testxt = self.testX.T
        for i in range(trainxt.shape[0]):
            x_scaler = MinMaxScaler(feature_range=self.scale_range)
            trainxt[i] = x_scaler.fit_transform(
                    trainxt[i].reshape(-1, 1)).reshape(-1,)
            valxt[i] = x_scaler.transform(
                    valxt[i].reshape(-1, 1)).reshape(-1,)
            testxt[i] = x_scaler.transform(
                    testxt[i].reshape(-1, 1)).reshape(-1,)

    def _difference(self, array):
        return np.append([0.0], np.diff(array.ravel()), axis=0).reshape(-1, 1)

    def _deseasonalize(self, y, seasonal_period):
        """Deseason the train, validation and test series
        """
        train, val = deseasonalize(y[:self.train_end],
                                   y[self.train_end:self.val_end],
                                   seasonal_period)
        dummy, test = deseasonalize(y[:self.val_end], y[self.val_end:],
                                    seasonal_period)
        return np.concatenate([train, val, test])

    def revert(self, yhat, mode="train", flat_list=False):
        """Take yhat (forecasted series) and turn it to levels.
        """
        reverted = deepcopy(yhat)

        # de-scale
        if self.y_scaler:
            reverted = self.y_scaler.inverse_transform(reverted.reshape(-1, 1))

        return reverted.flatten().tolist() if flat_list else reverted

    def attach_exogs(self, endogs, exogs):
        """Attach exog features to X
        """
        trainX2 = []
        for i, x in enumerate(endogs):
            # exogs are at the end of the lag period of the endogs
            exog = []
            for lag_i in range(self.lags, 0, -1):
                exog.extend(exogs[i+self.lags-lag_i])
            trainX2.append(np.concatenate([x, exog]))
        return np.array(trainX2)

    def create_ar_vars(self, dataset):
        """Create autoregressive X variables
        """
        dataX, dataY = [], []
        for i in range(len(dataset)-self.lags):
            dataX.append(dataset[i:i + self.lags, 0])
            dataY.append(dataset[i + self.lags, 0])
        return np.array(dataX), np.array(dataY)


class Data2d(Data):

    def split(self, df):
        """Set sizes of parts and split into train, validation and test
        """
        vals = df.values.astype('float32')
        train = vals[:self.train_end]
        val = vals[self.train_end:self.val_end]
        test = vals[self.val_end:]
        LOGGER.debug("input df shape %s" % str(df.shape))
        return train, val, test

    def preprocess(self, df):
        """Split into train and test for all methods except LSTM
        """
        train, val, test = self.split(df)

        # extract exogenous variables
        test_size = int((df.shape[0] + self.lags) * self.test_split)
        train_size = df.shape[0] - test_size*2
        self.endog_train = train[:, -1].reshape(train_size, 1)
        self.endog_test = test[:, -1].reshape(test_size, 1)
        self.endog_val = val[:, -1].reshape(test_size, 1)

        self.mean_endog_train = self.endog_train.mean()

        # create auto-regressive X variables
        self.trainX, self.trainY = self.create_ar_vars(self.endog_train)
        self.testX, self.testY = self.create_ar_vars(self.endog_test)
        self.valX, self.valY = self.create_ar_vars(self.endog_val)
        self.feature_names = ["lag%d" % x for x in range(self.lags, 0, -1)]

        # attach exogenous
        if self.use_exog:

            self.exog_train = train[:, :-1]
            self.exog_test = test[:, :-1]
            self.exog_val = val[:, :-1]

            self.trainX = self.attach_exogs(self.trainX, self.exog_train)
            self.testX = self.attach_exogs(self.testX, self.exog_test)
            self.valX = self.attach_exogs(self.valX, self.exog_val)
            self.feature_names.extend(self.exog_names)

        # scale all variables to [0, 1]
        self.scale()

        if self.feature_selection > 0 and self.rfe_step == 0:
            self.select_features()

        self.feature_names_orig = deepcopy(self.feature_names)

        self.feature_names_orig = deepcopy(self.feature_names)

    def pearson_r(self, x, y):
        c = [np.corrcoef(x[:, col_i], y)[0, 1] for col_i in range(x.shape[1])]
        c = np.abs(np.array(c))
        c[np.isnan(c)] = 0.0
        return c

    def select_features(self):
        """Select the most informative features, keeping all lag features
        :param df: input dataframe
        :param ratio: the number of features to select ([0, 1])
        """

        # train
        num_sel = int((self.trainX.shape[1] - self.lags) * self.feature_selection)
        if num_sel == 0:
            raise Exception(
                f"Feature_selection={self.feature_selection} removes all "+
                "features, review settings")
        LOGGER.debug(f"Will select {num_sel} exog features")

        scores = self.pearson_r(self.trainX[:, self.lags:], self.trainY)
        selected = Counter(dict(zip(self.feature_names[self.lags:], scores))
            ).most_common(num_sel)

        # index of columns to be deleted
        name2id = list(zip(self.feature_names[self.lags:],
                           range(self.lags, self.trainX.shape[1])))
        idx = [v for k, v in name2id if k not in dict(selected)]

        # delete de-selected columns
        self.trainX = np.delete(self.trainX, idx, 1)
        self.valX = np.delete(self.valX, idx, 1)
        self.testX = np.delete(self.testX, idx, 1)
        self.feature_names = self.feature_names[:self.lags] + \
            [k for k, v in name2id if k in dict(selected)]

        return


class Data3d(Data):

    def split(self, df):
        """Set sizes of parts and split into train, validation and test
        """
        vals = df.values.astype('float32')
        train = vals[self.train_start:self.train_end]
        val = vals[self.val_start:self.val_end]
        test = vals[self.test_start:self.test_end]
        return train, val, test

    def scale(self):

        if self.y_scaler is None:
            return

        # y's
        self.trainY = self.y_scaler.fit_transform(
                self.trainY.reshape(-1, 1)).ravel()
        self.valY = self.y_scaler.transform(
                self.valY.reshape(-1, 1)).ravel()
        self.testY = self.y_scaler.transform(
                self.testY.reshape(-1, 1)).ravel()

        # x's
        trainxt = self.trainX.T
        valxt = self.valX.T
        testxt = self.testX.T
        for i in range(trainxt.shape[0]):
            x_scaler = MinMaxScaler(feature_range=self.scale_range)
            trainxt[i] = x_scaler.fit_transform(
                    trainxt[i].ravel().reshape(-1, 1)).reshape(
                            trainxt.shape[1], trainxt.shape[2])
            valxt[i] = x_scaler.transform(
                    valxt[i].ravel().reshape(-1, 1)).reshape(
                            valxt.shape[1], valxt.shape[2])
            testxt[i] = x_scaler.transform(
                    testxt[i].ravel().reshape(-1, 1)).reshape(
                            testxt.shape[1], testxt.shape[2])

    def preprocess(self, df):
        """Split data into train and test for LSTM
        """

        df = self.series_to_supervised(df, n_in=self.lags)
        train, val, test = self.split(df)

        # split into X and Y
        self.trainX, self.trainY = train[:, :-1], train[:, -1]
        self.valX, self.valY = val[:, :-1], val[:, -1]
        self.testX, self.testY = test[:, :-1], test[:, -1]

        # reshape input to be 3D [samples, timesteps, features]
        self.trainX = self.trainX.reshape((self.trainX.shape[0], self.lags,
            int(self.trainX.shape[1]/self.lags)))
        self.valX = self.valX.reshape((self.valX.shape[0], self.lags,
            int(self.valX.shape[1]/self.lags)))
        self.testX = self.testX.reshape((self.testX.shape[0], self.lags,
            int(self.testX.shape[1]/self.lags)))

        # scale
        self.scale()

        return

    def series_to_supervised(self, data, n_in=1, n_out=1):
        """Creates a new df with each row being an instance.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        """

        if not self.use_exog:
            # delete exog values
            col_list = [x for x in range(data.shape[1]-1)]
            data.drop(data.columns[col_list], axis=1, inplace=True)
        else:
            # shift the exog values back by `intent_distance`
            data2 = []
            dfvals = data.values
            for i, x in enumerate(dfvals):
                if i >= 0:
                    exog = dfvals[i, :-1]
                else:
                    exog = [None]*(data.shape[1] - 1)
                data2.append(np.concatenate([exog, [x[-1]]]))
            data = np.array(data2)

        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()

        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names

        # drop unwanted columns
        start_to_del = n_vars * n_in
        end_to_del = n_vars * (n_in + 1) - 1

        col_list = [x for x in range(start_to_del, end_to_del)]
        agg.drop(agg.columns[col_list], axis=1, inplace=True)

        return agg
