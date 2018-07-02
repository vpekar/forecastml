import logging
import numpy as np
import pandas as pd
import stldecompose

from copy import deepcopy
from collections import Counter

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_regression


LOGGER = logging.getLogger('main.data')


class Data:

    def __init__(self, df, config):
        """
        :param config: preprocessing config
        """

        # options
        self.y_orig = deepcopy(df['dep_var'])
        self.index = df.index.tolist()
        self.use_exog = config['use_exog']
        self.lags = config['lags']
        self.test_split = config['test_split']
        self.intent_distance = config['intent_distance']
        self.horizon = config['horizon']
        self.y_scaler = None
        if config['scale_range'][0] != 0 or config['scale_range'][1] != 0:
            self.y_scaler = MinMaxScaler(feature_range=config['scale_range'])
        self.scale_range = config['scale_range']
        self.do_difference = config['difference']
        self.do_detrend = config['detrend']
        self.do_deseason = config['deseason']
        self.seasonal_period = config['seasonal_period']

        self.stl_forecast = {}

        # train, validation and test positions in the dataset
        test_size = int((df.shape[0] + self.lags) * self.test_split)
        self.train_start = self.lags
        self.train_end = df.shape[0] - test_size*2
        self.val_start = self.train_end + self.lags
        self.val_end = self.val_start + test_size - self.lags
        self.test_start = self.val_end + self.lags
        self.test_end = df.shape[0]

        if config['feature_selection'] != 0:
            df = self.select_features(df, config['feature_selection'])

        # set original level Y's
        self.trainYref = self.y_orig[self.train_start+self.horizon-1:self.train_end]
        self.valYref = self.y_orig[self.val_start+self.horizon-1:self.val_end]
        self.testYref = self.y_orig[self.test_start+self.horizon-1:self.test_end]

        # feature_names
        self.exog_names = deepcopy(df.columns).tolist()
        self.exog_names.remove('dep_var')
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
        return "trainX %s, trainY %s, valX %s, valY %s, testX %s, testY %s" % (
            self.trainX.shape, self.trainY.shape, self.valX.shape,
            self.valY.shape, self.testX.shape, self.testY.shape)

    def select_features(self, df, ratio):
        """Select the most informative features
        :param df: input dataframe
        :param ratio: the number of features to select ([0, 1])
        """
        x_columns = df.columns[:-1]
        num_sel = int((len(x_columns)) * ratio)
        if num_sel == 0:
            raise Exception("The feature_selection setting will remove "+
                            "all features, review settings.py")

        LOGGER.debug("Will select %d features" % num_sel)
        scores = f_regression(df.values[:, :-1], df.values[:, -1], False)[0]
        selected = Counter(dict(zip(x_columns, scores))).most_common(num_sel)

        LOGGER.info("Selected features:")
        for feature, score in selected:
            LOGGER.info("%s\t%.6f" % (feature, score))

        # delete de-selected columns
        selected_names = [x for x, y in selected]
        for col in x_columns:
            if col not in selected_names:
                del df[col]

        return df

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

    def get_stl(self, series):
        shape_before = series.shape
        series = series.resample('D').mean()
        shape_after = series.shape
        if shape_before != shape_after:
            raise Exception("After resampling series, " +
                            "the shape changed. Missing dates in the data?")
        return stldecompose.decompose(series, period=self.seasonal_period)

    def decompose_is(self):
        """Extract residuals from STL and create an in-sample forecast as
        estimated seasonality and/or trend on train data
        """
        series = self.y_orig[:self.train_end]
        stl = self.get_stl(series)
        if self.do_deseason and self.do_detrend:
            extract = stl.trend + stl.seasonal
            resid = stl.resid
        elif self.do_deseason:
            extract = stl.seasonal
            resid = stl.resid + stl.trend
        else:
            extract = stl.trend
            resid = stl.resid + stl.seasonal
        self.stl_forecast['train'] = \
            extract[self.train_start+self.horizon-1:].values.\
            astype('float32').reshape(-1, 1)
        endog_train = resid.values.astype('float32').reshape(-1, 1)
        return endog_train

    def decompose_val(self):
        """Obtain the series until the end of validation"""
        start = self.train_end
        end = self.val_end
        endog_val, forecast = self._decompose(start, end)
        self.stl_forecast['val'] = forecast[self.lags+self.horizon-1:]
        return endog_val

    def decompose_oos(self):
        """Obtain the series until the start of test from data and y_orig
        """
        start = self.val_end
        end = self.test_end
        endog_test, forecast = self._decompose(start, end)
        self.stl_forecast['test'] = forecast[self.lags+self.horizon-1:]
        return endog_test

    def decompose(self):
        """Decompose all endogenous variables, return residuals as new
        endogenous variables; store relevant sections of trend+seasonal as
        forecasts
        """
        self.endog_train = self.decompose_is()
        self.endog_val = self.decompose_val()
        self.endog_test = self.decompose_oos()

    def _decompose(self, start, end):
        """Decomposition for val and test. Assuming future values of val (test)
        are not known, trend, seasonal component, and residual are created for
        every item in val/test added to previous items.
        :return resid: residuals of stl
        :return forecast: seasonal+trend of stl
        """
        residuals = []
        forecast = []
        for i in range(1, end - start + 1):
            series = self.y_orig[0:start + i]
            stl = self.get_stl(series)
            if self.do_deseason and self.do_detrend:
                extract = stl.trend + stl.seasonal
                resid = stl.resid
            elif self.do_deseason:
                extract = stl.seasonal
                resid = stl.resid + stl.trend
            else:
                extract = stl.trend
                resid = stl.resid + stl.seasonal
            residuals.append(resid[-1])
            forecast.append(extract[-1])
        return np.array(residuals).reshape(-1, 1), \
                np.array(forecast).reshape(-1, 1)

    def _difference(self, array):
        return np.append([0.0], np.diff(array.ravel()), axis=0).reshape(-1, 1)

    def revert(self, series, mode="train"):
        """Take yhat (forecasted series) and turn it to levels.
        """
        reverted = deepcopy(series)

        # de-scale
        if self.y_scaler:
            reverted = self.y_scaler.inverse_transform(reverted.reshape(-1, 1))

        # de-difference
        if self.do_difference:
            if mode == "train":
                start = self.train_start - 1
            elif mode == "val":
                start = self.val_start - 1
            else:
                start = self.test_start - 1
            reverted = self.y_orig.iloc[start] + reverted.cumsum()

        # add trend and seasonal component
        if self.do_detrend or self.do_deseason:
            reverted = reverted + self.stl_forecast[mode]

        return reverted

    def attach_exogs(self, endogs, exogs):
        """Attach exog features to X
        """
        trainX2 = []
        for i, x in enumerate(endogs):
            # exogs are at the end of the lag period of the endogs
            exog = exogs[i+self.lags-self.intent_distance]
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
        LOGGER.debug("df shape %s" % str(df.shape))
        LOGGER.debug("train shape %s" % str(train.shape))
        LOGGER.debug("val shape %s" % str(val.shape))
        LOGGER.debug("test shape %s" % str(test.shape))
        return train, val, test

    def difference(self):
        """Difference endogenous variables
        """
        self.endog_train = self._difference(self.endog_train)
        self.endog_val = self._difference(self.endog_val)
        self.endog_test = self._difference(self.endog_test)

    def preprocess(self, df):
        """Split into train and test for all methods except LSTM
        :param intent_distance: 0 means the nouns from the same day are used
            to forecast
        """
        train, val, test = self.split(df)

        # extract exogenous variables
        test_size = int((df.shape[0] + self.lags) * self.test_split)
        train_size = df.shape[0] - test_size*2
        self.endog_train = train[:, -1].reshape(train_size, 1)
        self.endog_test = test[:, -1].reshape(test_size, 1)
        self.endog_val = val[:, -1].reshape(test_size, 1)

        # stl-decompose endogenous variables
        if self.do_deseason or self.do_detrend:
            self.decompose()

        # difference endogenous variables
        if self.do_difference:
            self.difference()

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

    def difference(self, df):
        """Difference the last column in the dataframe
        """
        df[df.columns[-1]] = self._difference(df[df.columns[-1]])
        return df

    def decompose(self, df):
        """Decompose all endogenous variables, return residuals as new
        endogenous variables
        """
        endog_train = self.decompose_is()
        endog_val = self.decompose_val()
        endog_test = self.decompose_oos()
        # put endog_train, endog_val and endog_test into df
        df[df.columns[-1]] = np.concatenate(
                [endog_train, endog_val, endog_test], axis=0)
        return df

    def preprocess(self, df):
        """Split data into train and test for LSTM
        """

        # decompose
        if self.do_deseason or self.do_detrend:
            df = self.decompose(df)

        # difference
        if self.do_difference:
            df = self.difference(df)

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
                if i >= self.intent_distance:
                    exog = dfvals[i-self.intent_distance, :-1]
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
