# -*- coding: utf-8 -*-

"""Wrappers around GB and XGB estimators to make them usable with RFE
"""

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

from early_stopping import Monitor


class GBWrapper(GradientBoostingRegressor):

    num_train_instances = 0
    early_stopping = None

    def __init__(self, loss='ls', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None, random_state=None,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, presort='auto', validation_fraction=0.1,
                 n_iter_no_change=None, tol=1e-4,
                 early_stopping=None, num_train=0):
        self.early_stopping = early_stopping
        self.num_train = num_train
        super().__init__(loss, learning_rate, n_estimators, subsample,
             criterion, min_samples_split, min_samples_leaf,
             min_weight_fraction_leaf, max_depth, min_impurity_decrease,
             min_impurity_split, init, random_state, max_features, alpha,
             verbose, max_leaf_nodes, warm_start, presort, validation_fraction,
             n_iter_no_change, tol)

    def fit(self, x, y):
        if self.early_stopping:
            trainX, valX = x[:self.num_train, :], x[self.num_train:, :]
            trainY, valY = y[:self.num_train], y[self.num_train:]
            super().fit(trainX, trainY,
                  monitor=Monitor(valX.astype("float32"),
                                  valY.astype("float32"),
                                  max_consecutive_decreases=self.early_stopping))
        else:
            super().fit(x, y)


class XGBWrapper(XGBRegressor):

    num_train_instances = 0
    early_stopping = None

    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100,
                 silent=True, objective='reg:linear', booster='gbtree',
                 n_jobs=1, nthread=None, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1,
                 colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
                 scale_pos_weight=1, base_score=0.5, random_state=0,
                 seed=None, missing=None,
                 early_stopping=None, num_train=0):
        self.early_stopping = early_stopping
        self.num_train = num_train
        super().__init__(max_depth, learning_rate, n_estimators, silent,
             objective, booster, n_jobs, nthread, gamma, min_child_weight,
             max_delta_step, subsample, colsample_bytree, colsample_bylevel,
             reg_alpha, reg_lambda, scale_pos_weight, base_score, random_state,
             seed, missing)

    def fit(self, x, y):
        if self.early_stopping:
            trainX, valX = x[:self.num_train, :], x[self.num_train:, :]
            trainY, valY = y[:self.num_train], y[self.num_train:]
            eval_set = [[valX.astype("float32"),
                        valY.astype("float32")]]
            super().fit(trainX, trainY, eval_set=eval_set, eval_metric='rmse',
                      early_stopping_rounds=self.early_stopping, verbose=False)
        else:
            super().fit(x, y)
