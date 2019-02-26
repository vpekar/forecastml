"""Given results of two runs, test the difference between them using
Diebold-Mariano test

The script require dm_test.py from [here](https://github.com/johntwk/Diebold-Mariano-Test) to be placed n nthe same folder
"""

import numpy as np
import importlib
import logging

from sklearn.metrics import mean_squared_error
from learner_configs import ConfigLSTM
from dm_test import dm_test
from utils import run_config, prepare_data


HORIZON = 1

LOGGER = logging.getLogger('dmtest')


def get_forecasts(learner, dataset_name, prepr_name, c, pc):

    # create config instances
    LearnerConfig = getattr(importlib.import_module("learner_configs"),
                               "Config%s" % learner)

    # load data
    dim = "3d" if LearnerConfig is ConfigLSTM else "2d"
    data = prepare_data(pc, dim=dim)
    LOGGER.debug("Prepared data")
    learner_config = LearnerConfig(c, pc)
    result_test = run_config([data, learner_config, 'test'])
    LOGGER.debug(f"Ran config: {c}")
    actual = data.testYref
    pred = data.revert(result_test.yhat_oos, "test")
    rmse = np.sqrt(mean_squared_error(actual, pred))
    print(f"{learner}, {dataset_name}, {prepr_name}, RMSE: {rmse:.3f}")

    return actual.flatten(), pred.flatten()


run1 = ('RFR', 'autoreg', 'autoreg', {'n_estimators': 50, 'max_features': 0.6, 'max_depth': 50, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_leaf_nodes': None}, {'data_file': 'data/3day.1000.all.facebook.csv', 'date_format': '%d/%m/%Y', 'test_split': 0.2, 'difference': 1, 'deseason': 0, 'seasonal_period': 7, 'horizon': HORIZON, 'feature_selection': 0, 'rfe_step': 0, 'use_exog': 0, 'lags': 7, 'scale_range': [0, 1], 'n_jobs': 2, 'random_state': 7})
run2 = ('RFR', 'tw100', 'mcgc', {'n_estimators': 25, 'max_features': 0.8, 'max_depth': 5, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_leaf_nodes': None}, {'data_file': 'data/3day.majorclust-gc.twitter.short.csv', 'date_format': '%d/%m/%Y', 'test_split': 0.2, 'difference': 1, 'deseason': 0, 'seasonal_period': 7, 'horizon': 1, 'feature_selection': 0, 'rfe_step': 0, 'use_exog': 1, 'lags': 7, 'scale_range': [0, 1], 'n_jobs': 2, 'random_state': 7})

actual1, pred1 = get_forecasts(*run1)
actual2, pred2 = get_forecasts(*run2)

pval = dm_test(actual1, pred1, pred2, h=HORIZON, crit="MSE").p_value

print(f"P value: {pval}")
