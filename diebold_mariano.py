"""Given results of two runs, test the difference between them using
Diebold-Mariano test

The script require dm_test.py from [here](https://github.com/johntwk/Diebold-Mariano-Test) to be placed n nthe same folder
"""

import importlib
import logging
from learner_configs import ConfigLSTM
from dm_test import dm_test
from utils import run_config, prepare_data


HORIZON = 1

LOGGER = logging.getLogger('dmtest')


def get_forecasts(learner, c, pc):

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

    return data.testYref.flatten(), result_test.yhat_oos.flatten()


run1 = ('RFR', 'autoreg', 'autoreg', {'n_estimators': 50, 'max_features': 0.6, 'max_depth': 50, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_leaf_nodes': None}, {'data_file': 'data/3day.1000.all.facebook.csv', 'date_format': '%d/%m/%Y', 'test_split': 0.2, 'difference': 1, 'deseason': 0, 'seasonal_period': 7, 'horizon': HORIZON, 'feature_selection': 0, 'rfe_step': 0, 'use_exog': 0, 'lags': 7, 'scale_range': [0, 1], 'n_jobs': 2, 'random_state': 7})
run2 = ('RFR', 'fb100', 'mcgc', {'n_estimators': 500, 'max_features': 0.6, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 5, 'max_leaf_nodes': None}, {'data_file': 'data/3day.majorclust-gc.facebook.csv', 'date_format': '%d/%m/%Y', 'test_split': 0.2, 'difference': 1, 'deseason': 0, 'seasonal_period': 7, 'horizon': 1, 'feature_selection': 0, 'rfe_step': 0, 'use_exog': 1, 'lags': 7, 'scale_range': [0, 1], 'n_jobs': 2, 'random_state': 7})

actual1, pred1 = get_forecasts(run1[0], run1[3], run1[4])
actual2, pred2 = get_forecasts(run2[0], run2[3], run2[4])

pval = dm_test(actual1, pred1, pred2, h=HORIZON, crit="MSE").p_value

print(f"P value: {pval}")
