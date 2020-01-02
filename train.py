# -*- coding: utf-8 -*-
"""
Run multiple configurations

Created on Mon May 21 19:03:56 2018

@author: user
"""

import settings
import importlib
import json
import time
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import run_config_space
from get_logger import get_logger
from learner_configs import ConfigSpace
from run import get_val_results


LOGGER = get_logger('main', 'logs/all-learners.log')
START = time.time()
TOTAL_RUNS = 0
N_RUNS = 0


def save(result):
    if os.path.exists("results.json"):
        all_results = json.load(open("results.json"))
    else:
        all_results = []
    all_results.append(result)
    out = open("results.json", "w")
    json.dump(all_results, out)
    out.close()


def log_time():
    global START, TOTAL_RUNS, N_RUNS
    m, s = divmod(time.time()-START, 60)
    h, m = divmod(m, 60)
    LOGGER.info("Completed %d of %d, %d hrs %02d min %02d sec" %
                (N_RUNS, TOTAL_RUNS, h, m, s))


def do_one_config(LearnerConfig, learner_config_settings, preproc_config):
    global N_RUNS
    config_space = ConfigSpace(LearnerConfig,
                               learner_config_settings,
                               preproc_config)

    result = run_config_space(preproc_config, config_space,
                              get_val_results)
    save(result)
    N_RUNS += 1
    log_time()


def get_learner_config(learner):

    if learner == "BiLSTM":
        learner = "LSTM"
        bidirectional = [True]
    else:
        bidirectional = [False]

    # set up parameter space for the learning method
    LearnerConfig = getattr(importlib.import_module("learner_configs"),
                           "Config%s" % learner)
    learner_config_settings = settings.__dict__[learner]

    if learner == "LSTM":
        learner_config_settings["bidirectional"] = bidirectional
    return LearnerConfig, learner_config_settings


def do_one_learner(learner):

    # parameters of data preprocessing
    preproc_config = settings.PREPROCESSING

    LearnerConfig, learner_config_settings = get_learner_config(learner)

    data_files = [
            "data/macroeconomy/macroeconomy.csv",
        ]

    horizons = [1]#[3, 7, 14]
    n_features_settings = [0]#[0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
    rfe_steps = [0]

    preproc_config['difference'] = 1
    preproc_config['deseason'] = 0
    preproc_config['date_format'] = '%d/%m/%Y'

    for data_file in data_files:
        print(data_file)
        preproc_config['data_file'] = data_file

        for rfe_step in rfe_steps:

            preproc_config['rfe_step'] = rfe_step

            for horizon in horizons:

                preproc_config['horizon'] = horizon

                # baseline
                preproc_config['use_exog'] = 0
                preproc_config['feature_selection'] = 0
                do_one_config(LearnerConfig, learner_config_settings,
                              preproc_config)

                # exogenous: feature selection
                preproc_config['use_exog'] = 1
                for n_features in n_features_settings:
                    preproc_config['feature_selection'] = n_features
                    do_one_config(LearnerConfig, learner_config_settings,
                                  preproc_config)


def main():
    # ['AdaBoost', 'GB', 'RFR', 'LSTM', 'BiLSTM', 'XGBoost', 'Lasso',
    # 'LSVR', 'SVRrbf', 'SVRsigmoid', 'SVRpoly', 'KNN', 'ElasticNet',
    # 'KernelRidge']
    learners = ['RFR']
    for learner in learners:
        do_one_learner(learner)


if __name__ == "__main__":

    main()
