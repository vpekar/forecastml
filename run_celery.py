# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 09:59:32 2018

@author: vpekar
"""

import os
import sys
import json
import importlib
import time
import logging

from celery import group
from collections import Counter

import settings

from workers_celery import work
from utils import run_config_space
from get_logger import get_logger
from learner_configs import ConfigSpace


learner = sys.argv[1]
assert learner in settings.__dict__
LOGGER = get_logger('main', 'logs/run_celery_%s.log' % learner)

logging.getLogger("matplotlib").disabled = True


def get_val_results(data, learner_config_space, pc):

    mse_scores = Counter()
    results = {}

    for result in group(work.s(x) for x in generate_jobs(
                                    learner_config_space, data))().get():
        LOGGER.debug("Got worker result: %s" % result)
        mse_scores[result.config_vals] = result.test_mse
        results[result.config_vals] = result

    return mse_scores, results


def generate_jobs(learner_config_space, data):
    for c in learner_config_space.generate_config():
        yield [data, c, 'val']


def main():

    global learner

    start = time.time()

    # parameters of data preprocessing
    preproc_config = settings.PREPROCESSING

    # set up parameter space for the learning method
    LearnerConfig = getattr(importlib.import_module("learner_configs"),
                           "Config%s" % learner)
    learner_config_settings = settings.__dict__[learner]
    config_space = ConfigSpace(LearnerConfig, learner_config_settings,
                               preproc_config)

    # train and test
    test_result = run_config_space(preproc_config, config_space,
                                   get_val_results)

    if os.path.exists("results.json"):
        all_results = json.load(open("results.json"))
    else:
        all_results = []
    all_results.append(test_result)
    out = open("results.json", "w")
    json.dump(all_results, out)
    out.close()

    m, s = divmod(time.time()-start, 60)
    h, m = divmod(m, 60)
    LOGGER.info("Took %d hours %02d minutes %02d seconds" % (h, m, s))


if __name__ == "__main__":

    main()
