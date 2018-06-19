import sys
import os
import multiprocessing
import importlib
import json
import time

from collections import Counter

import settings
import numpy as np
np.random.seed(settings.PREPROCESSING['random_state'])

from utils import run_config_space, run_config
from get_logger import get_logger
from learner_configs import ConfigSpace


learner = sys.argv[1]
assert learner in settings.__dict__
LOGGER = get_logger('main', 'logs/%s.log' % learner)


def get_val_results(d, learner_config_space, pc):
    """Search for best parameters on the validation set
    :param pc: preprocessing config
    """

    mse_scores = Counter()
    results = {}
    if pc['n_jobs'] == 1:
        for c in learner_config_space.generate_config():
            x = run_config([d, c, 'val'])
            mse_scores[x.config_vals] = x.test_mse
            results[x.config_vals] = x
    else:
        inputs = iter([d, c, 'val']
                        for c in learner_config_space.generate_config())
        pool = multiprocessing.Pool(pc['n_jobs'])
        outputs = pool.imap(run_config, inputs)
        for x in outputs:
            mse_scores[x.config_vals] = x.test_mse
            results[x.config_vals] = x

    return mse_scores, results


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
