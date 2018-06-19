# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 22:29:17 2018

Based on: http://mdup.fr/blog/easy-cluster-parallelization-with-zeromq

@author: vpekar
"""

import os
import sys
import time
import json
import zmq
import importlib

from collections import Counter

import settings

from utils import run_config_space
from get_logger import get_logger
from learner_configs import ConfigSpace


learner = sys.argv[1]
assert learner in settings.__dict__
LOGGER = get_logger('main', 'logs/run_zmq_%s.log' % learner)


def get_val_results(data, learner_config_space, pc):

    # Setup ZMQ.
    context = zmq.Context()
    sock = context.socket(zmq.REP)
    sock.bind(settings.ZMQ["master_address"])

    # How many calculations are expected?
    n_total = learner_config_space.num_configs

    # Generate the json messages for all computations.
    job_generator = generate_jobs(learner_config_space, data)

    # Loop until all results arrived.
    mse_scores = Counter()
    results = {}

    while len(results) < n_total:

        # Receive;
        response = sock.recv_pyobj()

        # First case: worker says "I'm available". Send him some work.
        if response['msg'] == "available":
            send_next_job(sock, job_generator)

        # Second case: worker says "Here's your result". Store it, say thanks.
        elif response['msg'] == "result":

            result = response['result']
            mse_scores[result.config_vals] = result.test_mse
            results[result.config_vals] = result

            if len(results) == n_total:
                sock.send(b"quit")
            else:
                sock.send(b"thanks")

    return mse_scores, results


def generate_jobs(learner_config_space, data):
    for c in learner_config_space.generate_config():
        yield [data, c, 'val']


def send_next_job(sock, job_generator):
    try:
        job = next(job_generator)
        LOGGER.debug("sending job %s" % job[1])
        sock.send_pyobj({"msg": "job", "data": job})
    except StopIteration:
        sock.send_pyobj({"msg": "quit"})


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
