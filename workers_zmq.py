# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 22:28:44 2018

Based on http://mdup.fr/blog/easy-cluster-parallelization-with-zeromq

@author: vpekar
"""

import sys
import zmq
from multiprocessing import Process

import settings
from get_logger import get_logger
from utils import run_config


learner = sys.argv[1]
assert learner in settings.__dict__
LOGGER = get_logger('main', 'logs/workers_zmq_%s.log' % learner)


def slave(worker_id):

    import logging
    logging.getLogger("matplotlib").disabled = True

    # Setup ZMQ.
    context = zmq.Context()
    sock = context.socket(zmq.REQ)
    sock.connect(settings.ZMQ["master_address"])

    while True:

        LOGGER.debug("%s: Available" % worker_id)
        sock.send_pyobj({"msg": "available"})

        # Retrieve work and run the computation.
        job = sock.recv_pyobj()
        if job.get("msg") == "quit":
            LOGGER.debug("%s: Received a quit msg, exiting" % worker_id)
            break

        LOGGER.debug("%s: Running config %s" % (worker_id, job["data"][1]))
        result = run_config(job["data"])

        LOGGER.debug("%s: Sending result back" % worker_id)
        sock.send_pyobj({"msg": "result", "result": result})
        LOGGER.debug("%s: Done sending result" % worker_id)

        msg = sock.recv()
        if msg == b"quit":
            LOGGER.debug("%s Received msg %s" % (worker_id, msg))
            break


if __name__ == "__main__":

    # Create a pool of workers to distribute work to
    for _id in range(settings.PREPROCESSING['n_jobs']):
        Process(target=slave, args=(_id,)).start()
