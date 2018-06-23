# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 09:59:50 2018

@author: vpekar

Usage:

celery -A workers_celery worker

Purge scheduled tasks:

celery -A workers_celery purge
"""

import sys
import os
import celery
os.environ["FORKED_BY_MULTIPROCESSING"] = "1"

import settings
from get_logger import get_logger
from utils import run_config


learner = sys.argv[-1]
assert learner in settings.__dict__
LOGGER = get_logger('main', 'logs/workers_celery_%s.log' % learner)

app = celery.Celery('workers_celery', broker='amqp://localhost//')
app.config_from_object('celeryconfig')


@app.task
def work(x):
    LOGGER.debug("Running worker")
    result = run_config(x)
    LOGGER.debug("Done")
    return result
