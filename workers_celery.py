# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 09:59:50 2018

@author: vpekar

Usage:

celery -A workers_celery worker

Purge scheduled tasks:

celery -A workers_celery purge
"""

import os
import celery
import logging
os.environ["FORKED_BY_MULTIPROCESSING"] = "1"

from utils import run_config


app = celery.Celery('workers_celery', broker='amqp://localhost//')
app.config_from_object('celeryconfig')

logging.getLogger("matplotlib").disabled = True
logging.getLogger("amqp").disabled = True


@app.task
def work(x):
    return run_config(x)
