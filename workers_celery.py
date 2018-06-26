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
import logging
os.environ["FORKED_BY_MULTIPROCESSING"] = "1"

from utils import run_config


app = celery.Celery('workers_celery', broker='amqp://localhost//')
app.config_from_object('celeryconfig')

logging.getLogger("matplotlib").disabled = True


@app.task
def work(x):
    start = time.time()
    print("Running config", file=sys.stderr)
    result = run_config(x)
    m, s = divmod(time.time()-start, 60)
    h, m = divmod(m, 60)
    print("Result: %s, took %d:%02d:%02d" % (result, h, m, s),
          file=sys.stderr)
    return result
