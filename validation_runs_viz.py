# -*- coding: utf-8 -*-
"""
Created on Wed May 23 14:52:06 2018

@author: vpekar
"""

import sys
import json
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
import settings


def compare_dicts(d1, d2, excepted_param):
    for k, v in d1.items():
        if k == excepted_param:
            continue
        if k not in d2 or v != d2[k]:
            return False
    return True


def chunks(alist, n):
    """Yield successive n-sized chunks from alist."""
    for i in range(0, len(alist), n):
        yield alist[i:i + n]


# select the test run:
learner = sys.argv[1]
assert learner in settings.__dict__
dataset = "ise"
score = 'mse'
plots_per_row = 3

test_results = json.load(open("results.json"))

# get the requested entry
entry = None
for result in test_results:
    if (result['learner'] == learner and
        dataset in result['preproc_config']['data_file']):
        entry = result

# build plots for every config parameter
print("Best config: %s" % entry['best_learner_config'])

for k in entry['feature_scores']:
    print("%s: %.3f" % tuple(k))

plots = []

for param, best_val in entry['best_learner_config'].items():
    param_vals = []
    scores = []

    for x in entry['validation_runs']:
        if compare_dicts(x['config'], entry['best_learner_config'], param):
            param_vals.append(x['config'][param])
            scores.append(x['scores'][score])

    df = pd.DataFrame({"Value": param_vals, "MSE": scores})

    if type(best_val) not in [int, float] or len(df['Value'].values) == 1:
        val_ranges = [str(x) for x in df['Value'].values]
        p = figure(x_range=val_ranges, title='%s, %s' % (learner, param), width=500, height=200)
        p.vbar(x=val_ranges, top=df['MSE'].values, width=0.75)
        p.y_range.start = min(df['MSE'].values) - min(df['MSE'].values)*0.2
    else:
        p = figure(title='%s, %s' % (learner, param), width=500, height=200)
        p.line(df['Value'].values, df['MSE'].values, line_width=1)
        p.circle(df['Value'].values, df['MSE'].values)
    p.xaxis.axis_label = 'parameter value'
    p.yaxis.axis_label = 'RMSE'
    plots.append(p)

grid = gridplot(list(chunks(plots, plots_per_row)))
show(grid)

