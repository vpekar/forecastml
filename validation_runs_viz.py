# -*- coding: utf-8 -*-
"""

Create visualizations of validation runs stored in results.json.

python validation_runs_viz.py <RUN_NUMBER>

If RUN_NUMBER is omitted, the last run is used.

@author: vpekar
"""

import sys
import json
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot
from bokeh.io import output_file
import settings

SMOOTHING_FUNC_KIND = "quadratic" # None, slinear, quadratic, cubic


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


# select the test run
entry_num = int(sys.argv[1]) if len(sys.argv) > 1 else None
score = 'mse'
plots_per_row = 3

# get the requested entry
entry = None
results = json.load(open("results.json"))
if entry_num is not None:
    for i, result in enumerate(results):
        if entry_num == i + 1:
            entry = result
            break
else:
    entry = results[-1]

if not entry:
    raise("Could not find the required entry in results.json.")

learner_name = entry['learner']

output_file(f"{learner_name}, run {entry_num} validation.html",
            title=f"Run {entry_num}")

# build plots for every config parameter
print("Best config: %s" % entry['best_learner_config'])

for k in entry['feature_scores'][:5]:
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
        p = figure(x_range=val_ranges, title='%s, %s' % (learner_name, param),
                   width=500, height=200)
        p.vbar(x=val_ranges, top=df['MSE'].values, width=0.75)
        p.y_range.start = min(df['MSE'].values) - min(df['MSE'].values)*0.2
    else:
        p = figure(title='%s, %s' % (learner_name, param), width=500, height=200)
        x, y = df['Value'].values, df['MSE'].values

        # smooth
        if SMOOTHING_FUNC_KIND:
            smoothing_func = interp1d(x, y, kind=SMOOTHING_FUNC_KIND)
            xnew = np.linspace(x.min(), x.max(), num=50, endpoint=True)
            p.line(xnew, smoothing_func(xnew), line_width=1)
        else:
            p.line(x, y, line_width=1)

        p.circle(x, y)
    p.xaxis.axis_label = param
    p.yaxis.axis_label = 'RMSE'
    plots.append(p)

grid = gridplot(list(chunks(plots, plots_per_row)))
show(grid)

