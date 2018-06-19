# -*- coding: utf-8 -*-
"""Change random seeds, run a config setting multiple times, and plot
confidence intervals as well as traces
"""

import sys
import json

import numpy as np
import pandas as pd

import importlib
import statsmodels.stats.api as sms
import settings
from utils import run_config, prepare_data
from learner_configs import ConfigLSTM

from bokeh.plotting import figure, show
from bokeh.models import Range1d
from bokeh.palettes import all_palettes
from bokeh.layouts import gridplot
from bokeh.models.formatters import DatetimeTickFormatter


# define the test run:
LEARNER = sys.argv[1]
assert LEARNER in settings.__dict__
DATASET = "ise"
SCORE = 'mse'
PLOTS_PER_ROW = 1
NUM_RUNS = 50
ALPHA = 0.05


def extract_result_record():
    """Get the requested results entry
    """
    test_results = json.load(open("results.json"))
    for result in test_results:
        if (result['learner'] == LEARNER and
            DATASET in result['preproc_config']['data_file']):
            return result


def chunks(alist, n):
    """Yield successive n-sized chunks from alist."""
    for i in range(0, len(alist), n):
        yield alist[i:i + n]


def create_plot(runs, y, idx, title, kind="ci"):
    """
    :param kind: "ci" (confidence bands) or "t" (traces)
    """

    mean = np.array(runs).mean(axis=0)
    idx = np.array(idx)
    yhat = pd.Series(mean, index=idx).values

    palette = all_palettes['Viridis'][11]
    bg_color = 'beige'
    y_color = palette[3]
    yhat_color = palette[6]

    p = figure(x_axis_type="datetime", title=title)
    p.grid.grid_line_alpha=0.7
    p.xaxis.axis_label = 'Date'
    p.xaxis.axis_label_text_font_size = "10pt"
    p.xaxis[0].formatter = DatetimeTickFormatter(days = ["%d %b %Y"])
    p.yaxis.axis_label = 'CSI'
    p.yaxis.axis_label_text_font_size = "10pt"
    min_y = min(y.min(), yhat.min())
    max_y = max(y.max(), yhat.max())
    p.y_range = Range1d(min_y*0.95, max_y*1.05)

    p.background_fill_color = bg_color
    p.background_fill_alpha = 0.7

    p.line(idx, y, line_width=1.5, color=y_color, legend='Actual')
    p.line(idx, yhat, line_width=1.5, color=yhat_color, legend='Predicted')

    if kind == "ci":
        # confidence bands
        lower, upper = sms.DescrStatsW(np.array(runs)).tconfint_mean(alpha=ALPHA)
        band_x = np.append(idx, idx[::-1])
        band_y = np.append(lower, upper[::-1])
        p.patch(band_x, band_y, color=yhat_color, fill_alpha=0.2)
    else:
        # traces for each run
        for run in runs:
            p.line(idx, run, line_width=1, color=yhat_color, line_alpha=0.3)

    p.legend.location = "top_left"
    p.plot_height = 350
    p.plot_width = 950

    return p

runs_is = []
runs_val = []
runs_oos = []
result_record = extract_result_record()

# create config instances
LearnerConfig = getattr(importlib.import_module("learner_configs"),
                           "Config%s" % LEARNER)
config_settings = result_record['best_learner_config']
preproc_config = result_record['preproc_config']

# load data
dim = "3d" if LearnerConfig is ConfigLSTM else "2d"
data = prepare_data(preproc_config, dim=dim)

for num_run, random_state in enumerate(range(NUM_RUNS)):

    if num_run%10 == 0:
        print("%.2f percent complete" % (100*num_run/NUM_RUNS))

    preproc_config['random_state'] = random_state
    learner_config = LearnerConfig(config_settings, preproc_config)

    # run on validation set
    result = run_config([data, learner_config, 'val'])
    yhat_val = data.revert(result.yhat_oos, 'val').flatten().tolist()

    # run on test set
    result = run_config([data, learner_config, 'test'])
    yhat_oos = data.revert(result.yhat_oos, 'test').flatten().tolist()
    yhat_is = data.revert(result.yhat_is, 'train').flatten().tolist()

    runs_is.append(yhat_is)
    runs_val.append(yhat_val)
    runs_oos.append(yhat_oos)
    print("Run %d: %s" % (num_run, result))

print("100.00% percent complete")

# plots with confidence bands

plots = [
    create_plot(runs_is, data.trainYref,
                    data.index[data.train_start:data.train_end],
                    title="In-sample, %.2f confidence bands" % (1.-ALPHA),
                    kind="ci"),
    create_plot(runs_val, data.valYref,
                    data.index[data.val_start:data.val_end],
                    title="Validation, %.2f confidence bands" % (1.-ALPHA),
                    kind="ci"),
    create_plot(runs_oos, data.testYref,
                    data.index[data.test_start:data.test_end],
                    title="Out-of-sample, %.2f confidence bands" % (1.-ALPHA),
                    kind="ci"),
    create_plot(runs_is, data.trainYref,
                    data.index[data.train_start:data.train_end],
                    title="In-sample, traces", kind="t"),
    create_plot(runs_val, data.valYref,
                    data.index[data.val_start:data.val_end],
                    title="Validation, traces", kind="t"),
    create_plot(runs_oos, data.testYref,
                    data.index[data.test_start:data.test_end],
                    title="Out-of-sample, traces", kind="t")
]

show(gridplot(list(chunks(plots, PLOTS_PER_ROW))))
