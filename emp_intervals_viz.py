# -*- coding: utf-8 -*-
"""Change random seeds, run a config setting multiple times, and plot
confidence intervals as well as traces
"""

import json

import numpy as np
import pandas as pd

import importlib
import statsmodels.stats.api as sms
import settings
from utils import run_config, prepare_data
from learner_configs import ConfigLSTM

from bokeh.plotting import figure, show, save, output_file
from bokeh.models import Range1d
from bokeh.palettes import all_palettes
from bokeh.layouts import gridplot
from bokeh.models.formatters import DatetimeTickFormatter


# define the test run:
PLOTS_PER_ROW = 1
NUM_RUNS = 50
ALPHA = 0.05
HORIZON = 1


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
    bg_color = 'white'# 'beige'
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


experiment_configs = [
    ('AdaBoost', 'autoreg', 'autoreg', {'n_estimators': 15, 'learning_rate': 1.25, 'loss': 'exponential'}, {'data_file': 'examples/ise.csv', 'date_format': '%d/%m/%Y', 'test_split': 0.2, 'difference': 1,'detrend': False, 'deseason': False, 'seasonal_period': 7, 'horizon': HORIZON, 'feature_selection': 0, 'rfe_step': 0, 'use_exog': 0, 'lags': 7, 'scale_range': [0, 1], 'n_jobs': 2, 'random_state': 7, 'dep_var_name': 'dep_var'}),
]

for LEARNER, CORPUS, DATASET, config_settings, pc in experiment_configs:

    assert LEARNER in settings.__dict__

    print(f"{LEARNER} {CORPUS} {DATASET}")

    # create config instances
    LearnerConfig = getattr(importlib.import_module("learner_configs"),
                               "Config%s" % LEARNER)

    # load data
    dim = "3d" if LearnerConfig is ConfigLSTM else "2d"
    data = prepare_data(pc, dim=dim)

    runs_is = []
    runs_val = []
    runs_oos = []

    scores = {
        'train': {
            'mse': {'obs': []},
            'mae': {'obs': []},
            'mape': {'obs': []},
        },
        'val': {
            'mse': {'obs': []},
            'mae': {'obs': []},
            'mape': {'obs': []},
        },
        'test': {
            'mse': {'obs': []},
            'mae': {'obs': []},
            'mape': {'obs': []},
        }
    }

    for random_state in range(NUM_RUNS):

        if random_state%10 == 0:
            print("%.2f percent complete" % (100*random_state/NUM_RUNS))

        pc['random_state'] = random_state
        learner_config = LearnerConfig(config_settings, pc)

        # run on validation set
        result_val = run_config([data, learner_config, 'val'])
        yhat_val = data.revert(result_val.yhat_oos, 'val').flatten().tolist()

        # run on test set
        result_test = run_config([data, learner_config, 'test'])
        yhat_oos = data.revert(result_test.yhat_oos, 'test').flatten().tolist()
        yhat_is = data.revert(result_test.yhat_is, 'train').flatten().tolist()

        runs_is.append(yhat_is)
        runs_val.append(yhat_val)
        runs_oos.append(yhat_oos)

        scores['train']['mse']['obs'].append(result_test.train_mse)
        scores['val']['mse']['obs'].append(result_val.test_mse)
        scores['test']['mse']['obs'].append(result_test.test_mse)
        scores['train']['mae']['obs'].append(result_test.train_mae)
        scores['val']['mae']['obs'].append(result_val.test_mae)
        scores['test']['mae']['obs'].append(result_test.test_mae)
        scores['train']['mape']['obs'].append(result_test.train_mape)
        scores['val']['mape']['obs'].append(result_val.test_mape)
        scores['test']['mape']['obs'].append(result_test.test_mape)

        print(f"Run {random_state}: {result_test}")

    print("100.00% percent complete")

    # calculate mean and stdev of the scores
    for set_name in ['train', 'val', 'test']:
        for measure in ['mse', 'mae', 'mape']:
            a = np.array(scores[set_name][measure]['obs'])
            scores[set_name][measure]['mean'] = a.mean()
            scores[set_name][measure]['std'] = a.std()

    # plots with confidence bands
    plots = [
        create_plot(runs_is, data.trainYref,
                        data.index[data.train_start+HORIZON-1:data.train_end],
                        title="In-sample, %.2f confidence bands" % (1.-ALPHA),
                        kind="ci"),
        create_plot(runs_val, data.valYref,
                        data.index[data.val_start+HORIZON-1:data.val_end],
                        title="Validation, %.2f confidence bands" % (1.-ALPHA),
                        kind="ci"),
        create_plot(runs_oos, data.testYref,
                        data.index[data.test_start+HORIZON-1:data.test_end],
                        title="Out-of-sample, %.2f confidence bands" % (1.-ALPHA),
                        kind="ci"),
        create_plot(runs_is, data.trainYref,
                        data.index[data.train_start+HORIZON-1:data.train_end],
                        title="In-sample, traces", kind="t"),
        create_plot(runs_val, data.valYref,
                        data.index[data.val_start+HORIZON-1:data.val_end],
                        title="Validation, traces", kind="t"),
        create_plot(runs_oos, data.testYref,
                        data.index[data.test_start+HORIZON-1:data.test_end],
                        title="Out-of-sample, traces", kind="t")
    ]

    gp = gridplot(list(chunks(plots, PLOTS_PER_ROW)))
    #show(gp)

    # save html file
    output_file("results-h=%d/performance-scores--%s-%s-%s.html" % (HORIZON,
        CORPUS, DATASET, LEARNER))
    save(gp)

    # dump performance scores
    out = open("results-h=%d/performance-scores--%s-%s-%s.json" % (HORIZON,
        CORPUS, DATASET, LEARNER), "w")
    json.dump(scores, out)
    out.close()

    # print the results in the tab-separated format:
    # AB PurInt Mean [6 means]
    # tab tab Std [6 Stds]
    out = open("out.txt", "a")
    print("%s-%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (CORPUS, LEARNER, DATASET,
        scores['train']['mse']['mean'], scores['val']['mse']['mean'], scores['test']['mse']['mean'],
        scores['train']['mae']['mean'], scores['val']['mae']['mean'], scores['test']['mae']['mean']), file=out)
    print("\t\t%s\t%s\t%s\t%s\t%s\t%s" % (
        scores['train']['mse']['std'], scores['val']['mse']['std'], scores['test']['mse']['std'],
        scores['train']['mae']['std'], scores['val']['mae']['std'], scores['test']['mae']['std']), file=out)
    out.close()
