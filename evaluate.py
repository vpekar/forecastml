# -*- coding: utf-8 -*-
"""
Runs a selected model on the test set

python evaluate.py <RUN_NUMBER>

If RUN_NUMBER is omitted, the last run is used.

@author: vpekar
"""

import sys
import json
import importlib

from utils import prepare_data, run_config


def get_entry(num, results):
    # get the requested entry
    if num is None:
        return results[-1]

    entry = None
    for i, result in enumerate(results):
        if entry_num == i + 1:
            entry = result
            break

    if not entry:
        raise("Could not find the required entry in the results. Check criteria?")
    return entry


# select the run, arg1 the number of the entry, if missing use last
entry_num = int(sys.argv[1]) if len(sys.argv) > 1 else None

results = json.load(open("results.json"))
entry = get_entry(entry_num, results)

LearnerConfig = getattr(importlib.import_module("learner_configs"),
                           "Config%s" % entry["learner"])

best_config = LearnerConfig(entry['best_learner_config'], entry['preproc_config'])

print(f"{entry['learner']}: best config: {entry['best_learner_config']}")

dim = dim = "3d" if entry['learner'] == "LSTM" else "2d"
data = prepare_data(entry['preproc_config'], dim=dim)
res = run_config([data, best_config, 'test'])

print('\t'.join(str(x) for x in [
    {'date': entry['date'],
     'best_learner_config': entry['best_learner_config'],
     'preproc_config': entry['preproc_config']},
      entry['mse']['train']['mean'],
      entry['mae']['train']['mean'],
      entry['mse']['val']['mean'],
      entry['mae']['val']['mean'],
      res.test_mse,
      res.test_mae
]))

# print feature scores
for k in entry['feature_scores'][:5]:
    print("%s: %.3f" % tuple(k))
