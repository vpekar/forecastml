import pandas as pd
from datetime import datetime, timedelta


def get_preproc_config(lags=4, use_exog=False, intent_distance=0,
                       detrend=False, deseason=False, difference=False,
                       scale=[0, 1], horizon=7):
    adict = {
        "data_file": "infile",
        "date_format": "%Y-%m-%d",
        "test_split": 0.2,
        "difference": difference,
        "detrend": detrend,
        "deseason": deseason,
        "horizon": horizon,
        "use_exog": use_exog,
        "intent_distance": intent_distance,
        "lags": lags,
        "scale_range": scale,
        "n_jobs": 1,
        "random_state": 7
    }
    return adict


def get_date_list():
    base = datetime(2000, 1, 1)
    return [base+timedelta(days=x) for x in range(0, 40)]


def get_df():
    columns = ['dim0','dim1', 'dep_var']
    idx = get_date_list()
    data = [[10, 11, 12],
            [20, 21, 22],
            [30, 31, 32],
            [40, 41, 42],
            [50, 51, 52],
            [60, 61, 62],
            [70, 71, 72],
            [80, 81, 82],
            [90, 91, 92],
            [100, 101, 102],
            [110, 111, 112],
            [120, 121, 122],
            [130, 131, 132],
            [140, 141, 142],
            [150, 151, 152],
            [160, 161, 162],
            [170, 171, 172],
            [180, 181, 182],
            [190, 191, 192],
            [200, 201, 202],
            [210, 211, 212],
            [220, 221, 222],
            [230, 231, 232],
            [240, 241, 242],
            [250, 251, 252],
            [260, 261, 262],
            [270, 271, 272],
            [280, 281, 282],
            [290, 291, 292],
            [300, 301, 302],
            [310, 311, 312],
            [320, 321, 322],
            [330, 331, 332],
            [340, 341, 342],
            [350, 351, 352],
            [360, 361, 362],
            [370, 371, 372],
            [380, 381, 382],
            [390, 391, 392],
            [400, 401, 402]]
    df = pd.DataFrame(data=data, index=idx, columns=columns)
    return df
