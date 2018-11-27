PREPROCESSING = {
        "data_file": "examples/AirQualityUCI.csv", # 'examples/ise.csv',
        "date_format": "%d/%m/%Y %H:%M:%S", # "%d-%b-%y",
        "test_split": 0.2,
        "difference": 0,
        "predifference": 0,
        "detrend": 0,
        "deseason": 0,
        "seasonal_period": 7,
        "horizon": 7,
        "feature_selection": 0,
        "rfe_step": 0,
        "use_exog": 0,
        "lags": 7,
        "scale_range": [0, 1],
        "n_jobs": 2,
        "random_state": 7
        }


ZMQ = {
       "master_address": "tcp://127.0.0.1:5557"
       }

GB = {
        "n_estimators": [200, 300, 500, 700, 1000, 1500, 2000],
        "learning_rate": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
        "loss": ['ls', 'lad', 'huber', 'quantile'],
        "max_features": [0.05, 0.1, 0.2, 0.6], #["auto", "sqrt", 0.1, 0.7],
        "max_depth": [3, 5, 10],
        "min_samples_split": [5],#[2, 5, 10, 20],
        "min_samples_leaf": [5], # [1, 5, 10],
        "max_leaf_nodes": [None], #[None, 3, 10],
        "min_weight_fraction_leaf": [0.0],
        "subsample": [0.7], #[1.0, 0.8],
        "alpha": [0.7],
        "warm_start": [True], #[True, False]
        "early_stopping": [3, 5, 7, 10]
        }

XGBoost = {
        "max_depth": [3],
        "learning_rate": [0.1],
        "n_estimators": [100],
        "objective": ["reg:linear"], # "reg:logistic", "binary:logistic"
        "booster": ["gbtree"], # gbtree, gblinear, dart
        "n_jobs": [1],
        "reg_alpha": [0], # 0.01
        }

AdaBoost = {
        "n_estimators": [5, 10, 15, 25, 50, 75, 100, 125, 150, 200],
        "learning_rate": [0.01, 0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        "loss": ['linear', 'square', 'exponential'],
        }

SVR = {
        "kernel": ["linear", "rbf", "sigmoid", "poly"],
        "degree": [1, 2, 3, 4, 5, 7, 10],
        "c": [0.001, 0.003, 0.005, 0.007, 0.01, 0.05, 0.1, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 9.0],
        "eps": [0.05, 0.07, 0.1, 0.15, 0.17, 0.25, 0.5]
        }

RFR = {
        "n_estimators": [15, 25, 50, 100],
        "max_features": ["auto", "sqrt", 0.05, 0.1, 0.2, 0.4, 0.6],
        "max_depth": [None, 5, 10, 50],
        "min_samples_split": [2, 3, 5, 10],
        "min_samples_leaf": [1, 3, 5, 10],
        "max_leaf_nodes": [None, 3, 5, 10]
        }

LSTM = {
        "bidirectional": [False, True],
        "topology": [(7, 3, 1)], #, (25, 25, 1)],#, (7, 14, 1)],
        "epochs": [2000],#, 5000],#, 3000],
        "batch_size": [2],#, 10],#, 5, 10],
        "activation": ["sigmoid", "relu", "tanh", "softmax"],#, None
        "dropout_rate": [0],#, 0.2, 0.4],# [0, 0.2, 0.4],
        "optimizer": ["adam"], # sgd, rmsprop, adagrad, adadelta, adamax, nadam
        "kernel_regularizer": [(0.0, 0.0)],#, (0.01, 0.0), (0.0, 0.01), (0.01, 0.01)],
        "bias_regularization": [(0.0, 0.0)],#, (0.01, 0.0), (0.0, 0.01), (0.01, 0.01)],
        "early_stopping": [3],#, 5],#, 2, 3],
        "stateful": [False] # True
        }
