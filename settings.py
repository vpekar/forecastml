PREPROCESSING = {
        "data_file": "data/3day.1000.w2v.50d.twitter.csv",
        "date_format": "%d/%m/%Y",# "%Y-%m-%d",
        "test_split": 0.2,
        "difference": 0,
        "deseason": 0,
        "seasonal_period": 7,
        "log_dep_var": False,
        "log_indep_var": False,
        "horizon": 7,
        "feature_selection": 0,
        "rfe_step": 0,
        "use_exog": 0,
        "lags": 7,
        "scaler_name": "standard", # minmax
        "scale_range": [0, 1],
        "n_jobs": 1,
        "freq_threshold": 0,
        "dep_var_name": "dep_var",
        "num_random_seeds": 10,
        "random_state": None
        }

ZMQ = {
       "master_address": "tcp://127.0.0.1:5557"
       }

GB = {
        "n_estimators": [200, 300, 500, 3000], #[200, 300, 1000], #
        "learning_rate": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],#, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0],
        "loss": ['ls', 'huber'], # , 'quantile', 'lad',
        "max_features": [0.05, 0.1, 0.2, 0.6, 0.8], #["auto", "sqrt", 0.1, 0.7],
        "max_depth": [5, 10, 20],
        "min_samples_split": [5],#[2, 5, 10, 20],
        "min_samples_leaf": [5], # [1, 5, 10],
        "max_leaf_nodes": [None], #[None, 3, 10],
        "min_weight_fraction_leaf": [0.0],
        "subsample": [0.7], #[1.0, 0.8],
        "alpha": [0.7],
        "warm_start": [True], #[True, False]
        "early_stopping": [3, 5],#, 10]
        }

XGBoost = {
        "max_depth": [3, 5, 10, 15, 20],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [10, 100, 200, 300],
        "objective": ["reg:linear"], # "reg:logistic", "binary:logistic"
        "booster": ["gbtree"], # gbtree, gblinear, dart
        "n_jobs": [1],
        "reg_alpha": [0, 0.01, 0.1], # 0.01
        "early_stopping": [3, 5, 7] #[3, 5],#, 10]
        }

AdaBoost = {
        "n_estimators": [10, 15, 25, 50, 75, 100, 200], #[200, 300, 1000],#[5, 10, 15, 25, ],
        "learning_rate": [0.01, 0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        "loss": ['linear', 'square', 'exponential'],
        }

LSVR = {
        "c": [0.001, 0.005, 0.01, 0.05, 0.1, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0],#, 20.0, 50.0, 100.0],
        "eps": [0.001, 0.01, 0.05, 0.07, 0.1, 0.15, 0.17, 0.25, 0.5, 1.0],
        "tol": [0.00001, 0.0001, 0.001, 0.01, 0.1],
        "max_iter": [1000],
        "dual": [True],
        }

SVRsigmoid = {
        "c": [0.001, 0.005, 0.01, 0.05, 0.1, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0],#, 20.0, 50.0, 100.0],
        "eps": [0.001, 0.01, 0.05, 0.07, 0.1, 0.15, 0.17, 0.25, 0.5, 1.0],
        "tol": [0.00001, 0.0001, 0.001, 0.01, 0.1],
        "max_iter": [-1],
        "dual": [True, False],
        "coef0": [0.0, 0.001, 0.01, 0.1, 1.0],
        "gamma": ["scale"]
        }

SVRpoly = {
        "c": [0.001, 0.005, 0.01, 0.05, 0.1, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0],#, 20.0, 50.0, 100.0],
        "eps": [0.001, 0.01, 0.05, 0.07, 0.1, 0.15, 0.17, 0.25, 0.5, 1.0],
        "tol": [0.00001, 0.0001, 0.001, 0.01, 0.1],
        "max_iter": [-1],
        "dual": [True, False],
        "degree": [1, 2, 3],
        "coef0": [0.0, 0.001, 0.01, 0.1, 1.0],
        "gamma": ["scale"]
        }

SVRrbf = {
        "c": [0.001, 0.005, 0.01, 0.05, 0.1, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0],#, 20.0, 50.0, 100.0],
        "eps": [0.001, 0.01, 0.05, 0.07, 0.1, 0.15, 0.17, 0.25, 0.5, 1.0],
        "tol": [0.00001, 0.0001, 0.001, 0.01, 0.1],
        "max_iter": [-1],
        "dual": [True, False],
        "gamma": ["scale", "auto"]
        }

RFR = {
        "n_estimators": [10, 15, 25, 50, 75, 100, 200], #[200, 300, 1000], #[15, 25, ],
        "max_features": [0.05, 0.1, 0.2, 0.6, 0.8],
        "max_depth": [5, 10, 20, 50],
        "min_samples_split": [5],
        "min_samples_leaf": [5],
        "max_leaf_nodes": [None]
        }

Lasso = {
        "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 1.5, 2],
        "max_iter": [1000]
}

KNN = {
        "n_neighbors": [1, 2, 3, 5, 7, 10, 15, 20],
        "p": [1]
}

LSTM = {
        "bidirectional": [True],
        "topology": [(5, 1), (20, 1)],#, (25, 25, 1), (12, 12, 1), (7, 1), (25, 1), (12, 1)],
        "epochs": [2000],#, 3000],
        "batch_size": [1],#, 10],#, 5, 10],
        "activation": ["relu"], #["sigmoid", "relu", "tanh"],#, softmax, relu, sigmoid, tanh, None
        "dropout_rate": [0, 0.2],# [0, 0.2, 0.4],
        "optimizer": ["adam"],#, "rmsprop"], # sgd, rmsprop, adagrad, adadelta, adamax, nadam
        "kernel_regularizer": [(0.0, 0.0), (0.1, 0.1)],# [(0.0, 0.0), (0.01, 0.0), (0.0, 0.01), (0.01, 0.01)], # (0.0, 0.01), (0.01, 0.01)
        "bias_regularization": [(0.0, 0.0)], # [(0.0, 0.0), (0.01, 0.0), (0.0, 0.01), (0.01, 0.01)], # (0.0, 0.01), (0.01, 0.01)
        "early_stopping": [None], #3, 5],#, 2, 3],
        "stateful": [False] # True
        }
