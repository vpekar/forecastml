# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock
import numpy as np

from learner_configs import ConfigLSTM, ConfigSVR

from tests.mock_data import get_preproc_config


def get_call_args(call):
    return call[0][0].tolist()


class TestConfig2d(TestCase):

    def test_general_case(self):
        model = Mock()
        model.predict = Mock(return_value=[1.0])
        testX = np.array([
                [0.1, 0.2, 0.3, 1.5],
                [0.2, 0.3, 0.4, 1.6],
                [0.3, 0.4, 0.5, 1.7],
                [0.4, 0.5, 0.6, 1.8],
                [0.5, 0.6, 0.7, 1.9],
                [0.6, 0.7, 0.8, 2.0],
            ])

        pc = get_preproc_config(lags=3, horizon=1)
        adict= {'kernel': 1., 'degree': 1., 'c': 1., 'eps': 1.}
        c = ConfigSVR(adict, pc)

        yhat = c.forecast(model, testX)
        self.assertEqual(yhat.tolist(), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def test_instances(self):
        """Ensure predictions are inserted in the right places for
        new instances
        """
        model = Mock()
        model.predict = Mock(return_value=[1.0])
        # the first 3 cols are exogenous vars (from earlier to later),
        # the last col is endogenous
        testX = np.array([
                [0.1, 0.2, 0.3, 1.5],
                [0.2, 0.3, 0.4, 1.6],
                [0.3, 0.4, 0.5, 1.7],
                [0.4, 0.5, 0.6, 1.8],
                [0.5, 0.6, 0.7, 1.9],
                [0.6, 0.7, 0.8, 2.0],
            ])

        pc = get_preproc_config(lags=3, horizon=2)
        adict= {'kernel': 1., 'degree': 1., 'c': 1., 'eps': 1.}
        c = ConfigSVR(adict, pc)

        yhat = c.forecast(model, testX)
        calls = model.predict.call_args_list
        self.assertEqual(get_call_args(calls[0]), [[0.1, 0.2, 0.3, 1.5]])
        self.assertEqual(get_call_args(calls[1]), [[0.2, 0.3, 1. , 1.6]])
        self.assertEqual(get_call_args(calls[2]), [[0.3, 1. , 1. , 1.7]])
        self.assertEqual(get_call_args(calls[3]), [[0.4, 0.5, 0.6, 1.8]])
        self.assertEqual(get_call_args(calls[4]), [[0.5, 0.6, 1. , 1.9]])
        self.assertEqual(get_call_args(calls[5]), [[0.6, 1. , 1. , 2.]])


class TestConfig3d(TestCase):

    def test_general_case(self):
        model = Mock()
        model.predict = Mock(return_value=[[1.0]])
        testX = np.array([
                [[0.1, 1.5], [0.2, 1.6]],
                [[0.2, 1.6], [0.3, 1.7]],
                [[0.3, 1.7], [0.4, 1.8]],
                [[0.4, 1.8], [0.5, 1.9]],
                [[0.5, 1.9], [0.6, 2.0]],
                [[0.6, 2.0], [0.7, 2.1]],
            ])

        pc = get_preproc_config(lags=2, horizon=1)
        adict= {"bidirectional": False, "topology": [3, 4], "epochs": 100,
            "batch_size": 10, "activation": None, "dropout_rate": 0.,
            "optimizer": None, "kernel_regularizer": (0.0, 0.0),
            "bias_regularization": (0.0, 0.0), "early_stopping": None,
            "stateful": False}
        c = ConfigLSTM(adict, pc)

        yhat = c.forecast(model, testX)
        self.assertEqual(yhat.tolist(), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def test_instances(self):
        """Ensure predictions are inserted in the right places for
        new instances
        """
        model = Mock()
        model.predict = Mock(return_value=[[1.0]])
        # the first col is endogenous vars, the last col is exogenous
        testX = np.array([
                [[0.1, 1.5], [0.2, 1.6]],
                [[0.2, 1.6], [0.3, 1.7]],
                [[0.3, 1.7], [0.4, 1.8]],
                [[0.4, 1.8], [0.5, 1.9]],
                [[0.5, 1.9], [0.6, 2.0]],
                [[0.6, 2.0], [0.7, 2.1]],
            ])
        pc = get_preproc_config(lags=2, horizon=2)
        adict= {"bidirectional": False, "topology": [3, 4], "epochs": 100,
            "batch_size": 10, "activation": None, "dropout_rate": 0.,
            "optimizer": None, "kernel_regularizer": (0.0, 0.0),
            "bias_regularization": (0.0, 0.0), "early_stopping": None,
            "stateful": False}
        c = ConfigLSTM(adict, pc)

        yhat = c.forecast(model, testX)
        calls = model.predict.call_args_list
        self.assertEqual(get_call_args(calls[0][0]), [[0.1, 1.5], [0.2, 1.6]])
        self.assertEqual(get_call_args(calls[1][0]), [[0.2, 1.6], [0.3, 1.0]])
        self.assertEqual(get_call_args(calls[2][0]), [[0.3, 1.0], [0.4, 1.0]])
        self.assertEqual(get_call_args(calls[3][0]), [[0.4, 1.8], [0.5, 1.9]])
        self.assertEqual(get_call_args(calls[4][0]), [[0.5, 1.9], [0.6, 1.0]])
        self.assertEqual(get_call_args(calls[5][0]), [[0.6, 1.0], [0.7, 1.0]])

