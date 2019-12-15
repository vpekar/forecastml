# -*- coding: utf-8 -*-

import numpy as np
from unittest import TestCase
from unittest.mock import Mock, create_autospec

import utils
import data
from utils import run_config
from utils import prepare_data
from learner_configs import ConfigLSTM, ConfigLSVR

from tests.mock_data import get_df, get_preproc_config


def get_model(d, ret_val=[1.0]):
    model = Mock()
    model.predict = Mock(return_value=ret_val)
    model.feature_importances_ = [x for x in range(len(d.feature_names))]
    return model


def get_mock_svr(d, pc):
    mockConfigSVR = create_autospec(ConfigLSVR)
    mockConfigSVR.name = "SVR"
    side_effect = lambda x, y: np.array([.1]*y.shape[0])
    mockConfigSVR.forecast = Mock(side_effect=side_effect)
    mockConfigSVR.train = Mock(return_value=get_model(d))
    mockConfigSVR.vals = {"a": 1, "b": 2, "c": 3}
    mockConfigSVR.pc = pc
    return mockConfigSVR


def get_mock_lstm(d, pc):
    mockConfigLSTM = create_autospec(ConfigLSTM)
    mockConfigLSTM.name = "LSTM"
    mockConfigLSTM.train = Mock(return_value=get_model(d, [[1.0]]))
    side_effect = lambda x, y: np.array([.1]*y.shape[0])
    mockConfigLSTM.forecast = Mock(side_effect=side_effect)
    mockConfigLSTM.vals = {"a": 1, "b": 2, "c": 3}
    mockConfigLSTM.pc = pc
    return mockConfigLSTM


class TestRunConfig(TestCase):

    def setUp(self):
        try:
            reload(data)
            reload(utils)
        except NameError:
            import importlib
            importlib.reload(data)
            importlib.reload(utils)
        utils.pd.read_csv = Mock(return_value=get_df())

    def test_2d_test_mode(self):
        pc = get_preproc_config(lags=3, horizon=1)
        d = prepare_data(pc)
        mockConfigSVR = get_mock_svr(d, pc)

        r = run_config([d, mockConfigSVR, 'test'])

        self.assertTrue(r.train_mse > 0.0)
        self.assertTrue(r.test_mse > 0.0)
        self.assertTrue(r.train_mae > 0.0)
        self.assertTrue(r.train_mae > 0.0)
        # yhat are scaled
        self.assertTrue(r.yhat_is[0] <= 1.0)
        self.assertTrue(r.yhat_oos[0] <= 1.0)

    def test_2d_val_mode(self):
        pc = get_preproc_config(lags=3, horizon=1)
        d = prepare_data(pc)
        mockConfigSVR = get_mock_svr(d, pc)

        r = run_config([d, mockConfigSVR, 'val'])

        self.assertTrue(r.train_mse > 0.0)
        self.assertTrue(r.test_mse > 0.0)
        self.assertTrue(r.train_mae > 0.0)
        self.assertTrue(r.train_mae > 0.0)
        # yhat are scaled
        self.assertTrue(r.yhat_is[0] <= 1.0)
        self.assertTrue(r.yhat_oos[0] <= 1.0)

    def test_2d_feature_importances_(self):
        pc = get_preproc_config(lags=3, horizon=1)
        d = prepare_data(pc)
        mockConfigSVR = get_mock_svr(d, pc)

        r = run_config([d, mockConfigSVR, 'val'])
        self.assertEqual(len(r.feature_scores), 3)
        self.assertEqual(r.feature_scores[0], ('lag1', 2))

    def test_3d_test_mode(self):
        pc = get_preproc_config(lags=3, horizon=1)
        d = prepare_data(pc, dim="3d")
        mockConfigLSTM = get_mock_lstm(d, pc)

        r = run_config([d, mockConfigLSTM, 'test'])

        self.assertTrue(r.train_mse > 0.0)
        self.assertTrue(r.test_mse > 0.0)
        self.assertTrue(r.train_mae > 0.0)
        self.assertTrue(r.train_mae > 0.0)
        # yhat are scaled
        self.assertTrue(r.yhat_is[0] <= 1.0)
        self.assertTrue(r.yhat_oos[0] <= 1.0)

    def test_3d_val_mode(self):
        pc = get_preproc_config(lags=3, horizon=1)
        d = prepare_data(pc, dim="3d")
        mockConfigLSTM = get_mock_lstm(d, pc)

        print(mockConfigLSTM.__class__.__dict__.keys())

        r = run_config([d, mockConfigLSTM, 'val'])

        self.assertTrue(r.train_mse > 0.0)
        self.assertTrue(r.test_mse > 0.0)
        self.assertTrue(r.train_mae > 0.0)
        self.assertTrue(r.train_mae > 0.0)
        # yhat are scaled
        self.assertTrue(r.yhat_is[0] <= 1.0)
        self.assertTrue(r.yhat_oos[0] <= 1.0)
