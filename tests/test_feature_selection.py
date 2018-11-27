# -*- coding: utf-8 -*-

import logging
import numpy as np
from unittest import TestCase
from unittest.mock import Mock

import utils
import data
from utils import prepare_data
from learner_configs import ConfigSVR, ConfigGB, ConfigXGBoost
from tests.mock_data import get_df, get_df2, get_preproc_config


logging.getLogger("matplotlib").disabled = True
logging.getLogger("tensorflow").disabled = True


class TestFeatureSelection(TestCase):

    def setUp(self):
        try:
            reload(data)
            reload(utils)
        except NameError:
            import importlib
            importlib.reload(data)
            importlib.reload(utils)
        utils.pd.read_csv = Mock(return_value=get_df())
        data.Data2d.decompose = Mock()
        data.Data2d.scale = Mock()

    def test_select_features(self):
        c = get_preproc_config(lags=2, use_exog=True, feature_selection=0.5)
        d = prepare_data(c)

        # keeping 4 features after selection: 2 original lags and 2 exogs
        self.assertEqual(d.trainX.shape[1], 4)
        self.assertEqual(d.valX.shape[1], 4)
        self.assertEqual(d.testX.shape[1], 4)

        # ensure that feature_names is updated too
        self.assertEqual(len(d.feature_names), 4)


class TestFeatureScoring(TestCase):

    def setUp(self):
        try:
            reload(data)
            reload(utils)
        except NameError:
            import importlib
            importlib.reload(data)
            importlib.reload(utils)
        utils.pd.read_csv = Mock(return_value=get_df2())
        data.Data2d.decompose = Mock()
        data.Data2d.scale = Mock()

    def test_scoring(self):
        c = get_preproc_config(lags=2, use_exog=True, feature_selection=0.5)
        d = prepare_data(c)

        # ensure that the lags + most informative exogs are kept
        self.assertEqual(tuple(d.trainX[0].tolist()), (100, 20, 20, 100))

        # ensure relevant feature_names is left
        self.assertEqual(tuple(d.feature_names), ('lag2', 'lag1', 'dim10', 'dim11'))


class TestRfeFeatureSelection(TestCase):

    def setUp(self):
        try:
            reload(data)
            reload(utils)
        except NameError:
            import importlib
            importlib.reload(data)
            importlib.reload(utils)
        utils.pd.read_csv = Mock(return_value=get_df())

    def get_forecast(self, config, config_dict):
        pc = get_preproc_config(lags=3, use_exog=True, horizon=1,
                                feature_selection=0.5, rfe_step=1)
        d = prepare_data(pc)
        c = config(config_dict, pc)
        model = c.train(d)
        yhat = c.forecast(model, np.array([d.testX[0]]))
        return yhat

    def test_2d_svr(self):
        yhat = self.get_forecast(ConfigSVR,
            {'kernel': 'linear', 'degree': 1., 'c': 1., 'eps': 1.})
        self.assertAlmostEqual(yhat.tolist()[0][0], 0.5, 1)

    def test_2d_gb(self):
        yhat = self.get_forecast(ConfigGB, {})
        self.assertAlmostEqual(yhat.tolist()[0][0], 1.0, 1)

    def test_2d_xgb(self):
        yhat = self.get_forecast(ConfigXGBoost, {})
        self.assertAlmostEqual(yhat.tolist()[0][0], 1.0, 1)
