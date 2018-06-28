# -*- coding: utf-8 -*-

import unittest
import logging
from mock import Mock

import utils
import data
from utils import prepare_data
from tests.mock_data import get_df, get_preproc_config

logging.getLogger("matplotlib").disabled = True
logging.getLogger("tensorflow").disabled = True


class TestFeatureSelection(unittest.TestCase):

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
        c = get_preproc_config(lags=3, use_exog=True, feature_selection=0.5)
        d = prepare_data(c)

        # keeping 1 exog variable after feature selection and 3 endog vars
        self.assertEqual(d.trainX.shape[1], 4)
        self.assertEqual(d.valX.shape[1], 4)
        self.assertEqual(d.testX.shape[1], 4)

        # ensure that the exog value is from col "dim1" (the selected var)
        self.assertTrue(d.trainX[0][0], 11)
