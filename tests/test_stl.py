
import sys
from unittest import TestCase
from unittest.mock import Mock

import utils
import data
from utils import prepare_data
from tests.mock_data import get_df, get_preproc_config

import traceback
import warnings


def warn_with_traceback(message, category, filename, lineno, file=None,
                        line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno,
                                     line))
warnings.showwarning = warn_with_traceback


class TestDecompose2d(TestCase):

    def setUp(self):
        try:
            reload(data)
            reload(utils)
        except NameError:
            import importlib
            importlib.reload(data)
            importlib.reload(utils)
        utils.pd.read_csv = Mock(return_value=get_df())
        data.Data2d.scale = Mock()
        data.Data2d.revert = Mock(side_effect=lambda x: x)

    def test_decompose_is(self):
        c = get_preproc_config(lags=3, detrend=True, deseason=True, horizon=1)
        d = prepare_data(c)
        # assert that stl forecast is the same shape as original trainY
        self.assertEqual(d.stl_forecast['train'].shape, (21-c['horizon']+1, 1))
        # assert that the new data.trainY is the same shape as original trainY
        self.assertEqual(d.trainY.shape, (21,))

    def test_decompose_val(self):
        c = get_preproc_config(lags=3, detrend=True, deseason=True, horizon=1)
        d = prepare_data(c)
        self.assertEqual(d.stl_forecast['val'].shape, (5-c['horizon']+1, 1))

    def test_decompose_oos(self):
        c = get_preproc_config(lags=3, detrend=True, deseason=True, horizon=1)
        d = prepare_data(c)
        self.assertEqual(d.stl_forecast['test'].shape, (5-c['horizon']+1, 1))

    def test_revert_decompose__train(self):
        c = get_preproc_config(lags=3, detrend=True, deseason=True, horizon=1)
        d = prepare_data(c)
        # assert that the data was decomposed and forecast + resid is
        # equal to ref
        self.assertEqual(d.trainY[0] + d.stl_forecast['train'][0],
                         d.trainYref[0])
        self.assertEqual(d.trainY[-1] + d.stl_forecast['train'][-1],
                         d.trainYref[-1])

    def test_revert_decompose__val(self):
        c = get_preproc_config(lags=3, detrend=True, deseason=True, horizon=1)
        d = prepare_data(c)
        # assert that the data was decomposed and forecast + resid is
        # equal to ref
        self.assertEqual(d.valY[0] + d.stl_forecast['val'][0], d.valYref[0])
        self.assertEqual(d.valY[-1] + d.stl_forecast['val'][-1], d.valYref[-1])

    def test_revert_decompose__test(self):
        c = get_preproc_config(lags=3, detrend=True, deseason=True, horizon=1)
        d = prepare_data(c)
        # assert that the data was decomposed and forecast + resid is
        # equal to ref
        self.assertEqual(d.testY[0] + d.stl_forecast['test'][0],
                         d.testYref[0])
        self.assertEqual(d.testY[-1] + d.stl_forecast['test'][-1],
                         d.testYref[-1])


class TestDecompose3d(TestCase):

    def setUp(self):
        try:
            reload(data)
            reload(utils)
        except NameError:
            import importlib
            importlib.reload(data)
            importlib.reload(utils)
        utils.pd.read_csv = Mock(return_value=get_df())
        data.Data3d.scale = Mock()
        data.Data3d.revert = Mock(side_effect=lambda x: x)

    def test_decompose_is(self):
        c = get_preproc_config(lags=3, detrend=True, deseason=True, horizon=1)
        d = prepare_data(c, dim="3d")
        # assert that stl forecast is the same shape as original trainY
        self.assertEqual(d.stl_forecast['train'].shape, (21-c['horizon']+1,1))
        # assert that the new data.trainY is the same shape as original trainY
        self.assertEqual(d.trainY.shape, (21,))

    def test_decompose_val(self):
        c = get_preproc_config(lags=3, detrend=True, deseason=True, horizon=1)
        d = prepare_data(c, dim="3d")
        self.assertEqual(d.stl_forecast['val'].shape, (5-c['horizon']+1, 1))

    def test_decompose_oos(self):
        c = get_preproc_config(lags=3, detrend=True, deseason=True, horizon=1)
        d = prepare_data(c, dim="3d")
        self.assertEqual(d.stl_forecast['test'].shape, (5-c['horizon']+1, 1))

    def test_revert_decompose__train(self):
        c = get_preproc_config(lags=3, detrend=True, deseason=True, horizon=1)
        d = prepare_data(c, dim="3d")
        # assert that the data was decomposed and forecast + resid is equal to
        # ref
        self.assertAlmostEqual(d.trainY[0] + d.stl_forecast['train'][0],
                               d.trainYref[0], delta=0.9)
        self.assertAlmostEqual(d.trainY[-1] + d.stl_forecast['train'][-1],
                               d.trainYref[-1], delta=0.9)

    def test_revert_decompose__val(self):
        c = get_preproc_config(lags=3, detrend=True, deseason=True, horizon=1)
        d = prepare_data(c, dim="3d")
        # assert that the data was decomposed and forecast + resid is
        # equal to ref
        self.assertAlmostEqual(d.valY[0] + d.stl_forecast['val'][0],
                               d.valYref[0], delta=0.9)
        self.assertAlmostEqual(d.valY[-1] + d.stl_forecast['val'][-1],
                               d.valYref[-1], delta=0.9)

    def test_revert_decompose__test(self):
        c = get_preproc_config(lags=3, detrend=True, deseason=True, horizon=1)
        d = prepare_data(c, dim="3d")
        # assert that the data was decomposed and forecast + resid is
        # equal to ref
        self.assertAlmostEqual(d.testY[0] + d.stl_forecast['test'][0],
                               d.testYref[0], delta=0.9)
        self.assertAlmostEqual(d.testY[-1] + d.stl_forecast['test'][-1],
                               d.testYref[-1], delta=0.9)

