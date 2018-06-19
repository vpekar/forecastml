import unittest
from mock import Mock

import utils
import data
from utils import prepare_data
from tests.mock_data import get_df, get_preproc_config


class TestScaling2d(unittest.TestCase):

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

    def test_scaleY(self):
        # ensure scaling works
        c = get_preproc_config(use_exog=True)
        d = prepare_data(c)

        # train
        self.assertEqual(0.0, d.trainY[0])
        self.assertEqual(1.0, d.trainY[-1])

        # val
        self.assertTrue(d.valY[0] != d.valYref[0])
        self.assertTrue(d.valY[-1] != d.valYref[-1])

        # test
        self.assertTrue(d.testY[0] != d.testYref[0])
        self.assertTrue(d.testY[-1] != d.testYref[-1])

    def test_scale_revert(self):
        c = get_preproc_config(use_exog=True)
        d = prepare_data(c)
        self.assertTrue(d.revert(d.trainY)[5], d.trainYref[5])

    def test_scaleYref(self):
        # assert original Y's are not changed after scaling
        c = get_preproc_config(use_exog=True)
        d = prepare_data(c)
        self.assertEqual(52.0, d.trainYref[0])
        self.assertEqual(242.0, d.trainYref[-1])
        self.assertEqual(292.0, d.valYref[0])
        self.assertEqual(322.0, d.valYref[-1])
        self.assertEqual(372.0, d.testYref[0])
        self.assertEqual(402.0, d.testYref[-1])

    def test_scaleX(self):
        c = get_preproc_config(use_exog=True)
        d = prepare_data(c)

        # train
        self.assertEqual(0, int(d.trainX[0][0]))
        self.assertEqual(1, int(d.trainX[-1][1]))


class TestScaling3d(unittest.TestCase):

    def setUp(self):
        try:
            reload(data)
            reload(utils)
        except NameError:
            import importlib
            importlib.reload(data)
            importlib.reload(utils)
        utils.pd.read_csv = Mock(return_value=get_df())
        data.Data3d.decompose = Mock()

    def test_scaleY(self):
        # ensure scaling works
        c = get_preproc_config(use_exog=True)
        d = prepare_data(c)
        d = prepare_data(c, dim="3d")

        # train
        self.assertEqual(0.0, d.trainY[0])
        self.assertEqual(1.0, d.trainY[-1])

        # val
        self.assertTrue(d.valY[0] != d.valYref[0])
        self.assertTrue(d.valY[-1] != d.valYref[-1])

        # test
        self.assertTrue(d.testY[0] != d.testYref[0])
        self.assertTrue(d.testY[-1] != d.testYref[-1])

    def test_scale_revert(self):
        c = get_preproc_config(use_exog=True)
        d = prepare_data(c)
        d = prepare_data(c, dim="3d")
        self.assertTrue(d.revert(d.trainY)[5], d.trainYref[5])

    def test_scaleYref(self):
        # assert original Y's are not changed after scaling
        c = get_preproc_config(use_exog=True)
        d = prepare_data(c)
        d = prepare_data(c, dim="3d")
        self.assertEqual(52.0, d.trainYref[0])
        self.assertEqual(242.0, d.trainYref[-1])
        self.assertEqual(292.0, d.valYref[0])
        self.assertEqual(322.0, d.valYref[-1])
        self.assertEqual(372.0, d.testYref[0])
        self.assertEqual(402.0, d.testYref[-1])

    def test_scaleX(self):
        c = get_preproc_config(use_exog=True)
        d = prepare_data(c)
        d = prepare_data(c, dim="3d")

        # train
        self.assertAlmostEqual(0.0, d.trainX[0][0][0], 2)
        self.assertAlmostEqual(1.0, d.trainX[-1][-1][-1], 2)
