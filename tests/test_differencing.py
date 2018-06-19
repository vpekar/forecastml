import unittest
from mock import Mock

import utils
import data
from utils import prepare_data
from tests.mock_data import get_df, get_preproc_config


class TestDifferencing2d(unittest.TestCase):

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

    def test_get_data__do_difference(self):
        # ensure differencing works
        c = get_preproc_config(difference=True)
        d = prepare_data(c)

        # train
        self.assertEqual(10.0, d.trainY[0])
        self.assertEqual(10.0, d.trainY[-1])

        # val
        self.assertEqual(10.0, d.valY[0])
        self.assertEqual(10.0, d.valY[-1])

        # test
        self.assertEqual(10.0, d.testY[0])
        self.assertEqual(10.0, d.testY[-1])

    def test_get_data__revert_difference_train(self):
        # ensure trainY are restored to previous values
        c = get_preproc_config(difference=True, scale=[0., 0.])
        d = prepare_data(c)
        reverted = d.revert(d.trainY)
        self.assertEqual(d.trainYref[0], reverted[0])
        self.assertEqual(d.trainYref[3], reverted[3])
        self.assertEqual(d.trainYref[-1], reverted[-1])

    def test_get_data__revert_difference_val(self):
        # ensure valY are restored to previous values
        c = get_preproc_config(difference=True, scale=[0., 0.])
        d = prepare_data(c)
        reverted = d.revert(d.valY, "val")
        self.assertEqual(d.valYref[0], reverted[0])
        self.assertEqual(d.valYref[3], reverted[3])
        self.assertEqual(d.valYref[-1], reverted[-1])

    def test_get_data__revert_difference_test(self):
        # ensure testY are restored to previous values
        c = get_preproc_config(difference=True, scale=[0., 0.])
        d = prepare_data(c)
        reverted = d.revert(d.testY, "test")
        self.assertEqual(d.testYref[0], reverted[0])
        self.assertEqual(d.testYref[3], reverted[3])
        self.assertEqual(d.testYref[-1], reverted[-1])


class TestDifferencing3d(unittest.TestCase):

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
        data.Data3d.scale = Mock()

    def test_get_data__do_difference(self):
        # ensure differencing works
        c = get_preproc_config(difference=True)
        d = prepare_data(c, dim="3d")

        # train
        self.assertEqual(10.0, d.trainY[0])
        self.assertEqual(10.0, d.trainY[-1])

        # val
        self.assertEqual(10.0, d.valY[0])
        self.assertEqual(10.0, d.valY[-1])

        # test
        self.assertEqual(10.0, d.testY[0])
        self.assertEqual(10.0, d.testY[-1])

    def test_get_data__revert_difference_train(self):
        # ensure trainY are restored to previous values
        c = get_preproc_config(difference=True, scale=[0., 0.])
        d = prepare_data(c, dim="3d")

        reverted = d.revert(d.trainY)
        self.assertEqual(d.trainYref[0], reverted[0])
        self.assertEqual(d.trainYref[3], reverted[3])
        self.assertEqual(d.trainYref[-1], reverted[-1])

    def test_get_data__revert_difference_val(self):
        # ensure valY are restored to previous values
        c = get_preproc_config(difference=True, scale=[0., 0.])
        d = prepare_data(c, dim="3d")
        reverted = d.revert(d.valY, "val")
        self.assertEqual(d.valYref[0], reverted[0])
        self.assertEqual(d.valYref[3], reverted[3])
        self.assertEqual(d.valYref[-1], reverted[-1])

    def test_get_data__revert_difference_test(self):
        # ensure testY are restored to previous values
        c = get_preproc_config(difference=True, scale=[0., 0.])
        d = prepare_data(c, dim="3d")
        reverted = d.revert(d.testY, "test")
        self.assertEqual(d.testYref[0], reverted[0])
        self.assertEqual(d.testYref[3], reverted[3])
        self.assertEqual(d.testYref[-1], reverted[-1])
