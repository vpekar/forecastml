import unittest
from unittest.mock import Mock

import utils
import data
from utils import prepare_data, separate_exogs
from tests.mock_data import get_df, get_preproc_config


class TestCompatibleSetSizes(unittest.TestCase):

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
        data.Data3d.scale = Mock()

    def test_sets(self):
        c = get_preproc_config(lags=3)
        d_ml = prepare_data(c)
        d_lstm = prepare_data(c, dim="3d")
        # same shapes
        self.assertEqual(d_ml.trainY.shape[0], d_lstm.trainY.shape[0])
        self.assertEqual(d_ml.valY.shape[0], d_lstm.valY.shape[0])
        self.assertEqual(d_ml.testY.shape[0], d_lstm.testY.shape[0])
        # same values
        self.assertEqual(d_ml.trainY[0], d_lstm.trainY[0])
        self.assertEqual(d_ml.valY[0], d_lstm.valY[0])
        self.assertEqual(d_ml.testY[0], d_lstm.testY[0])


class TestUtils3d(unittest.TestCase):

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
        data.Data3d.decompose = Mock(side_effect=lambda x: x)
        data.Data3d.revert = Mock(side_effect=lambda x: x)

    def test_Yref(self):
        c = get_preproc_config(lags=3)
        d = prepare_data(c, dim="3d")
        self.assertEqual(d.trainY[0], d.trainYref[0])
        self.assertEqual(d.valY[0], d.valYref[0])
        self.assertEqual(d.testY[0], d.testYref[0])

    def test_get_data__no_exog(self):
        c = get_preproc_config(lags=3)
        d = prepare_data(c, dim="3d")
        self.assertEqual(d.trainX.shape, (21, 3, 1))
        self.assertEqual(d.valX.shape, (5, 3, 1))
        self.assertEqual(d.testX.shape, (5, 3, 1))
        self.assertEqual(d.trainX[0].tolist(), [[12], [22], [32]])
        self.assertEqual(d.trainY[0], 42)
        self.assertEqual(d.valX[0].tolist(), [[252], [262], [272]])
        self.assertEqual(d.valY[0], 282)
        self.assertEqual(d.testX[0].tolist(), [[332], [342], [352]])
        self.assertEqual(d.testY[0], 362)

    def test_get_data_with_exog__intent0(self):
        c = get_preproc_config(lags=3, use_exog=True)
        d = prepare_data(c, dim="3d")
        self.assertEqual(d.trainX.shape, (21, 3, 3))
        self.assertEqual(d.valX.shape, (5, 3, 3))
        self.assertEqual(d.testX.shape, (5, 3, 3))
        self.assertEqual(d.trainX[0].tolist(),
                         [[10, 11, 12], [20, 21, 22], [30, 31, 32]])
        self.assertEqual(d.trainY[0], 42)
        self.assertEqual(d.valX[0].tolist(),
                         [[250, 251, 252], [260, 261, 262], [270, 271, 272]])
        self.assertEqual(d.valY[0], 282)
        self.assertEqual(d.testX[0].tolist(),
                         [[330, 331, 332], [340, 341, 342], [350, 351, 352]])
        self.assertEqual(d.testY[0], 362)

    @unittest.skip("Not using intent distance for LSTM")
    def test_get_data_with_exog__intent2(self):
        c = get_preproc_config(lags=3, use_exog=True, intent_distance=2)
        d = prepare_data(c, dim="3d")
        self.assertEqual(d.trainX[0].tolist(),
                         [[10, 11, 32], [20, 21, 42], [30, 31, 52]])
        self.assertEqual(d.trainY[0], 62)
        self.assertEqual(d.valX[0].tolist(),
                         [[250, 251, 272], [260, 261, 282], [270, 271, 292]])
        self.assertEqual(d.valY[0], 302)
        self.assertEqual(d.testX[0].tolist(),
                         [[330, 331, 352], [340, 341, 362], [350, 351, 372]])
        self.assertEqual(d.testY[0], 382)


class TestUtils2d(unittest.TestCase):

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
        data.Data2d.decompose = Mock(side_effect=lambda x: x)
        data.Data2d.revert = Mock(side_effect=lambda x: x)

    def test_Yref(self):
        c = get_preproc_config(lags=3)
        d = prepare_data(c)
        self.assertEqual(d.trainYref[0], d.trainY[0])
        self.assertEqual(d.valY[0], d.valYref[0])
        self.assertEqual(d.testY[0], d.testYref[0])

    def test_get_data__no_exog(self):
        c = get_preproc_config(lags=3)
        d = prepare_data(c)
        self.assertEqual(d.trainX.shape, (21, 3))
        self.assertEqual(d.valX.shape, (5, 3))
        self.assertEqual(d.testX.shape, (5, 3))
        self.assertEqual(d.trainX[0].tolist(), [12, 22, 32])
        self.assertEqual(d.trainY[0], 42)
        self.assertEqual(d.valX[0].tolist(), [252, 262, 272])
        self.assertEqual(d.valY[0], 282)
        self.assertEqual(d.testX[0].tolist(), [332, 342, 352])
        self.assertEqual(d.testY[0], 362)

    def test_get_data_with_exog__intent0(self):
        c = get_preproc_config(lags=3, use_exog=True)
        d = prepare_data(c)
        self.assertEqual(d.trainX.shape, (21, 5))
        self.assertEqual(d.valX.shape, (5, 5))
        self.assertEqual(d.testX.shape, (5, 5))
        self.assertEqual(d.trainX[0].tolist(), [12, 22, 32, 40, 41])
        self.assertEqual(d.trainY[0], 42)
        self.assertEqual(d.valX[0].tolist(), [252, 262, 272, 280, 281])
        self.assertEqual(d.valY[0], 282)
        self.assertEqual(d.testX[0].tolist(), [332, 342, 352, 360, 361])
        self.assertEqual(d.testY[0], 362)

    def test_get_data_with_exog__intent2(self):
        c = get_preproc_config(lags=3, use_exog=True, intent_distance=2)
        d = prepare_data(c)
        # 12, 22, 32 are endogs, 2 and 21 are exogs at distance=2
        self.assertEqual(d.trainX[0].tolist(), [12, 22, 32, 20, 21])
        self.assertEqual(d.trainY[0], 42)
        self.assertEqual(d.valX[0].tolist(), [252, 262, 272, 260, 261])
        self.assertEqual(d.valY[0], 282)
        self.assertEqual(d.testX[0].tolist(), [332, 342, 352, 340, 341])
        self.assertEqual(d.testY[0], 362)

    def test_get_data__with_exog_only(self):
        c = get_preproc_config(lags=3, use_exog=True, intent_distance=2)
        d = prepare_data(c)
        trainX1, trainX2 = separate_exogs(d.trainX, lags=3)
        self.assertEqual(trainX1[0].tolist(), [12, 22, 32])
        self.assertEqual(trainX2[0].tolist(), [20, 21])
        self.assertEqual(d.trainY[0], 42)
