import numpy as np
import logging
logging.getLogger("matplotlib").disabled = True
logging.getLogger("tensorflow").disabled = True

from unittest import TestCase
from unittest.mock import Mock, create_autospec

import utils
import data
from utils import prepare_data
from tests.mock_data import get_df3, get_preproc_config


class TestDeseasonalize(TestCase):

    def setUp(self):
        try:
            reload(data)
            reload(utils)
        except NameError:
            import importlib
            importlib.reload(data)
            importlib.reload(utils)
        data.Data2d.scale = Mock()
        data.Data2d.preprocess = Mock()

    def test_input_and_output_shapes(self):
        c = get_preproc_config(deseason=False)
        df = get_df3()
        d = data.Data2d(df, c)

        y2 = d._deseasonalize(df['dep_var'], 4)
        self.assertEqual(df['dep_var'].shape, y2.shape)

    def test_deseasonalized_reference_values(self):
        c = get_preproc_config(deseason=True, seasonal_period=4, horizon=1, lags=1)
        df = get_df3()
        d = data.Data2d(df, c)

        # all deseasonalized values equal to 10.0
        self.assertTrue(np.array_equal(d.trainYref, [10.0]*11))
        self.assertTrue(np.array_equal(d.valYref, [10.0]*3))
        self.assertTrue(np.array_equal(d.testYref, [10.0]*3))

