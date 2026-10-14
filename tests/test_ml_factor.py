import unittest

import numpy as np
import pandas as pd

from ml.ml_factor import (build_factor_panel, rolling_train_validation_dates,
                          _fit_transform)


class WalkForwardMLTests(unittest.TestCase):
    def test_build_factor_panel_aligns_features_and_target(self):
        dates = pd.date_range("2022-01-31", periods=2, freq="ME")
        codes = ["000001", "000002"]
        factor = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=dates, columns=codes)
        target = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], index=dates, columns=codes)
        panel = build_factor_panel({"value": factor}, target)
        self.assertEqual(list(panel.index.names), ["date", "code"])
        self.assertEqual(panel.loc[(dates[0], "000001"), "value"], 1.0)
        self.assertEqual(panel.loc[(dates[1], "000002"), "forward_return"], 0.4)

    def test_rolling_windows_are_contiguous_and_end_before_origin(self):
        dates = pd.date_range("2020-01-31", periods=40, freq="ME")
        windows = rolling_train_validation_dates(dates, 36, 24, 6)
        self.assertEqual(len(windows.train_dates), 24)
        self.assertEqual(len(windows.valid_dates), 6)
        self.assertLess(windows.train_dates[-1], windows.valid_dates[0])
        self.assertEqual(windows.valid_dates[-1], dates[35])

    def test_transform_bounds_and_medians_come_from_train_only(self):
        train = pd.DataFrame({"f": [1.0, 2.0, 3.0, np.nan]})
        test = pd.DataFrame({"f": [1000.0, np.nan]})
        _, transformed = _fit_transform(train, test, ["f"])
        self.assertLess(transformed[0, 0], 1000.0)
        self.assertTrue(np.isfinite(transformed).all())


if __name__ == "__main__":
    unittest.main()
