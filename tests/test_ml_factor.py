import unittest

import numpy as np
import pandas as pd

from ml.ml_factor import (build_factor_panel, rolling_train_validation_dates,
                          _fit_transform)
from ml.market_models import (cross_sectional_binary_target,
                              efficient_frontier_weights,
                              monthly_kmeans_diagnostics,
                              progressive_feature_sets,
                              walk_forward_random_forest)


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

    def test_binary_target_is_within_month_median(self):
        dates = pd.to_datetime(["2024-01-31"] * 4)
        index = pd.MultiIndex.from_arrays([dates, ["a", "b", "c", "d"]],
                                          names=["date", "code"])
        panel = pd.DataFrame({"forward_return": [0.1, 0.2, 0.3, 0.4]}, index=index)
        target = cross_sectional_binary_target(panel)
        self.assertEqual(target.tolist(), [0.0, 0.0, 1.0, 1.0])

    def test_feature_addition_sets_are_cumulative_and_preordered(self):
        steps = progressive_feature_sets(["dividend_yield", "low_volatility", "momentum_12m"])
        self.assertEqual(list(steps), ["add_dividend", "add_low_volatility", "add_price_volume"])
        self.assertEqual(steps["add_dividend"], ["dividend_yield"])
        self.assertEqual(steps["add_price_volume"],
                         ["dividend_yield", "low_volatility", "momentum_12m"])

    def test_rf_prediction_does_not_use_same_month_return_label(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("optional scikit-learn is not installed")
        rng = np.random.default_rng(11)
        dates = pd.date_range("2023-11-30", periods=5, freq="ME")
        codes = [f"{i:06d}" for i in range(40)]
        rows = []
        for date in dates:
            for i, code in enumerate(codes):
                rows.append((date, code, float(i) / 40 + rng.normal(0, .02),
                             float(i) / 40 + rng.normal(0, .03)))
        panel = pd.DataFrame(rows, columns=["date", "code", "dividend_yield", "forward_return"])
        panel = panel.set_index(["date", "code"]).sort_index()
        split = dates[3]
        first = walk_forward_random_forest(panel, ["dividend_yield"], split,
                                           min_train_rows=40, min_names_per_month=20)
        changed = panel.copy()
        changed.loc[(dates[3], slice(None)), "forward_return"] = np.linspace(-1, 1, len(codes))
        second = walk_forward_random_forest(changed, ["dividend_yield"], split,
                                            min_train_rows=40, min_names_per_month=20)
        left = first.loc[first.date == split].probability_above_median.to_numpy()
        right = second.loc[second.date == split].probability_above_median.to_numpy()
        np.testing.assert_allclose(left, right)

    def test_kmeans_diagnostic_only_uses_feature_columns(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("optional scikit-learn is not installed")
        dates = pd.to_datetime(["2024-01-31"] * 40)
        index = pd.MultiIndex.from_arrays([dates, [f"{i:06d}" for i in range(40)]],
                                          names=["date", "code"])
        panel = pd.DataFrame({"f1": np.arange(40), "f2": np.arange(40)[::-1],
                              "forward_return": np.linspace(-0.2, 0.2, 40)}, index=index)
        assignments, summary = monthly_kmeans_diagnostics(panel, ["f1", "f2"], n_clusters=4)
        self.assertEqual(len(assignments), 40)
        self.assertEqual(assignments.cluster.nunique(), 4)
        self.assertIn("mean_forward_return", summary.columns)

    def test_frontier_weights_are_long_only_and_capped(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("optional scikit-learn is not installed")
        rng = np.random.default_rng(7)
        names = [f"{i:06d}" for i in range(10)]
        history = pd.DataFrame(rng.normal(0.01, 0.05, size=(24, 10)), columns=names)
        expected = pd.Series(np.linspace(0.01, 0.03, 10), index=names)
        weights = efficient_frontier_weights(expected, history, max_weight=0.10)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=7)
        self.assertGreaterEqual(float(weights.min()), -1e-9)
        self.assertLessEqual(float(weights.max()), 0.100001)


if __name__ == "__main__":
    unittest.main()
