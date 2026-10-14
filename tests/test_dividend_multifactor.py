import unittest

import numpy as np
import pandas as pd

from analysis.dividend_multifactor import (equal_weight_rank_composite,
                                            fama_macbeth_multifactor,
                                            rank_ic_series, quintile_returns)
from factors.dividend_factors import trailing_cash_dividend_yield
from factors.risk_factors import low_volatility


class DividendFactorTests(unittest.TestCase):
    def test_yield_uses_only_public_implemented_events_and_raw_close(self):
        dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
        close = pd.DataFrame({"000001": [10.0, 10.0, 10.0]}, index=dates)
        events = pd.DataFrame([
            {"event_id": "a", "code": "000001", "version_date": "2024-02-01",
             "status": "implemented", "cash_per_share": 1.0, "ex_date": "2024-02-15"},
            # Not yet public at the January signal: must not leak backward.
            {"event_id": "b", "code": "000001", "version_date": "2024-03-01",
             "status": "implemented", "cash_per_share": 2.0, "ex_date": "2024-03-10"},
        ])
        actual = trailing_cash_dividend_yield(events, close)
        self.assertEqual(actual.loc[dates[0], "000001"], 0.0)
        self.assertAlmostEqual(actual.loc[dates[1], "000001"], 0.1)
        self.assertAlmostEqual(actual.loc[dates[2], "000001"], 0.3)

    def test_later_cancelled_version_removes_event_only_after_announcement(self):
        dates = pd.to_datetime(["2024-02-29", "2024-03-31"])
        close = pd.DataFrame({"000001": [10.0, 10.0]}, index=dates)
        events = pd.DataFrame([
            {"event_id": "a", "code": "000001", "version_date": "2024-02-01",
             "status": "implemented", "cash_per_share": 1.0, "ex_date": "2024-02-15"},
            {"event_id": "a", "code": "000001", "version_date": "2024-03-15",
             "status": "cancelled", "cash_per_share": 1.0, "ex_date": "2024-02-15"},
        ])
        actual = trailing_cash_dividend_yield(events, close)
        self.assertAlmostEqual(actual.loc[dates[0], "000001"], 0.1)
        self.assertEqual(actual.loc[dates[1], "000001"], 0.0)

    def test_cancellation_without_ex_date_still_supersedes_old_version(self):
        dates = pd.to_datetime(["2024-02-29", "2024-03-31"])
        close = pd.DataFrame({"000001": [10.0, 10.0]}, index=dates)
        events = pd.DataFrame([
            {"event_id": "a", "code": "000001", "version_date": "2024-02-01",
             "status": "implemented", "cash_per_share": 1.0, "ex_date": "2024-02-15"},
            {"event_id": "a", "code": "000001", "version_date": "2024-03-15",
             "status": "cancelled", "cash_per_share": np.nan, "ex_date": pd.NaT},
        ])
        actual = trailing_cash_dividend_yield(events, close)
        self.assertAlmostEqual(actual.loc[dates[0], "000001"], 0.1)
        self.assertEqual(actual.loc[dates[1], "000001"], 0.0)

    def test_low_vol_higher_for_constant_return_series(self):
        index = pd.date_range("2024-01-01", periods=8)
        close = pd.DataFrame({"steady": np.arange(10, 18),
                              "volatile": [10, 11, 9, 12, 8, 13, 7, 14]}, index=index)
        factor = low_volatility(close, window=3, min_periods=3)
        self.assertGreater(factor.loc[index[-1], "steady"], factor.loc[index[-1], "volatile"])

    def test_equal_rank_composite_requires_complete_pair(self):
        ix = pd.date_range("2024-01-31", periods=1, freq="ME")
        a = pd.DataFrame([[1.0, 2.0, 3.0]], index=ix, columns=list("abc"))
        b = pd.DataFrame([[3.0, 2.0, np.nan]], index=ix, columns=list("abc"))
        combo = equal_weight_rank_composite({"a": a, "b": b})
        self.assertEqual(combo.loc[ix[0], "a"], 0.75)
        self.assertEqual(combo.loc[ix[0], "b"], 0.75)
        self.assertTrue(np.isnan(combo.loc[ix[0], "c"]))

    def test_rank_ic_detects_known_positive_ordering(self):
        ix = pd.date_range("2024-01-31", periods=2, freq="ME")
        f = pd.DataFrame([range(60), range(60)], index=ix)
        r = f.copy()
        ic = rank_ic_series(f, r)
        self.assertTrue((ic > 0.99).all())

    def test_quintile_returns_outputs_five_bins(self):
        ix = pd.date_range("2024-01-31", periods=1, freq="ME")
        f = pd.DataFrame([range(100)], index=ix)
        r = f / 100.0
        table = quintile_returns(f, r, min_names=50)
        self.assertEqual(set(table.quintile), {1, 2, 3, 4, 5})
        self.assertGreater(table.loc[table.quintile.eq(5), "mean_return"].iloc[0],
                           table.loc[table.quintile.eq(1), "mean_return"].iloc[0])

    def test_joint_fama_macbeth_returns_conditional_factor_coefficients(self):
        rng = np.random.default_rng(11)
        dates = pd.date_range("2021-01-31", periods=3, freq="ME")
        codes = [f"{i:06d}" for i in range(100)]
        a = pd.DataFrame(rng.normal(size=(3, 100)), index=dates, columns=codes)
        b = pd.DataFrame(rng.normal(size=(3, 100)), index=dates, columns=codes)
        r = 0.02 * a - 0.01 * b + pd.DataFrame(
            rng.normal(scale=0.001, size=(3, 100)), index=dates, columns=codes)
        result = fama_macbeth_multifactor({"dividend_yield": a, "low_volatility": b}, r)
        self.assertEqual(len(result), 3)
        self.assertGreater(result.dividend_yield.mean(), 0)
        self.assertLess(result.low_volatility.mean(), 0)


if __name__ == "__main__":
    unittest.main()
