import unittest

import numpy as np
import pandas as pd

from analysis.staged import (cross_sectional_rank_correlations, demean_by_group,
                             monthly_long_only_backtest, quarterly_long_only_backtest,
                             rank_ic)


class StagedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-28"])
        self.codes = [f"{i:06d}" for i in range(60)]

    def test_rank_ic_detects_monotone_relation(self):
        factor = pd.DataFrame([np.arange(60)] * 3, index=self.dates, columns=self.codes)
        returns = factor * 0.001
        ic = rank_ic(factor, returns, min_names=30)
        self.assertTrue(np.allclose(ic.values, 1.0))

    def test_cross_factor_correlation_is_datewise_ranked(self):
        x = pd.DataFrame([np.arange(60)] * 3, index=self.dates, columns=self.codes)
        y = x * -1
        result = cross_sectional_rank_correlations({"x": x, "y": y})
        value = result.mean_daily_rank_corr.dropna().iloc[0]
        self.assertAlmostEqual(value, -1.0)

    def test_group_demean_removes_group_means(self):
        x = pd.DataFrame([np.arange(60, dtype=float)], index=self.dates[:1], columns=self.codes)
        groups = pd.Series(["a"] * 30 + ["b"] * 30, index=self.codes)
        residual = demean_by_group(x, groups).iloc[0]
        self.assertAlmostEqual(residual.iloc[:30].mean(), 0.0)
        self.assertAlmostEqual(residual.iloc[30:].mean(), 0.0)

    def test_monthly_top_quantile_cost_and_missing_prices(self):
        factor = pd.DataFrame([np.arange(60, dtype=float)] * 3, index=self.dates, columns=self.codes)
        ret = pd.DataFrame(0.01, index=self.dates, columns=self.codes)
        ret.loc[self.dates[0], self.codes[-1]] = np.nan
        portfolio, _ = monthly_long_only_backtest(factor, ret, slippage_bps=10)
        self.assertEqual(portfolio.iloc[0].missing_return_count, 1)
        self.assertTrue(np.isnan(portfolio.iloc[0].gross_return))
        # Initial funding is buy-only: 3bp commission + 10bp slippage.
        self.assertAlmostEqual(portfolio.iloc[0].cost_return, 0.0013)

    def test_quarterly_backtest_uses_four_period_annualization(self):
        dates = pd.to_datetime(["2024-03-31", "2024-06-30"])
        factor = pd.DataFrame([np.arange(60, dtype=float)] * 2, index=dates, columns=self.codes)
        returns = pd.DataFrame(0.03, index=dates, columns=self.codes)
        benchmark = pd.Series(0.01, index=dates)
        panel, summary = quarterly_long_only_backtest(factor, returns, benchmark)
        row = summary[(summary.period == "test_2024_2025") &
                      (summary.return_series == "gross_return")].iloc[0]
        self.assertEqual(len(panel), 2)
        self.assertAlmostEqual(row.annualized_return, 1.03 ** 4 - 1)


if __name__ == "__main__":
    unittest.main()
