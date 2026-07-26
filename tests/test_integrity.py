"""Tests for factorkit.integrity. / 泄漏探针与执行现实性自测。

关键一条:探针必须能抓到**故意写坏**的因子。
一个从来没报过警的检查等于没有检查——所以这里专门写一个偷看未来的因子给它抓。
"""

import unittest

from factorkit import (
    filter_executable,
    limit_up_open,
    make_synthetic_panel,
    truncation_invariance,
    unbuyable_at_open,
)


def clean_factor(panel):
    """只回看:5 日反转。/ backward-looking only."""
    return -(panel.close / panel.close.shift(5) - 1.0)


def leaky_factor(panel):
    """故意偷看未来一天。/ deliberately peeks one day ahead."""
    return panel.close.shift(-1) / panel.close - 1.0


class IntegrityTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.panel = make_synthetic_panel(n_days=400, n_stocks=120, seed=5)
        cls.cuts = [cls.panel.dates[200], cls.panel.dates[300], cls.panel.dates[350]]

    def test_clean_factor_passes(self):
        report = truncation_invariance(clean_factor, self.panel, self.cuts)
        self.assertTrue(report["passed"].all())
        self.assertEqual(report["n_mismatch"].sum(), 0)
        self.assertLess(report["max_abs_diff"].max(), 1e-12)

    def test_leaky_factor_is_caught(self):
        report = truncation_invariance(leaky_factor, self.panel, self.cuts)
        self.assertFalse(report["passed"].any(), "偷看未来的因子必须被抓到")
        self.assertTrue(report["has_nan_mismatch"].all())

    def test_limit_up_mask_is_found(self):
        mask = limit_up_open(self.panel)
        self.assertGreater(int(mask.to_numpy().sum()), 0, "合成数据里注入了一字涨停,应当被识别")

    def test_unbuyable_includes_limit_up(self):
        limit = limit_up_open(self.panel)
        unbuyable = unbuyable_at_open(self.panel)
        self.assertTrue((unbuyable | ~limit).to_numpy().all(), "涨停开盘必须算作买不进")

    def test_filter_executable_drops_unbuyable_and_shifts_by_one_day(self):
        signal = (self.panel.close / self.panel.close.shift(5) - 1.0) < -0.05
        executed = filter_executable(signal, self.panel)
        unbuyable = unbuyable_at_open(self.panel)
        self.assertFalse((executed & unbuyable).to_numpy().any(), "成交日不得落在买不进的格子上")
        self.assertLessEqual(int(executed.to_numpy().sum()), int(signal.to_numpy().sum()))
        # 首日不可能有成交(信号要平移一天)
        self.assertEqual(int(executed.iloc[0].sum()), 0)


if __name__ == "__main__":
    unittest.main()
