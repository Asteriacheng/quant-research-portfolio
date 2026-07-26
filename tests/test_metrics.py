"""Tests for factorkit.metrics. / metrics 模块自测。

思路:合成数据里埋了一个**已知方向**的 5 日反转 alpha。
如果尺子是准的,反转因子必须测出显著为正的 IC、单调的分组收益;
动量因子必须测出为负。测不出来 = 工具坏了,而不是"市场变了"。
"""

import unittest

from factorkit import (
    annual_cost_drag,
    decay_curve,
    factor_autocorr,
    forward_return,
    ic_summary,
    make_synthetic_panel,
    quantile_stats,
    rank_ic,
    top_turnover,
)


class MetricsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.panel = make_synthetic_panel(n_days=500, n_stocks=200, seed=7)
        cls.fwd = forward_return(cls.panel, horizon=5, mode="close_to_close")
        past5 = cls.panel.close / cls.panel.close.shift(5) - 1.0
        cls.reversal = -past5          # 跌得多的排前面
        cls.momentum = past5

    def test_reversal_ic_is_positive_and_significant(self):
        s = ic_summary(rank_ic(self.reversal, self.fwd))
        self.assertGreater(s.ic_mean, 0.02, "埋入的反转 alpha 应当被测出来")
        self.assertGreater(abs(s.t_stat), 3.0, "多重检验环境下 |t| > 3 才算站住")
        self.assertTrue(s.direction_stable, "前后半段方向必须一致")

    def test_momentum_ic_is_the_mirror_image(self):
        s = ic_summary(rank_ic(self.momentum, self.fwd))
        self.assertLess(s.ic_mean, 0.0, "同一段数据里动量必须为负")

    def test_quantile_monotonicity(self):
        table, rho = quantile_stats(self.reversal, self.fwd, q=10)
        self.assertEqual(len(table), 10)
        self.assertGreater(rho, 0.7, "线性 alpha 应当分组单调,只有两头 = 尾部效应")
        self.assertGreater(table["mean_ret"].iloc[-1], table["mean_ret"].iloc[0])

    def test_turnover_is_a_fraction(self):
        to = top_turnover(self.reversal, top_pct=0.1)
        self.assertTrue((to >= 0).all() and (to <= 1).all())
        self.assertGreater(to.mean(), 0.0)

    def test_slow_factor_has_higher_autocorr_than_fast_factor(self):
        slow = self.panel.close.rolling(60).mean()
        fast = self.panel.close.pct_change()
        self.assertGreater(factor_autocorr(slow), factor_autocorr(fast))
        self.assertGreater(factor_autocorr(slow), 0.9)

    def test_decay_curve_shape(self):
        curve = decay_curve(self.reversal, self.panel, ks=(1, 2, 3, 5, 10))
        self.assertEqual(list(curve["k"]), [1, 2, 3, 5, 10])
        self.assertAlmostEqual(curve["retained"].iloc[0], 1.0, places=9)
        self.assertGreater(curve["ic_mean"].iloc[0], 0.0)

    def test_cost_drag_math(self):
        # 日换手 0.3、双边 25bp、244 个交易日 → 约 18.3%/年
        self.assertAlmostEqual(annual_cost_drag(0.3, 25.0, 244), 0.183, places=3)
        self.assertAlmostEqual(annual_cost_drag(0.0, 25.0), 0.0, places=9)

    def test_forward_return_modes_differ(self):
        c2c = forward_return(self.panel, horizon=5, mode="close_to_close")
        o2o = forward_return(self.panel, horizon=5, mode="open_to_open")
        self.assertEqual(c2c.shape, o2o.shape)
        # 可执行口径整体向后错一天,不应与研究口径完全相同
        self.assertFalse(c2c.equals(o2o))

    def test_rank_ic_rejects_thin_cross_sections(self):
        thin = self.reversal.iloc[:, :5]
        ic = rank_ic(thin, self.fwd.iloc[:, :5], min_obs=30)
        self.assertEqual(len(ic), 0, "截面股票数不足时应当整日丢弃,而不是算个噪声出来")


if __name__ == "__main__":
    unittest.main()
