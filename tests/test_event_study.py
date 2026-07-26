"""Tests for factorkit.event_study. / 事件研究模块自测。

两条底线:
1. 一个**纯随机**的信号,净抬升必须约等于 0 —— 否则基线本身有偏,
   之后所有"净抬升为正"的结论都不可信;
2. 一个**踩在已知 alpha 上**的信号,净抬升必须为正且显著。
"""

import unittest

import numpy as np

from factorkit import (
    bucketed_event_study,
    drop_one_ablation,
    event_study,
    forward_return,
    make_synthetic_panel,
)


class EventStudyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.panel = make_synthetic_panel(n_days=500, n_stocks=200, seed=11)
        cls.fwd = forward_return(cls.panel, horizon=5, mode="close_to_close")
        cls.past5 = cls.panel.close / cls.panel.close.shift(5) - 1.0

    def test_random_signal_has_no_net_lift(self):
        rng = np.random.default_rng(3)
        noise = rng.random(self.fwd.shape) < 0.01
        signal = self.fwd.notna() & noise
        res = event_study(signal, self.fwd, n_draws=300, seed=1)
        self.assertLess(abs(res.net_lift), 0.005, "随机信号不该有净抬升,有就是基线坏了")
        self.assertEqual(res.verdict(), res.verdict())  # verdict 不抛异常

    def test_signal_on_known_alpha_beats_the_baseline(self):
        signal = (self.past5 < -0.08) & self.fwd.notna()      # 5 日跌超 8% 的票
        res = event_study(signal, self.fwd, n_draws=300, seed=1)
        self.assertGreater(res.n_events, 100)
        self.assertGreater(res.net_lift, 0.0)
        self.assertLess(res.p_value, 0.05)
        self.assertTrue(res.verdict().startswith("PASS"))

    def test_win_rate_alone_can_mislead(self):
        """裸胜率 > 50% 不代表有 alpha —— 基线的胜率往往也 > 50%。"""
        signal = (self.past5 > 0.08) & self.fwd.notna()
        res = event_study(signal, self.fwd, n_draws=300, seed=1)
        self.assertGreaterEqual(res.win_rate, 0.0)
        self.assertLessEqual(res.win_rate, 1.0)
        # 该信号方向与埋入的 alpha 相反,净抬升应当为负
        self.assertLess(res.net_lift, 0.0)

    def test_bucketed_table_structure(self):
        signal = (self.past5 < -0.08) & self.fwd.notna()
        year = self.fwd.index.to_series().dt.year.rename("bucket")
        table = bucketed_event_study(signal, self.fwd, year, n_draws=100, seed=1)
        self.assertGreaterEqual(len(table), 2)
        self.assertIn("bucket", table.columns)

    def test_drop_one_ablation_structure(self):
        conditions = {
            "dip(5日跌超5%)": self.past5 < -0.05,
            "liquid(成交额过半)": self.panel.amount.gt(self.panel.amount.median(axis=1), axis=0),
            "mom_confirm(20日为正)": (self.panel.close / self.panel.close.shift(20) - 1.0) > 0,
        }
        table = drop_one_ablation(conditions, self.fwd, n_draws=100, seed=1)
        self.assertEqual(len(table), 4)                       # FULL + 三个 drop
        self.assertEqual(table["variant"].iloc[0], "FULL(全条件)")
        self.assertAlmostEqual(table["delta_vs_full"].iloc[0], 0.0, places=12)
        self.assertTrue(set(table["role"].iloc[1:]) <= {"承重柱 load-bearing", "毒药 harmful", "搭车客 free-rider"})

    def test_too_few_events_is_an_error_not_a_number(self):
        signal = self.past5 < -0.99
        with self.assertRaises(ValueError):
            event_study(signal, self.fwd, n_draws=50, seed=1, min_events=20)


if __name__ == "__main__":
    unittest.main()
