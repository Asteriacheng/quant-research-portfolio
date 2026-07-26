"""factorkit — a small, honest single-factor validation toolkit.

factorkit —— 一个小而诚实的单因子检验工具包。

它不负责挖因子,只负责**判因子的死活**:IC / 换手 / 成本 / 衰减 / 事件净抬升 /
前视泄漏。所有阈值都写在文档里,所有结论都可复现。

Quick start / 快速开始
---------------------
    from factorkit import make_synthetic_panel, forward_return, rank_ic, ic_summary

    panel = make_synthetic_panel()
    fwd = forward_return(panel, horizon=5)
    factor = -(panel.close / panel.close.shift(5) - 1)      # 5 日反转
    print(ic_summary(rank_ic(factor, fwd)).to_dict())
"""

from .panel import Panel, make_synthetic_panel
from .metrics import (
    ICSummary,
    annual_cost_drag,
    correlation_with_pool,
    decay_curve,
    factor_autocorr,
    factor_correlation,
    forward_return,
    ic_summary,
    quantile_stats,
    rank_ic,
    residualize,
    single_day_forward_return,
    top_turnover,
)
from .event_study import EventResult, bucketed_event_study, drop_one_ablation, event_study
from .integrity import (
    filter_executable,
    limit_up_open,
    shuffled_future_test,
    truncation_invariance,
    unbuyable_at_open,
)

__version__ = "0.1.0"

__all__ = [
    "Panel",
    "make_synthetic_panel",
    "forward_return",
    "single_day_forward_return",
    "rank_ic",
    "ic_summary",
    "ICSummary",
    "quantile_stats",
    "top_turnover",
    "factor_autocorr",
    "factor_correlation",
    "correlation_with_pool",
    "residualize",
    "decay_curve",
    "annual_cost_drag",
    "event_study",
    "bucketed_event_study",
    "drop_one_ablation",
    "EventResult",
    "truncation_invariance",
    "limit_up_open",
    "unbuyable_at_open",
    "filter_executable",
    "shuffled_future_test",
    "__version__",
]
