"""Leakage probes and execution-reality filters.

前视泄漏探针与执行现实性过滤。

一条判断标准:一个回测框架靠不靠谱,看它**拒绝**结果的流程,不看它产生结果的流程。
本模块提供三件在因子层最常用的机械检查:

1. 截断不变性(truncation invariance)—— 判因子有没有偷看未来的**定版探针**;
2. T+1 可执行性 —— 信号在 T 日收盘产生,能不能在 T+1 开盘真的买到;
3. shuffled-future 置换 —— 组合级好用,**因子层对反转/持续类因子有结构性假阳性**,
   只当零假设带对照,不当判分依据(这条是踩过坑之后写下来的)。
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

__all__ = [
    "truncation_invariance",
    "limit_up_open",
    "unbuyable_at_open",
    "filter_executable",
    "shuffled_future_test",
]


# --------------------------------------------------------------------------
# 1. 截断不变性 / truncation invariance
# --------------------------------------------------------------------------
def truncation_invariance(
    factor_fn: Callable[[object], pd.DataFrame],
    panel,
    cut_dates,
    tol: float = 1e-9,
) -> pd.DataFrame:
    """Does the factor value at date t change when future data is removed?

    把数据截断到某天,重算因子,和全量算出来的值逐格比较。

    原理:一个只回看的因子,在 T 日的取值只依赖 ≤T 的数据,因此**无论后面有没有
    数据,它在 T 日的值都必须一模一样**。只要有一格不同(含「一边有值一边是 NaN」),
    就说明该因子用到了未来信息。

    这是因子级泄漏判分的**定版口径**——它是结构性证明,不是统计检验,
    不受因子类型影响,也不会被样本噪声糊弄。

    Returns
    -------
    DataFrame: 每个截断点一行,列含 max_abs_diff / n_mismatch / passed。
    """
    full = factor_fn(panel)
    rows = []
    for cut in cut_dates:
        cut = pd.Timestamp(cut)
        sub = factor_fn(panel.slice_to(cut))
        idx = sub.index.intersection(full.index)
        cols = sub.columns.intersection(full.columns)
        a = full.loc[idx, cols].to_numpy(dtype=float)
        b = sub.loc[idx, cols].to_numpy(dtype=float)

        both_nan = np.isnan(a) & np.isnan(b)
        diff = np.abs(a - b)
        diff = np.where(both_nan, 0.0, diff)
        diff = np.where(np.isnan(diff), np.inf, diff)   # 一边 NaN 一边有值 = 明确不一致

        n_bad = int((diff > tol).sum())
        finite = diff[np.isfinite(diff)]
        rows.append(
            {
                "cut_date": cut.date().isoformat(),
                "cells_compared": int(diff.size),
                "n_mismatch": n_bad,
                "max_abs_diff": float(finite.max()) if finite.size else 0.0,
                "has_nan_mismatch": bool(np.isinf(diff).any()),
                "passed": n_bad == 0,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. T+1 可执行性 / execution reality
# --------------------------------------------------------------------------
def limit_up_open(panel, limit_pct: float = 0.10, tol: float = 0.005) -> pd.DataFrame:
    """Mask of "opened at the limit-up price". / 开盘即涨停的掩码。

    A 股主板 ±10%、创业板/科创板 ±20%、ST ±5%——limit_pct 请按标的实际制度传,
    不要一个数走天下。
    """
    limit_price = (panel.prev_close() * (1 + limit_pct)).round(2)
    return panel.open >= (limit_price - tol)


def unbuyable_at_open(panel, limit_pct: float = 0.10) -> pd.DataFrame:
    """Cannot be bought at the open. / T+1 开盘买不进的掩码。

    两种情况:①开盘即涨停(挂单排不上);②停牌(数据缺行 → NaN)。
    这两件事是**确定性约束**,必须写成硬闸门,绝不能作为特征喂进模型让它「学」——
    确定性的东西用确定性方式实现,学出来的 99% 正确率在剩下 1% 上就是事故。
    """
    return limit_up_open(panel, limit_pct=limit_pct) | panel.open.isna()


def filter_executable(signal: pd.DataFrame, panel, limit_pct: float = 0.10) -> pd.DataFrame:
    """Move a T-close signal to T+1 open and drop the unbuyable ones.

    把 T 日收盘产生的信号平移到 T+1 开盘,并剔除买不进的部分。

    返回的宽表:行 = 实际成交日(T+1),True = 该日该股确实可以买入。
    报告里必须写清「弃单率」:弃单率高本身就是策略缺陷(信号总在涨停日出现,
    意味着回测里赚的那部分收益现实中拿不到)。
    """
    shifted = signal.fillna(False).astype(bool).shift(1).fillna(False).astype(bool)
    return shifted & ~unbuyable_at_open(panel, limit_pct=limit_pct)


# --------------------------------------------------------------------------
# 3. 置换检验(带告示的工具)/ permutation test (use with the caveat)
# --------------------------------------------------------------------------
def shuffled_future_test(
    factor: pd.DataFrame,
    fwd: pd.DataFrame,
    n_draws: int = 200,
    seed: int = 0,
) -> dict:
    """Null band from date-shuffled labels. / 打乱标签日期得到的零假设带。

    做法:把前向收益的**日期顺序**打乱,重算 IC,得到「没有真实对应关系时」
    IC 该有多大。真信号应当显著跑出这条带。

    ⚠️ 已知局限(踩过的坑,必须随工具一起交付):
    对**反转/持续类因子**,打乱后的标签窗口可能落进因子的「过去」,而反转因子
    与过去收益相关本身是合法信息 → 结构性假阳性;个股固定效应(如低波股在
    熊市段整体占优)同样污染该探针。
    因此因子层判泄漏一律以 `truncation_invariance` 为准,本函数只作零假设带对照。
    组合级 shuffled-future(在回测框架里做)不受此影响,照用。
    """
    from .metrics import ic_summary, rank_ic

    actual = ic_summary(rank_ic(factor, fwd)).ic_mean
    rng = np.random.default_rng(seed)
    dates = fwd.index.to_numpy()
    null = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        shuffled = fwd.copy()
        shuffled.index = pd.DatetimeIndex(rng.permutation(dates))
        shuffled = shuffled.sort_index()
        try:
            null[i] = ic_summary(rank_ic(factor, shuffled)).ic_mean
        except ValueError:
            null[i] = np.nan
    null = null[~np.isnan(null)]
    return {
        "actual_ic": float(actual),
        "null_mean": float(null.mean()),
        "null_abs_p95": float(np.percentile(np.abs(null), 95)),
        "outside_null_band": bool(abs(actual) > np.percentile(np.abs(null), 95)),
        "n_draws": int(null.size),
        "caveat": "因子层判泄漏以截断不变性为准;本检验仅作零假设带对照。",
    }
