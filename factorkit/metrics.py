"""Single-factor evaluation: IC / ICIR / monotonicity / turnover / decay / cost.

单因子检验:IC / ICIR / 分组单调性 / 换手率 / 衰减曲线 / 成本估算。

顺序是固定的,前一关不过后面全免:
    1. 有效性(RankIC、ICIR、t 值、分组单调性)
    2. 换手率(平滑是拿 IC 换 turnover,不是无脑加窗口)
    3. 成本(高换手因子必须 after-cost 才有发言权)
    4. 衰减(决定持有期与执行速度要求)
经验判读(A 股日频):|IC| > 0.02 可用,> 0.05 很强;ICIR > 0.3 尚可;
多重检验环境下 |t| > 3 才算站住(2 不够)。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "forward_return",
    "single_day_forward_return",
    "rank_ic",
    "ICSummary",
    "ic_summary",
    "quantile_stats",
    "top_turnover",
    "factor_autocorr",
    "factor_correlation",
    "correlation_with_pool",
    "residualize",
    "decay_curve",
    "annual_cost_drag",
]

TRADING_DAYS = 244


# --------------------------------------------------------------------------
# 前向收益 / forward returns
# --------------------------------------------------------------------------
def forward_return(panel, horizon: int = 5, mode: str = "open_to_open") -> pd.DataFrame:
    """Forward return over `horizon` days. / 未来 horizon 日收益。

    mode:
      - ``open_to_open``(默认,**可执行口径**):T 日收盘出信号 → T+1 开盘买入
        → T+1+H 开盘卖出。研究口径必须与执行口径一致,否则回测赚的是
        「T 日尾盘就能成交」这个不存在的假设。
      - ``close_to_close``:纯截面研究口径,便于与文献对齐,不可直接当收益。
    """
    if horizon < 1:
        raise ValueError("horizon 必须 >= 1")
    if mode == "close_to_close":
        px = panel.close
        return px.shift(-horizon) / px - 1.0
    if mode == "open_to_open":
        px = panel.open
        return px.shift(-(1 + horizon)) / px.shift(-1) - 1.0
    raise ValueError(f"未知 mode: {mode}")


def single_day_forward_return(panel, k: int) -> pd.DataFrame:
    """Return of day t+k alone (not cumulative). / 仅第 t+k 天的单日收益。

    衰减曲线用它:因子在 T0 的值,分别对 t+1、t+2 … 的**单日**收益算 IC,
    看信息在第几天耗尽。用累计收益画衰减会被前几天的强信号糊住。
    """
    px = panel.close
    return px.shift(-k) / px.shift(-(k - 1)) - 1.0


# --------------------------------------------------------------------------
# IC
# --------------------------------------------------------------------------
def rank_ic(factor: pd.DataFrame, fwd: pd.DataFrame, min_obs: int = 30) -> pd.Series:
    """Daily cross-sectional Spearman IC. / 日度截面 Spearman 相关(RankIC)。

    只在两边同时非空的股票上排名——先 mask 再 rank,否则缺失票会污染秩次。
    """
    f, r = factor.align(fwd, join="inner")
    mask = f.notna() & r.notna()
    f = f.where(mask)
    r = r.where(mask)
    ic = f.rank(axis=1).corrwith(r.rank(axis=1), axis=1)
    return ic.where(mask.sum(axis=1) >= min_obs).dropna()


@dataclass(frozen=True)
class ICSummary:
    n_days: int
    ic_mean: float
    ic_std: float
    icir: float
    t_stat: float
    positive_ratio: float
    ic_first_half: float
    ic_second_half: float

    @property
    def direction_stable(self) -> bool:
        """前后半段方向是否一致——不一致的因子不许进下一关。"""
        return np.sign(self.ic_first_half) == np.sign(self.ic_second_half)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["direction_stable"] = self.direction_stable
        return d


def ic_summary(ic: pd.Series) -> ICSummary:
    """Aggregate a daily IC series. / 汇总日度 IC 序列。"""
    ic = ic.dropna()
    n = int(ic.size)
    if n < 2:
        raise ValueError("IC 序列太短 / IC series too short")
    mean = float(ic.mean())
    std = float(ic.std(ddof=1))
    half = n // 2
    return ICSummary(
        n_days=n,
        ic_mean=mean,
        ic_std=std,
        icir=float(mean / std) if std > 0 else float("nan"),
        t_stat=float(mean / std * np.sqrt(n)) if std > 0 else float("nan"),
        positive_ratio=float((ic > 0).mean()),
        ic_first_half=float(ic.iloc[:half].mean()),
        ic_second_half=float(ic.iloc[half:].mean()),
    )


# --------------------------------------------------------------------------
# 分组单调性 / quantile monotonicity
# --------------------------------------------------------------------------
def quantile_stats(factor: pd.DataFrame, fwd: pd.DataFrame, q: int = 10) -> tuple[pd.DataFrame, float]:
    """Mean forward return per factor quantile. / 按因子分 q 组的平均前向收益。

    返回 (每组统计表, 单调性 rho)。单调性 = 组序号与组均值的 Spearman 相关。
    只有两头有、中间乱 = 尾部效应,不是线性 alpha——这种因子不能直接当排序分。
    """
    f, r = factor.align(fwd, join="inner")
    mask = f.notna() & r.notna()
    f = f.where(mask)
    r = r.where(mask)
    pct = f.rank(axis=1, pct=True)
    group = np.ceil(pct * q).clip(1, q)

    rows = []
    for g in range(1, q + 1):
        sel = r.where(group == g)
        daily = sel.mean(axis=1)
        rows.append(
            {
                "quantile": g,
                "mean_ret": float(daily.mean()),
                "std_ret": float(daily.std(ddof=1)),
                "n_obs": int(sel.notna().to_numpy().sum()),
            }
        )
    table = pd.DataFrame(rows)
    rho = float(stats.spearmanr(table["quantile"], table["mean_ret"]).statistic)
    return table, rho


# --------------------------------------------------------------------------
# 换手率与自相关 / turnover and autocorrelation
# --------------------------------------------------------------------------
def top_turnover(factor: pd.DataFrame, top_pct: float = 0.1) -> pd.Series:
    """Daily turnover of the top-`top_pct` bucket. / top 组合的日换手率。

    定义 = 今日 top 组中「昨天不在组里」的比例。
    经验门槛:日换手 < 0.3 较舒服,> 0.6 必须处理(平滑或判死)。
    """
    if not 0 < top_pct < 1:
        raise ValueError("top_pct 必须在 (0,1)")
    pct = factor.rank(axis=1, pct=True)
    top = pct > (1 - top_pct)
    prev = top.shift(1).fillna(False).astype(bool)
    entered = (top & ~prev).sum(axis=1)
    size = top.sum(axis=1)
    turnover = (entered / size.replace(0, np.nan)).iloc[1:]
    return turnover.dropna()


def factor_autocorr(factor: pd.DataFrame, lag: int = 1) -> float:
    """Mean cross-sectional rank autocorrelation. / 截面秩自相关均值(换手的镜像读数)。

    > 0.95 慢因子, < 0.5 快因子。
    """
    a = factor.rank(axis=1)
    b = factor.shift(lag).rank(axis=1)
    return float(a.corrwith(b, axis=1).dropna().mean())


# --------------------------------------------------------------------------
# 与池内相关性 / correlation against the existing pool
# --------------------------------------------------------------------------
def factor_correlation(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Mean daily cross-sectional Spearman between two factors. / 两因子的日度截面秩相关均值。"""
    x, y = a.align(b, join="inner")
    mask = x.notna() & y.notna()
    x = x.where(mask).rank(axis=1)
    y = y.where(mask).rank(axis=1)
    return float(x.corrwith(y, axis=1).dropna().mean())


def correlation_with_pool(factor: pd.DataFrame, pool: dict[str, pd.DataFrame]) -> pd.Series:
    """Correlation against every factor already in the pool. / 对池内每一个已有因子算相关。

    门槛:|corr| < 0.7 才谈得上入池,< 0.3 才算真新鲜。
    陷阱:同一条指标公式里拆出来的多个变量,常常是同一信息源的不同写法,
    互相 |corr| 0.8~0.95,只能留一个。
    """
    return pd.Series({name: factor_correlation(factor, other) for name, other in pool.items()})


def residualize(factor: pd.DataFrame, pool: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Cross-sectional residual of `factor` against the pool. / 对池内因子做截面回归取残差。

    降相关的**治本**手法:新因子对旧因子逐日截面回归,只保留残差部分。
    残差还有 IC,才说明它带来的是新钱,而不是把同一笔钱换个写法再赚一遍。
    (换数据源、换构造逻辑通常只是治标——同一想法换个写法往往还是 0.8+ 相关。)
    """
    if not pool:
        return factor.copy()
    frames = [factor] + list(pool.values())
    aligned = [frames[0]]
    for f in frames[1:]:
        aligned.append(f.reindex(index=factor.index, columns=factor.columns))
    y_all = aligned[0]
    x_all = aligned[1:]

    out = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
    for date in factor.index:
        y = y_all.loc[date]
        xs = [x.loc[date] for x in x_all]
        mask = y.notna()
        for x in xs:
            mask &= x.notna()
        if int(mask.sum()) < 30:
            continue
        yv = stats.zscore(y[mask].rank().to_numpy(dtype=float))
        design = np.column_stack([np.ones(int(mask.sum()))] + [
            stats.zscore(x[mask].rank().to_numpy(dtype=float)) for x in xs
        ])
        beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
        out.loc[date, mask.to_numpy()] = yv - design @ beta
    return out


# --------------------------------------------------------------------------
# 衰减与成本 / decay and cost
# --------------------------------------------------------------------------
def decay_curve(factor: pd.DataFrame, panel, ks=(1, 2, 3, 5, 10)) -> pd.DataFrame:
    """IC against the single-day return at t+k. / 因子对 t+k 单日收益的 IC。

    读法:t+3 就归零 = 快因子,必须快执行、对滑点极敏感;
    t+10 仍有 60% 强度 = 慢因子,容量大、可配长持有。
    """
    rows = []
    for k in ks:
        ic = rank_ic(factor, single_day_forward_return(panel, k))
        s = ic_summary(ic)
        rows.append({"k": k, "ic_mean": s.ic_mean, "icir": s.icir, "t_stat": s.t_stat, "n_days": s.n_days})
    out = pd.DataFrame(rows)
    base = out.loc[out["k"] == min(ks), "ic_mean"].iloc[0]
    out["retained"] = out["ic_mean"] / base if base != 0 else np.nan
    return out


def annual_cost_drag(daily_turnover: float, roundtrip_bp: float = 25.0, trading_days: int = TRADING_DAYS) -> float:
    """Rough annual cost drag. / 年化成本粗估。

    年化成本 ≈ 日换手 × 年交易日 × 双边费率。
    示例:日换手 0.3 × 244 × 25bp ≈ 18.3%/年 —— 什么 alpha 都喂不起。
    这一步的作用不是精确,是**在建模之前就先把喂不起的因子劝退**。
    """
    return float(daily_turnover * trading_days * roundtrip_bp / 1e4)
