"""Research-stage utilities for monthly cross-sectional factor diagnostics.

These routines operate on a signal observed at month-end and an open-to-open
return realized over the following calendar month. They are intentionally
separate from the overlapping five-session diagnostic in run_real_study.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rank_ic(factor: pd.DataFrame, forward_return: pd.DataFrame,
            min_names: int = 30) -> pd.Series:
    """Daily/monthly cross-sectional Spearman correlation with complete pairs."""
    f, r = factor.align(forward_return, join="inner", axis=0)
    cols = f.columns.intersection(r.columns)
    f, r = f[cols], r[cols]
    valid = f.notna() & r.notna()
    fr = f.where(valid).rank(axis=1, method="average")
    rr = r.where(valid).rank(axis=1, method="average")
    out = fr.corrwith(rr, axis=1)
    return out.where(valid.sum(axis=1) >= min_names).dropna()


def cross_sectional_rank_correlations(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pairwise correlations of within-date percentile ranks, averaged by period."""
    names = list(factors)
    rows: list[dict[str, object]] = []
    rank_panels = {k: v.rank(axis=1, pct=True) for k, v in factors.items()}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = rank_panels[left].align(rank_panels[right], join="inner")
            daily = a.corrwith(b, axis=1).dropna()
            for split, mask in (("train_2021_2023", daily.index.year <= 2023),
                                ("test_2024_2025", daily.index.year >= 2024)):
                s = daily.loc[mask]
                rows.append({"factor_a": left, "factor_b": right, "period": split,
                             "mean_daily_rank_corr": s.mean(), "n_dates": len(s)})
    return pd.DataFrame(rows)


def demean_by_group(factor: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Cross-sectionally demean by a supplied group label (e.g. industry).

    Caller is responsible for ensuring group labels are point-in-time valid.
    """
    group = groups.reindex(factor.columns)
    out = factor.copy()
    for date in factor.index:
        row = factor.loc[date]
        out.loc[date] = row - row.groupby(group).transform("mean")
    return out


def monthly_long_only_backtest(
    factor: pd.DataFrame,
    forward_return: pd.DataFrame,
    benchmark_return: pd.Series | None = None,
    quantile: float = 0.2,
    commission_bps: float = 3.0,
    slippage_bps: float = 10.0,
    stamp_pre_20230828_bps: float = 10.0,
    stamp_post_20230828_bps: float = 5.0,
    test_start: str = "2024-01-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top-quantile equal-weight monthly portfolio and summary by return period.

    ``forward_return`` is already aligned to signal month-end and represents
    next-month first-open to last-open return. The one-way turnover is half the
    L1 change in portfolio weights. Missing returns are excluded and counted;
    they must not be read as an executable treatment of suspensions/delistings.
    """
    f, r = factor.align(forward_return, join="inner")
    cols = f.columns.intersection(r.columns)
    f, r = f[cols], r[cols]
    prev_weights = pd.Series(dtype=float)
    rows: list[dict[str, object]] = []
    for signal_month in f.index:
        score, ret = f.loc[signal_month], r.loc[signal_month]
        valid = score.notna()
        if int(valid.sum()) < 50:
            continue
        cutoff = score.loc[valid].quantile(1.0 - quantile)
        members = score.index[valid & score.ge(cutoff)]
        if len(members) < 10:
            continue
        weights = pd.Series(1.0 / len(members), index=members)
        shared = weights.index.union(prev_weights.index)
        w0 = prev_weights.reindex(shared, fill_value=0.0)
        w1 = weights.reindex(shared, fill_value=0.0)
        turnover = 1.0 if prev_weights.empty else 0.5 * float((w1 - w0).abs().sum())
        priced = ret.reindex(members)
        gross = float(priced.mean()) if len(priced) == len(members) and priced.notna().all() else np.nan
        sale_month_end = signal_month + pd.offsets.MonthEnd(1)
        stamp = stamp_pre_20230828_bps if sale_month_end < pd.Timestamp("2023-08-28") else stamp_post_20230828_bps
        buy_rate = (commission_bps + slippage_bps) / 10000.0
        sell_rate = (commission_bps + slippage_bps + stamp) / 10000.0
        # Initial funding is buy-only; rebalances have equal buy/sell notionals.
        cost = turnover * (buy_rate if prev_weights.empty else buy_rate + sell_rate)
        bench = benchmark_return.get(signal_month, np.nan) if benchmark_return is not None else np.nan
        rows.append({"signal_month": signal_month, "gross_return": gross,
                     "benchmark_return": bench, "one_way_turnover": turnover,
                     "missing_return_count": int(ret.reindex(members).isna().sum()),
                     "n_holdings": len(members), "cost_return": cost,
                     "net_return": gross - cost if np.isfinite(gross) else np.nan})
        prev_weights = weights
    panel = pd.DataFrame(rows)
    summaries: list[dict[str, object]] = []
    split = pd.Timestamp(test_start)
    for period, mask in (("development", panel.signal_month < split),
                         ("test", panel.signal_month >= split)):
        p = panel.loc[mask]
        # Compare the benchmark over exactly the months where the portfolio has
        # a complete priced holding set; do not give the benchmark extra months.
        matched = p.loc[p.net_return.notna()]
        for series_name in ("gross_return", "net_return", "benchmark_return"):
            source = matched if series_name == "benchmark_return" else p
            x = source[series_name].dropna()
            if x.empty:
                continue
            wealth = (1.0 + x).cumprod()
            dd = wealth / wealth.cummax() - 1.0
            ann = float(wealth.iloc[-1] ** (12.0 / len(x)) - 1.0)
            vol = float(x.std(ddof=1) * np.sqrt(12)) if len(x) > 1 else np.nan
            rf = float(x.mean() * 12.0)
            summaries.append({"period": period, "return_series": series_name,
                              "months": len(x), "cumulative_return": float(wealth.iloc[-1] - 1),
                              "annualized_return": ann, "annualized_volatility": vol,
                              "max_drawdown": float(dd.min()),
                              "sharpe_zero_rf": rf / vol if vol and vol > 0 else np.nan,
                              "mean_one_way_turnover": float(matched.one_way_turnover.mean()) if "one_way_turnover" in matched else np.nan,
                              "mean_cost_per_month": float(matched.cost_return.mean()) if "cost_return" in matched else np.nan})
    return panel, pd.DataFrame(summaries)


def quarterly_long_only_backtest(
    factor: pd.DataFrame,
    quarterly_return: pd.DataFrame,
    benchmark_return: pd.Series | None = None,
    quantile: float = 0.2,
    commission_bps: float = 3.0,
    slippage_bps: float = 10.0,
    stamp_pre_20230828_bps: float = 10.0,
    stamp_post_20230828_bps: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quarter-end Q portfolio evaluated on the following quarter's return.

    Returns must be indexed by quarter-end signal date and cover the following
    three calendar months. Annualization uses four observations per year.
    """
    f, r = factor.align(quarterly_return, join="inner")
    cols = f.columns.intersection(r.columns)
    f, r = f[cols], r[cols]
    prev_weights = pd.Series(dtype=float)
    rows: list[dict[str, object]] = []
    for signal_date in f.index:
        score = f.loc[signal_date]
        valid = score.notna()
        if int(valid.sum()) < 50:
            continue
        cutoff = score.loc[valid].quantile(1.0 - quantile)
        members = score.index[valid & score.ge(cutoff)]
        if len(members) < 10:
            continue
        weights = pd.Series(1.0 / len(members), index=members)
        shared = weights.index.union(prev_weights.index)
        w0, w1 = prev_weights.reindex(shared, fill_value=0.0), weights.reindex(shared, fill_value=0.0)
        turnover = 1.0 if prev_weights.empty else 0.5 * float((w1 - w0).abs().sum())
        priced = r.loc[signal_date].reindex(members)
        gross = float(priced.mean()) if priced.notna().all() else np.nan
        exit_month = signal_date.to_period("M") + 3
        stamp = stamp_pre_20230828_bps if exit_month.end_time < pd.Timestamp("2023-08-28") else stamp_post_20230828_bps
        buy_rate = commission_bps + slippage_bps
        sell_rate = commission_bps + slippage_bps + stamp
        cost = turnover * (buy_rate if prev_weights.empty else buy_rate + sell_rate) / 10000.0
        bench = benchmark_return.get(signal_date, np.nan) if benchmark_return is not None else np.nan
        rows.append({"signal_month": signal_date, "gross_return": gross,
                     "benchmark_return": bench, "one_way_turnover": turnover,
                     "missing_return_count": int(priced.isna().sum()),
                     "n_holdings": len(members), "cost_return": cost,
                     "net_return": gross - cost if np.isfinite(gross) else np.nan})
        prev_weights = weights
    panel = pd.DataFrame(rows)
    summaries = []
    for period, mask in (("train_2021_2023", panel.signal_month.dt.year <= 2023),
                         ("test_2024_2025", panel.signal_month.dt.year >= 2024)):
        p = panel.loc[mask]
        matched = p.loc[p.net_return.notna()]
        for series_name in ("gross_return", "net_return", "benchmark_return"):
            source = matched if series_name == "benchmark_return" else p
            x = source[series_name].dropna()
            if x.empty:
                continue
            wealth = (1 + x).cumprod()
            dd = wealth / wealth.cummax() - 1
            ann = float(wealth.iloc[-1] ** (4.0 / len(x)) - 1)
            vol = float(x.std(ddof=1) * np.sqrt(4.0)) if len(x) > 1 else np.nan
            summaries.append({"period": period, "return_series": series_name,
                              "periods": len(x), "cumulative_return": float(wealth.iloc[-1] - 1),
                              "annualized_return": ann, "annualized_volatility": vol,
                              "max_drawdown": float(dd.min()),
                              "sharpe_zero_rf": float(x.mean() * 4 / vol) if vol and vol > 0 else np.nan,
                              "mean_one_way_turnover": float(matched.one_way_turnover.mean()),
                              "mean_cost_per_period": float(matched.cost_return.mean())})
    return panel, pd.DataFrame(summaries)
