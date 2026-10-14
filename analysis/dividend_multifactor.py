"""Small, transparent evaluation helpers for dividend + low-volatility studies."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analysis.extended import newey_west_mean


def equal_weight_rank_composite(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Average cross-sectional percentile ranks, requiring all factors per stock."""
    if len(factors) < 2:
        raise ValueError("a composite needs at least two factors")
    names = list(factors)
    base = factors[names[0]].copy()
    aligned = {name: frame.reindex(index=base.index, columns=base.columns)
               for name, frame in factors.items()}
    common = pd.DataFrame(True, index=base.index, columns=base.columns)
    for frame in aligned.values():
        common &= frame.notna()
    # Rank every leg on the same complete-case universe; otherwise missing values
    # make each factor's percentiles refer to a different cross-section.
    ranks = {name: frame.where(common).rank(axis=1, pct=True, method="average")
             for name, frame in aligned.items()}
    stack = pd.concat(ranks, axis=1)
    combo = stack.T.groupby(level=1).mean().T
    return combo.where(common)


def rank_ic_series(factor: pd.DataFrame, forward_returns: pd.DataFrame,
                   min_names: int = 30) -> pd.Series:
    f, r = factor.align(forward_returns, join="inner", axis=0)
    cols = f.columns.intersection(r.columns)
    f, r = f[cols], r[cols]
    valid = f.notna() & r.notna()
    fr = f.where(valid).rank(axis=1, method="average")
    rr = r.where(valid).rank(axis=1, method="average")
    return fr.corrwith(rr, axis=1).where(valid.sum(axis=1) >= min_names).dropna()


def summarize_factor_tests(
    factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    split_date: str = "2024-01-01",
    min_names: int = 30,
    nw_lag: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return monthly IC observations and train/test summaries.

    Factors and returns must already be sampled at the same monthly signal dates;
    this function never chooses factors or weights from the test window.
    """
    split = pd.Timestamp(split_date)
    rows, summaries = [], []
    for name, factor in factors.items():
        ic = rank_ic_series(factor, forward_returns, min_names=min_names)
        rows.extend({"factor": name, "date": date, "rank_ic": value}
                    for date, value in ic.items())
        for period, mask in (("development", ic.index < split), ("test", ic.index >= split)):
            values = ic.loc[mask]
            nw = newey_west_mean(values, max_lag=nw_lag)
            summaries.append({
                "factor": name, "period": period, **nw,
                "ic_std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "icir": float(values.mean() / values.std(ddof=1))
                        if len(values) > 1 and values.std(ddof=1) > 0 else np.nan,
                "positive_ratio": float((values > 0).mean()) if len(values) else np.nan,
            })
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def quintile_returns(factor: pd.DataFrame, forward_returns: pd.DataFrame,
                     q: int = 5, min_names: int = 50) -> pd.DataFrame:
    """Equal-weight forward return by monthly factor quintile."""
    f, r = factor.align(forward_returns, join="inner", axis=0)
    cols = f.columns.intersection(r.columns)
    rows = []
    for date in f.index:
        valid = f.loc[date].notna() & r.loc[date].notna()
        if int(valid.sum()) < max(min_names, q * 2):
            continue
        ranks = f.loc[date, valid].rank(method="first", pct=True)
        groups = np.minimum((ranks * q).astype(int) + 1, q)
        for group in range(1, q + 1):
            members = groups.index[groups == group]
            if len(members):
                rows.append({"date": date, "quintile": group,
                             "n_stocks": len(members),
                             "mean_return": float(r.loc[date, members].mean())})
    return pd.DataFrame(rows)


def monotonicity(quintile_table: pd.DataFrame) -> float:
    """Spearman correlation of quintile number and average quintile return."""
    avg = quintile_table.groupby("quintile")["mean_return"].mean().dropna()
    if len(avg) < 3:
        return float("nan")
    return float(stats.spearmanr(avg.index.to_numpy(), avg.to_numpy()).statistic)


def fama_macbeth_multifactor(
    factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    min_names: int = 50,
) -> pd.DataFrame:
    """Monthly joint cross-sectional regressions of return on all factor ranks.

    Each factor is transformed to a within-date percentile rank and standardized
    cross-sectionally. Coefficients are return associations per one cross-sectional
    standard deviation, conditional on the other supplied factors; they are not
    causal effects. The returned monthly coefficient panel can be summarized with
    a Newey-West mean test by development/test period.
    """
    if not factors:
        raise ValueError("at least one factor is required")
    first = next(iter(factors.values()))
    aligned = {name: value.reindex(index=first.index, columns=first.columns)
               for name, value in factors.items()}
    returns = forward_returns.reindex(index=first.index, columns=first.columns)
    rows = []
    for date in first.index:
        frame = pd.DataFrame({name: value.loc[date] for name, value in aligned.items()})
        frame["forward_return"] = returns.loc[date]
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < max(min_names, len(factors) + 5):
            continue
        x_cols = list(factors)
        for name in x_cols:
            ranks = frame[name].rank(pct=True, method="average")
            sd = ranks.std(ddof=1)
            frame[name] = (ranks - ranks.mean()) / sd if sd > 0 else 0.0
        x = np.column_stack([np.ones(len(frame)), frame[x_cols].to_numpy(float)])
        beta = np.linalg.lstsq(x, frame.forward_return.to_numpy(float), rcond=None)[0]
        rows.append({"date": date, "n_stocks": len(frame),
                     **{name: beta[i + 1] for i, name in enumerate(x_cols)}})
    return pd.DataFrame(rows)
