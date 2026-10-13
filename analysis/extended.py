"""AlphaLab-inspired, small-sample-safe analysis helpers for real-data studies."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def newey_west_mean(series: pd.Series, max_lag: int = 5) -> dict[str, float]:
    x = series.dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 2:
        return {"mean": np.nan, "nw_se": np.nan, "t_stat": np.nan, "p_value": np.nan, "n": n}
    mu = x.mean()
    u = x - mu
    # Bartlett HAC covariance of the sample mean.
    long_run = np.dot(u, u) / n
    lag = min(max_lag, n - 1)
    for j in range(1, lag + 1):
        gamma = np.dot(u[j:], u[:-j]) / n
        long_run += 2.0 * (1.0 - j / (lag + 1.0)) * gamma
    se = float(np.sqrt(max(long_run, 0.0) / n))
    t = mu / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), df=n - 1) if np.isfinite(t) else np.nan
    return {"mean": float(mu), "nw_se": se, "t_stat": float(t), "p_value": float(p), "n": n}


def quintile_panel(factor: pd.DataFrame, forward_return: pd.DataFrame,
                   turnover: pd.DataFrame | None = None, q: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily cross-sectional equal-weight quintile returns and average membership counts."""
    f, r = factor.align(forward_return, join="inner")
    t = turnover.reindex_like(f) if turnover is not None else None
    ret_rows, meta_rows = [], []
    for date in f.index:
        valid = f.loc[date].notna() & r.loc[date].notna()
        if valid.sum() < q * 5:
            continue
        # rank(method=first) resolves ties deterministically by input column ordering.
        rank = f.loc[date, valid].rank(method="first", pct=True)
        group = np.minimum((rank * q).astype(int) + 1, q)
        for g in range(1, q + 1):
            codes = group.index[group == g]
            ret_rows.append({"date": date, "quintile": g, "return": r.loc[date, codes].mean()})
            meta_rows.append({"date": date, "quintile": g, "n": len(codes),
                              "turnover": float(t.loc[date, codes].mean()) if t is not None else np.nan})
    return pd.DataFrame(ret_rows), pd.DataFrame(meta_rows)


def fama_macbeth_monthly(factor: pd.DataFrame, fwd: pd.DataFrame,
                         controls: pd.DataFrame | None = None) -> pd.DataFrame:
    """Monthly cross-sectional OLS, with complete-case controls and explicit coefficient series.

    This avoids the AlphaLab pasted snippet's fillna(0) for controls, which can manufacture
    an economically meaningless category. HAC aggregation is applied by summarize_fm.
    """
    f, r = factor.align(fwd, join="inner")
    c = controls.reindex(index=f.index, columns=f.columns) if controls is not None else None
    rows = []
    months = f.index.to_period("M")
    for month in months.unique():
        ix = months == month
        # use the final available date in each month as the cross section
        date = f.index[ix][-1]
        x = f.loc[date]
        y = r.loc[date]
        frame = pd.DataFrame({"factor": x, "return": y})
        if c is not None:
            for name in c.columns:
                frame[name] = c.loc[date, name]
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < 50:
            continue
        cols = ["factor"] + (list(c.columns) if c is not None else [])
        X = np.column_stack([np.ones(len(frame)), frame[cols].to_numpy(float)])
        beta = np.linalg.lstsq(X, frame["return"].to_numpy(float), rcond=None)[0]
        rows.append({"month": str(month), **{name: beta[i + 1] for i, name in enumerate(cols)}, "n": len(frame)})
    return pd.DataFrame(rows)


def summarize_fm(beta_history: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    rows = []
    for col in beta_history.columns.drop(["month", "n"], errors="ignore"):
        out = newey_west_mean(beta_history[col], max_lag=max_lag)
        rows.append({"variable": col, **out})
    return pd.DataFrame(rows)
