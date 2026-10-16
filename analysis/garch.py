"""Optional walk-forward GARCH market-volatility feature and diagnostic."""
from __future__ import annotations

import numpy as np
import pandas as pd


def walk_forward_garch_market_volatility(
    daily_returns: pd.Series,
    signal_dates: pd.DatetimeIndex | list[pd.Timestamp],
    lookback: int = 500,
    min_observations: int = 120,
    horizon: int = 21,
) -> pd.DataFrame:
    """Forecast next-horizon market volatility from returns available by signal.

    Returns are decimals (e.g. 0.01 for 1%). The fitted GARCH(1,1) is a
    conventional time-series model, not machine learning. It is intended as an
    optional market-regime feature; no social-media sentiment is synthesized.
    """
    try:
        from arch import arch_model
    except ImportError as exc:
        raise ImportError("Install optional GARCH dependency with `pip install -r requirements-garch.txt`") from exc
    series = daily_returns.copy()
    series.index = pd.to_datetime(series.index)
    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if series.index.has_duplicates:
        raise ValueError("daily market returns contain duplicate dates")
    if not np.isfinite(series.to_numpy()).all():
        raise ValueError("daily market returns must be finite")
    rows = []
    for signal_date in pd.DatetimeIndex(pd.to_datetime(signal_dates)).sort_values().unique():
        history = series.loc[series.index <= signal_date].tail(lookback)
        if len(history) < min_observations:
            rows.append({"date": signal_date, "observations": len(history),
                         "forecast_volatility_horizon": np.nan,
                         "status": "skipped_insufficient_history"})
            continue
        model = arch_model(history * 100.0, mean="Constant", vol="GARCH",
                           p=1, q=1, dist="normal", rescale=False)
        fitted = model.fit(disp="off", show_warning=False)
        variance = fitted.forecast(horizon=horizon, reindex=False).variance.iloc[-1]
        predicted_vol = float(np.sqrt(np.maximum(variance.to_numpy(), 0.0).sum()) / 100.0)
        rows.append({"date": signal_date, "observations": len(history),
                     "forecast_volatility_horizon": predicted_vol, "status": "ok",
                     "convergence_flag": int(fitted.convergence_flag)})
    return pd.DataFrame(rows)
