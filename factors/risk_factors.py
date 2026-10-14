"""Risk factors derived from historical daily prices."""
from __future__ import annotations

import pandas as pd


def low_volatility(close: pd.DataFrame, window: int = 60,
                   min_periods: int | None = None) -> pd.DataFrame:
    """Negative rolling standard deviation of daily close-to-close returns.

    Higher values mean lower realized volatility. The current day's close is
    included, so the value is available only after that close.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    min_periods = min_periods if min_periods is not None else window
    if not 2 <= min_periods <= window:
        raise ValueError("min_periods must be between 2 and window")
    returns = close.pct_change(fill_method=None)
    return -returns.rolling(window, min_periods=min_periods).std(ddof=1)
