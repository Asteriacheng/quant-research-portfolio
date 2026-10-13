"""Price/volume factors implemented independently from AlphaLab's definitions.

AlphaLab CN is the formula reference; see docs/research_plan.md for attribution.
All outputs are long-is-good except where a *_raw function is explicitly defined.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def amihud_illiquidity(close: pd.DataFrame, amount: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling mean |daily close return| / traded value (yuan); higher is less liquid."""
    ret = close.pct_change(fill_method=None)
    ratio = ret.abs() / amount.where(amount > 0)
    return ratio.rolling(window, min_periods=max(10, window // 2)).mean() * 1e6


def turnover_20d(turn_pct: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Mean daily turnover fraction; source field is percent, e.g. 1.2 means 1.2%."""
    return (turn_pct / 100.0).rolling(window, min_periods=max(10, window // 2)).mean()


def reversal_5d(close: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Negative prior 5-session close-to-close return; higher means stronger reversal."""
    return -(close / close.shift(window) - 1.0)


def volume_volatility(volume: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Negative rolling coefficient of variation; higher means more stable volume."""
    mean = volume.rolling(window, min_periods=max(10, window // 2)).mean()
    std = volume.rolling(window, min_periods=max(10, window // 2)).std(ddof=1)
    return -(std / mean.where(mean > 0))


def momentum_12_1(close: pd.DataFrame, formation: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-1 momentum: return from t-252 to t-21, matching AlphaLab's source convention."""
    return close.shift(skip) / close.shift(formation) - 1.0


def info_weighted_momentum(close: pd.DataFrame, volume: pd.DataFrame,
                           formation: int = 231, skip: int = 21,
                           volume_window: int = 20, max_weight: float = 5.0) -> pd.DataFrame:
    """AlphaLab-style momentum: sum daily returns weighted by volume / trailing mean.

    Source formula is documented in AlphaLab CN src/factors/momentum.py. This implementation
    uses only same-day and trailing volume and returns up to t-skip, so it is point-in-time.
    """
    daily_ret = close.pct_change(fill_method=None)
    vol_ma = volume.rolling(volume_window, min_periods=volume_window).mean()
    weight = (volume / vol_ma.where(vol_ma > 0)).clip(upper=max_weight)
    weighted_ret = daily_ret * weight
    return weighted_ret.shift(skip).rolling(formation, min_periods=int(formation * .8)).sum()


def factor_map(close: pd.DataFrame, volume: pd.DataFrame, amount: pd.DataFrame,
               turn_pct: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "amihud_20d": -amihud_illiquidity(close, amount),
        "turnover_20d": -turnover_20d(turn_pct),
        "reversal_5d": reversal_5d(close),
        "volume_volatility_20d": volume_volatility(volume),
        "momentum_12_1": momentum_12_1(close),
        "info_weighted_momentum": info_weighted_momentum(close, volume),
    }
