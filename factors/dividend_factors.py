"""Point-in-time dividend factors from a user-supplied, versioned event table.

No data provider is called here. See data/templates/README.md for the CSV contract.
"""
from __future__ import annotations

import pandas as pd


EVENT_COLUMNS = {
    "event_id", "code", "version_date", "status", "cash_per_share", "ex_date"
}


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    missing = EVENT_COLUMNS.difference(events.columns)
    if missing:
        raise ValueError(f"dividend events missing columns: {sorted(missing)}")
    out = events.copy()
    for col in ("version_date", "ex_date"):
        out[col] = pd.to_datetime(out[col], errors="coerce")
    out["cash_per_share"] = pd.to_numeric(out["cash_per_share"], errors="coerce")
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["event_id"] = out["event_id"].astype(str)
    out["status"] = out["status"].astype(str).str.lower().str.strip()
    out = out.dropna(subset=["version_date"])
    if out.duplicated(["code", "event_id", "version_date"]).any():
        raise ValueError("duplicate dividend-event version; resolve duplicate event_id/code/version_date rows")
    return out


def trailing_cash_dividend_yield(
    events: pd.DataFrame,
    raw_close: pd.DataFrame,
    signal_dates: pd.DatetimeIndex | None = None,
    lookback_days: int = 365,
    implemented_status: str = "implemented",
) -> pd.DataFrame:
    """Return trailing realized cash dividend yield known at each signal date.

    For each signal date, select the latest public version of each dividend event
    available by that date. Sum per-share cash distributions whose ex-date is in
    the trailing ``lookback_days`` and no later than the signal date, then divide
    by that date's *unadjusted* close. This deliberately excludes future ex-dates.
    Rows with no eligible dividend events receive 0 only when a valid close exists.
    """
    if lookback_days < 1:
        raise ValueError("lookback_days must be positive")
    events = _prepare_events(events)
    prices = raw_close.copy()
    prices.index = pd.to_datetime(prices.index)
    prices.columns = prices.columns.astype(str).str.zfill(6)
    dates = pd.DatetimeIndex(signal_dates if signal_dates is not None else prices.index)
    dates = dates.intersection(prices.index).sort_values()
    out = pd.DataFrame(float("nan"), index=dates, columns=prices.columns)

    for date in dates:
        eligible = events.loc[events["version_date"] <= date]
        if eligible.empty:
            out.loc[date] = 0.0
            out.loc[date] = out.loc[date].where(prices.loc[date].notna())
            continue
        latest = (eligible.sort_values(["version_date"])
                  .drop_duplicates(["code", "event_id"], keep="last"))
        latest = latest.loc[
            latest["status"].eq(implemented_status)
            & latest["ex_date"].le(date)
            & latest["ex_date"].ge(date - pd.Timedelta(days=lookback_days))
            & latest["cash_per_share"].gt(0)
        ]
        sums = latest.groupby("code")["cash_per_share"].sum()
        close = prices.loc[date]
        out.loc[date] = 0.0
        for code, cash in sums.items():
            if code in out.columns and pd.notna(close.get(code)) and close[code] > 0:
                out.loc[date, code] = cash / close[code]
        out.loc[date] = out.loc[date].where(close.notna() & close.gt(0))
    return out
