"""Create monthly price/volume ML inputs from the pinned public snapshot.

This is an exploratory real-data case. The source OHLC is forward-adjusted and
the universe is not point-in-time complete, so outputs must not be presented as
an executable or survivorship-free A-share backtest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "astock"
OUT = ROOT / "data" / "manual_ml_snapshot"
SOURCE_COMMIT = "0babe4cf6c1a175c6b84add10803b4b1f9c6000c"
SIGNAL_START, SIGNAL_END = "2021-01-01", "2025-12-31"


def _features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("date").copy()
    close = pd.to_numeric(g["close"], errors="coerce")
    daily_return = close.pct_change(fill_method=None)
    volume = pd.to_numeric(g["volume"], errors="coerce")
    amount = pd.to_numeric(g["amount"], errors="coerce").replace(0, np.nan)
    turnover = pd.to_numeric(g["turn"], errors="coerce")
    g["momentum_5d"] = close.div(close.shift(5)).sub(1)
    g["momentum_20d"] = close.div(close.shift(20)).sub(1)
    g["momentum_60d"] = close.div(close.shift(60)).sub(1)
    g["momentum_12_1"] = close.shift(21).div(close.shift(252)).sub(1)
    g["reversal_5d"] = -g["momentum_5d"]
    g["low_volatility_20d"] = -daily_return.rolling(20, min_periods=15).std()
    g["low_volatility_60d"] = -daily_return.rolling(60, min_periods=45).std()
    mean_volume = volume.rolling(20, min_periods=15).mean()
    g["volume_stability_20d"] = -volume.rolling(20, min_periods=15).std().div(mean_volume)
    g["low_turnover_20d"] = -turnover.rolling(20, min_periods=15).mean()
    g["amihud_liquidity_20d"] = -(daily_return.abs().div(amount).rolling(20, min_periods=15).mean() * 1e6)
    return g


def main() -> None:
    if not list(RAW.glob("kline_*.parquet")):
        raise SystemExit("Run `python data/fetch_real_data.py` first to fetch the pinned source snapshot.")
    chunks = [pd.read_parquet(p) for p in sorted(RAW.glob("kline_*.parquet"))]
    d = pd.concat(chunks, ignore_index=True)
    d["date"] = pd.to_datetime(d["date"])
    d["code"] = d["code"].astype(str).str.zfill(6)
    d = d.sort_values(["code", "date"], kind="stable")
    if d.duplicated(["date", "code"]).any():
        raise SystemExit("duplicate (date, code) records in pinned source")

    # Use one common month-end signal date per exchange calendar. A stock without
    # a row on that date (e.g. missing/suspended) is not assigned a stale signal.
    trading_days = pd.DatetimeIndex(d.date.drop_duplicates().sort_values())
    month_end = pd.Series(trading_days, index=trading_days.to_period("M")).groupby(level=0).last()
    signal_dates = pd.DatetimeIndex(month_end.loc[
        (month_end >= pd.Timestamp(SIGNAL_START)) & (month_end <= pd.Timestamp(SIGNAL_END))
    ].to_numpy())
    day_period = pd.Series(trading_days.to_period("M"), index=trading_days)
    first_days = pd.Series(trading_days, index=trading_days.to_period("M")).groupby(level=0).first()
    last_days = month_end
    signal_months = {date.to_period("M"): date for date in signal_dates}

    # Keep only source fields needed for features and the next-calendar-month label.
    keep = ["code", "date", "open", "close", "volume", "amount", "turn"]
    records = []
    for code, group in d[keep].groupby("code", sort=False):
        g = _features(group)
        by_date = g.set_index("date")
        for period, signal_date in signal_months.items():
            if signal_date not in by_date.index:
                continue
            row = by_date.loc[signal_date]
            next_period = period + 1
            entry_date = first_days.get(next_period)
            exit_date = last_days.get(next_period)
            forward_return = np.nan
            if entry_date in by_date.index and exit_date in by_date.index:
                entry, exit_ = by_date.at[entry_date, "open"], by_date.at[exit_date, "open"]
                if pd.notna(entry) and pd.notna(exit_) and entry > 0:
                    forward_return = float(exit_ / entry - 1)
            records.append({"date": signal_date, "code": code,
                            **{name: row[name] for name in FEATURE_COLUMNS},
                            "forward_return": forward_return})

    panel = pd.DataFrame.from_records(records).sort_values(["date", "code"])
    factors = panel[["date", "code", *FEATURE_COLUMNS]]
    returns = panel[["date", "code", "forward_return"]]
    OUT.mkdir(parents=True, exist_ok=True)
    factors.to_csv(OUT / "factors.csv", index=False, date_format="%Y-%m-%d")
    returns.to_csv(OUT / "forward_returns.csv", index=False, date_format="%Y-%m-%d")
    metadata = {
        "source": "https://github.com/newbiestring-lang/astock",
        "source_commit": SOURCE_COMMIT,
        "source_retrieved_at_utc": json.loads((ROOT / "data/processed/data_manifest.json").read_text())["retrieved_at_utc"],
        "source_ohlc_adjustment": "forward-adjusted; not raw executable prices",
        "universe": "securities present in the source snapshot; survivorship and historical membership are not resolved",
        "requested_signal_window": [SIGNAL_START, SIGNAL_END],
        "actual_month_end_signal_range": [str(signal_dates.min().date()), str(signal_dates.max().date())],
        "development_signal_period": ["2021-01", "2023-12"],
        "reserved_test_signal_period": ["2024-01", "2025-12"],
        "execution_label": "next calendar month's first common trading-session open to last common trading-session open, from source adjusted open prices",
        "features": {"momentum_5d": "close[t]/close[t-5]-1",
                     "momentum_20d": "close[t]/close[t-20]-1",
                     "momentum_60d": "close[t]/close[t-60]-1",
                     "momentum_12_1": "close[t-21]/close[t-252]-1",
                     "reversal_5d": "-momentum_5d",
                     "low_volatility_20d": "negative 20-session standard deviation of daily close returns; minimum 15",
                     "low_volatility_60d": "negative 60-session standard deviation of daily close returns; minimum 45",
                     "volume_stability_20d": "-20-session std(volume)/mean(volume); minimum 15",
                     "low_turnover_20d": "negative 20-session mean source turnover field; minimum 15",
                     "amihud_liquidity_20d": "-1e6 * 20-session mean(abs(daily return)/amount); minimum 15"},
        "row_count": len(panel),
        "codes": int(panel.code.nunique()),
        "signals": int(panel.date.nunique()),
        "labeled_rows": int(panel.forward_return.notna().sum()),
        "unlabeled_rows": int(panel.forward_return.isna().sum()),
        "factors_sha256": hashlib.sha256((OUT / "factors.csv").read_bytes()).hexdigest(),
        "returns_sha256": hashlib.sha256((OUT / "forward_returns.csv").read_bytes()).hexdigest(),
        "limitations": ["no dividends or announcement-version history",
                        "no point-in-time historical universe or delisted-name recovery",
                        "no historical ST, industry, size, or limit-trading status",
                        "forward-adjusted price history can embed later corporate-action adjustments; results are exploratory only"],
    }
    (OUT / "input_manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


FEATURE_COLUMNS = ["momentum_5d", "momentum_20d", "momentum_60d", "momentum_12_1",
                   "reversal_5d", "low_volatility_20d", "low_volatility_60d",
                   "volume_stability_20d", "low_turnover_20d", "amihud_liquidity_20d"]


if __name__ == "__main__":
    main()
