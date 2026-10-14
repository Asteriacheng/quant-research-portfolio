"""Offline runner for dividend-yield + low-volatility factor diagnostics.

Requires manually supplied point-in-time CSV inputs; it does not call any API.
The current public price snapshot is adjusted and therefore intentionally rejected
as a source for this study's raw-price yield and dividend-inclusive return accounting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.dividend_multifactor import (equal_weight_rank_composite,
                                           fama_macbeth_multifactor,
                                           quintile_returns,
                                           rank_ic_series,
                                           summarize_factor_tests)
from analysis.extended import fama_macbeth_monthly, newey_west_mean, summarize_fm
from analysis.staged import (cross_sectional_rank_correlations,
                             monthly_long_only_backtest,
                             quarterly_long_only_backtest)
from factors.dividend_factors import trailing_cash_dividend_yield
from factors.risk_factors import low_volatility

INPUT = ROOT / "data" / "manual_dividend"
OUT = ROOT / "reports" / "dividend_multifactor"
PRICE_FILE = INPUT / "prices_daily.csv"
EVENT_FILE = INPUT / "dividend_events.csv"
BENCHMARK_FILE = INPUT / "benchmark_daily.csv"
SIGNAL_START, SIGNAL_END = "2021-01-01", "2025-12-31"


def main() -> None:
    missing = [str(path.relative_to(ROOT)) for path in (PRICE_FILE, EVENT_FILE, BENCHMARK_FILE) if not path.exists()]
    if missing:
        raise SystemExit(
            "No empirical dividend result generated. Supply the offline CSVs described in "
            "data/templates/README.md; missing: " + ", ".join(missing)
        )
    prices = pd.read_csv(PRICE_FILE, dtype={"code": str}, parse_dates=["date"])
    events = pd.read_csv(EVENT_FILE, dtype={"code": str}, parse_dates=["version_date", "record_date", "ex_date", "pay_date", "fiscal_period_end"])
    benchmark = pd.read_csv(BENCHMARK_FILE, parse_dates=["date"]).set_index("date").sort_index()
    required = {"date", "code", "open_raw", "close_raw"}
    if required.difference(prices.columns):
        raise SystemExit(f"prices_daily.csv missing columns: {sorted(required.difference(prices.columns))}")
    prices["code"] = prices["code"].str.zfill(6)
    if prices.duplicated(["date", "code"]).any():
        raise SystemExit("duplicate (date, code) rows in prices_daily.csv")
    prices = prices.sort_values(["date", "code"])
    if benchmark.index.has_duplicates:
        raise SystemExit("duplicate dates in benchmark_daily.csv")
    if {"open_raw", "close_raw"}.difference(benchmark.columns):
        raise SystemExit("benchmark_daily.csv must contain open_raw and close_raw")
    event_required = {"event_id", "code", "version_date", "status", "cash_per_share", "ex_date"}
    if event_required.difference(events.columns):
        raise SystemExit(f"dividend_events.csv missing columns: {sorted(event_required.difference(events.columns))}")
    events["code"] = events["code"].str.zfill(6)
    events["cash_per_share"] = pd.to_numeric(events["cash_per_share"], errors="coerce")
    if events.duplicated(["code", "event_id", "version_date"]).any():
        raise SystemExit("duplicate code/event_id/version_date rows in dividend_events.csv")
    close = prices.pivot(index="date", columns="code", values="close_raw").sort_index()
    opn = prices.pivot(index="date", columns="code", values="open_raw").reindex_like(close)
    month_dates = pd.Series(close.index, index=close.index.to_period("M")).groupby(level=0).last()
    signal_dates = pd.DatetimeIndex(month_dates.loc[
        (month_dates >= pd.Timestamp(SIGNAL_START)) & (month_dates <= pd.Timestamp(SIGNAL_END))
    ].to_numpy())
    dividend_yield = trailing_cash_dividend_yield(events, close, signal_dates=signal_dates)
    low_vol = low_volatility(close, window=60, min_periods=45).reindex(signal_dates)
    factors = {"trailing_cash_dividend_yield": dividend_yield,
               "low_volatility_60d": low_vol}
    factors["equal_rank_dividend_low_vol"] = equal_weight_rank_composite(
        {"dividend": dividend_yield, "low_volatility": low_vol}
    )

    # Non-overlapping next-calendar-month raw-price return. Cash dividends with
    # ex-date in the holding interval are added once to P&L; signals never use them
    # before their public version dates. Incomplete stock-month labels remain NaN.
    final_events = (events.sort_values("version_date")
                    .drop_duplicates(["code", "event_id"], keep="last"))
    final_events = final_events.loc[final_events["status"].astype(str).str.lower().eq("implemented")]
    month_first = prices.groupby(prices.date.dt.to_period("M")).date.min()
    month_last = prices.groupby(prices.date.dt.to_period("M")).date.max()
    returns = []
    benchmark_returns = {}
    for signal_date in signal_dates:
        next_month = signal_date.to_period("M") + 1
        if next_month not in month_first.index:
            continue
        entry, exit_ = month_first.loc[next_month], month_last.loc[next_month]
        pnl = opn.loc[exit_] / opn.loc[entry] - 1.0
        cash = final_events.loc[
            final_events["ex_date"].gt(entry) & final_events["ex_date"].le(exit_)
        ].groupby("code")["cash_per_share"].sum()
        for code, amount in cash.items():
            if code in pnl.index and pd.notna(opn.loc[entry, code]) and opn.loc[entry, code] > 0:
                pnl.loc[code] += amount / opn.loc[entry, code]
        returns.append((signal_date, pnl))
        if entry in benchmark.index and exit_ in benchmark.index:
            benchmark_returns[signal_date] = benchmark.loc[exit_, "open_raw"] / benchmark.loc[entry, "open_raw"] - 1.0
        else:
            benchmark_returns[signal_date] = float("nan")
    forward = pd.DataFrame({date: row for date, row in returns}).T.reindex(columns=close.columns)
    forward.index = pd.DatetimeIndex(forward.index)
    if len(signal_dates) < 36:
        raise SystemExit("at least 36 month-end observations are required for development and a 24-month test window")
    # Reserve the final 24 available signal months, independent of their observed results.
    split_date = pd.Timestamp(signal_dates[-24])
    test_label_months = sum(date >= split_date for date, _ in returns)
    if test_label_months != 24:
        raise SystemExit(
            f"reserved test has {test_label_months}/24 complete monthly labels; "
            "include complete execution prices through January 2026 in the input files"
        )
    ic_rows, summary = summarize_factor_tests(factors, forward, split_date=str(split_date.date()))
    OUT.mkdir(parents=True, exist_ok=True)
    ic_rows.to_csv(OUT / "monthly_rank_ic.csv", index=False)
    summary.to_csv(OUT / "ic_summary.csv", index=False)
    correlations = cross_sectional_rank_correlations({
        "trailing_cash_dividend_yield": dividend_yield,
        "low_volatility_60d": low_vol,
    })
    correlations.to_csv(OUT / "factor_rank_correlations.csv", index=False)
    fm_rows, fm_summaries = [], []
    backtest_summaries = []
    benchmark_returns = pd.Series(benchmark_returns, dtype=float)
    # Separate descriptive stability summaries by calendar year and market state.
    annual_rows, regime_rows = [], []
    ma200 = benchmark["close_raw"].rolling(200, min_periods=200).mean()
    regimes = pd.Series("unknown", index=benchmark.index)
    regimes.loc[benchmark["close_raw"].gt(ma200)] = "above_200d"
    regimes.loc[benchmark["close_raw"].le(ma200)] = "at_or_below_200d"
    signal_regime = regimes.reindex(signal_dates)
    for name, factor in factors.items():
        ic = rank_ic_series(factor, forward)
        for year, values in ic.groupby(ic.index.year):
            stat = newey_west_mean(values, max_lag=3)
            annual_rows.append({"factor": name, "year": int(year), **stat,
                                "positive_ratio": float((values > 0).mean())})
        joined = pd.DataFrame({"rank_ic": ic}).join(signal_regime.rename("regime"), how="left")
        for regime, values in joined.groupby("regime")["rank_ic"]:
            for period, mask in (("development", values.index < split_date), ("test", values.index >= split_date)):
                part = values.loc[mask]
                if len(part):
                    stat = newey_west_mean(part, max_lag=3)
                    regime_rows.append({"factor": name, "regime": regime, "period": period, **stat,
                                        "positive_ratio": float((part > 0).mean())})
    pd.DataFrame(annual_rows).to_csv(OUT / "annual_ic_summary.csv", index=False)
    pd.DataFrame(regime_rows).to_csv(OUT / "regime_ic_summary.csv", index=False)

    sensitivity_rows = []
    for window in (180, 365, 730):
        alt = trailing_cash_dividend_yield(events, close, signal_dates=signal_dates,
                                           lookback_days=window)
        values = rank_ic_series(alt, forward)
        test = values.loc[values.index >= split_date]
        sensitivity_rows.append({"factor": "dividend_yield", "window": window,
                                 **newey_west_mean(test, max_lag=3)})
    for window in (20, 60, 120):
        alt = low_volatility(close, window=window, min_periods=max(15, int(window * 0.75))).reindex(signal_dates)
        values = rank_ic_series(alt, forward)
        test = values.loc[values.index >= split_date]
        sensitivity_rows.append({"factor": "low_volatility", "window": window,
                                 **newey_west_mean(test, max_lag=3)})
    pd.DataFrame(sensitivity_rows).to_csv(OUT / "formula_sensitivity_test.csv", index=False)

    def run_fama_macbeth(name: str, factor: pd.DataFrame) -> None:
        for period, mask in (("development", factor.index < split_date),
                             ("test", factor.index >= split_date)):
            fm = fama_macbeth_monthly(factor.loc[mask], forward.reindex(factor.index[mask]))
            if not fm.empty:
                fm.insert(0, "period", period)
                fm.insert(0, "factor_name", name)
                fm_rows.append(fm)
                fm_summary = summarize_fm(fm.drop(columns=["factor_name", "period"]))
                fm_summary.insert(0, "period", period)
                fm_summary.insert(0, "factor_name", name)
                fm_summaries.append(fm_summary)

    # Joint Fama–MacBeth tests whether dividend yield and low volatility have
    # conditional cross-sectional associations when included together. It does
    # not assign causal contribution or claim an investable long-short return.
    for period, mask in (("development", dividend_yield.index < split_date),
                         ("test", dividend_yield.index >= split_date)):
        joint = fama_macbeth_multifactor(
            {"dividend_yield": dividend_yield.loc[mask],
             "low_volatility": low_vol.loc[mask]},
            forward.reindex(dividend_yield.index[mask]),
        )
        if not joint.empty:
            joint.insert(0, "period", period)
            fm_rows.append(joint.rename(columns={"dividend_yield": "beta_dividend_yield",
                                                 "low_volatility": "beta_low_volatility"}))
            for column in ("beta_dividend_yield", "beta_low_volatility"):
                stat = newey_west_mean(joint[column], max_lag=3)
                fm_summaries.append(pd.DataFrame([{"period": period,
                                                   "factor_name": "joint_dividend_low_vol",
                                                   "variable": column, **stat}]))

    for name, factor in factors.items():
        table = quintile_returns(factor, forward)
        table.to_csv(OUT / f"quintiles_{name}.csv", index=False)
        run_fama_macbeth(name, factor)
        for slippage in (5.0, 10.0, 20.0):
            portfolio, portfolio_summary = monthly_long_only_backtest(
                factor, forward, benchmark_return=benchmark_returns,
                quantile=0.2, commission_bps=3.0,
            slippage_bps=slippage, stamp_pre_20230828_bps=10.0,
                stamp_post_20230828_bps=5.0, test_start=str(split_date.date()),
            )
            portfolio.to_csv(OUT / f"backtest_monthly_{name}_{int(slippage)}bp.csv", index=False)
            if not portfolio_summary.empty:
                portfolio_summary.insert(0, "slippage_bps_per_side", slippage)
                portfolio_summary.insert(0, "factor_name", name)
                backtest_summaries.append(portfolio_summary)

    # Frequency sensitivity: retain only calendar quarter-end signals and hold
    # through the next three calendar months. Require a complete Q1-2026 label
    # for the December-2025 signal, so the quarterly holdout has eight quarters.
    quarter_dates = pd.DatetimeIndex([
        date for date in signal_dates if date.month in (3, 6, 9, 12)
    ])
    quarterly_returns, quarterly_benchmark = [], {}
    for signal_date in quarter_dates:
        months = [signal_date.to_period("M") + offset for offset in (1, 2, 3)]
        if any(month not in month_first.index for month in months):
            continue
        entry, exit_ = month_first.loc[months[0]], month_last.loc[months[-1]]
        pnl = opn.loc[exit_] / opn.loc[entry] - 1.0
        cash = final_events.loc[
            final_events["ex_date"].gt(entry) & final_events["ex_date"].le(exit_)
        ].groupby("code")["cash_per_share"].sum()
        for code, amount in cash.items():
            if code in pnl.index and pd.notna(opn.loc[entry, code]) and opn.loc[entry, code] > 0:
                pnl.loc[code] += amount / opn.loc[entry, code]
        quarterly_returns.append((signal_date, pnl))
        if entry in benchmark.index and exit_ in benchmark.index:
            quarterly_benchmark[signal_date] = (
                benchmark.loc[exit_, "open_raw"] / benchmark.loc[entry, "open_raw"] - 1.0
            )
        else:
            quarterly_benchmark[signal_date] = float("nan")
    quarterly_forward = pd.DataFrame({date: row for date, row in quarterly_returns}).T
    quarterly_forward = quarterly_forward.reindex(columns=close.columns)
    quarterly_forward.index = pd.DatetimeIndex(quarterly_forward.index)
    expected_quarters = sum(date >= split_date for date in quarter_dates)
    actual_quarters = sum(date >= split_date for date, _ in quarterly_returns)
    if actual_quarters != expected_quarters:
        raise SystemExit(
            f"quarterly sensitivity has {actual_quarters}/{expected_quarters} complete labels; "
            "include prices through March 2026 for the December 2025 signal"
        )
    quarterly_benchmark = pd.Series(quarterly_benchmark, dtype=float)
    quarterly_summaries = []
    for name, factor in factors.items():
        quarter_factor = factor.reindex(quarter_dates)
        panel, summary = quarterly_long_only_backtest(
            quarter_factor, quarterly_forward,
            benchmark_return=quarterly_benchmark, quantile=0.2,
            commission_bps=3.0, slippage_bps=10.0,
            stamp_pre_20230828_bps=10.0, stamp_post_20230828_bps=5.0,
        )
        panel.insert(0, "factor_name", name)
        panel.to_csv(OUT / f"backtest_quarterly_{name}.csv", index=False)
        if not summary.empty:
            summary.insert(0, "factor_name", name)
            quarterly_summaries.append(summary)
    if quarterly_summaries:
        pd.concat(quarterly_summaries, ignore_index=True).to_csv(
            OUT / "quarterly_backtest_summary.csv", index=False
        )
    if fm_rows:
        pd.concat(fm_rows, ignore_index=True).to_csv(OUT / "fama_macbeth_monthly_coefficients.csv", index=False)
    if fm_summaries:
        pd.concat(fm_summaries, ignore_index=True).to_csv(OUT / "fama_macbeth_summary.csv", index=False)
    if backtest_summaries:
        pd.concat(backtest_summaries, ignore_index=True).to_csv(OUT / "backtest_summary.csv", index=False)
    metadata = {
        "data_mode": "manual offline CSV; no provider API called",
        "signal_window": [SIGNAL_START, SIGNAL_END],
        "signal": "month-end close; next calendar month first open entry; January 2026 prices complete the December 2025 label",
        "label": "next calendar month last open plus cash dividends with ex-date during holding interval",
        "dividend_factor": "trailing 365-day implemented cash per share known by signal / raw close",
        "low_vol_factor": "negative 60-session standard deviation of raw-price daily returns",
        "composite": "equal-weight average of monthly cross-sectional percentile ranks, complete cases",
        "development": "signals before 2024-01-01",
        "reserved_test_start": str(split_date.date()),
        "reserved_test": "last 24 observed signal months; rule fixed without inspecting outcomes",
        "frequency_sensitivity": "quarter-end signals, hold next three full calendar months, same fixed top-quintile rule; requires data through March 2026",
        "factor_selection_and_weights": "none selected from test; equal ranks fixed before test",
        "cost_scenarios": "commission 3bp per side; slippage 5/10/20bp per side; sell tax 10bp before 2023-08-28 then 5bp",
        "input_rows": int(len(prices)),
        "codes": int(prices.code.nunique()),
        "events": int(events.event_id.nunique()),
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
