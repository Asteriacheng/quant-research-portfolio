# Dividend research CSV contract (manual/offline input)

This project does not call AKShare, Tushare, or another provider in this workflow.
Place locally obtained, licensed files here; raw files are intentionally ignored by Git.

## `prices_daily.csv`

Required columns: `date,code,open_raw,close_raw,volume,amount,turnover`.
Prices must be unadjusted raw prices (CNY/share); `date` is the exchange trading date;
`code` is a six-digit string. Include the full calendar history needed for rolling
signals and the following holding period. Do not replace missing suspensions with
forward-filled prices. The current `astock` snapshot is forward-adjusted and is not a
substitute for `open_raw` / `close_raw` in dividend-yield or cash-distribution accounting.
As of 2026-10-16, the latest five complete calendar years are 2021–2025. Include
prices from at least 2020-01-01 (for the first trailing-365-day dividend window) through
2026-03-31. The 36 development signal months are 2021-01 through 2023-12; the final
24 test signal months are 2024-01 through 2025-12. Monthly labels need January 2026
prices. Full quarterly-frequency sensitivity needs March 2026 prices to finish the
three-month holding period after the December 2025 signal. Do not extend signals into
2026 year-to-date in this frozen study version; that year is incomplete. Signals are
formed only through 2025-12-31.

## `dividend_events.csv`

One row per public version of a dividend event; preserve every amendment rather than
only the latest snapshot. Required columns:

`event_id,code,version_date,status,cash_per_share,record_date,ex_date,pay_date,fiscal_period_end`

- `event_id`: stable id for the same proposed distribution across amendments.
- `version_date`: date this exact version became public (not report period or ex-date).
- `status`: normalized to `proposed`, `approved`, `implemented`, or `cancelled`.
- `cash_per_share`: cash distribution in CNY per share, not per 10 shares; tax basis must be consistent and documented.
- `ex_date`: ex-dividend date. Empty until known; never infer it from the report period.
- Keep cancelled/amended versions so the as-of routine can reconstruct what was known then.

The primary signal implemented here is trailing-12-month realized cash yield:
sum implemented per-share cash distributions with ex-date in the prior 365 days and
public version available by the signal date, divided by unadjusted close. This is a
backward-looking dividend-yield definition, not a forecast of the next dividend.

## Point-in-time requirements

At each month-end signal, select only the latest version for each event with
`version_date <= signal_date`. A current snapshot containing only final distributions
cannot recreate proposal revisions and must not be backdated to fiscal/report dates.
No historical factor/backtest result should be published until the actual files pass
coverage, duplicate, date, unit, and revision-history checks. Do not commit licensed or
large source data; record source, retrieval date, file hashes, coverage, and license in
a local manifest.


## `benchmark_daily.csv`

Required for benchmark-relative backtests and market-regime breakdowns. Columns are
`date,open_raw,close_raw`; use the same trading calendar and unadjusted price convention
as the stock data. The runner uses the next-month first-open to last-open return for
comparison and a 200-session close moving average for a descriptive regime label.
