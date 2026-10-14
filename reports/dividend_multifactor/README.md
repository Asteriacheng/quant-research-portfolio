# Dividend yield + low-volatility study: implementation status

## Scope

This is the offline research path requested for the next study stage: validate a
point-in-time cash-dividend-yield factor first, then compare it with a predeclared
equal-rank dividend-yield/low-volatility composite. It adds no API calls and does not
consume a provider token. Inputs are local CSV files documented in
[`data/templates/README.md`](../../data/templates/README.md).

## What is implemented

- Point-in-time trailing-12-month realized cash-dividend yield in
  `factors/dividend_factors.py`. At each signal date, it selects each event's latest
  public version known by that date, counts only implemented cash distributions with
  ex-dates in the preceding 365 days, and divides by unadjusted closing price.
- A 60-session low-volatility signal in `factors/risk_factors.py`.
- Equal-weight cross-sectional rank combination, monthly Rank IC, quintile sorts,
  individual and joint Fama–MacBeth summaries, factor-rank correlations, and
  top-quintile costed backtests in the research runner and shared analysis helpers.
- The final 24 observed signal months are reserved for testing; earlier signals are
  development. Composite weights are fixed equal weights; no test-period tuning.
  Prices through January 2026 are required to label the December 2025 signal.
- Unit tests for announcement timing, amendments/cancellations, factor direction,
  common-universe ranking, Rank IC, and quintile sorting.
- A benchmark-relative top-quintile backtest with 5/10/20 bp per-side slippage
  scenarios, annual/regime summaries, and formula-window sensitivity.
- An eight-chart export script. Figures are generated only from actual pipeline
  outputs, and the test-period boundary is read from run metadata.

## What has not been run

No dividend-event history with a version/announcement log and no unadjusted-price
CSV is present in the repository. The existing public `astock` snapshot has
forward-adjusted prices and no dividend-announcement history. Therefore the runner
has not generated empirical dividend ICs, portfolio returns, or performance numbers.
The CSV headers under `data/templates/` are schemas only; they contain no market
observations. Do not cite any dividend-strategy result until real licensed inputs are
provided and pass the documented quality checks.

The data source adapter is intentionally not connected. Once local files are placed
at `data/manual_dividend/prices_daily.csv`, `data/manual_dividend/dividend_events.csv`,
and `data/manual_dividend/benchmark_daily.csv`, run:

```bash
python scripts/run_dividend_multifactor.py
```

Outputs are written under this directory: monthly IC series and summaries, quintile
returns, Fama–MacBeth coefficients, factor rank correlation, annual and market-regime
summaries, formula-window sensitivity, monthly net portfolio returns, and run metadata.
Then run `python scripts/make_dividend_figures.py` to export eight PNG figures. No
quarterly sensitivity is currently included in this dividend-specific runner.
Transaction costs use the existing scenario assumptions
(3 bp commission per side, 10 bp slippage per side, and sell-side stamp tax of 10 bp
before 2023-08-28 and 5 bp thereafter). They are assumptions, not broker-specific
historical fills.

## Required next data work

Before interpreting results, supply point-in-time event versions with public dates,
cash per share, ex-date, implementation/cancellation status and stable event IDs;
unadjusted OHLC prices; and a survivorship-aware historical security universe. The
current runner does not claim to handle historical delistings, ST status, suspensions,
limit fills, industry/size neutrality, or dividend reinvestment. Those remain research
limitations until data fields are available and the corresponding checks are added.
