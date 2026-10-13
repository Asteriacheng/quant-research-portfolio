# Real-data factor study — plan and provenance

## Scope fixed before reading test results

- Universe: ordinary six-digit A-share codes in the public snapshot; the universe at date *t* is proxied by securities with a row observed by *t*. It is not a complete point-in-time exchange listing universe.
- Research window: 2021-01-04 through 2025-12-31 (five complete calendar years available in the snapshot).
- Exploration/development: through 2023-12-29. Reserved test: 2024-01-01 through 2025-12-31. Formula definitions are copied from AlphaLab CN source conventions before reading this run's test statistics. No test-based tuning has been performed.
- Signal/label: close at date *t*; enter at the next observed session's open; exit at open after five further observed sessions. The resulting five-session labels overlap and are factor diagnostics, not a monthly portfolio backtest.
- Baseline: published Shanghai Composite daily series, supplied as an index parquet file. Not a broad-market total-return benchmark; this initial factor diagnostic does not report benchmark-relative strategy returns.

## Data source and limits

The source snapshot is [newbiestring-lang/astock](https://github.com/newbiestring-lang/astock), pinned to commit `0babe4cf6c1a175c6b84add10803b4b1f9c6000c`. Retrieved 2026-10-13 UTC. Raw parquet files and the source README are stored in `data/raw/astock/`; the processed 2021–2025 rows and SHA256 manifest are in `data/processed/`.

The source README reports 5,197 stocks, daily OHLCV/amount/turnover/percent-change fields, 2020-06 to 2026-04 coverage, forward-adjusted prices, approximately 685 stocks with early-history gaps and approximately 504 with terminal suspensions. The downloaded snapshot's kline parquet files contain indices as well as stocks; after filtering six-digit codes and 2021–2025, this study has 5,504,248 rows and 4,902 codes. Per-code missing dates are retained as missing; duplicate `(code,date)` rows were audited.

Important unavailable fields: historic delisted-security master/listing dates, daily ST and suspension flags, point-in-time industry membership, market capitalization, dividend amount, announcement/publication date, ex-dividend date, and share adjustment factors. Therefore: (1) the present universe retains survivorship and coverage bias; (2) ST/limit/suspension executable filters and sector/size controls are not yet defensible; (3) cash dividend returns are excluded; (4) forward-adjusted source prices may reflect later adjustment information, and the source snapshot does not expose adjustment factors. This version must not be presented as a dividend-factor study or investable strategy backtest.

## Factors and conventions

Implementation references AlphaLab CN `src/factors/sentiment.py` and `src/factors/momentum.py`; this project recalculates signals from the raw source fields, without copying AlphaLab output tables:

1. Amihud illiquidity: `mean_20(abs(close_t/close_{t-1}-1) / amount_t) * 1e6`; implementation flips sign so higher is more liquid / more desirable. Amount is yuan.
2. 20-day turnover: `-mean_20(turn_pct / 100)`; aligned to the AlphaLab source's negative-turnover convention.
3. 5-day reversal: `-(close_t/close_{t-5}-1)`.
4. Volume volatility: `-std_20(volume)/mean_20(volume)`.
5. 12-1 momentum: `close_{t-21}/close_{t-252}-1`.
6. Info-weighted momentum (optional/reference formula): daily return times `min(volume / trailing_20d_mean_volume, 5)`, shifted 21 sessions and summed over 231 sessions (80% minimum observations).

The first five definitions follow the identified AlphaLab formulas, except Amihud uses actual `amount` (yuan) instead of the reference implementation's `close*volume` proxy. Their differences are documented and can change results. No dividends are double-counted because no dividend cash return is added; adjusted close is used solely as the snapshot's price series.

## Analysis sequence and AlphaLab method reuse

Used/adapted: Rank IC, HAC/Newey–West mean standard error and t statistic; daily quintile sorts; factor autocorrelation/turnover output; Fama–MacBeth monthly cross-sections with complete-case controls; subperiod and regime grouping; turnover/cost framework; output tables/plots. AlphaLab's pasted Fama–MacBeth example fills missing controls with zero; this project intentionally uses complete cases instead. Factor-zoo redundancy and multi-factor portfolio are later stages after factor screening. ML/LLM report generation is out of scope until there is valid point-in-time fundamental/dividend data.

Costs are not yet subtracted from strategy returns. For a future investable backtest, preregister base one-way commission 3bp, sell stamp duty at the legally applicable historical rate, and 10bp one-way slippage; run 5/10/20bp one-way slippage sensitivities. These are assumptions to test, not observed broker costs. The present study outputs top-20% membership replacement as a diagnostic, not an executable filled turnover series.

## Quality checks (12)

Run/record these against the preserved raw inputs and processed output: (1) expected files exist; (2) required columns/types; (3) valid six-digit code; (4) dates inside declared coverage; (5) no duplicate code-date; (6) positive OHLC prices; (7) OHLC ordering; (8) nonnegative volume/amount/turnover; (9) null counts; (10) per-code date gaps and first/last observation coverage; (11) source/processed row, code, and date reconciliation; (12) processed-file/source commit/hash manifest. Checks 1–9 and 11–12 are feasible now; gap coverage is reported but cannot distinguish suspension from source omission without status history.

## How to reproduce

From the project root (Python 3.10+ recommended):

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-real.txt
python data/fetch_real_data.py
python scripts/run_real_study.py
python scripts/make_figures.py
python -m unittest discover -s tests -t .
```

The first command reads the included raw snapshot and produces the processed parquet plus a timestamp/hash manifest. The factor run writes `reports/real_data_v1/`. Use the saved source snapshot for exact reproduction; upstream `main` may update.
