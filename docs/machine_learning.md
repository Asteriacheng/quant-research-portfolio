# Walk-forward ML composite factor

This optional module follows AlphaLab CN's rolling ML-factor idea, adapted to the
project's date-by-stock factor panels. It is deliberately separate from the ordinary
IC/Fama–MacBeth workflow and does not call a market-data API.

## Point-in-time protocol

1. `factors.csv` contains only factor values observable at each signal close.
   `forward_returns.csv` contains the corresponding next-period labels, aligned to
   the signal date. Never put the target itself into the feature file.
2. Candidate XGBoost/LightGBM settings are compared using rolling validation folds
   strictly before `--split-date` (24 development months for training and 6 months
   for validation by default). Winsorization bounds and missing-value medians are
   fit on each training fold only.
3. The selected settings are frozen for the reserved test. For each test signal,
   model weights are refit on up to the preceding 36 labeled signal months. Earlier
   test labels can enter later walk-forward fits after those returns become known;
   they never tune parameters. This is a chronological prequential test, not a claim
   that the entire holdout is predicted once from a static model.
4. Output includes stock/date predictions, monthly Rank IC, and per-date tree feature
   importance. Importance is a model diagnostic, not causal attribution.

## Offline input schema

Place local files under `data/manual_ml/` (ignored by Git):

- `factors.csv`: one row per `date,code`, plus one numeric column per point-in-time
  feature, for example `dividend_yield,low_volatility,quality,momentum`.
- `forward_returns.csv`: one row per `date,code`, with a `forward_return` column.

The two files must have unique date/code keys and the same signal-date convention.
Forward labels must begin only after the factor could have been traded. A practical
monthly convention is month-end close signal and next-month open-to-open return.
Keep actual historical delisted names in the input where possible. Do not substitute
the current universe for historical membership.

## Run

Install the normal project requirements, then only if using ML install the optional
estimators and invoke:

```bash
pip install -r requirements-ml.txt
python scripts/run_ml_factor.py --model xgboost --split-date 2024-01-01
# Or: --model lightgbm
```

Results are written to `reports/ml_factor/`: stock/date predictions, monthly Rank IC,
feature importance and the validation scores/selected parameter set. Missing input
data or optional estimator dependencies produce an explicit error; no demo/synthetic
model performance is presented as an A-share result. With a 2021–2025 monthly panel and a 2024 test split,
the development history is relatively short for hyperparameter selection. Treat the
ML model as an exploratory composite and compare its test Rank IC and costed portfolio
with the fixed equal-rank baseline. Do not keep changing the grid after seeing test
results.

## Limitations

This module does not fetch source data, establish historical ST/suspension/limit-up
execution, infer delisting returns, or create point-in-time accounting features.
Those quality and value features must be supplied with publication-date-safe values.
The output is factor research, not a production trading system.
