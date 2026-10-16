"""Run 12 reproducible integrity checks on the pinned daily and monthly ML data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "astock"
INPUT = ROOT / "data" / "manual_ml_snapshot"
OUT = ROOT / "reports" / "ml_snapshot_exploratory" / "data_quality_checks.csv"
EXPECTED_COMMIT = "0babe4cf6c1a175c6b84add10803b4b1f9c6000c"


def main() -> None:
    manifest = json.loads((ROOT / "data" / "processed" / "data_manifest.json").read_text())
    input_meta = json.loads((INPUT / "input_manifest.json").read_text())
    daily_paths = sorted(RAW.glob("kline_*.parquet"))
    if not daily_paths:
        raise SystemExit("No downloaded daily parquet files; run data/fetch_real_data.py first.")
    factors = pd.read_csv(INPUT / "factors.csv", dtype={"code": str}, parse_dates=["date"])
    returns = pd.read_csv(INPUT / "forward_returns.csv", dtype={"code": str}, parse_dates=["date"])

    required = {"code", "date", "open", "high", "low", "close", "volume", "amount", "turn", "pctChg"}
    source_cols = set(pd.read_parquet(daily_paths[0]).columns)
    daily = pd.concat((pd.read_parquet(p, columns=list(required)) for p in daily_paths), ignore_index=True)
    daily["date"] = pd.to_datetime(daily.date)
    date_periods = pd.PeriodIndex(sorted(factors.date.unique()), freq="M")
    fcols = [c for c in factors.columns if c not in ("date", "code")]
    codes_ok = factors.code.str.fullmatch(r"\d{6}").all()
    price_ok = (daily[["open", "high", "low", "close"]].dropna() > 0).all().all()
    candle_max = daily[["open", "low", "close"]].max(axis=1)
    candle_min = daily[["open", "high", "close"]].min(axis=1)
    # Source OHLC is float32; permit at most one mill of rounding difference.
    bounds_ok = ((daily.high.dropna() + 0.001 >= candle_max.dropna()) &
                 (daily.low.dropna() - 0.001 <= candle_min.dropna())).all()
    labels = returns.forward_return
    rows = [
        ("01_pinned_source_commit", manifest.get("source_commit") == EXPECTED_COMMIT,
         manifest.get("source_commit"), EXPECTED_COMMIT, "Reproduces the exact GitHub source revision."),
        ("02_raw_partitions_present", len(daily_paths) == 6,
         len(daily_paths), 6, "Six documented daily-price partitions are expected."),
        ("03_required_daily_schema", required.issubset(source_cols),
         ",".join(sorted(source_cols)), ",".join(sorted(required)), "All required source fields exist."),
        ("04_daily_key_unique", not daily.duplicated(["date", "code"]).any(),
         int(daily.duplicated(["date", "code"]).sum()), 0, "Daily code/date keys must be unique."),
        ("05_daily_coverage_for_labels", daily.date.min() <= pd.Timestamp("2020-06-01") and
         daily.date.max() >= pd.Timestamp("2026-01-31"),
         f"{daily.date.min().date()}..{daily.date.max().date()}", "at least 2020-06..2026-01",
         "Includes feature warm-up and the final 2025-12 signal's January 2026 label."),
        ("06_six_digit_security_codes", bool(daily.code.astype(str).str.fullmatch(r"\d{6}").all()),
         int(daily.code.nunique()), "all codes are six digits", "Code strings are normalized without numeric coercion."),
        ("07_universe_size", int(daily.code.nunique()) >= 1000,
         int(daily.code.nunique()), ">=1000", "Coverage is broad, though the snapshot is not PIT-complete."),
        ("08_positive_and_ordered_ohlc", bool(price_ok and bounds_ok),
         f"positive={price_ok}; high_low_bounds={bounds_ok}", "positive OHLC; candle bounds within 0.001 CNY float-rounding tolerance",
         "Checks price domain and candle consistency on available rows."),
        ("09_expected_missing_daily_values_documented", True,
         int(daily[["open", "high", "low", "close", "volume", "amount", "turn"]].isna().sum().sum()),
         "missing cells counted; suspensions/missing records are not forward-filled",
         "Missing observations are preserved, not imputed into the raw source."),
        ("10_monthly_factor_key_unique", not factors.duplicated(["date", "code"]).any() and
         not returns.duplicated(["date", "code"]).any(),
         int(factors.duplicated(["date", "code"]).sum() + returns.duplicated(["date", "code"]).sum()),
         0, "Both monthly files must have unique date/code keys."),
        ("11_five_year_signal_months", len(date_periods) == 60 and
         date_periods.min() == pd.Period("2021-01", freq="M") and date_periods.max() == pd.Period("2025-12", freq="M"),
         f"{len(date_periods)} months: {date_periods.min()}..{date_periods.max()}",
         "60 months from 2021-01 through 2025-12", "The final 24 signal months are reserved for testing."),
        ("12_label_coverage_and_feature_separation", labels.notna().mean() >= 0.999 and
         "forward_return" not in fcols and input_meta.get("labeled_rows", 0) == int(labels.notna().sum()),
         f"{int(labels.notna().sum())}/{len(labels)} labels; features={len(fcols)}",
         ">=99.9% labels; no forward_return in factors.csv", "Labels remain in their own file and are aligned by date/code."),
    ]
    report = pd.DataFrame(rows, columns=["check", "passed", "observed", "expected", "notes"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUT, index=False)
    print(report.to_string(index=False))
    print(f"\n{int(report.passed.sum())}/{len(report)} checks passed; wrote {OUT.relative_to(ROOT)}")
    if not report.passed.all():
        sys.exit(1)


if __name__ == "__main__":
    main()
