"""Run the optional walk-forward ML factor on offline CSV factor/return panels."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.ml_factor import build_factor_panel, walk_forward_predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("xgboost", "lightgbm"), default="xgboost")
    parser.add_argument("--split-date", default="2024-01-01")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "manual_ml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "ml_factor")
    args = parser.parse_args()
    factor_path = args.input_dir / "factors.csv"
    return_path = args.input_dir / "forward_returns.csv"
    missing = [str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
               for path in (factor_path, return_path) if not path.exists()]
    if missing:
        raise SystemExit(
            "No ML result generated. Supply offline point-in-time panel CSVs using "
            "docs/machine_learning.md; missing: " + ", ".join(missing)
        )
    factor_data = pd.read_csv(factor_path, dtype={"code": str}, parse_dates=["date"])
    returns_data = pd.read_csv(return_path, dtype={"code": str}, parse_dates=["date"])
    for frame, label in ((factor_data, "factors.csv"), (returns_data, "forward_returns.csv")):
        if {"date", "code"}.difference(frame.columns):
            raise SystemExit(f"{label} must include date and code columns")
        frame["code"] = frame.code.str.zfill(6)
        if frame.duplicated(["date", "code"]).any():
            raise SystemExit(f"duplicate date/code rows in {label}")
    if "forward_return" not in returns_data:
        raise SystemExit("forward_returns.csv must include a forward_return column")
    feature_cols = [col for col in factor_data.columns if col not in ("date", "code")]
    if not feature_cols:
        raise SystemExit("factors.csv must contain at least one factor feature column")
    factors = {name: factor_data.pivot(index="date", columns="code", values=name)
               for name in feature_cols}
    forward = returns_data.pivot(index="date", columns="code", values="forward_return")
    panel = build_factor_panel(factors, forward)
    predictions, monthly_ic, importance, parameter_selection = walk_forward_predict(
        panel, feature_cols, split_date=args.split_date, model_type=args.model
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    monthly_ic.to_csv(args.output_dir / "monthly_rank_ic.csv", index=False)
    importance.to_csv(args.output_dir / "feature_importance.csv", index=False)
    parameter_selection.to_csv(args.output_dir / "parameter_selection.csv", index=False)
    print(monthly_ic.to_string(index=False))


if __name__ == "__main__":
    main()
