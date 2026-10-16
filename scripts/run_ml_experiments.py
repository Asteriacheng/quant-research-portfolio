"""Compare a fixed ML model set, K-means diagnostics and frontier portfolios."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.staged import monthly_long_only_backtest
from analysis.garch import walk_forward_garch_market_volatility
from ml.market_models import (efficient_frontier_weights,
                              monthly_kmeans_diagnostics,
                              progressive_feature_sets,
                              walk_forward_random_forest,
                              walk_forward_ridge)
from ml.ml_factor import build_factor_panel, walk_forward_predict, _panel_dates


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    factor_path, return_path = input_dir / "factors.csv", input_dir / "forward_returns.csv"
    missing = [str(path) for path in (factor_path, return_path) if not path.exists()]
    if missing:
        raise SystemExit("No market results generated: missing input(s): " + ", ".join(missing))
    factors = pd.read_csv(factor_path, dtype={"code": str}, parse_dates=["date"])
    returns = pd.read_csv(return_path, dtype={"code": str}, parse_dates=["date"])
    for frame, name in ((factors, "factors.csv"), (returns, "forward_returns.csv")):
        if {"date", "code"}.difference(frame.columns):
            raise SystemExit(f"{name} must include date and code columns")
        frame["code"] = frame.code.str.zfill(6)
        if frame.duplicated(["date", "code"]).any():
            raise SystemExit(f"duplicate date/code rows in {name}")
    if "forward_return" not in returns:
        raise SystemExit("forward_returns.csv must include forward_return")
    feature_cols = [c for c in factors.columns if c not in ("date", "code")]
    if not feature_cols or "forward_return" in feature_cols:
        raise SystemExit("factors.csv must contain factor columns only, never forward_return")
    factor_panels = {c: factors.pivot(index="date", columns="code", values=c)
                     for c in feature_cols}
    ret_panel = returns.pivot(index="date", columns="code", values="forward_return")
    return build_factor_panel(factor_panels, ret_panel), feature_cols


def prediction_pivot(predictions: pd.DataFrame, score_col: str) -> pd.DataFrame:
    return predictions.pivot(index="date", columns="code", values=score_col).sort_index()


def summarize_rf(predictions: pd.DataFrame) -> pd.DataFrame:
    try:
        from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
    except ImportError as exc:
        raise SystemExit("Install ML dependencies: pip install -r requirements-ml.txt") from exc
    rows = []
    for date, group in predictions.dropna(subset=["realized_class"]).groupby("date"):
        y = group.realized_class.astype(int)
        p = group.probability_above_median
        rows.append({"date": date, "n_stocks": len(group),
                     "accuracy": accuracy_score(y, p.ge(0.5)),
                     "roc_auc": roc_auc_score(y, p) if y.nunique() == 2 else np.nan,
                     "brier_score": brier_score_loss(y, p)})
    return pd.DataFrame(rows)


def frontier_backtest(predictions: pd.DataFrame, panel: pd.DataFrame,
                      split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = prediction_pivot(predictions, "prediction")
    returns = panel.forward_return.unstack("code").sort_index()
    previous = pd.Series(dtype=float)
    rows = []
    for date in scores.index[scores.index >= pd.Timestamp(split_date)]:
        expected = scores.loc[date].dropna().nlargest(min(50, scores.loc[date].notna().sum()))
        history = returns.loc[returns.index < date].reindex(columns=expected.index).tail(36)
        coverage = history.notna().sum()
        names = coverage[coverage >= 12].index
        expected = expected.reindex(names)
        history = history.reindex(columns=names).dropna(axis=0, how="any")
        if len(names) < 10 or len(history) < 12:
            rows.append({"date": date, "status": "skipped_insufficient_history"})
            continue
        expected = expected.reindex(history.columns)
        try:
            weights = efficient_frontier_weights(expected, history, previous_weights=previous,
                                                 risk_aversion=4.0, turnover_penalty=0.001,
                                                 max_weight=0.10)
        except (ValueError, RuntimeError) as exc:
            rows.append({"date": date, "status": f"skipped:{exc}"})
            continue
        realized = returns.loc[date].reindex(weights.index)
        gross = float((weights * realized).sum()) if realized.notna().all() else np.nan
        union = weights.index.union(previous.index)
        old, new = previous.reindex(union, fill_value=0), weights.reindex(union, fill_value=0)
        turnover = 1.0 if previous.empty else 0.5 * float((new - old).abs().sum())
        sale_date = date + pd.offsets.MonthEnd(1)
        stamp = 10.0 if sale_date < pd.Timestamp("2023-08-28") else 5.0
        buy_cost, sell_cost = 13.0 / 10000, (13.0 + stamp) / 10000
        cost = turnover * (buy_cost if previous.empty else buy_cost + sell_cost)
        rows.append({"date": date, "status": "ok", "n_holdings": len(weights),
                     "gross_return": gross, "cost_return": cost,
                     "net_return": gross - cost if np.isfinite(gross) else np.nan,
                     "one_way_turnover": turnover,
                     "max_weight": float(weights.max())})
        previous = weights
    result = pd.DataFrame(rows)
    valid = result.loc[(result.status == "ok") & result.net_return.notna()]
    if valid.empty:
        summary = pd.DataFrame([{"period": "test", "months": 0,
                                 "note": "No complete optimized returns; no frontier result generated."}])
    else:
        r = valid.net_return
        wealth = (1 + r).cumprod()
        vol = float(r.std(ddof=1) * np.sqrt(12)) if len(r) > 1 else np.nan
        summary = pd.DataFrame([{
            "period": "test", "months": len(r), "cumulative_return": wealth.iloc[-1] - 1,
            "annualized_return": wealth.iloc[-1] ** (12 / len(r)) - 1,
            "annualized_volatility": vol,
            "max_drawdown": (wealth / wealth.cummax() - 1).min(),
            "sharpe_zero_rf": (r.mean() * 12 / vol) if np.isfinite(vol) and vol > 0 else np.nan,
            "mean_one_way_turnover": valid.one_way_turnover.mean(),
        }])
    return result, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "manual_ml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports" / "ml_experiments")
    parser.add_argument("--split-date", default="2024-01-01")
    parser.add_argument("--regression-model", choices=("xgboost", "lightgbm", "both"), default="both")
    parser.add_argument("--market-returns", type=Path,
                        help="optional date,market_return CSV to add a GARCH(1,1) market-regime feature")
    args = parser.parse_args()
    panel, features = load_inputs(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    garch_status = "not_requested"
    if args.market_returns:
        market = pd.read_csv(args.market_returns, parse_dates=["date"])
        if {"date", "market_return"}.difference(market.columns):
            raise SystemExit("market-returns CSV must include date and market_return columns")
        if market.duplicated("date").any():
            raise SystemExit("market-returns CSV contains duplicate dates")
        market = market.set_index("date").market_return
        garch = walk_forward_garch_market_volatility(market, _panel_dates(panel))
        garch.to_csv(args.output_dir / "garch_market_regime.csv", index=False)
        values = garch.set_index("date").forecast_volatility_horizon
        panel["market_garch_vol_21d"] = panel.index.get_level_values("date").map(values)
        features = features + ["market_garch_vol_21d"]
        garch_status = "requested; forecast is an optional conventional time-series feature"

    # Main regression model, plus fixed ridge and cross-sectional rank baselines.
    models = {}
    regression_models = (["xgboost", "lightgbm"] if args.regression_model == "both"
                         else [args.regression_model])
    regressions = {}
    for model_name in regression_models:
        regression, monthly_ic, importance, tuning = walk_forward_predict(
            panel, features, split_date=args.split_date, model_type=model_name)
        regression.to_csv(args.output_dir / f"{model_name}_predictions.csv", index=False)
        monthly_ic.to_csv(args.output_dir / f"{model_name}_monthly_rank_ic.csv", index=False)
        importance.to_csv(args.output_dir / f"{model_name}_feature_importance.csv", index=False)
        tuning.to_csv(args.output_dir / f"{model_name}_validation.csv", index=False)
        regressions[model_name] = regression
        models[model_name] = prediction_pivot(regression, "prediction")
    realized = panel.forward_return.unstack("code").sort_index()

    ridge = walk_forward_ridge(panel, features, args.split_date)
    ridge.to_csv(args.output_dir / "ridge_predictions.csv", index=False)
    models["ridge"] = prediction_pivot(ridge, "prediction")

    # Equal-weight average of within-month feature percentiles is the fixed rule.
    feature_panels = [panel[c].unstack("code").sort_index().rank(axis=1, pct=True)
                      for c in features]
    rank_factor = sum(feature_panels) / len(feature_panels)
    models["equal_weight_rank"] = rank_factor.loc[rank_factor.index >= pd.Timestamp(args.split_date)]

    rf_metric_tables = []
    portfolio_summaries = []
    for step_name, selected_features in progressive_feature_sets(features).items():
        rf = walk_forward_random_forest(panel, selected_features, args.split_date)
        rf.to_csv(args.output_dir / f"random_forest_{step_name}_predictions.csv", index=False)
        rf_metrics = summarize_rf(rf)
        rf_metrics.insert(0, "feature_step", step_name)
        rf_metrics["features"] = ",".join(selected_features)
        rf_metric_tables.append(rf_metrics)
        score = prediction_pivot(rf, "probability_above_median")
        rf_score, rf_summary = monthly_long_only_backtest(score, realized,
                                                           test_start=args.split_date)
        rf_score.insert(0, "model", f"random_forest_{step_name}")
        rf_summary.insert(0, "model", f"random_forest_{step_name}")
        rf_score.to_csv(args.output_dir / f"portfolio_random_forest_{step_name}.csv", index=False)
        portfolio_summaries.append(rf_summary)
    pd.concat(rf_metric_tables, ignore_index=True).to_csv(
        args.output_dir / "random_forest_feature_addition_metrics.csv", index=False)

    for name, score in models.items():
        monthly, summary = monthly_long_only_backtest(score, realized,
                                                       test_start=args.split_date)
        monthly.insert(0, "model", name)
        summary.insert(0, "model", name)
        monthly.to_csv(args.output_dir / f"portfolio_{name}.csv", index=False)
        portfolio_summaries.append(summary)
    pd.concat(portfolio_summaries, ignore_index=True).to_csv(
        args.output_dir / "model_comparison.csv", index=False)

    cluster_rows, cluster_summaries = monthly_kmeans_diagnostics(panel, features)
    cluster_rows.loc[cluster_rows.date >= pd.Timestamp(args.split_date)].to_csv(
        args.output_dir / "kmeans_test_assignments.csv", index=False)
    cluster_summaries.loc[cluster_summaries.date >= pd.Timestamp(args.split_date)].to_csv(
        args.output_dir / "kmeans_test_summary.csv", index=False)

    frontier_model = "xgboost" if "xgboost" in regressions else "lightgbm"
    frontier, frontier_summary = frontier_backtest(regressions[frontier_model], panel,
                                                    args.split_date)
    frontier.to_csv(args.output_dir / "efficient_frontier_monthly.csv", index=False)
    frontier_summary.to_csv(args.output_dir / "efficient_frontier_summary.csv", index=False)
    metadata = {
        "split_date": args.split_date,
        "regression_models": regression_models,
        "features": features,
        "target": "next-period return for regression; above same-month cross-sectional median for RF classification",
        "random_forest_settings": {"n_estimators": 300, "max_depth": 5,
                                   "min_samples_leaf": 15, "max_features": 0.7},
        "rank_baseline": "equal-weight mean of within-month percentile ranks; feature directions must be pre-specified",
        "portfolio_cost_bps": {"commission_each_side": 3, "slippage_each_side": 10,
                               "sell_stamp_before_2023_08_28": 10,
                               "sell_stamp_after_2023_08_28": 5},
        "frontier": {"prediction_source": frontier_model, "max_weight": 0.10, "risk_aversion": 4.0,
                     "covariance": "Ledoit-Wolf shrinkage of up to 36 prior realized monthly returns",
                     "minimum_history_months": 12},
        "data_caveat": "Inputs are not checked for source accuracy, survivorship, delistings, or point-in-time corporate actions by this runner.",
        "garch": garch_status,
        "feature_addition_order": progressive_feature_sets(features),
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.concat(portfolio_summaries, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
