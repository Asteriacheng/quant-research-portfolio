"""Walk-forward ML composite factor for monthly cross-sectional A-share research.

The API accepts already point-in-time factor panels and forward-return labels.
Hyperparameters are selected only on development-period validation folds. During
the reserved test, model weights may update with labels that have become observable
by each prediction date, while the chosen hyperparameters remain frozen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DEFAULT_PARAM_GRID: dict[str, list[dict[str, Any]]] = {
    "xgboost": [
        {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 80,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
        {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 120,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 2.0},
    ],
    "lightgbm": [
        {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 80,
         "num_leaves": 7, "subsample": 0.8, "colsample_bytree": 0.8},
        {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 120,
         "num_leaves": 15, "subsample": 0.8, "colsample_bytree": 0.8},
    ],
}


@dataclass(frozen=True)
class TemporalWindows:
    train_dates: pd.DatetimeIndex
    valid_dates: pd.DatetimeIndex


def build_factor_panel(
    factors: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Combine wide date-by-stock factor panels into a long MultiIndex panel."""
    if not factors:
        raise ValueError("at least one factor panel is required")
    base = forward_returns.copy()
    base.index = pd.to_datetime(base.index)
    base.columns = base.columns.astype(str)
    base.index.name = "date"
    base.columns.name = "code"
    frames = []
    for name, values in factors.items():
        panel = values.copy()
        panel.index = pd.to_datetime(panel.index)
        panel.columns = panel.columns.astype(str)
        panel = panel.reindex(index=base.index, columns=base.columns)
        frames.append(panel.rename_axis(index="date", columns="code").stack(dropna=False).rename(name))
    target = base.rename_axis(index="date", columns="code").stack(dropna=False).rename("forward_return")
    result = pd.concat(frames + [target], axis=1)
    result.index = result.index.set_names(["date", "code"])
    result = result.sort_index()
    if len(set(factors).intersection({"forward_return"})):
        raise ValueError("a feature cannot be named forward_return")
    return result


def rolling_train_validation_dates(
    dates: pd.DatetimeIndex,
    validation_end_position: int,
    train_months: int,
    valid_months: int,
) -> TemporalWindows:
    """Return contiguous monthly training/validation dates ending before origin."""
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values().unique()
    start = validation_end_position - train_months - valid_months
    if train_months < 1 or valid_months < 1 or start < 0:
        raise ValueError("not enough earlier dates for requested temporal windows")
    train = dates[start:start + train_months]
    valid = dates[start + train_months:validation_end_position]
    if len(train) != train_months or len(valid) != valid_months:
        raise ValueError("temporal windows do not have the requested lengths")
    return TemporalWindows(train_dates=train, valid_dates=valid)


def _make_estimator(model_type: str, params: dict[str, Any], random_state: int):
    if model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("Install optional ML dependencies with `pip install -r requirements-ml.txt` (xgboost missing)") from exc
        return XGBRegressor(objective="reg:squarederror", n_jobs=1,
                            random_state=random_state, verbosity=0,
                            tree_method="hist", **params)
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("Install optional ML dependencies with `pip install -r requirements-ml.txt` (lightgbm missing)") from exc
        return LGBMRegressor(objective="regression", n_jobs=1,
                             random_state=random_state, verbosity=-1,
                             deterministic=True, force_col_wise=True, **params)
    raise ValueError("model_type must be 'xgboost' or 'lightgbm'")


def _fit_transform(train: pd.DataFrame, other: pd.DataFrame,
                   features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fit clipping bounds and median imputation on training rows only."""
    lo = train[features].quantile(0.01)
    hi = train[features].quantile(0.99)
    clipped_train = train[features].clip(lower=lo, upper=hi, axis=1)
    clipped_other = other[features].clip(lower=lo, upper=hi, axis=1)
    median = clipped_train.median().fillna(0.0)
    return (clipped_train.fillna(median).to_numpy(dtype=float),
            clipped_other.fillna(median).to_numpy(dtype=float))


def _date_ic(frame: pd.DataFrame, prediction_col: str = "prediction") -> float:
    if len(frame) < 3:
        return float("nan")
    result = spearmanr(frame[prediction_col], frame["forward_return"]).statistic
    return float(result) if np.isfinite(result) else float("nan")


def _panel_dates(panel: pd.DataFrame) -> pd.DatetimeIndex:
    if not isinstance(panel.index, pd.MultiIndex) or panel.index.nlevels != 2:
        raise ValueError("panel index must be a MultiIndex(date, code)")
    names = list(panel.index.names)
    if names != ["date", "code"]:
        raise ValueError("panel index levels must be named ('date', 'code')")
    return pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()


def _validation_score(
    panel: pd.DataFrame,
    features: list[str],
    train_dates: pd.DatetimeIndex,
    valid_dates: pd.DatetimeIndex,
    model_type: str,
    params: dict[str, Any],
    min_train_rows: int,
    min_valid_names: int,
    random_state: int,
) -> list[float]:
    train = panel.loc[panel.index.get_level_values("date").isin(train_dates)]
    train = train.dropna(subset=["forward_return"])
    if len(train) < min_train_rows:
        return []
    x_train, _ = _fit_transform(train, train, features)
    estimator = _make_estimator(model_type, params, random_state)
    estimator.fit(x_train, train.forward_return.to_numpy(dtype=float))
    scores = []
    for date in valid_dates:
        valid = panel.xs(date, level="date", drop_level=False).dropna(subset=["forward_return"])
        if len(valid) < min_valid_names:
            continue
        _, x_valid = _fit_transform(train, valid, features)
        sample = valid.copy()
        sample["prediction"] = estimator.predict(x_valid)
        score = _date_ic(sample)
        if np.isfinite(score):
            scores.append(score)
    return scores


def select_hyperparameters(
    panel: pd.DataFrame,
    feature_cols: list[str],
    development_end: str | pd.Timestamp,
    model_type: str = "xgboost",
    selection_train_months: int = 24,
    validation_months: int = 6,
    min_train_rows: int = 300,
    min_valid_names: int = 30,
    random_state: int = 42,
    param_grid: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Choose parameters using rolling validation folds entirely before cutoff."""
    dates = _panel_dates(panel)
    cutoff = pd.Timestamp(development_end)
    dev_dates = dates[dates < cutoff]
    grid = param_grid or DEFAULT_PARAM_GRID.get(model_type)
    if not grid:
        raise ValueError(f"no parameter grid configured for {model_type}")
    rows = []
    for config_id, params in enumerate(grid):
        fold_scores = []
        first_end = selection_train_months + validation_months
        for end_position in range(first_end, len(dev_dates) + 1, validation_months):
            windows = rolling_train_validation_dates(
                dev_dates, end_position, selection_train_months, validation_months
            )
            fold_scores.extend(_validation_score(
                panel, feature_cols, windows.train_dates, windows.valid_dates,
                model_type, params, min_train_rows, min_valid_names, random_state
            ))
        rows.append({"config_id": config_id, "model_type": model_type,
                     "params": repr(params), "mean_validation_rank_ic":
                     float(np.mean(fold_scores)) if fold_scores else np.nan,
                     "validation_months_scored": len(fold_scores)})
    scores = pd.DataFrame(rows)
    eligible = scores.dropna(subset=["mean_validation_rank_ic"])
    if eligible.empty:
        raise RuntimeError("no valid development-period validation folds; check history and sample thresholds")
    chosen = int(eligible.sort_values(["mean_validation_rank_ic", "config_id"],
                                     ascending=[False, True]).iloc[0].config_id)
    return grid[chosen], scores


def walk_forward_predict(
    panel: pd.DataFrame,
    feature_cols: list[str],
    split_date: str | pd.Timestamp,
    model_type: str = "xgboost",
    train_months: int = 36,
    selection_train_months: int = 24,
    validation_months: int = 6,
    min_train_rows: int = 300,
    min_valid_names: int = 30,
    random_state: int = 42,
    param_grid: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune only on development data, then produce prequential test predictions.

    At a test signal date, the model is refit using up to ``train_months`` prior
    labeled signal months. Earlier test-period labels may enter later training once
    their forward returns are observable, but they never select hyperparameters.
    """
    if not feature_cols or "forward_return" in feature_cols:
        raise ValueError("provide feature columns and keep forward_return out of features")
    missing = set(feature_cols + ["forward_return"]).difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    if panel.index.has_duplicates:
        raise ValueError("panel contains duplicate date/code rows")
    dates = _panel_dates(panel)
    split = pd.Timestamp(split_date)
    chosen, validation = select_hyperparameters(
        panel, feature_cols, split, model_type=model_type,
        selection_train_months=selection_train_months,
        validation_months=validation_months, min_train_rows=min_train_rows,
        min_valid_names=min_valid_names, random_state=random_state,
        param_grid=param_grid,
    )
    validation["selected"] = validation.params.eq(repr(chosen))
    train_dates = dates[dates < split]
    predictions, importances = [], []
    for date in dates[dates >= split]:
        prior_dates = dates[dates < date]
        if len(prior_dates) > train_months:
            prior_dates = prior_dates[-train_months:]
        train = panel.loc[panel.index.get_level_values("date").isin(prior_dates)]
        train = train.dropna(subset=["forward_return"])
        current = panel.loc[panel.index.get_level_values("date").eq(date)]
        if len(train) < min_train_rows or current.empty:
            continue
        x_train, x_current = _fit_transform(train, current, feature_cols)
        estimator = _make_estimator(model_type, chosen, random_state)
        estimator.fit(x_train, train.forward_return.to_numpy(dtype=float))
        prediction = current.copy()
        prediction["prediction"] = estimator.predict(x_current)
        prediction["date"] = date
        prediction["code"] = prediction.index.get_level_values("code").to_numpy()
        predictions.append(prediction.reset_index(drop=True)[
            ["date", "code", "prediction", "forward_return"]
        ])
        importance = getattr(estimator, "feature_importances_", None)
        if importance is not None:
            importances.extend({"date": date, "feature": name, "importance": float(value)}
                               for name, value in zip(feature_cols, importance))
    if not predictions:
        raise RuntimeError("no test predictions were generated; check date span and row thresholds")
    pred = pd.concat(predictions, ignore_index=True).sort_values(["date", "code"])
    pred["date"] = pd.to_datetime(pred.date)
    ic_rows = []
    for date, group in pred.groupby("date"):
        ic_rows.append({"date": date, "rank_ic": _date_ic(group), "n_stocks": len(group)})
    return pred, pd.DataFrame(ic_rows), pd.DataFrame(importances), validation
