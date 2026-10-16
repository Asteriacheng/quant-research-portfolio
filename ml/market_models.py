"""Optional cross-sectional ML experiments for monthly A-share panels.

All functions expect point-in-time features and signal-month-aligned forward
returns. They do not fetch data or make a claim that any model is profitable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ml.ml_factor import _fit_transform, _panel_dates


def cross_sectional_binary_target(panel: pd.DataFrame,
                                  target: str = "forward_return") -> pd.Series:
    """1 if a stock beats that signal month's median return, else 0.

    The median is calculated only among names with an observed target in the
    same month. This target is for classification diagnostics, not a tradable
    benchmark return.
    """
    if target not in panel:
        raise ValueError(f"panel is missing {target!r}")
    values = panel[target]
    medians = values.groupby(level="date").transform("median")
    return values.gt(medians).where(values.notna()).astype("float64")


def progressive_feature_sets(feature_cols: list[str]) -> dict[str, list[str]]:
    """Return fixed, cumulative factor groups for an RF feature-addition test.

    Names that do not match a known family enter the final `other` step. The
    mapping is intentionally transparent and should be reviewed before a run;
    it is not inferred from test-period performance.
    """
    families = {
        "dividend": ("dividend", "div_yield", "cash_yield", "payout"),
        "low_volatility": ("low_vol", "realized_vol", "idiosyncratic_vol", "market_garch", "beta"),
        "fundamentals": ("quality", "profit", "roe", "roa", "leverage", "earnings", "value"),
        "price_volume": ("momentum", "reversal", "amihud", "turnover", "liquidity", "volume", "size"),
    }
    buckets: dict[str, list[str]] = {key: [] for key in families}
    buckets["other"] = []
    for feature in feature_cols:
        normalized = feature.lower()
        match = next((family for family, tokens in families.items()
                      if any(token in normalized for token in tokens)), "other")
        buckets[match].append(feature)
    result: dict[str, list[str]] = {}
    cumulative: list[str] = []
    for family in (*families.keys(), "other"):
        additions = buckets[family]
        if not additions:
            continue
        cumulative.extend(additions)
        result[f"add_{family}"] = cumulative.copy()
    if not result:
        raise ValueError("no features supplied")
    return result


def walk_forward_random_forest(
    panel: pd.DataFrame,
    feature_cols: list[str],
    split_date: str | pd.Timestamp,
    train_months: int = 36,
    min_train_rows: int = 300,
    min_names_per_month: int = 30,
    random_state: int = 42,
    n_estimators: int = 300,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Fit a fixed random-forest classifier month by month after the split.

    Classifier settings are fixed in advance; no test-period tuning is done.
    Each prediction uses features at the current signal date and labels from
    earlier signal dates only. The RF score is the probability of beating the
    contemporaneous cross-sectional median.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise ImportError("Install optional ML dependencies with `pip install -r requirements-ml.txt` (scikit-learn missing)") from exc
    if not feature_cols or "forward_return" in feature_cols:
        raise ValueError("provide feature columns and exclude forward_return")
    missing = set(feature_cols + ["forward_return"]).difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    if panel.index.has_duplicates:
        raise ValueError("panel contains duplicate date/code rows")

    dates = _panel_dates(panel)
    split = pd.Timestamp(split_date)
    labels = cross_sectional_binary_target(panel)
    outputs = []
    for date in dates[dates >= split]:
        prior = dates[dates < date]
        if train_months and len(prior) > train_months:
            prior = prior[-train_months:]
        train_mask = panel.index.get_level_values("date").isin(prior)
        train = panel.loc[train_mask].copy()
        train["_target"] = labels.loc[train.index]
        train = train.dropna(subset=["_target"])
        current = panel.xs(date, level="date", drop_level=False)
        current = current.dropna(subset=feature_cols)
        if len(train) < min_train_rows or current.empty:
            continue
        if train._target.nunique() < 2:
            continue
        if len(current) < min_names_per_month:
            continue
        x_train, x_current = _fit_transform(train, current, feature_cols)
        estimator = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=5, min_samples_leaf=15,
            max_features=0.7, class_weight="balanced_subsample",
            random_state=random_state, n_jobs=n_jobs,
        )
        estimator.fit(x_train, train["_target"].astype(int).to_numpy())
        positive_col = list(estimator.classes_).index(1)
        result = current[["forward_return"]].copy()
        result["probability_above_median"] = estimator.predict_proba(x_current)[:, positive_col]
        result["date"] = date
        result["code"] = result.index.get_level_values("code")
        result["realized_class"] = labels.loc[result.index].to_numpy()
        outputs.append(result.reset_index(drop=True))
    if not outputs:
        raise RuntimeError("no RF predictions; check development history and sample thresholds")
    return pd.concat(outputs, ignore_index=True).sort_values(["date", "code"])


def walk_forward_ridge(
    panel: pd.DataFrame,
    feature_cols: list[str],
    split_date: str | pd.Timestamp,
    train_months: int = 36,
    min_train_rows: int = 300,
) -> pd.DataFrame:
    """Fixed ridge-regression benchmark with strictly prior training labels."""
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("Install optional ML dependencies with `pip install -r requirements-ml.txt` (scikit-learn missing)") from exc
    if not feature_cols or "forward_return" in feature_cols:
        raise ValueError("provide feature columns and exclude forward_return")
    missing = set(feature_cols + ["forward_return"]).difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    dates = _panel_dates(panel)
    split = pd.Timestamp(split_date)
    outputs = []
    for date in dates[dates >= split]:
        prior = dates[dates < date]
        if train_months and len(prior) > train_months:
            prior = prior[-train_months:]
        train = panel.loc[panel.index.get_level_values("date").isin(prior)]
        train = train.dropna(subset=["forward_return"])
        current = panel.xs(date, level="date", drop_level=False)
        current = current.dropna(subset=feature_cols)
        if len(train) < min_train_rows or current.empty:
            continue
        x_train, x_current = _fit_transform(train, current, feature_cols)
        scaler = StandardScaler().fit(x_train)
        estimator = Ridge(alpha=10.0).fit(scaler.transform(x_train),
                                           train.forward_return.to_numpy(dtype=float))
        result = current[["forward_return"]].copy()
        result["prediction"] = estimator.predict(scaler.transform(x_current))
        result["date"] = date
        result["code"] = result.index.get_level_values("code")
        outputs.append(result.reset_index(drop=True))
    if not outputs:
        raise RuntimeError("no Ridge predictions; check development history and sample thresholds")
    return pd.concat(outputs, ignore_index=True).sort_values(["date", "code"])


def monthly_kmeans_diagnostics(
    panel: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = 4,
    min_names: int = 30,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster each signal-month cross-section from features only.

    Cluster IDs are arbitrary and are re-fit monthly; they must not be treated
    as persistent labels or as a return forecast. Forward returns are joined
    only after clustering to describe realized cluster outcomes.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("Install optional ML dependencies with `pip install -r requirements-ml.txt` (scikit-learn missing)") from exc
    missing = set(feature_cols).difference(panel.columns)
    if missing:
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    assignments, summaries = [], []
    for date, group in panel.groupby(level="date", sort=True):
        sample = group.dropna(subset=feature_cols)
        if len(sample) < max(min_names, n_clusters):
            continue
        x = sample[feature_cols].to_numpy(dtype=float)
        x = StandardScaler().fit_transform(x)
        labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state).fit_predict(x)
        part = pd.DataFrame({"date": date,
                             "code": sample.index.get_level_values("code"),
                             "cluster": labels,
                             "forward_return": sample["forward_return"].to_numpy()
                             if "forward_return" in sample else np.nan})
        assignments.append(part)
        summaries.append(part.groupby("cluster", as_index=False).agg(
            n_stocks=("code", "size"), mean_forward_return=("forward_return", "mean"),
            median_forward_return=("forward_return", "median"),
        ).assign(date=date))
    if not assignments:
        raise RuntimeError("no monthly clusters generated; check feature coverage")
    return pd.concat(assignments, ignore_index=True), pd.concat(summaries, ignore_index=True)


def efficient_frontier_weights(
    expected_returns: pd.Series,
    realized_history: pd.DataFrame,
    previous_weights: pd.Series | None = None,
    risk_aversion: float = 4.0,
    turnover_penalty: float = 0.001,
    max_weight: float = 0.10,
) -> pd.Series:
    """Long-only mean-variance weights using a shrunk historical covariance.

    `expected_returns` must be a pre-computed prediction available at rebalance
    time. `realized_history` must contain only returns already realized by that
    time. This small-sample optimizer is an exploratory benchmark, not a robust
    production allocation method.
    """
    names = expected_returns.dropna().index
    if len(names) < 2 or max_weight * len(names) < 1.0:
        raise ValueError("need enough assets for the requested long-only max_weight")
    history = realized_history.reindex(columns=names).dropna(axis=0, how="any")
    if len(history) < 12:
        raise ValueError("at least 12 complete prior return observations are required")
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError as exc:
        raise ImportError("Install optional ML dependencies with `pip install -r requirements-ml.txt` (scikit-learn missing)") from exc
    mu = expected_returns.loc[names].to_numpy(dtype=float)
    covariance = LedoitWolf().fit(history.to_numpy(dtype=float)).covariance_
    covariance = covariance + np.eye(len(names)) * 1e-10
    previous = (previous_weights.reindex(names).fillna(0.0).to_numpy(dtype=float)
                if previous_weights is not None else np.full(len(names), 1.0 / len(names)))

    def objective(weights: np.ndarray) -> float:
        return float(-mu @ weights + 0.5 * risk_aversion * weights @ covariance @ weights
                     + turnover_penalty * np.abs(weights - previous).sum())

    result = minimize(objective, previous, method="SLSQP",
                      bounds=[(0.0, max_weight)] * len(names),
                      constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                      options={"maxiter": 500, "ftol": 1e-10})
    if not result.success:
        raise RuntimeError(f"efficient-frontier optimization failed: {result.message}")
    weights = pd.Series(result.x, index=names, name="weight")
    return weights / weights.sum()
