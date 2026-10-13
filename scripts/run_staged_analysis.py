"""Run the second-stage, monthly non-overlapping diagnostics on the real snapshot.

No settings are selected from the reserved test-period returns. Static industry
labels are used only for an explicitly non-confirmatory sensitivity; no market
capitalization or historical ST/suspension/delisting fields exist in this source.
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.extended import newey_west_mean
from analysis.staged import (cross_sectional_rank_correlations, demean_by_group,
                             monthly_long_only_backtest, quarterly_long_only_backtest,
                             rank_ic)
from factors.price_factors import (amihud_illiquidity, info_weighted_momentum,
                                   momentum_12_1, reversal_5d, turnover_20d,
                                   volume_volatility)

DATA = ROOT / "data/processed/daily_2021_2025.parquet"
RAW = ROOT / "data/raw/astock"
OUT = ROOT / "reports/real_data_v2"
OUT.mkdir(parents=True, exist_ok=True)


def wide(frame: pd.DataFrame, field: str, dates: pd.DatetimeIndex,
         codes: list[str]) -> pd.DataFrame:
    return frame.pivot(index="date", columns="code", values=field).reindex(index=dates, columns=codes).sort_index()


def month_dates(dates: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, dict[pd.Period, pd.Timestamp], dict[pd.Period, pd.Timestamp]]:
    period = dates.to_period("M")
    first = pd.Series(dates, index=period).groupby(level=0).first()
    last = pd.Series(dates, index=period).groupby(level=0).last()
    return pd.DatetimeIndex(last.to_numpy()), first.to_dict(), last.to_dict()


def next_month_returns(open_prices: pd.DataFrame, signal_dates: pd.DatetimeIndex,
                       first: dict[pd.Period, pd.Timestamp], last: dict[pd.Period, pd.Timestamp]) -> pd.DataFrame:
    """Return at signal month-end = next calendar month's first-open to last-open."""
    out = []
    for signal_date in signal_dates:
        target = signal_date.to_period("M") + 1
        if target not in first or target not in last:
            continue
        entry, exit_ = first[target], last[target]
        out.append(open_prices.loc[exit_] / open_prices.loc[entry] - 1.0)
    idx = signal_dates[:len(out)]
    # The last signal month has no following month; preserve exact labels explicitly.
    idx = pd.DatetimeIndex([d for d in signal_dates if d.to_period("M") + 1 in first])
    return pd.DataFrame(out, index=idx, columns=open_prices.columns)


def ic_tables(daily_dir: Path, index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = index.set_index("date").sort_index()
    close = idx["close"].astype(float)
    ma = close.rolling(200, min_periods=200).mean()
    regime = pd.Series(np.select([close > ma, close <= ma], ["above_200d", "at_or_below_200d"], default="unknown"), index=close.index)
    yearly_rows, regime_rows = [], []
    for path in sorted(daily_dir.glob("daily_ic_*.csv")):
        factor = path.stem.removeprefix("daily_ic_")
        frame = pd.read_csv(path, parse_dates=["date"]).set_index("date")["rank_ic"].dropna()
        for year, s in frame.groupby(frame.index.year):
            for period, x in (("full_year", s),):
                stat = newey_west_mean(x, max_lag=5)
                yearly_rows.append({"factor": factor, "year": int(year), "period": period, **stat,
                                    "icir": x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else np.nan,
                                    "positive_ratio": float((x > 0).mean())})
        aligned = frame.to_frame("rank_ic").join(regime.rename("regime"), how="inner")
        aligned = aligned[aligned.index >= "2021-01-01"]
        aligned = aligned[aligned.index <= "2025-12-31"]
        for split, start, end in (("train_2021_2023", "2021-01-01", "2023-12-31"),
                                  ("test_2024_2025", "2024-01-01", "2025-12-31")):
            for state, s in aligned.loc[start:end].groupby("regime")["rank_ic"]:
                stat = newey_west_mean(s, max_lag=5)
                regime_rows.append({"factor": factor, "period": split, "regime": state,
                                    **stat, "positive_ratio": float((s > 0).mean())})
    return pd.DataFrame(yearly_rows), pd.DataFrame(regime_rows)


def main() -> None:
    d = pd.read_parquet(DATA)
    print(f"Loaded {len(d):,} rows; constructing month-end factors...", flush=True)
    d["date"] = pd.to_datetime(d.date)
    dates = pd.DatetimeIndex(sorted(d.date.unique()))
    codes = sorted(d.code.unique())
    close = wide(d, "close", dates, codes).astype(float)
    opn = wide(d, "open", dates, codes).astype(float)
    volume = wide(d, "volume", dates, codes).astype(float)
    amount = wide(d, "amount", dates, codes).astype(float)
    turnover = wide(d, "turn", dates, codes).astype(float)
    signal_dates, first, last = month_dates(dates)
    monthly_factor: dict[str, pd.DataFrame] = {}
    builders = {
        "amihud_20d": lambda: -amihud_illiquidity(close, amount),
        "turnover_20d": lambda: -turnover_20d(turnover),
        "reversal_5d": lambda: reversal_5d(close),
        "volume_volatility_20d": lambda: volume_volatility(volume),
        "momentum_12_1": lambda: momentum_12_1(close),
        "info_weighted_momentum": lambda: info_weighted_momentum(close, volume),
    }
    for name, builder in builders.items():
        factor = builder()
        monthly_factor[name] = factor.reindex(signal_dates).copy()
        del factor
        gc.collect()
        print(f"Prepared {name}", flush=True)
    monthly_return = next_month_returns(opn, signal_dates, first, last)
    quarter_dates = pd.DatetimeIndex([d for d in signal_dates if d.month in (3, 6, 9, 12)])
    quarter_ret_rows, quarter_ret_dates = [], []
    for signal_date in quarter_dates:
        entry_month = signal_date.to_period("M") + 1
        exit_month = signal_date.to_period("M") + 3
        if entry_month in first and exit_month in last:
            quarter_ret_rows.append(opn.loc[last[exit_month]] / opn.loc[first[entry_month]] - 1.0)
            quarter_ret_dates.append(signal_date)
    quarterly_return = pd.DataFrame(quarter_ret_rows, index=pd.DatetimeIndex(quarter_ret_dates), columns=opn.columns)

    # Shanghai Composite regime and one-month benchmark return, known at signal close.
    ix = pd.read_parquet(RAW / "index_sh.parquet").drop_duplicates("date").set_index("date").sort_index()
    ix.index = pd.to_datetime(ix.index)
    ix_close, ix_open = ix.close.astype(float), ix.open.astype(float)
    ix_ma = ix_close.rolling(200, min_periods=200).mean()
    regime_daily = pd.Series(np.select([ix_close > ix_ma, ix_close <= ix_ma],
                                       ["above_200d", "at_or_below_200d"],
                                       default="unknown"), index=ix.index)
    regime_monthly = regime_daily.reindex(signal_dates)
    benchmark = []
    for signal_date in monthly_return.index:
        target = signal_date.to_period("M") + 1
        benchmark.append(ix_open.loc[last[target]] / ix_open.loc[first[target]] - 1.0)
    benchmark_return = pd.Series(benchmark, index=monthly_return.index, name="benchmark_return")
    quarterly_benchmark = []
    for signal_date in quarterly_return.index:
        entry_month = signal_date.to_period("M") + 1
        exit_month = signal_date.to_period("M") + 3
        quarterly_benchmark.append(ix_open.loc[last[exit_month]] / ix_open.loc[first[entry_month]] - 1.0)
    quarterly_benchmark = pd.Series(quarterly_benchmark, index=quarterly_return.index,
                                    name="benchmark_return")

    # Cross-sectional IC, factor redundancy, and factor-monthly regression diagnostics.
    cor = cross_sectional_rank_correlations(monthly_factor)
    train_ref = pd.read_csv(ROOT / "reports/real_data_v1/ic_summary.csv")
    train_ref = train_ref[train_ref.period.eq("train_2021_2023")]
    candidates = train_ref[(train_ref["mean"] > 0) & (train_ref.t_stat >= 1.96)]
    candidates = candidates.sort_values("t_stat", ascending=False).factor.tolist()
    # Redundancy gate is also estimated only in the development window.
    selected: list[str] = []
    train_cor = cor[cor.period.eq("train_2021_2023")].set_index(["factor_a", "factor_b"])
    for candidate in candidates:
        redundant = False
        for kept in selected:
            key = (candidate, kept) if (candidate, kept) in train_cor.index else (kept, candidate)
            if key in train_cor.index and abs(float(train_cor.loc[key, "mean_daily_rank_corr"])) >= 0.70:
                redundant = True
                break
        if not redundant:
            selected.append(candidate)
    fm_rows, fm_summary = [], []
    monthly_ic_rows, neutral_ic_rows = [], []
    meta = pd.read_parquet(RAW / "stock_list.parquet")
    # Exclude index pseudo-securities and unclassified rows; labels remain static snapshot metadata.
    meta = meta[(meta.code.str.fullmatch(r"\d{6}")) & (meta.sector.notna()) & (meta.sector != "未分类")]
    groups = meta.drop_duplicates("code", keep="last").set_index("code").sector.reindex(codes)
    for name, f_daily_monthly in monthly_factor.items():
        f = f_daily_monthly.reindex(monthly_return.index)
        ret = monthly_return.reindex(f.index)
        ic = rank_ic(f, ret, min_names=50)
        for date, value in ic.items():
            monthly_ic_rows.append({"factor": name, "signal_month": date, "rank_ic": value,
                                    "regime": regime_monthly.get(date, "unknown")})
        # Optional static-industry sensitivity. It is not point-in-time validated.
        neutral = demean_by_group(f, groups)
        nic = rank_ic(neutral, ret, min_names=50)
        for date, value in nic.items():
            neutral_ic_rows.append({"factor": name, "signal_month": date, "rank_ic": value,
                                    "regime": regime_monthly.get(date, "unknown")})

        # Univariate and static-industry-controlled Fama-MacBeth monthly slopes.
        for date in f.index:
            frame = pd.DataFrame({"factor": f.loc[date], "return": ret.loc[date]})
            frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
            if len(frame) < 50:
                continue
            score = frame.factor.rank(pct=True).to_numpy() - 0.5
            y = frame["return"].to_numpy(float)
            design = np.column_stack([np.ones(len(frame)), score])
            beta = np.linalg.lstsq(design, y, rcond=None)[0]
            fm_rows.append({"factor": name, "signal_month": date, "model": "univariate",
                            "factor_slope": beta[1], "n_stocks": len(frame)})
            sector_frame = frame.join(groups.rename("industry"), how="left").dropna(subset=["industry"])
            if len(sector_frame) >= 50 and sector_frame.industry.nunique() > 1:
                sector_score = sector_frame.factor.rank(pct=True).to_numpy() - 0.5
                dummy = pd.get_dummies(sector_frame.industry, drop_first=True, dtype=float).to_numpy()
                sector_design = np.column_stack([np.ones(len(sector_frame)), sector_score, dummy])
                beta = np.linalg.lstsq(sector_design, sector_frame["return"].to_numpy(float), rcond=None)[0]
                fm_rows.append({"factor": name, "signal_month": date,
                                "model": "static_industry_fixed_effects",
                                "factor_slope": beta[1], "n_stocks": len(sector_frame)})
        panel, _ = monthly_long_only_backtest(f, ret, benchmark_return.reindex(f.index),
                                               quantile=0.2, commission_bps=3,
                                               slippage_bps=10)
        panel["factor"] = name
        panel["cost_scenario"] = "base_10bp_slippage"
        yield_frames = [panel]
        for slip in (5.0, 20.0):
            alt, _ = monthly_long_only_backtest(f, ret, benchmark_return.reindex(f.index),
                                                quantile=0.2, commission_bps=3,
                                                slippage_bps=slip)
            alt["factor"] = name
            alt["cost_scenario"] = f"{int(slip)}bp_slippage"
            yield_frames.append(alt)
        portfolio_rows.extend(yield_frames)
        print(f"Analyzed {name}", flush=True)

    # Freeze an equal-weight composite from the development-period screen only.
    combo_name = "equal_rank_combo"
    if selected:
        ranks = [monthly_factor[name].rank(axis=1, pct=True) for name in selected]
        combo = pd.concat(ranks, keys=selected).groupby(level=1).mean()
        # Require all preselected components; do not silently change the composite.
        coverage = pd.concat([x.notna() for x in ranks], keys=selected).groupby(level=1).sum()
        combo = combo.where(coverage.eq(len(selected)))
        combo = combo.reindex(monthly_return.index)
        ret = monthly_return.reindex(combo.index)
        for date, value in rank_ic(combo, ret, min_names=50).items():
            monthly_ic_rows.append({"factor": combo_name, "signal_month": date, "rank_ic": value,
                                    "regime": regime_monthly.get(date, "unknown")})
        neutral = demean_by_group(combo, groups)
        for date, value in rank_ic(neutral, ret, min_names=50).items():
            neutral_ic_rows.append({"factor": combo_name, "signal_month": date, "rank_ic": value,
                                    "regime": regime_monthly.get(date, "unknown")})
        for date in combo.index:
            frame = pd.DataFrame({"factor": combo.loc[date], "return": ret.loc[date]}).replace([np.inf, -np.inf], np.nan).dropna()
            if len(frame) < 50:
                continue
            score = frame.factor.rank(pct=True).to_numpy() - 0.5
            y = frame["return"].to_numpy(float)
            beta = np.linalg.lstsq(np.column_stack([np.ones(len(frame)), score]), y, rcond=None)[0]
            fm_rows.append({"factor": combo_name, "signal_month": date, "model": "univariate",
                            "factor_slope": beta[1], "n_stocks": len(frame)})
            sector_frame = frame.join(groups.rename("industry"), how="left").dropna(subset=["industry"])
            if len(sector_frame) >= 50 and sector_frame.industry.nunique() > 1:
                sec_score = sector_frame.factor.rank(pct=True).to_numpy() - 0.5
                dummy = pd.get_dummies(sector_frame.industry, drop_first=True, dtype=float).to_numpy()
                beta = np.linalg.lstsq(np.column_stack([np.ones(len(sec_score)), sec_score, dummy]),
                                       sector_frame["return"].to_numpy(float), rcond=None)[0]
                fm_rows.append({"factor": combo_name, "signal_month": date,
                                "model": "static_industry_fixed_effects", "factor_slope": beta[1],
                                "n_stocks": len(sector_frame)})
        combined, _ = monthly_long_only_backtest(combo, ret, benchmark_return.reindex(combo.index),
                                                quantile=0.2, commission_bps=3, slippage_bps=10)
        combined["factor"] = combo_name
        combined["cost_scenario"] = "base_10bp_slippage"
        yield_frames = [combined]
        for slip in (5.0, 20.0):
            alt, _ = monthly_long_only_backtest(combo, ret, benchmark_return.reindex(combo.index),
                                                quantile=0.2, commission_bps=3, slippage_bps=slip)
            alt["factor"] = combo_name
            alt["cost_scenario"] = f"{int(slip)}bp_slippage"
            yield_frames.append(alt)
        portfolio_rows.extend(yield_frames)
        all_factor_panels = {**monthly_factor, combo_name: combo}
        cor = cross_sectional_rank_correlations(all_factor_panels)
    else:
        all_factor_panels = monthly_factor

    # Predeclared window-length alternatives. These are sensitivity variants,
    # not candidates for changing the frozen training-screened composite.
    variants = {
        "turnover_60d": -turnover.rolling(60, min_periods=30).mean().reindex(signal_dates),
        "volume_volatility_60d": -(volume.rolling(60, min_periods=30).std(ddof=1) /
                                    volume.rolling(60, min_periods=30).mean().where(
                                        volume.rolling(60, min_periods=30).mean() > 0)).reindex(signal_dates),
        "reversal_10d": -(close / close.shift(10) - 1.0).reindex(signal_dates),
    }
    formula_rows = []
    for variant, values in variants.items():
        x = values.reindex(monthly_return.index)
        monthly_ic_values = rank_ic(x, monthly_return.reindex(x.index), min_names=50)
        test_ic = monthly_ic_values[monthly_ic_values.index.year >= 2024]
        stat = newey_west_mean(test_ic, max_lag=3)
        formula_rows.append({"variant": variant, "test_months": len(test_ic), **stat,
                             "positive_ratio": float((test_ic > 0).mean()) if len(test_ic) else np.nan})
    pd.DataFrame(formula_rows).to_csv(OUT / "factor_formula_sensitivity.csv", index=False)

    # Quarterly rebalance sensitivity on the frozen composite and components.
    cadence_rows = []
    for name in selected + ([combo_name] if selected else []):
        _, qsummary = quarterly_long_only_backtest(
            all_factor_panels[name].reindex(quarterly_return.index), quarterly_return,
            benchmark_return=quarterly_benchmark, commission_bps=3, slippage_bps=10)
        qsummary["factor"] = name
        qsummary["frequency"] = "quarterly"
        cadence_rows.append(qsummary)
    if cadence_rows:
        pd.concat(cadence_rows, ignore_index=True).to_csv(OUT / "quarterly_backtest_sensitivity.csv", index=False)
    cor.to_csv(OUT / "factor_rank_correlations.csv", index=False)

    fm = pd.DataFrame(fm_rows)
    fm.to_csv(OUT / "fama_macbeth_monthly_coefficients.csv", index=False)
    for (name, model), grp in fm.groupby(["factor", "model"]):
        for split, mask in (("train_2021_2023", grp.signal_month.dt.year <= 2023),
                            ("test_2024_2025", grp.signal_month.dt.year >= 2024)):
            x = grp.loc[mask, "factor_slope"]
            s = newey_west_mean(x, max_lag=3)
            fm_summary.append({"factor": name, "model": model, "period": split, **s})

    monthly_ic = pd.DataFrame(monthly_ic_rows)
    monthly_ic.to_csv(OUT / "monthly_nonoverlap_rank_ic.csv", index=False)
    pd.DataFrame(neutral_ic_rows).to_csv(OUT / "static_industry_neutral_rank_ic.csv", index=False)
    pd.DataFrame(fm_summary).to_csv(OUT / "fama_macbeth_summary.csv", index=False)
    portfolio = pd.concat(portfolio_rows, ignore_index=True)
    portfolio.to_csv(OUT / "monthly_top_quintile_backtests.csv", index=False)
    summary_rows = []
    for (factor_name, scenario), group in portfolio.groupby(["factor", "cost_scenario"]):
        # Monthly strategy outcomes use non-overlapping month labels; preserve missing month counts.
        _, sm = monthly_long_only_backtest(
            all_factor_panels[factor_name].reindex(monthly_return.index), monthly_return,
            benchmark_return, commission_bps=3,
            slippage_bps=float(scenario.split("_")[0].replace("bp", "")) if scenario != "base_10bp_slippage" else 10.0)
        sm["factor"] = factor_name
        sm["cost_scenario"] = scenario
        summary_rows.append(sm)
    pd.concat(summary_rows, ignore_index=True).to_csv(OUT / "monthly_portfolio_summary.csv", index=False)

    ic_summary_rows = []
    for name, group in monthly_ic.groupby("factor"):
        for split, mask in (("train_2021_2023", monthly_ic.signal_month.dt.year <= 2023),
                            ("test_2024_2025", monthly_ic.signal_month.dt.year >= 2024)):
            x = group.loc[mask.reindex(group.index, fill_value=False), "rank_ic"]
            if x.empty:
                continue
            stat = newey_west_mean(x, max_lag=3)
            ic_summary_rows.append({"factor": name, "period": split, **stat,
                                    "icir": x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else np.nan,
                                    "positive_ratio": float((x > 0).mean())})
    pd.DataFrame(ic_summary_rows).to_csv(OUT / "monthly_nonoverlap_rank_ic_summary.csv", index=False)
    by_regime_rows = []
    for (name, regime), group in monthly_ic.groupby(["factor", "regime"]):
        for split, mask in (("train_2021_2023", group.signal_month.dt.year <= 2023),
                            ("test_2024_2025", group.signal_month.dt.year >= 2024)):
            x = group.loc[mask, "rank_ic"]
            if x.empty:
                continue
            by_regime_rows.append({"factor": name, "regime": regime, "period": split,
                                   **newey_west_mean(x, max_lag=3),
                                   "positive_ratio": float((x > 0).mean())})
    pd.DataFrame(by_regime_rows).to_csv(OUT / "monthly_nonoverlap_rank_ic_by_regime.csv", index=False)

    neutral_frame = pd.DataFrame(neutral_ic_rows)
    neutral_summary_rows = []
    for name, group in neutral_frame.groupby("factor"):
        for split, mask in (("train_2021_2023", group.signal_month.dt.year <= 2023),
                            ("test_2024_2025", group.signal_month.dt.year >= 2024)):
            x = group.loc[mask, "rank_ic"]
            if x.empty:
                continue
            neutral_summary_rows.append({"factor": name, "period": split,
                                         **newey_west_mean(x, max_lag=3),
                                         "positive_ratio": float((x > 0).mean())})
    pd.DataFrame(neutral_summary_rows).to_csv(OUT / "static_industry_neutral_rank_ic_summary.csv", index=False)
    if selected:
        focus = set(selected + [combo_name])
        test_ic = pd.DataFrame(ic_summary_rows)
        test_ic = test_ic[(test_ic.factor.isin(focus)) & test_ic.period.eq("test_2024_2025")]
        test_port = pd.read_csv(OUT / "monthly_portfolio_summary.csv")
        test_port = test_port[(test_port.factor.isin(focus)) & test_port.period.eq("test_2024_2025") &
                              test_port.cost_scenario.eq("base_10bp_slippage") &
                              test_port.return_series.eq("net_return")]
        test_ic.merge(test_port, on="factor", suffixes=("_ic", "_portfolio")).to_csv(
            OUT / "combo_vs_components_test.csv", index=False)
        base = portfolio[(portfolio.cost_scenario == "base_10bp_slippage") &
                         portfolio.factor.isin(focus) & (portfolio.signal_month.dt.year >= 2024)]
        net = base.pivot(index="signal_month", columns="factor", values="net_return")
        common = net.dropna(subset=sorted(focus))
        bench = base.drop_duplicates("signal_month").set_index("signal_month").benchmark_return.reindex(common.index)
        pair_rows = []
        for name in sorted(focus):
            x = common[name]
            wealth = (1 + x).cumprod()
            drawdown = wealth / wealth.cummax() - 1
            n = len(x)
            ann_return = float(wealth.iloc[-1] ** (12 / n) - 1)
            ann_vol = float(x.std(ddof=1) * np.sqrt(12))
            bwealth = (1 + bench).cumprod()
            bret = float(bwealth.iloc[-1] ** (12 / n) - 1)
            one = base[base.factor.eq(name)].set_index("signal_month").reindex(common.index)
            pair_rows.append({"factor": name, "common_test_months": n,
                              "cumulative_net_return": float(wealth.iloc[-1] - 1),
                              "annualized_net_return": ann_return,
                              "annualized_volatility": ann_vol,
                              "max_drawdown": float(drawdown.min()),
                              "sharpe_zero_rf": float(x.mean() * 12 / ann_vol) if ann_vol > 0 else np.nan,
                              "benchmark_annualized_return_same_months": bret,
                              "annualized_excess_return": ann_return - bret,
                              "mean_one_way_turnover": float(one.one_way_turnover.mean()),
                              "mean_cost_per_month": float(one.cost_return.mean())})
        pd.DataFrame(pair_rows).to_csv(OUT / "combo_vs_components_paired_test.csv", index=False)

    yearly, regimes = ic_tables(ROOT / "reports/real_data_v1", ix.reset_index())
    yearly.to_csv(OUT / "overlapping_label_yearly_ic.csv", index=False)
    regimes.to_csv(OUT / "overlapping_label_regime_ic.csv", index=False)
    metadata = {
        "source_file": DATA.name,
        "signal_date": "last observed trading session close of month",
        "execution": "next calendar month's first observed open; exit at that month's last observed open",
        "return_label": "non-overlapping next-calendar-month open-to-open adjusted-price return",
        "universe": "securities with source rows; not historical full listing universe",
        "benchmark": "Shanghai Composite open-to-open over same next-month interval",
        "industry_sensitivity": "static 2026 snapshot sector labels; exploratory only, not point-in-time",
        "size_control": "not run: source has no market capitalization/shares outstanding history",
        "corporate_actions": "source prices are forward-adjusted; no separate cash dividend was added",
        "missing_exit": "if any selected holding lacks a monthly price observation, monthly strategy return is NA; count retained; no forced liquidation assumed",
        "execution_limits": "no historical ST, suspension, limit-up/down or delisting state; costs are assumptions",
        "cost_assumptions": {
            "commission_bps_per_side": 3,
            "slippage_bps_per_side_scenarios": [5, 10, 20],
            "stamp_sell_bps": "10 before 2023-08-28, 5 from 2023-08-28; official reduction notice linked in research plan"
        },
        "static_industry_rows": int(groups.notna().sum()),
        "development_screen_rule": "positive 2021-2023 overlapping-label mean Rank IC and Newey-West t >= 1.96; remove pairs with absolute mean monthly cross-sectional rank correlation >= 0.70",
        "selected_development_factors": selected,
        "composite": "equal-weight average of within-month percentile ranks; requires all selected component signals; no fitted weights",
        "sensitivity": "tested 60-day turnover/volume windows and 10-day reversal; quarterly rebalancing evaluated separately; these variants do not alter the frozen monthly composite",
        "securities": len(codes),
        "signal_months": len(signal_dates),
        "test_window": ["2024-01", "2025-12"],
        "confirmatory_limit": "No strategy claims until a point-in-time universe, status, size and classification history is obtained."
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote staged analysis to {OUT}")
    print(pd.DataFrame(fm_summary).to_string(index=False))


if __name__ == "__main__":
    portfolio_rows: list[pd.DataFrame] = []
    main()
