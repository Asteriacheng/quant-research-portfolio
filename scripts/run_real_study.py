"""Run the first real-data factor diagnostics on the pinned public A-share snapshot."""
from __future__ import annotations
import sys
from pathlib import Path
import json
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from factors.price_factors import (amihud_illiquidity, turnover_20d, reversal_5d,
                                   volume_volatility, momentum_12_1, info_weighted_momentum)
from analysis.extended import newey_west_mean, quintile_panel

DATA = ROOT / "data" / "processed" / "daily_2021_2025.parquet"
OUT = ROOT / "reports" / "real_data_v1"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)


def wide(frame: pd.DataFrame, field: str) -> pd.DataFrame:
    return frame.pivot(index="date", columns="code", values=field).sort_index()


def main() -> None:
    d = pd.read_parquet(DATA)
    dates = pd.DatetimeIndex(sorted(d.date.unique()))
    # Historical investability proxy: each code first becomes eligible at its first observed date.
    first_seen = d.groupby("code").date.min()
    eligible = pd.DataFrame({c: dates >= first_seen.loc[c] for c in d.code.unique()}, index=dates)
    codes = sorted(d.code.unique())
    eligible = eligible.reindex(columns=codes)
    close, opn = wide(d, "close").reindex(index=dates, columns=codes), wide(d, "open").reindex(index=dates, columns=codes)
    high, low = wide(d, "high").reindex(index=dates, columns=codes), wide(d, "low").reindex(index=dates, columns=codes)
    volume, amount = wide(d, "volume").reindex(index=dates, columns=codes), wide(d, "amount").reindex(index=dates, columns=codes)
    turn = wide(d, "turn").reindex(index=dates, columns=codes)
    # Source has adjusted prices but no adjustment factor: treat published prices as its return series;
    # note that dividend cash returns cannot be added without corporate-action data.
    turnover_frac = turnover_20d(turn)
    # Signal at close t; entry open t+1; exit open t+6 = five open-to-open holding sessions.
    fwd5 = opn.shift(-6) / opn.shift(-1) - 1.0
    # Never use a missing-price forward observation. Eligibility starts at first observed date.
    fwd5 = fwd5.where(eligible)
    train_end, test_start = pd.Timestamp("2023-12-29"), pd.Timestamp("2024-01-01")
    summaries, monthly, group_tables, decay_rows = [], [], {}, []
    factor_builders = {
        "amihud_20d": lambda: -amihud_illiquidity(close, amount),
        "turnover_20d": lambda: -turnover_frac,
        "reversal_5d": lambda: reversal_5d(close),
        "volume_volatility_20d": lambda: volume_volatility(volume),
        "momentum_12_1": lambda: momentum_12_1(close),
        "info_weighted_momentum": lambda: info_weighted_momentum(close, volume),
    }
    for name, build_factor in factor_builders.items():
        factor = build_factor().where(eligible)
        common = factor.notna() & fwd5.notna()
        frank = factor.where(common).rank(axis=1)
        rrank = fwd5.where(common).rank(axis=1)
        daily_ic = frank.corrwith(rrank, axis=1).where(common.sum(axis=1) >= 30).dropna()
        daily_ic.name=name
        pd.DataFrame({"date": daily_ic.index, "rank_ic": daily_ic.values}).to_csv(
            OUT / f"daily_ic_{name}.csv", index=False)
        for k in (1, 2, 3, 5, 10):
            ret_k = close.shift(-k) / close.shift(-(k - 1)) - 1.0
            ok=factor.notna() & ret_k.notna()
            series=factor.where(ok).rank(axis=1).corrwith(ret_k.where(ok).rank(axis=1),axis=1)
            series=series.where(ok.sum(axis=1)>=30)
            decay_rows.append({"factor": name, "horizon_day": k, "mean_rank_ic": series.mean(),
                               "positive_ic_ratio": series.dropna().gt(0).mean(),
                               "n_days": int(series.notna().sum())})
        for split, mask in [("train_2021_2023", daily_ic.index <= train_end),
                            ("test_2024_2025", daily_ic.index >= test_start)]:
            s = newey_west_mean(daily_ic.loc[mask], max_lag=5)
            std = daily_ic.loc[mask].std(ddof=1)
            summaries.append({"factor": name, "period": split, "ic_std": std,
                              "icir": s["mean"] / std if std and std > 0 else np.nan,
                              "positive_ic_ratio": float((daily_ic.loc[mask] > 0).mean()) if s["n"] else np.nan,
                              **s})
        icmonthly = daily_ic.resample("ME").mean()
        monthly.append(icmonthly.rename(name))
        gr_all, meta_all = quintile_panel(factor, fwd5, q=5)
        group_tables[name] = gr_all[gr_all.date <= train_end]
        gr_all.to_csv(OUT / f"quintile_daily_{name}.csv", index=False)
        meta_all.to_csv(OUT / f"quintile_membership_{name}.csv", index=False)
        # daily signal turnover among the top 20% (portfolio membership replacement rate)
        top=factor.rank(axis=1,pct=True)>=.8
        top=top.fillna(False)
        replacement=((top & ~top.shift(1).fillna(False)).sum(axis=1)
                     / top.sum(axis=1).replace(0,np.nan)).iloc[1:]
        replacement.name=name
        replacement.rename(name).to_csv(OUT / f"top20_turnover_{name}.csv", header=True,index_label="date")
        del factor, common, frank, rrank, top, ret_k, series
        gc.collect()

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "ic_summary.csv", index=False)
    pd.DataFrame(decay_rows).to_csv(OUT / "decay_summary.csv", index=False)
    pd.concat(monthly, axis=1).to_csv(OUT / "monthly_ic.csv", index_label="month")
    # Pre-registered 12 cost / universe / portfolio tests are planned; first baseline report is transparent.
    (OUT / "run_metadata.json").write_text(json.dumps({
        "research_window": ["2021-01-01", "2025-12-31"],
        "exploration_only_through": "2023-12-29",
        "reserved_test_window": ["2024-01-01", "2025-12-31"],
        "signal": "calendar-date close; execute next observed session open; exit at open after five further sessions",
        "source_adjusted_prices": True,
        "dividend_cash_return": "not included: source has no cash-dividend/announcement records",
        "limit_executable_filter": "not applied: source has no daily suspension/ST/listing status; missing OHLC rows are excluded",
        "costs": "not included in this signal diagnostic; turnover/cost assumptions are in docs/research_plan.md",
        "n_rows": len(d), "n_codes": int(d.code.nunique()), "n_sessions": len(dates),
        "source_file": str(DATA.name)
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    # chart 1 is generated from exported monthly_ic.csv in make_figures.py
    # chart 2: training quintile mean returns, by factor
    rows=[]
    for name, gr in group_tables.items():
        avg=gr.groupby("quintile")["return"].mean()
        rows += [{"factor":name,"quintile":q,"mean_forward_return":v} for q,v in avg.items()]
    gt=pd.DataFrame(rows); gt.to_csv(OUT / "quintile_summary_train.csv",index=False)
    # Test portfolio output is generated in a separate opt-in script to keep memory bounded.
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
