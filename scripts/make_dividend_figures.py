"""Create eight dividend/low-vol research plots from run_dividend_multifactor outputs."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports" / "dividend_multifactor"
FIG = OUT / "figures"
FACTORS = ["trailing_cash_dividend_yield", "low_volatility_60d", "equal_rank_dividend_low_vol"]


def read_csv(name: str) -> pd.DataFrame:
    path = OUT / name
    if not path.exists():
        raise SystemExit(f"Missing {path.relative_to(ROOT)}; run scripts/run_dividend_multifactor.py first")
    return pd.read_csv(path)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    ic = read_csv("monthly_rank_ic.csv")
    ic["date"] = pd.to_datetime(ic.date)
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, part in ic.groupby("factor"):
        ax.plot(part.date, part.rank_ic, marker=".", linewidth=1, label=name)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(title="Monthly non-overlapping Rank IC", ylabel="Rank IC")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "01_monthly_rank_ic.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, name in zip(axes, FACTORS):
        q = read_csv(f"quintiles_{name}.csv")
        means = q.groupby("quintile").mean_return.mean()
        ax.bar(means.index.astype(str), means.values)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(title=name, xlabel="Quintile", ylabel="Mean next-month total return")
    fig.tight_layout(); fig.savefig(FIG / "02_quintile_returns.png", dpi=160); plt.close(fig)

    base_name = "equal_rank_dividend_low_vol"
    bt = read_csv(f"backtest_monthly_{base_name}_10bp.csv")
    bt["signal_month"] = pd.to_datetime(bt.signal_month)
    metadata_path = OUT / "run_metadata.json"
    if not metadata_path.exists():
        raise SystemExit("Missing run_metadata.json; run scripts/run_dividend_multifactor.py first")
    test_start = pd.Timestamp(json.loads(metadata_path.read_text(encoding="utf-8"))["reserved_test_start"])
    bt = bt.loc[bt.signal_month >= test_start].sort_values("signal_month")
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, label in (("net_return", "Dividend + low-vol net"), ("benchmark_return", "Benchmark")):
        series = bt.set_index("signal_month")[col].dropna()
        if not series.empty:
            ax.plot(series.index, (1 + series).cumprod(), label=label)
    ax.set(title="Reserved-test cumulative wealth", ylabel="Growth of 1.0")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "03_test_cumulative.png", dpi=160); plt.close(fig)

    wealth = (1 + bt.set_index("signal_month").net_return.fillna(0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#b44", alpha=.35)
    ax.set(title="Reserved-test drawdown", ylabel="Drawdown")
    fig.tight_layout(); fig.savefig(FIG / "04_test_drawdown.png", dpi=160); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    for name in FACTORS:
        frame = read_csv(f"backtest_monthly_{name}_10bp.csv")
        frame["signal_month"] = pd.to_datetime(frame.signal_month)
        frame = frame.loc[frame.signal_month >= test_start]
        ax1.plot(frame.signal_month, frame.one_way_turnover, marker=".", label=name)
    ax1.set(title="One-way target-weight turnover (10 bp slippage scenario)", ylabel="Turnover")
    ax1.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIG / "05_turnover.png", dpi=160); plt.close(fig)

    corr = read_csv("factor_rank_correlations.csv")
    names = ["trailing_cash_dividend_yield", "low_volatility_60d"]
    matrix = pd.DataFrame(np.eye(2), index=names, columns=names)
    for _, row in corr.loc[corr.period.eq("train_2021_2023")].iterrows():
        if row.factor_a in names and row.factor_b in names:
            matrix.loc[row.factor_a, row.factor_b] = row.mean_daily_rank_corr
            matrix.loc[row.factor_b, row.factor_a] = row.mean_daily_rank_corr
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(2), names, rotation=20, ha="right")
    ax.set_yticks(range(2), names)
    for i in range(2):
        for j in range(2): ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Mean cross-sectional rank correlation")
    fig.tight_layout(); fig.savefig(FIG / "06_factor_correlation.png", dpi=160); plt.close(fig)

    annual = read_csv("annual_ic_summary.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    piv = annual.pivot(index="year", columns="factor", values="mean")
    piv.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(title="Annual Rank IC stability", ylabel="Mean Rank IC", xlabel="Year")
    fig.tight_layout(); fig.savefig(FIG / "07_annual_stability.png", dpi=160); plt.close(fig)

    regimes = read_csv("regime_ic_summary.csv")
    regimes = regimes.loc[regimes.period.eq("test")]
    fig, ax = plt.subplots(figsize=(10, 5))
    piv = regimes.pivot(index="regime", columns="factor", values="mean")
    piv.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(title="Reserved-test Rank IC by benchmark regime", ylabel="Mean Rank IC", xlabel="Regime")
    fig.tight_layout(); fig.savefig(FIG / "08_regime_stability.png", dpi=160); plt.close(fig)
    print(f"Wrote 8 figures to {FIG}")


if __name__ == "__main__":
    main()
