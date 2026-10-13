"""Render eight plots for the second-stage real-data study."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports/real_data_v2"
F = R / "figures"
F.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 1) Monthly non-overlapping Rank IC, including frozen composite.
ic = pd.read_csv(R / "monthly_nonoverlap_rank_ic.csv", parse_dates=["signal_month"])
fig, ax = plt.subplots(figsize=(12, 5))
for name, g in ic.groupby("factor"):
    ax.plot(g.signal_month, g.rank_ic, marker=".", ms=3, lw=1, label=name)
ax.axvline(pd.Timestamp("2024-01-01"), color="black", ls="--", lw=1)
ax.axhline(0, color="gray", lw=.7)
ax.set(title="月度非重叠 Rank IC（测试期自 2024 年起）", ylabel="Rank IC")
ax.legend(fontsize=7, ncol=4)
fig.tight_layout(); fig.savefig(F / "01_monthly_rank_ic.png", dpi=160); plt.close(fig)

# Common-month comparison for selected components and the equal-rank composite.
paired = pd.read_csv(R / "combo_vs_components_paired_test.csv")
names = paired.factor.tolist()
base = pd.read_csv(R / "monthly_top_quintile_backtests.csv", parse_dates=["signal_month"])
base = base[(base.cost_scenario == "base_10bp_slippage") & base.factor.isin(names) & (base.signal_month.dt.year >= 2024)]
wide = base.pivot(index="signal_month", columns="factor", values="net_return")
common = wide.dropna(subset=names)
bench = base.drop_duplicates("signal_month").set_index("signal_month").benchmark_return.reindex(common.index)

# 2) Compounded net wealth on identical complete months.
fig, ax = plt.subplots(figsize=(11, 5))
for name in names:
    ax.plot((1 + common[name]).cumprod(), label=name)
ax.plot((1 + bench).cumprod(), color="black", ls="--", label="上证综指")
ax.set(title=f"测试期 Q5 净值：共同完整月份 n={len(common)}", ylabel="累计净值（起点=1）")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(F / "02_test_cumulative_net.png", dpi=160); plt.close(fig)

# 3) Matched-month drawdowns.
fig, ax = plt.subplots(figsize=(11, 5))
for name in names + ["__benchmark__"]:
    x = bench if name == "__benchmark__" else common[name]
    wealth = (1 + x).cumprod()
    ax.plot(wealth / wealth.cummax() - 1, label="上证综指" if name == "__benchmark__" else name)
ax.axhline(0, color="gray", lw=.7)
ax.set(title="测试期同月样本回撤", ylabel="回撤")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(F / "03_test_drawdown.png", dpi=160); plt.close(fig)

# 4) Training/test mean monthly IC.
summ = pd.read_csv(R / "monthly_nonoverlap_rank_ic_summary.csv")
fig, ax = plt.subplots(figsize=(11, 5))
factors = list(summ.factor.drop_duplicates())
x = np.arange(len(factors)); width = .36
for j, period in enumerate(["train_2021_2023", "test_2024_2025"]):
    s = summ[summ.period == period].set_index("factor").reindex(factors)
    ax.bar(x + (j-.5)*width, s["mean"], width, label=period)
ax.axhline(0, color="black", lw=.7)
ax.set_xticks(x, factors, rotation=25, ha="right")
ax.set(title="月度 Rank IC 稳定性：开发期与预留测试期", ylabel="平均 Rank IC")
ax.legend()
fig.tight_layout(); fig.savefig(F / "04_ic_train_test.png", dpi=160); plt.close(fig)

# 5) Turnover and estimated cost for matched test months.
costs = base[base.signal_month.isin(common.index)]
costs = costs[costs.factor.isin(names)].groupby("factor").agg(
    turnover=("one_way_turnover", "mean"), cost=("cost_return", "mean")).reindex(names)
fig, ax1 = plt.subplots(figsize=(9, 5))
costs.turnover.plot.bar(ax=ax1, color="#427aa1", width=.65)
ax1.set_ylabel("平均单边换手（比例）"); ax1.set_xticklabels(names, rotation=25, ha="right")
ax2 = ax1.twinx(); (costs.cost * 100).plot(ax=ax2, color="#d1495b", marker="o", ls="--")
ax2.set_ylabel("平均月度成本（%）")
ax1.set_title("测试期共同月份：组合换手与成本估计")
fig.tight_layout(); fig.savefig(F / "05_turnover_cost.png", dpi=160); plt.close(fig)

# 6) Selected factors' training-only daily cross-sectional rank correlations.
cor = pd.read_csv(R / "factor_rank_correlations.csv")
cor = cor[(cor.period == "train_2021_2023") & cor.factor_a.isin(names) & cor.factor_b.isin(names)]
mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
for row in cor.itertuples():
    mat.loc[row.factor_a, row.factor_b] = row.mean_daily_rank_corr
    mat.loc[row.factor_b, row.factor_a] = row.mean_daily_rank_corr
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(mat.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
ax.set_yticks(range(len(names)), names)
for i in range(len(names)):
    for j in range(len(names)):
        ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, shrink=.8)
ax.set_title("开发期因子日截面秩相关（含组合）")
fig.tight_layout(); fig.savefig(F / "06_rank_correlation.png", dpi=160); plt.close(fig)

# 7) Static industry-neutral sensitivity against raw IC (exploratory, labels are static).
raw = ic[(ic.factor.isin(names)) & (ic.signal_month.dt.year >= 2024)].groupby("factor").rank_ic.mean().rename("raw")
neutral = pd.read_csv(R / "static_industry_neutral_rank_ic.csv")
neutral = neutral[(neutral.factor.isin(names)) & (neutral.signal_month.astype(str).str.startswith("2024"))]
neutral = neutral.groupby("factor").rank_ic.mean().rename("static_industry_demeaned")
comp = pd.concat([raw, neutral], axis=1)
ax = comp.plot.bar(figsize=(10, 5), color=["#427aa1", "#f4a261"])
ax.axhline(0, color="black", lw=.7); ax.set_ylabel("平均月度 Rank IC")
ax.set_title("测试期行业中性敏感性（静态行业标签，仅探索）")
ax.tick_params(axis="x", rotation=25)
plt.tight_layout(); plt.savefig(F / "07_industry_sensitivity.png", dpi=160); plt.close()

# 8) Market regime table visualized; retain the small-N warning in title.
reg = pd.read_csv(R / "monthly_nonoverlap_rank_ic_by_regime.csv")
reg = reg[(reg.period == "test_2024_2025") & reg.factor.isin(names)]
piv = reg.pivot(index="factor", columns="regime", values="mean").reindex(names)
ax = piv.plot.bar(figsize=(10, 5), color=["#2a9d8f", "#e76f51", "#999999"][:len(piv.columns)])
ax.axhline(0, color="black", lw=.7); ax.set_ylabel("平均月度 Rank IC")
ax.set_title("测试期市场状态稳定性（沪指 200 日均线；熊市子样本较少）")
ax.tick_params(axis="x", rotation=25); ax.legend(title="市场状态")
plt.tight_layout(); plt.savefig(F / "08_regime_stability.png", dpi=160); plt.close()

print(f"Saved 8 figures in {F}")
