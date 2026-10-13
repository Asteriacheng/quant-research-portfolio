"""Render compact research plots from exported real-data result tables."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/"reports"/"real_data_v1"; F=R/"figures"; F.mkdir(parents=True,exist_ok=True)
monthly=pd.read_csv(R/"monthly_ic.csv",parse_dates=["month"]).set_index("month")
ax=monthly.plot(figsize=(12,5),alpha=.8,lw=1)
ax.axvline(pd.Timestamp("2024-01-01"),color="black",ls="--",label="frozen test starts")
ax.set_title("Monthly Rank IC (5-session executable open-to-open label)"); ax.set_ylabel("Rank IC")
ax.legend(ncol=3,fontsize=8); plt.tight_layout(); plt.savefig(F/"01_monthly_ic.png",dpi=160); plt.close()
q=pd.read_csv(R/"quintile_summary_train.csv")
q.pivot(index="quintile",columns="factor",values="mean_forward_return").plot(marker="o",figsize=(10,5))
plt.title("Training-period quintile mean forward returns"); plt.ylabel("5-session return")
plt.tight_layout(); plt.savefig(F/"02_quintile_returns_train.png",dpi=160); plt.close()

# chart 3: cumulative compounded monthly equally weighted Q5 diagnostic returns in reserved test.
fig,ax=plt.subplots(figsize=(12,5))
for name in q.factor.unique():
    g=pd.read_csv(R/f"quintile_daily_{name}.csv",parse_dates=["date"])
    g=g[(g.quintile==5)&(g.date>="2024-01-01")].set_index("date")["return"].resample("ME").mean()
    ax.plot((1+g).cumprod(),label=name)
ax.axvline(pd.Timestamp("2024-01-01"),color="black",ls="--")
ax.set_title("Test top-quintile compounded monthly mean 5-session forward returns")
ax.set_ylabel("Index (overlapping-label diagnostic)"); ax.legend(fontsize=8,ncol=2)
plt.tight_layout(); plt.savefig(F/"03_test_top_quintile_cumulative.png",dpi=160); plt.close()

# chart 4: drawdown of the same diagnostic return series.
fig,ax=plt.subplots(figsize=(12,5))
for name in q.factor.unique():
    g=pd.read_csv(R/f"quintile_daily_{name}.csv",parse_dates=["date"])
    g=g[(g.quintile==5)&(g.date>="2024-01-01")].set_index("date")["return"].resample("ME").mean()
    wealth=(1+g).cumprod(); dd=wealth/wealth.cummax()-1
    ax.plot(dd,label=name)
ax.set_title("Test top-quintile drawdown (overlapping-label diagnostic)"); ax.set_ylabel("Drawdown")
ax.legend(fontsize=8,ncol=2); plt.tight_layout(); plt.savefig(F/"04_test_drawdown.png",dpi=160); plt.close()

# chart 5: mean daily top-20% membership replacement proxy.
fig,ax=plt.subplots(figsize=(12,5))
for p in R.glob("top20_turnover_*.csv"):
    t=pd.read_csv(p,parse_dates=["date"]).set_index("date").iloc[:,0].resample("ME").mean()
    ax.plot(t.index,t.values,label=p.stem.replace("top20_turnover_",""))
ax.set_title("Top-20% membership replacement rate"); ax.set_ylabel("Daily entrant fraction")
ax.legend(fontsize=8,ncol=2); plt.tight_layout(); plt.savefig(F/"05_turnover.png",dpi=160); plt.close()

# chart 6: monthly IC moving averages, useful as a stability view.
ax=monthly.rolling(6,min_periods=3).mean().plot(figsize=(12,5),lw=1)
ax.axvline(pd.Timestamp("2024-01-01"),color="black",ls="--")
ax.set_title("Six-month rolling mean Rank IC"); ax.set_ylabel("Rank IC")
ax.legend(ncol=3,fontsize=8); plt.tight_layout(); plt.savefig(F/"06_stability_rolling_ic.png",dpi=160); plt.close()

# chart 7: source's static industry labels. Labels are not point-in-time history.
meta=pd.read_parquet(ROOT/"data"/"raw"/"astock"/"stock_list.parquet")
meta=meta[meta.code.str.fullmatch(r"\d{6}")].drop_duplicates("code")
meta.sector.fillna("缺失").value_counts().head(15).sort_values().plot.barh(figsize=(10,6))
plt.title("Top 15 source industry labels (static metadata, not historical classifications)")
plt.xlabel("Number of codes"); plt.tight_layout(); plt.savefig(F/"07_industry_coverage.png",dpi=160); plt.close()

# chart 8: single-day forward-return IC decay.
decay=pd.read_csv(R/"decay_summary.csv")
ax=decay.pivot(index="horizon_day",columns="factor",values="mean_rank_ic").plot(marker="o",figsize=(10,5))
ax.axhline(0,color="black",lw=.7); ax.set_title("Single-day Rank IC by forward horizon")
ax.set_xlabel("Forward trading session k"); ax.set_ylabel("Mean Rank IC")
ax.legend(fontsize=8,ncol=2); plt.tight_layout(); plt.savefig(F/"08_ic_decay.png",dpi=160); plt.close()
