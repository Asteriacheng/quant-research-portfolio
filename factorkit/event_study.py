"""Event study with a mechanical random baseline and drop-one ablation.

事件研究:机械随机基线 + 逐条件消融。

为什么必须配随机基线
--------------------
布尔选股信号(通达信 XG 那类稀疏事件)不能算 IC,只能做事件研究。
但**裸胜率没有意义**:信号偏好在弱市反弹段触发,那段随机买也赚钱。
所以每个信号都要和一条机械基线比:**同信号日、同股票池、随机等量抽票、
同出场规则**。报「净抬升 = 信号均值 − 基线均值」,不报裸胜率。

真实教训:某指标的大 AND 条件在弱市桶 10 日裸收益 +0.94% 看似金矿,
分桶随机基线是 +1.38% —— 选股是负贡献。不分桶就看不出来。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

__all__ = ["EventResult", "event_study", "bucketed_event_study", "drop_one_ablation"]


@dataclass(frozen=True)
class EventResult:
    n_events: int
    mean_ret: float
    median_ret: float
    win_rate: float
    baseline_mean: float
    baseline_std: float
    net_lift: float          # 信号均值 − 基线均值(小数)
    p_value: float           # 置换 p:随机基线达到或超过信号的比例
    n_draws: int

    def to_dict(self) -> dict:
        return asdict(self)

    def verdict(self, alpha: float = 0.05) -> str:
        if self.net_lift <= 0:
            return "REJECT(净抬升 ≤ 0,跑输随机基线)"
        if self.p_value > alpha:
            return f"WEAK(净抬升为正但 p={self.p_value:.3f} 不显著)"
        return "PASS(净抬升为正且显著)"


def _eligible_arrays(fwd: pd.DataFrame, dates) -> dict:
    """每个信号日的可选池(前向收益非空的股票),预先取出来,避免重复切片。"""
    pool = {}
    for d in dates:
        vals = fwd.loc[d].to_numpy(dtype=float)
        pool[d] = vals[~np.isnan(vals)]
    return pool


def event_study(
    signal: pd.DataFrame,
    fwd: pd.DataFrame,
    n_draws: int = 500,
    seed: int = 0,
    min_events: int = 20,
) -> EventResult:
    """Compare a boolean signal against a same-day random baseline.

    对布尔信号做事件研究,并与「同日同池随机等量抽票」的机械基线比较。

    Parameters
    ----------
    signal : 与 fwd 同形状的布尔宽表,True = 该日该股触发。
    fwd    : 前向收益(建议用 forward_return(..., mode="open_to_open"))。
    n_draws: 随机基线的重抽次数,决定置换 p 的分辨率。
    """
    sig, r = signal.align(fwd, join="inner")
    sig = sig.fillna(False).astype(bool) & r.notna()
    counts = sig.sum(axis=1)
    active = counts[counts > 0].index
    if len(active) == 0:
        raise ValueError("没有任何触发日 / no trigger day")

    sig_vals = r.where(sig).to_numpy(dtype=float)
    events = sig_vals[~np.isnan(sig_vals)]
    n_events = int(events.size)
    if n_events < min_events:
        raise ValueError(f"事件数 {n_events} < {min_events},统计力不足 / too few events")

    pool = _eligible_arrays(r, active)
    per_day = counts.loc[active].to_numpy(dtype=int)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws, dtype=float)
    for i in range(n_draws):
        acc = np.empty(n_events, dtype=float)
        pos = 0
        for d, k in zip(active, per_day):
            candidates = pool[d]
            if candidates.size == 0:
                continue
            take = min(k, candidates.size)
            acc[pos : pos + take] = rng.choice(candidates, size=take, replace=False)
            pos += take
        draws[i] = acc[:pos].mean() if pos else np.nan

    draws = draws[~np.isnan(draws)]
    actual = float(events.mean())
    baseline = float(draws.mean())
    # 置换 p:加 1 平滑,避免 n_draws 有限时报出 p=0
    p_value = float((np.sum(draws >= actual) + 1) / (draws.size + 1))

    return EventResult(
        n_events=n_events,
        mean_ret=actual,
        median_ret=float(np.median(events)),
        win_rate=float((events > 0).mean()),
        baseline_mean=baseline,
        baseline_std=float(draws.std(ddof=1)),
        net_lift=actual - baseline,
        p_value=p_value,
        n_draws=int(draws.size),
    )


def bucketed_event_study(
    signal: pd.DataFrame,
    fwd: pd.DataFrame,
    buckets: pd.Series,
    n_draws: int = 300,
    seed: int = 0,
    min_events: int = 20,
) -> pd.DataFrame:
    """Run the event study separately inside each date bucket.

    按日期分桶(如牛/震荡/熊)分别做事件研究。

    存在的意义:整体净抬升为正,可能只是因为信号集中在某个整体好做的时段。
    分桶后基线随之抬高,「金矿」经常就消失了。
    """
    rows = []
    for label, dates in buckets.groupby(buckets):
        idx = dates.index
        try:
            res = event_study(
                signal.loc[signal.index.intersection(idx)],
                fwd.loc[fwd.index.intersection(idx)],
                n_draws=n_draws,
                seed=seed,
                min_events=min_events,
            )
        except ValueError as exc:
            rows.append({"bucket": label, "n_events": 0, "note": str(exc)})
            continue
        row = {"bucket": label, **res.to_dict(), "verdict": res.verdict()}
        rows.append(row)
    return pd.DataFrame(rows)


def drop_one_ablation(
    conditions: dict[str, pd.DataFrame],
    fwd: pd.DataFrame,
    n_draws: int = 300,
    seed: int = 0,
    min_events: int = 20,
) -> pd.DataFrame:
    """Drop each AND-condition in turn and see what happens to net lift.

    逐条件消融:把大 AND 链里的每个条件轮流删掉,看净抬升怎么变。

    读法:
      - 删掉后净抬升**下降** → 该条件是承重柱;
      - 删掉后净抬升**上升** → 该条件是毒药(常见于「动量确认」类条件把
        反转 alpha 毁掉),应当删;
      - 几乎不变 → 搭车客,只在减少样本,建议删。
    条件越多样本越少、过拟合越重,且分不清谁在贡献——大 AND 不配消融等于没做。
    """
    names = list(conditions)
    if len(names) < 2:
        raise ValueError("至少两个条件才谈得上消融")

    def _and(keys) -> pd.DataFrame:
        out = None
        for k in keys:
            cur = conditions[k].fillna(False).astype(bool)
            out = cur if out is None else (out & cur)
        return out

    full = event_study(_and(names), fwd, n_draws=n_draws, seed=seed, min_events=min_events)
    rows = [
        {
            "variant": "FULL(全条件)",
            "dropped": "-",
            "n_events": full.n_events,
            "mean_ret": full.mean_ret,
            "baseline_mean": full.baseline_mean,
            "net_lift": full.net_lift,
            "delta_vs_full": 0.0,
            "p_value": full.p_value,
        }
    ]
    for name in names:
        keys = [k for k in names if k != name]
        try:
            res = event_study(_and(keys), fwd, n_draws=n_draws, seed=seed, min_events=min_events)
        except ValueError as exc:
            rows.append({"variant": f"drop {name}", "dropped": name, "n_events": 0, "note": str(exc)})
            continue
        rows.append(
            {
                "variant": f"drop {name}",
                "dropped": name,
                "n_events": res.n_events,
                "mean_ret": res.mean_ret,
                "baseline_mean": res.baseline_mean,
                "net_lift": res.net_lift,
                "delta_vs_full": res.net_lift - full.net_lift,
                "p_value": res.p_value,
            }
        )
    table = pd.DataFrame(rows)
    table["role"] = np.where(
        table["variant"].eq("FULL(全条件)"),
        "-",
        np.where(table["delta_vs_full"] < -0.001, "承重柱 load-bearing",
                 np.where(table["delta_vs_full"] > 0.001, "毒药 harmful", "搭车客 free-rider")),
    )
    return table
