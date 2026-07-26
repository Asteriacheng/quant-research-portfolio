"""Event study with a mechanical baseline. / 带机械基线的事件研究。

    python examples/run_event_study.py

输出 examples/sample_output/event_study.md。

这个脚本演示的是一个具体的方法论主张:
**布尔选股信号的裸胜率没有意义,必须和「同日同池随机等量抽票」的机械基线比。**
不比基线,你分不清赚的是选股能力还是那几天整个市场都在涨。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd                                            # noqa: E402

from factorkit import (                                        # noqa: E402
    bucketed_event_study,
    drop_one_ablation,
    event_study,
    filter_executable,
    forward_return,
    make_synthetic_panel,
    unbuyable_at_open,
)

OUT = ROOT / "examples" / "sample_output" / "event_study.md"


def market_state(panel, window: int = 20) -> pd.Series:
    """按全市场 20 日累计收益分三档:弱市 / 震荡 / 强市。"""
    breadth = panel.close.pct_change().mean(axis=1)
    cum = breadth.rolling(window).sum()
    q = cum.quantile([1 / 3, 2 / 3])
    labels = pd.Series("震荡", index=cum.index, dtype=object)
    labels[cum <= q.iloc[0]] = "弱市"
    labels[cum >= q.iloc[1]] = "强市"
    return labels[cum.notna()].rename("bucket")


def main() -> None:
    panel = make_synthetic_panel(n_days=600, n_stocks=300, seed=20260726)
    fwd10 = forward_return(panel, horizon=10, mode="open_to_open")

    ret1 = panel.close.pct_change()
    past5 = panel.close / panel.close.shift(5) - 1.0
    past20 = panel.close / panel.close.shift(20) - 1.0
    vol_ma5 = panel.volume.rolling(5).mean()
    amt_median = panel.amount.median(axis=1)

    breakout = {
        "A 阳线实体>3%": ret1 > 0.03,
        "B 温和放量(量>1.2×5日均量)": panel.volume > 1.2 * vol_ma5,
        "C 中期动量为正(20日>0)": past20 > 0,
        "D 流动性(成交额高于中位)": panel.amount.gt(amt_median, axis=0),
    }
    dip = {
        "E 5日跌幅>5%": past5 < -0.05,
        "D 流动性(成交额高于中位)": panel.amount.gt(amt_median, axis=0),
    }

    def and_all(conds: dict[str, pd.DataFrame]) -> pd.DataFrame:
        out = None
        for v in conds.values():
            out = v if out is None else (out & v)
        return out.fillna(False)

    sig_breakout = and_all(breakout) & fwd10.notna()
    sig_dip = and_all(dip) & fwd10.notna()

    lines: list[str] = []
    add = lines.append
    add("# 事件研究报告(合成数据) / Event Study (synthetic data)")
    add("")
    add(f"- 样本 / sample: {panel.shape[0]} 个交易日 × {panel.shape[1]} 只股票")
    add("- 持有期 / holding: 10 个交易日,**可执行口径**(T 日收盘出信号 → T+1 开盘买 → T+11 开盘卖)")
    add("- 基线 / baseline: 同信号日、同股票池、随机等量抽票、同出场规则(重抽 500 次)")
    add("- ⚠️ 合成数据,内含一个已知的 5 日反转 alpha;本报告用于演示方法,不含市场结论。")
    add("")

    add("## 一、两个信号 vs 机械基线 / Two signals against the baseline")
    add("")
    add("| 信号 | 事件数 | 裸均值 | 裸胜率 | 基线均值 | 净抬升 | 置换 p | 判定 |")
    add("|---|---|---|---|---|---|---|---|")
    results = {}
    for name, sig in (("突破追涨(A∧B∧C∧D)", sig_breakout), ("回调低吸(E∧D)", sig_dip)):
        res = event_study(sig, fwd10, n_draws=500, seed=42)
        results[name] = res
        add(
            f"| {name} | {res.n_events:,} | {res.mean_ret:+.2%} | {res.win_rate:.1%} | "
            f"{res.baseline_mean:+.2%} | {res.net_lift * 100:+.2f}pt | {res.p_value:.3f} | {res.verdict()} |"
        )
    add("")
    add("**这张表就是整个方法论的全部意义**:裸均值和裸胜率单独看毫无信息,")
    add("把基线那一列盖住,两个信号看起来会像是同一个水平。")
    add("")

    add("## 二、分市场状态 / Bucketed by market state")
    add("")
    add("整体净抬升为正,也可能只是因为信号集中在整体好做的时段。分桶之后基线随之抬高,")
    add("「金矿」经常就消失了——这一步不做,等于没做事件研究。")
    add("")
    buckets = market_state(panel)
    for name, sig in (("突破追涨", sig_breakout), ("回调低吸", sig_dip)):
        table = bucketed_event_study(sig, fwd10, buckets, n_draws=200, seed=42)
        add(f"**{name}**")
        add("")
        add("| 市场状态 | 事件数 | 裸均值 | 基线均值 | 净抬升 | 置换 p |")
        add("|---|---|---|---|---|---|")
        for _, row in table.iterrows():
            if row.get("n_events", 0) == 0:
                add(f"| {row['bucket']} | 0 | - | - | - | 样本不足 |")
                continue
            add(
                f"| {row['bucket']} | {int(row['n_events']):,} | {row['mean_ret']:+.2%} | "
                f"{row['baseline_mean']:+.2%} | {row['net_lift'] * 100:+.2f}pt | {row['p_value']:.3f} |"
            )
        add("")

    add("## 三、逐条件消融 / Drop-one ablation")
    add("")
    add("对「突破追涨」这条 4 条件 AND 链,把每个条件轮流删掉:")
    add("")
    ablation = drop_one_ablation(breakout, fwd10, n_draws=200, seed=42)
    add("| 变体 | 事件数 | 裸均值 | 基线均值 | 净抬升 | 相对全条件 | 角色 |")
    add("|---|---|---|---|---|---|---|")
    for _, row in ablation.iterrows():
        if row.get("n_events", 0) == 0:
            add(f"| {row['variant']} | 0 | - | - | - | - | 样本不足 |")
            continue
        add(
            f"| {row['variant']} | {int(row['n_events']):,} | {row['mean_ret']:+.2%} | "
            f"{row['baseline_mean']:+.2%} | {row['net_lift'] * 100:+.2f}pt | "
            f"{row['delta_vs_full'] * 100:+.2f}pt | {row['role']} |"
        )
    add("")
    add("读法:删掉后净抬升**下降** = 承重柱;**上升** = 毒药(应当删);几乎不变 = 搭车客。")
    add("条件越多样本越少、过拟合越重,而且分不清谁在贡献——大 AND 不配消融等于没做。")
    add("")

    add("## 四、执行现实性 / Execution reality")
    add("")
    add("信号在 T 日收盘产生,能不能在 T+1 开盘真的买到?开盘即涨停买不进,停牌买不进。")
    add("")
    add("| 信号 | T 日触发数 | T+1 实际可买 | 弃单率 |")
    add("|---|---|---|---|")
    unbuyable = unbuyable_at_open(panel)
    for name, sig in (("突破追涨", sig_breakout), ("回调低吸", sig_dip)):
        raw = int(sig.to_numpy().sum())
        executed = int(filter_executable(sig, panel).to_numpy().sum())
        drop_rate = 1 - executed / raw if raw else float("nan")
        add(f"| {name} | {raw:,} | {executed:,} | {drop_rate:.2%} |")
    add("")
    add(f"(合成数据里注入的「开盘一字涨停」共 {int(unbuyable.to_numpy().sum()):,} 格。)")
    add("")
    add("弃单率必须写进报告:弃单率高本身就是策略缺陷——信号总落在买不进的日子上,")
    add("意味着回测里赚到的那部分收益,现实中根本拿不到。")
    add("")
    add("---")
    add("")
    add("生成命令 / reproduce: `python examples/run_event_study.py`")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
