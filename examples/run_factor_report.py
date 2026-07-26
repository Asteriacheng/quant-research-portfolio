"""Single-factor report on synthetic data. / 合成数据上的单因子检验报告。

    python examples/run_factor_report.py

输出 examples/sample_output/factor_report.md。
数据是合成的,**结论只用于证明工具算得对**,不是任何市场结论。
换成真实数据:把 make_synthetic_panel() 换成 Panel.from_csv_dir("你的目录")。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from factorkit import (                                        # noqa: E402
    annual_cost_drag,
    correlation_with_pool,
    decay_curve,
    factor_autocorr,
    forward_return,
    ic_summary,
    make_synthetic_panel,
    quantile_stats,
    rank_ic,
    residualize,
    top_turnover,
    truncation_invariance,
)

OUT = ROOT / "examples" / "sample_output" / "factor_report.md"

FACTORS = {
    "rev5 5日反转": lambda p: -(p.close / p.close.shift(5) - 1.0),
    "mom20 20日动量": lambda p: p.close / p.close.shift(20) - 1.0,
    "vol20 20日已实现波动率(取负)": lambda p: -p.close.pct_change().rolling(20).std(),
    "amt20 20日成交额(取负,小票代理)": lambda p: -p.amount.rolling(20).mean(),
}


def fmt(x: float, digits: int = 4) -> str:
    return "n/a" if x != x else f"{x:.{digits}f}"


def main() -> None:
    panel = make_synthetic_panel(n_days=600, n_stocks=300, seed=20260726)
    fwd5 = forward_return(panel, horizon=5, mode="close_to_close")

    lines: list[str] = []
    add = lines.append
    add("# 单因子检验报告(合成数据) / Single-Factor Report (synthetic data)")
    add("")
    add(f"- 样本 / sample: {panel.shape[0]} 个交易日 × {panel.shape[1]} 只股票")
    add(f"- 区间 / span: {panel.dates[0].date()} → {panel.dates[-1].date()}")
    add("- 标签 / label: 未来 5 日收益(close-to-close)")
    add("- ⚠️ 数据为合成生成,内含一个**已知的 5 日反转 alpha**;本报告用于验证工具口径,不含市场结论。")
    add("")

    add("## 一、有效性总表 / Effectiveness")
    add("")
    add("| 因子 Factor | RankIC | ICIR | t | IC>0 占比 | 前半 IC | 后半 IC | 方向稳定 | 判定 |")
    add("|---|---|---|---|---|---|---|---|---|")
    summaries = {}
    for name, fn in FACTORS.items():
        s = ic_summary(rank_ic(fn(panel), fwd5))
        summaries[name] = s
        verdict = "PASS" if abs(s.ic_mean) > 0.02 and abs(s.t_stat) > 3 and s.direction_stable else "REJECT"
        add(
            f"| {name} | {fmt(s.ic_mean)} | {fmt(s.icir, 3)} | {fmt(s.t_stat, 2)} | "
            f"{fmt(s.positive_ratio, 3)} | {fmt(s.ic_first_half)} | {fmt(s.ic_second_half)} | "
            f"{'✔' if s.direction_stable else '✘'} | {verdict} |"
        )
    add("")
    add("门槛(A 股日频经验值):|IC| > 0.02 可用、> 0.05 很强;ICIR > 0.3 尚可;")
    add("多重检验环境下 |t| > 3 才算站住(2 不够);前后半段方向必须一致。")
    add("")

    add("## 二、换手、成本与自相关 / Turnover, cost, autocorrelation")
    add("")
    add("| 因子 | top10% 日换手 | 秩自相关 | 年化成本(双边 25bp) | 备注 |")
    add("|---|---|---|---|---|")
    for name, fn in FACTORS.items():
        f = fn(panel)
        to = float(top_turnover(f, 0.1).mean())
        ac = factor_autocorr(f)
        cost = annual_cost_drag(to, 25.0)
        note = "慢因子" if ac > 0.95 else ("快因子" if ac < 0.5 else "中速")
        if to > 0.6:
            note += ",换手过高必须处理"
        add(f"| {name} | {fmt(to, 3)} | {fmt(ac, 3)} | {cost:.1%} | {note} |")
    add("")
    add("成本这一栏的作用不是精确,是**在建模之前就把喂不起的因子劝退**:")
    add("日频满换手 ≈ 年 244 次 × 双边 25bp ≈ 61%/年,什么 alpha 都撑不住。")
    add("")

    add("## 三、分组单调性 / Quantile monotonicity")
    add("")
    for name in ("rev5 5日反转", "mom20 20日动量"):
        table, rho = quantile_stats(FACTORS[name](panel), fwd5, q=10)
        add(f"**{name}** — 单调性 rho = {rho:.3f}")
        add("")
        add("| 组 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |")
        add("|---|---|---|---|---|---|---|---|---|---|---|")
        cells = " | ".join(f"{v * 100:.2f}%" for v in table["mean_ret"])
        add(f"| 平均 5 日收益 | {cells} |")
        add("")
    add("只有两头有、中间乱 = 尾部效应,不是线性 alpha,这种因子不能直接当排序分用。")
    add("")

    add("## 四、衰减曲线 / Alpha decay")
    add("")
    curve = decay_curve(FACTORS["rev5 5日反转"](panel), panel, ks=(1, 2, 3, 5, 10))
    add("| t+k | 1 | 2 | 3 | 5 | 10 |")
    add("|---|---|---|---|---|---|")
    add("| 单日 IC | " + " | ".join(fmt(v) for v in curve["ic_mean"]) + " |")
    add("| 保留强度 | " + " | ".join(f"{v:.0%}" for v in curve["retained"]) + " |")
    add("")
    add("读法:t+3 归零 = 快因子,必须快执行、对滑点极敏感;")
    add("t+10 仍有 60% 强度 = 慢因子,容量大、可配长持有。")
    add("")

    add("## 五、与池内相关性与正交化 / Correlation with the pool & orthogonalisation")
    add("")
    add("假设池内已有 `rev5`,现在有两个候选要入池:")
    add("")
    pool = {"rev5": FACTORS["rev5 5日反转"](panel)}
    candidates = {
        "mom20 20日动量": FACTORS["mom20 20日动量"](panel),
        "rev10 10日反转": -(panel.close / panel.close.shift(10) - 1.0),
    }
    add("| 候选 | 与 rev5 的相关 | 原始 IC | 对 rev5 残差化后的 IC | 增量判定 |")
    add("|---|---|---|---|---|")
    for name, f in candidates.items():
        corr = correlation_with_pool(f, pool)["rev5"]
        raw = ic_summary(rank_ic(f, fwd5)).ic_mean
        resid = ic_summary(rank_ic(residualize(f, pool), fwd5)).ic_mean
        verdict = "有增量" if abs(resid) >= 0.015 else "无增量,不入池"
        add(f"| {name} | {fmt(corr, 3)} | {fmt(raw)} | {fmt(resid)} | {verdict} |")
    add("")
    add("门槛:|corr| < 0.7 才谈得上入池,< 0.3 才算真新鲜;残差 IC ≥ 0.015 才算带来新钱。")
    add("判的是「赚的是不是同一笔钱」,不是「表达式长得像不像」——同一条公式拆出来的")
    add("几个变量常常互相 0.8~0.95 相关,只能留一个。")
    add("")

    add("## 六、前视泄漏探针 / Look-ahead probe")
    add("")
    add("截断不变性:把数据截到某天重算,因子在历史每一格的取值必须和全量算出来的完全一致。")
    add("")
    add("| 因子 | 截断点 | 比较格数 | 不一致格数 | 最大绝对差 | 通过 |")
    add("|---|---|---|---|---|---|")
    cuts = [panel.dates[300], panel.dates[450]]
    for name, fn in FACTORS.items():
        rep = truncation_invariance(fn, panel, cuts)
        for _, row in rep.iterrows():
            add(
                f"| {name} | {row['cut_date']} | {row['cells_compared']:,} | {row['n_mismatch']} | "
                f"{row['max_abs_diff']:.2e} | {'✔' if row['passed'] else '✘'} |"
            )
    add("")
    add("对照组:把 `close.shift(-1)/close - 1` 这种偷看未来一天的因子丢进同一个探针,")
    add("会在截断点当天出现「一边有值一边是 NaN」的不一致而被判死(见 tests/test_integrity.py)。")
    add("")
    add("---")
    add("")
    add("生成命令 / reproduce: `python examples/run_factor_report.py`")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
