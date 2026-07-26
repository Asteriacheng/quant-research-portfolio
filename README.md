# factorkit — Single-Factor Validation for A-Share Research

**中文版本见下方** | [English below](#english)

---

## 项目是什么

一个小而诚实的**单因子检验工具包**。不负责挖因子,只负责判因子的死活:
IC / ICIR / 换手 / 成本 / 衰减 / 与池内相关性 / 事件研究 / 前视泄漏排查。

所有阈值写在文档里,所有结论可复现,没有黑箱。

核心理念:**一个回测框架靠不靠谱,看它拒绝结果的流程,不看它产生结果的流程。**

---

## 快速开始

```bash
pip install pandas numpy scipy

python examples/run_factor_report.py
python examples/run_event_study.py
python -m unittest discover -s tests -t .
```

输出在 `examples/sample_output/`。数据是合成的,包含一个已知的 5 日反转 alpha,
用来验证工具的口径——不是市场结论。

---

## 文档导航

| 文档 | 内容 |
|---|---|
| [docs/01-factor-checklist.md](docs/01-factor-checklist.md) | 单因子研究五关检查清单 + 组合层基础理论 |
| [docs/02-case-tdx-indicator-teardown.md](docs/02-case-tdx-indicator-teardown.md) | 真实案例:一条商业指标的因子化拆解与判死 |
| [docs/03-backtest-integrity.md](docs/03-backtest-integrity.md) | 回测防自欺八项校验 + 一次已公开的方法修正 |
| [docs/04-pipeline-design.md](docs/04-pipeline-design.md) | 选股流水线的四层岗位设计与顺序约束 |
| [docs/05-research-engineering.md](docs/05-research-engineering.md) | 研究工程纪律:问题台账、冻结指纹、事故复盘 |

---

## 代码结构

```
factorkit/
├── __init__.py              # 导出所有公开接口
├── panel.py                 # 面板数据容器 + 合成行情生成器
├── metrics.py               # IC / ICIR / 换手 / 衰减 / 相关性
├── event_study.py           # 事件研究 + 机械基线 + 消融
└── integrity.py             # 前视泄漏探针 + 执行现实性

examples/
├── run_factor_report.py     # 单因子完整检验报告(合成数据)
├── run_event_study.py       # 事件研究案例(合成数据)
└── sample_output/
    ├── factor_report.md     # 生成的因子报告
    └── event_study.md       # 生成的事件研究报告

tests/
├── test_metrics.py          # 20 个单元测试(20/20 pass)
├── test_event_study.py
└── test_integrity.py
```

---

## 五关检查顺序(fixed pipeline)

```
1. 有效性(RankIC / ICIR / t / 单调性 / 方向稳定)
   ↓ (无效则后面全免)
2. 换手率(top 组合日换手 < 0.3 较舒服)
   ↓
3. 交易成本(年化 = 日换手 × 244 × 双边费率)
   ↓
4. Alpha 衰减(因子对 t+k 单日收益的 IC 曲线)
   ↓
5. 与池内相关性(|corr| < 0.7 才入池,< 0.3 才算新鲜)
   ↓
入池 + 组合层
```

前一关不过后面全免——不是为了严格,是为了省算力。

---

## 关键设计选择

### 1. 可执行口径写死在数据层
标签一律定义为:**T 日收盘出信号 → T+1 开盘买入 → T+1+H 开盘卖出的超额收益**。
用 T 日收盘当买入价,回测赚的是一个不存在的成交机会。

实现:
```python
from factorkit import forward_return, filter_executable

fwd = forward_return(panel, horizon=5, mode="open_to_open")
executed = filter_executable(signal, panel)  # 剔除涨停/停牌
```

### 2. 事件研究必须配机械基线
裸胜率 > 50% 不代表有 alpha——那段整个市场都在涨。
对照:同信号日、同股票池、**随机等量抽票**的基线。

实现:
```python
from factorkit import event_study, bucketed_event_study

result = event_study(signal, fwd, n_draws=500)
print(f"净抬升 = {result.net_lift * 100:.2f}pt")  # 信号均值 - 基线均值

# 分市场状态桶,因为整体净抬升为正可能只是因为信号集中在好做的时段
bucketed = bucketed_event_study(signal, fwd, buckets_by_market_regime, n_draws=200)
```

### 3. 因子层的前视泄漏用截断不变性判
打乱标签日期的置换检验在因子层对反转类因子有**结构性假阳性**。
改用**截断不变性测试**:数据截到某天重算,因子每一格的值必须和全量完全一致。

实现:
```python
from factorkit import truncation_invariance

report = truncation_invariance(my_factor_fn, panel, cut_dates)
assert report["passed"].all(), "有偷看未来的函数"
```

### 4. 与池内相关性:判「赚的是不是同一笔钱」
同一条公式拆出来的几个变量往往互相 0.8~0.95 相关,只能留一个。
正交化用截面回归取残差,残差还有 IC 才算带来新钱。

实现:
```python
from factorkit import correlation_with_pool, residualize

corr = correlation_with_pool(new_factor, existing_pool)
print(corr)  # 如果 |corr| > 0.7 对哪一个,只能二选一

resid = residualize(new_factor, existing_pool)
ic_resid = ic_summary(rank_ic(resid, fwd))
# 残差 IC >= 0.015 才算增量
```

---

## 测试

所有测试在合成数据上跑。合成数据**故意埋了一个已知方向的反转 alpha**:
如果工具测不出来,说明工具坏了,不是数据问题。

```bash
python -m unittest discover -s tests -t .
# 预期:20/20 pass
```

关键测试:
- `test_reversal_ic_is_positive_and_significant`:反转因子必须测出显著为正的 IC
- `test_leaky_factor_is_caught`:故意偷看未来的因子必须被抓到
- `test_random_signal_has_no_net_lift`:纯随机信号不该有净抬升

---

## 一句话定位

这个工具包的价值不在于「挖到了什么」,而在于**能可信地测量「井有多深」**。
一个能诚实告诉你「这里没矿」的流程,比一个总能挖到矿的流程值钱得多。

---

## 数据输入格式

如果想用真实数据,改写 `examples/run_factor_report.py` 中的:

```python
# 从这一行:
panel = make_synthetic_panel(...)

# 改成:
panel = Panel.from_csv_dir("你的 CSV 目录")
```

CSV 要求:每只股票一个文件,文件名即代码,包含列 `date, open, high, low, close, volume, amount`。
`volume` 单位为股(不是手)。

---

<a name="english"></a>

# factorkit — A Small, Honest Single-Factor Validation Toolkit

## What is this

A minimal Python toolkit for validating cross-sectional stock-picking factors.
It does **not** generate factors; it judges whether a factor is dead or alive:
IC / ICIR / turnover / cost / decay / orthogonality / event study / look-ahead audit.

All thresholds are documented, all results are reproducible, no black boxes.

**Core principle: A backtest framework is only as good as its rejection process, not its generative process.**

## Quick start

```bash
pip install pandas numpy scipy
python examples/run_factor_report.py
python examples/run_event_study.py
python -m unittest discover -s tests -t .
```

Sample output in `examples/sample_output/`. Data is synthetic, containing one known reversal
alpha — used to verify the toolkit's correctness, not to draw market conclusions.

## Documentation

- **01-factor-checklist.md**: Five-gate pipeline, numeric thresholds for A-share daily data
- **02-case-tdx-indicator-teardown.md**: Real case study: a retail TDX indicator deconstructed and killed
- **03-backtest-integrity.md**: Eight mechanical checks + one method correction I made public
- **04-pipeline-design.md**: Four-role pipeline architecture (gate, admission, rank, veto)
- **05-research-engineering.md**: Operating discipline: issue ledger, frozen config fingerprint, incident reviews

## Pipeline (fixed order)

```
1. Effectiveness (RankIC | ICIR | t-stat | monotonicity | direction stability)
   → if failed, stop
2. Turnover (top-10% daily < 0.3)
3. Cost drag (annual ≈ daily_turnover × 244 × 25bp)
4. Decay curve (IC against t+k single-day returns)
5. Correlation with pool (|corr| < 0.7 to enter, < 0.3 to be new)
   → if passed, enters combination layer
```

## Key design choices

**1. Executable pricing fixed at data level**
```python
fwd = forward_return(panel, horizon=5, mode="open_to_open")
# T close → T+1 open buy → T+1+H open sell
executed = filter_executable(signal, panel)  # drops limit-up, suspended
```

**2. Event study requires mechanical baseline**
```python
result = event_study(signal, fwd, n_draws=500)
# net_lift = signal.mean() - random_baseline.mean()
bucketed = bucketed_event_study(signal, fwd, market_buckets)
```

**3. Look-ahead audit via truncation invariance**
```python
report = truncation_invariance(my_factor, panel, cut_dates)
assert report["passed"].all()  # every cell matches after truncation
```

**4. Orthogonality via residualization**
```python
corr = correlation_with_pool(new_factor, existing_pool)
# if |corr| > 0.7 to any existing factor, choose one
resid = residualize(new_factor, existing_pool)
ic_resid = ic_summary(rank_ic(resid, fwd))
# ic_resid >= 0.015 = "brings new money"
```

## Tests

20 unit tests, all pass. Synthetic data carries a **known reversal alpha** to verify the toolkit
can find it — if the toolkit fails, the toolkit is broken, not the data.

```bash
python -m unittest discover -s tests -t .
# Expected: 20/20 pass
```

Key assertions:
- Reversal factor must show positive significant IC
- Deliberately leaky factor must be caught by truncation probe
- Random signal must have zero net lift vs. baseline

## Value proposition

The toolkit's worth is not in "what I found" but in **"how deep is the well" — a credible
measurement**. A process that can honestly tell you "there is no alpha here" is worth more than
one that always finds something.

---

## License & Citation

This is a research toolkit built from first principles, documented for reproducibility.
Use it to validate your own factors, not as a black-box signal generator.

When citing this work in research or reports, reference the docs:
- Methodology: docs/01-factor-checklist.md + docs/03-backtest-integrity.md
- Case study: docs/02-case-tdx-indicator-teardown.md  
- Pipeline: docs/04-pipeline-design.md
