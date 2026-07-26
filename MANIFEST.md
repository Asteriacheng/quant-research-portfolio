# File Manifest with Descriptions / 文件清单与说明

---

## Core Toolkit / 核心工具包

### `factorkit/__init__.py`
**EN**: Public API exports. All functions used in examples are listed here.  
**中文**: 公开接口导出。所有在例子里用到的函数都在这里暴露。

### `factorkit/panel.py`
**EN**: Panel data container (T×N wide DataFrames) and synthetic A-share-like market generator
with an **intentionally embedded reversal alpha** for self-testing.

**中文**: 面板数据容器(T×N 宽表)与合成行情生成器,内含一个**故意埋入的反转 alpha**,
用来自测工具的正确性。

### `factorkit/metrics.py`
**EN**: Single-factor evaluation pipeline: RankIC, ICIR, t-stat, quantile monotonicity,
turnover, cost drag, decay curve, and correlation with existing factors.

**中文**: 单因子检验流水线:RankIC、ICIR、t 值、分组单调性、换手率、成本、衰减曲线、
与池内相关性。

### `factorkit/event_study.py`
**EN**: Boolean signal validation: event study with mechanical random baseline (C4 method),
market-bucketed event study, and drop-one ablation for AND-chains.

**中文**: 布尔信号验证:事件研究(含机械随机基线 C4 方法)、分市场状态桶、逐条件消融。

### `factorkit/integrity.py`
**EN**: Leakage probes and execution-reality filters: truncation invariance (look-ahead audit),
limit-up detection, T+1 executability, and shuffled-future test (with known false-positive caveat).

**中文**: 泄漏探针与执行现实性过滤:截断不变性(前视审计)、涨停识别、T+1 可执行性、
置换检验(含已知假阳性注记)。

---

## Tests / 测试

### `tests/test_metrics.py`
**EN**: 9 unit tests for the metrics pipeline. All tests use synthetic data with known alpha.
Tests verify: reversal IC is positive & significant, momentum is the mirror image,
quantile monotonicity, turnover computation, autocorrelation, decay shape, cost math.

**中文**: 9 个单元测试,用已知 alpha 的合成数据验证:反转因子 IC 显著为正、动量是镜像、
分组单调性、换手计算、自相关、衰减形状、成本数学。

### `tests/test_event_study.py`
**EN**: 6 unit tests for event study. Verify: random signals have zero net lift,
signals on known alpha beat the baseline, win rate alone can mislead, bucketed table structure,
drop-one ablation, few-events edge case.

**中文**: 6 个单元测试,验证:随机信号无净抬升、已知 alpha 上的信号跑赢基线、
裸胜率会误导、分桶表结构、消融结构、样本不足处理。

### `tests/test_integrity.py`
**EN**: 6 unit tests for leakage probes and execution filters. Verify: clean factors pass
truncation invariance, deliberately leaky factors get caught, limit-up mask is found,
unbuyable includes limit-up, filter_executable shifts by one day and drops unbuyables.

**中文**: 6 个单元测试,验证:纯回看因子通过截断检验、偷看未来的被抓到、
涨停识别工作、买不进包括涨停、可执行过滤正确移位并去掉买不进的单子。

**Run all tests**: `python -m unittest discover -s tests -t .`

---

## Examples / 例子

### `examples/run_factor_report.py`
**EN**: Runnable example: complete single-factor validation pipeline on synthetic data.
Outputs a markdown report showing effectiveness table, turnover & cost analysis, quantile stats,
decay curve, truncation-invariance audit. Data contains known reversal alpha so you can verify
the toolkit detects it correctly.

**中文**: 可运行的例子:在合成数据上跑完整单因子检验流水线。输出 markdown 报告,
包含有效性表、换手成本分析、分组统计、衰减曲线、截断不变性审计。数据含已知反转 alpha,
验证工具能正确测出。

**Run**: `python examples/run_factor_report.py`

**Output**: `examples/sample_output/factor_report.md`

### `examples/run_event_study.py`
**EN**: Runnable example: boolean signal validation with mechanical baseline. Generates
a report showing two test signals vs. random baseline, bucketed by market state, and
drop-one ablation of a 4-condition AND chain. Demonstrates why raw win rate is misleading
without the baseline, and how to identify which conditions are load-bearing vs. harmful.

**中文**: 可运行的例子:布尔信号验证(含机械基线)。生成报告显示两个测试信号 vs 随机基线、
按市场状态分桶、4 条件 AND 链的消融。演示为什么裸胜率没意义、怎样识别承重条件 vs 毒药。

**Run**: `python examples/run_event_study.py`

**Output**: `examples/sample_output/event_study.md`

### `examples/sample_output/factor_report.md`
**EN**: Generated report from `run_factor_report.py`. Shows effectiveness table (RankIC, ICIR,
t-stat, etc.), turnover & cost analysis, quantile monotonicity plots for two factors,
alpha decay curve, and truncation-invariance audit results.

**中文**: 由 `run_factor_report.py` 生成的报告。显示有效性表、换手成本分析、
两个因子的分组单调性、衰减曲线、截断不变性审计。

### `examples/sample_output/event_study.md`
**EN**: Generated report from `run_event_study.py`. Shows two test signals' performance
vs. mechanical baseline across market states, drop-one ablation identifying
load-bearing/harmful/free-rider conditions, and execution-reality metrics (fill rates).

**中文**: 由 `run_event_study.py` 生成的报告。显示两个测试信号按市场状态的表现 vs 基线、
消融识别承重/毒药/搭车条件、成交现实性指标。

---

## Documentation / 文档

### `docs/01-factor-checklist.md`
**EN**: The five-gate factor research pipeline with numeric thresholds for A-share daily data:
effectiveness → turnover → cost → decay → correlation with pool. Includes combination-layer
basics (equal-weight baseline, linear/ML synthesis, portfolio optimization). The checklist
is the operational definition of "does my factor live or die?"

**中文**: 五关因子研究流水线,含 A 股日频阈值:有效性→换手→成本→衰减→相关性。
包括组合层基础(等权基线、线性/ML 合成、优化)。检查清单是「我的因子是死是活」的操作定义。

### `docs/02-case-tdx-indicator-teardown.md`
**EN**: Real case study: a retail-circuit commercial TDX indicator was decomposed into 6
continuous factors and tested cross-sectionally. Findings: (1) the indicator's narrative
direction is inverted in the data, (2) four of the six variables are 0.8–0.95 correlated,
(3) the full AND-chain loses to random baseline once bucketed by regime, (4) drop-one ablation
shows momentum-confirmation conditions destroy the reversal alpha. Includes reproducible methods.

**中文**: 真实案例研究:一条商业通达信选股指标拆成 6 个连续因子全市场测试。发现:
(1) 指标叙事方向与实测相反, (2) 6 个里有 4 个互相 0.8~0.95 相关,
(3) 整 AND 链分桶后跑输随机基线, (4) 消融表显示动量确认条件毁掉了吸筹反转 alpha。
包含可复现方法。

### `docs/03-backtest-integrity.md`
**EN**: Eight mechanical checks every backtest result must pass before you quote a number.
Includes T+1 constraint, look-ahead audit (truncation invariance), quantity units, date
alignment, cache consistency, permutation test, independent-source spot-check, execution
reality. Plus two structural defences: executable-price convention and frozen split-date.
Includes one method correction: shuffled-future gives false positives at factor level,
so factor-level leakage is judged by truncation invariance instead.

**中文**: 八项机械校验,任何数字出炉前必须过。包含 T+1、前视排查、量纲、日期、
缓存、置换、独立源抽价、执行现实性。加两层结构防线:可执行口径 + 冻结切分日。
含一条公开的方法修正:置换检验在因子层有假阳性,改用截断不变性判。

### `docs/04-pipeline-design.md`
**EN**: Stock-selection pipeline architecture: four roles (gate, admission, ranker, veto),
each with different power and evidence type. Gate = regime filter (HMM on market breadth + vol);
Admission = boolean thresholds; Ranker = cross-sectional RankIC; Veto = individual-name kill.
Includes label design, mandatory sequencing (industry diversification MUST come after ranking),
and distinction between statistical rules vs. deterministic constraints.

**中文**: 选股流水线架构:四层岗位(阀门、入围、排序、反筛),各有不同权力和证据要求。
阀门 = 状态模型(HMM 用市场广度 + 波动率); 入围 = 二值门槛; 排序 = 截面 RankIC;
反筛 = 个股一票否决。包含标签设计、强制顺序(行业打散必须在排序之后)、
统计规律 vs 物理约束的区分。

### `docs/05-research-engineering.md`
**EN**: Operating discipline for research under your own rules: append-only issue ledger
with severity levels, config+code fingerprint frozen at run start (prevents silent parameter
drift), post-mortems for infrastructure failures that own their root-cause status
("not yet determined" is honest). The operational definition of a healthy system:
"no known defect survives two freeze cycles without being fixed/mitigated/explicitly accepted,
and every fix has before/after numbers."

**中文**: 自规研究工程纪律:只许追加的问题台账,带严重度分级、冻结指纹(运行开始时固定,
防止参数偷偷漂移)、事故复盘诚实写清根因定没定。健康系统的操作定义:
「已知缺陷不能活过两个冻结周期而不被处理,每个修复都有前后数字对照」。

---

## Project Root Files / 项目根文件

### `README.md`
**EN**: Project entry point. What the toolkit is, why it matters, quick start,
file structure overview, five key design choices with code examples, and value proposition.

**中文**: 项目入口。工具包是什么、为什么重要、快速开始、文件结构、五个关键设计(含代码示例)、价值定位。

### `.gitignore` (recommended)
```
__pycache__/
*.pyc
.pytest_cache/
.DS_Store
*.egg-info/
dist/
build/
```

---

## Quick Navigation / 快速导航

**For first-time users / 新手入门**:
1. Read `README.md` for orientation
2. Run `python examples/run_factor_report.py` and `run_event_study.py` to see it in action
3. Read `docs/01-factor-checklist.md` to understand the five-gate pipeline
4. Run `python -m unittest discover -s tests -t .` to verify everything works

**For methodological details / 方法细节**:
- **Single-factor pipeline**: `docs/01-factor-checklist.md`
- **Real-world example**: `docs/02-case-tdx-indicator-teardown.md`
- **Backtest integrity**: `docs/03-backtest-integrity.md`
- **Pipeline architecture**: `docs/04-pipeline-design.md`
- **Engineering discipline**: `docs/05-research-engineering.md`

**For code reference / 代码参考**:
- **Data & metrics**: `factorkit/panel.py`, `factorkit/metrics.py`
- **Event study**: `factorkit/event_study.py`
- **Leakage audit**: `factorkit/integrity.py`
- **Tests**: `tests/test_*.py` (20 unit tests, all pass)
- **Working examples**: `examples/run_*.py`

---

## File Statistics / 文件统计

| Category | Count | Loc (approx) |
|---|---|---|
| Core toolkit files | 4 | ~1,200 |
| Test files | 3 | ~350 |
| Documentation | 5 | ~2,000 |
| Examples | 2 | ~400 |
| Generated reports | 2 | ~200 |
| **Total** | **19** | **~4,150** |

Unit tests: **20 total, all 20 passing**.

---

## Design Principles / 设计原则

✅ **Small**: ~1,200 LOC of core toolkit, easy to read and modify  
✅ **Honest**: All thresholds documented, all conclusions reproducible, no black boxes  
✅ **Tested**: 20 unit tests with synthetic alpha embedded for correctness verification  
✅ **Practical**: Includes real case study showing how to kill factors in practice  
✅ **Bilingual**: All documentation in English + Chinese for accessibility  

---

## Not Included (By Design) / 故意不包含

❌ **Factor zoo**: Alpha101, Qlib158, GTJA191 implementations (bring your own factors)  
❌ **Backtest harness**: Walk-forward validation, performance metrics, benchmarking  
❌ **Data pipeline**: Bloomberg, TDX, China Index APIs (use data_gate or your own)  
❌ **Production deployment**: Real-time scoring, portfolio rebalancing, execution  
❌ **Black-box models**: Neural networks, reinforcement learning without interpretation  

This toolkit is for **research-level validation only**. It is intentionally narrow in scope.

---

## Recommended Next Steps / 推荐后续

1. **Use with your own factors**: Load real A-share data via `Panel.from_csv_dir()`,
   run the same pipeline on your candidate factors
2. **Extend the examples**: Add your own factor definitions, benchmarks, market bucketing logic
3. **Document your findings**: Use the same reporting structure (`run_factor_report.py` style)
   to make your research reproducible and auditable
4. **Push to GitHub**: Create a repo, add this toolkit + your own research notes,
   make it a portfolio piece showing research methodology
