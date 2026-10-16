# A-Share Factor Research

This repository contains Python tools for factor diagnostics, A-share price/volume research, an offline dividend-yield study framework, and optional walk-forward machine-learning code.
本仓库包含因子检验工具、A 股价量研究、离线股息率研究框架，以及可选的机器学习滚动预测代码。

## Project contents / 项目内容

- `factorkit/`: IC, turnover, decay, event-study, and data-integrity utilities.  
  `factorkit/`：IC、换手、衰减、事件研究和数据完整性工具。
- `factors/`: price/volume, dividend-yield, and low-volatility factor calculations.  
  `factors/`：价量、股息率和低波因子计算。
- `analysis/`: portfolio sorts, Fama–MacBeth regressions, cost scenarios, and multifactor diagnostics.  
  `analysis/`：分组收益、Fama–MacBeth 回归、成本情景和多因子诊断。
- `ml/`: optional walk-forward regression/classification, clustering diagnostics, and portfolio optimization.
  `ml/`：可选的滚动回归/分类、聚类诊断和组合优化。
- `scripts/`: runnable research and chart-generation scripts.  
  `scripts/`：研究运行和图表生成脚本。
- `data/templates/`: offline dividend-study input schemas.  
  `data/templates/`：离线股息研究的数据字段说明和 CSV 表头模板。
- `docs/`: methods, data requirements, and run instructions.  
  `docs/`：研究方法、数据要求和运行说明。
- `reports/real_data_v1/` and `reports/real_data_v2/`: outputs from the price/volume studies.  
  `reports/real_data_v1/` 和 `reports/real_data_v2/`：价量因子研究输出。

## Current status / 当前状态

**Price/volume factors.** The repository includes single-factor and portfolio analyses using historical A-share market data. The available data lacks historical delisted names, ST and suspension status, limit-price execution flags, and point-in-time industry and market-cap classifications. The results should not be read as a fully executable market backtest.  
**价量因子：** 仓库包含基于 A 股历史行情的单因子和组合分析。现有数据缺少历史退市股票、ST 和停牌状态、涨跌停成交标记，以及点时行业和市值分类，因此结果不等同于完整可成交回测。

**Dividend yield and low volatility.** The code calculates a point-in-time trailing cash yield and low-volatility signal, then provides rank combination, IC, quintile, Fama–MacBeth, cost, and rebalance-frequency analyses. The repository does not contain versioned dividend-event history or unadjusted prices, so this study has no empirical results yet.  
**股息率与低波：** 代码实现了点时现金股息率和低波信号，并提供秩组合、IC、五分组、Fama–MacBeth、成本及调仓频率分析。仓库尚无分红公告版本历史和未复权价格，因此这部分还没有实证结果。

**Machine learning and statistical analysis.** The experiment runner compares a fixed random-forest classifier with progressively added feature groups, Ridge, XGBoost, LightGBM, and an equal-weight rank baseline on the same holdout months. K-means is an unsupervised diagnostic. IC, AUC/Brier, portfolio sorts, and GARCH volatility forecasts are reported as statistical/time-series analyses; mean-variance frontier construction is portfolio optimization, not machine learning. No point-in-time dividend panel has been supplied, so these additions do not yet produce real dividend-strategy results. See [the ML instructions](docs/machine_learning.md).
**机器学习与统计分析：** 实验脚本在相同留出月份比较逐步增加特征组的随机森林分类、Ridge、XGBoost、LightGBM，以及等权秩基准。K-means 是无监督诊断。IC、AUC/Brier、分组收益和 GARCH 波动率预测分别作为统计或时间序列分析报告；均值—方差有效前沿属于组合优化，不是机器学习。目前没有点时股息因子面板，因此这些代码尚未产生真实股息策略结果。详见[机器学习文档](docs/machine_learning.md)。

## Setup and tests / 环境与测试

Install the base dependencies and run the unit tests:
安装基础依赖并运行单元测试：

```bash
pip install pandas numpy scipy
python -m unittest discover -s tests -t .
```

The current suite has 42 tests covering calculations and time-split behavior. Passing unit tests checks code behavior; it does not establish that a factor works in the market.
当前共有 42 项单元测试，覆盖计算逻辑和时间切分。测试通过只说明代码行为符合预期，不代表因子在真实市场有效。

Machine-learning libraries are optional. Install them only if you plan to run that pipeline:
机器学习库为可选依赖；需要运行 ML 流程时再安装：

```bash
pip install -r requirements-ml.txt
```

## Run the ML pipeline / 运行机器学习流程

Prepare local CSV files; files under `data/manual_ml/` are ignored by Git:
准备本地 CSV 文件；`data/manual_ml/` 下的文件会被 Git 忽略：

- `factors.csv`: one row per `date,code`, with one column per point-in-time factor, such as dividend yield, low volatility, quality, or momentum.  
  `factors.csv`：每个 `date,code` 一行，每个点时因子一列，例如股息率、低波、质量或动量。
- `forward_returns.csv`: one row per `date,code` with a `forward_return` label that starts after the signal can be traded.  
  `forward_returns.csv`：每个 `date,code` 一行，包含 `forward_return`；收益区间必须从信号之后可成交的价格开始。

Run the comparison suite:
运行模型比较：

```bash
python scripts/run_ml_experiments.py --regression-model both --split-date 2024-01-01
# Or run one tree model with --regression-model xgboost or lightgbm
# 也可将参数改为 xgboost 或 lightgbm 只运行一种树模型
```

The script reads from `data/manual_ml/` and writes model predictions and comparisons, classification metrics, cluster assignments, portfolio returns, frontier diagnostics, and run metadata to `reports/ml_experiments/`.
脚本从 `data/manual_ml/` 读取数据，并将模型预测与比较、分类指标、聚类结果、组合收益、有效前沿诊断和运行信息写入 `reports/ml_experiments/`。

**Public-snapshot example.** To run the exploratory ML case on the repository's pinned public price/volume snapshot, install both dependency sets, fetch the files, build the monthly panel, validate it, then run the pre-fixed model suite:
**公开快照示例。** 如需在仓库固定版本的公开价量快照上运行探索性 ML 案例，安装两组依赖、下载行情、构建月度面板并完成质量检查，再运行预设模型：

```bash
pip install -r requirements-real.txt
pip install -r requirements-ml.txt
python data/fetch_real_data.py
python scripts/build_ml_inputs_from_snapshot.py
python scripts/check_ml_snapshot_data.py
python scripts/run_ml_experiments.py --input-dir data/manual_ml_snapshot \
  --output-dir reports/ml_snapshot_exploratory --regression-model both \
  --rf-estimators 20 --rf-jobs -1 --split-date 2024-01-01
```

The run summary is in [`reports/ml_snapshot_exploratory/README.md`](reports/ml_snapshot_exploratory/README.md). It covers 2021–2025 monthly signals with 2024–2025 reserved, uses the same 22 complete test months for the model comparison, and reports the source hashes and 12 integrity checks. Because the snapshot has forward-adjusted prices and no point-in-time universe or historical dividend revisions, these are exploratory price/volume diagnostics, not dividend-strategy or executable backtest results.
运行结果见[`reports/ml_snapshot_exploratory/README.md`](reports/ml_snapshot_exploratory/README.md)。信号覆盖 2021–2025，2024–2025 预留为测试期；模型比较使用相同的 22 个完整测试月，并记录源数据哈希和 12 项质量检查。由于快照使用前复权行情，且缺少点时股票池和历史分红修订，这些结果只是价量探索诊断，不是股息策略或可执行回测结果。

An optional `sentiment_score` may be supplied as another factor only when it has a documented historical source, timestamp, and release-time rule. This repository does not download archived social-media data or synthesize sentiment observations.
只有在历史来源、时间戳和可用时点规则都有记录时，才可将 `sentiment_score` 作为额外因子输入。本仓库不会下载社交媒体历史数据，也不会合成情绪观测值。

## Run the dividend study / 运行股息研究

The dividend workflow reads local CSV files and does not call a market-data API. See [the input specification](data/templates/README.md), then run:
股息研究读取本地 CSV，不调用行情数据 API。先查看[输入字段说明](data/templates/README.md)，然后运行：

```bash
python scripts/run_dividend_multifactor.py
python scripts/make_dividend_figures.py
```

**Date design (as of 16 October 2026).** The study uses the latest five complete calendar years, 2021–2025: 36 development signal months (January 2021–December 2023) and a pre-reserved 24-month test (January 2024–December 2025). We exclude 2026 year-to-date because it is not a complete calendar year. Monthly labels require prices through January 2026; the full quarterly-frequency sensitivity requires prices through March 2026 to complete the December 2025 signal's three-month holding period. Signal dates and the split are fixed in the runner for reproducibility. If versioned dividend history or raw prices are missing, it stops without generating market results.  
**日期設計（截至 2026 年 10 月 16 日）：** 研究採用最近五個完整年度，即 2021–2025 年：開發期 36 個信號月（2021 年 1 月至 2023 年 12 月），以及預先保留的 24 個月測試期（2024 年 1 月至 2025 年 12 月）。2026 年尚未結束，因此不納入本輪完整年度樣本。月度收益標籤需要行情截至 2026 年 1 月；完整季度調倉敏感性則需要行情截至 2026 年 3 月，以完成 2025 年 12 月信號後的三個月持有期。腳本固定信號區間與切分點以便復現。缺少分紅公告版本歷史或未復權價格時，腳本會停止，不會生成市場結果。

## Limitations / 研究限制

The exploratory ML suite has been run on real price/volume data, but the dataset cannot establish a point-in-time dividend strategy. Historical membership and delistings, ST/suspension/limit-price execution, point-in-time industry and size controls, and versioned dividend announcements remain unavailable. The 22-month common test result is not a trading claim.
探索性 ML 流程已在真实价量数据上运行，但该数据不能验证点时股息策略。历史股票池及退市处理、ST/停牌/涨跌停成交、点时行业和市值控制、分红公告版本仍未补齐。共同样本的 22 个月结果不代表可交易业绩。
