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
- `ml/`: optional XGBoost / LightGBM walk-forward model.  
  `ml/`：可选的 XGBoost / LightGBM 滚动训练模型。
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

**Machine learning.** The optional XGBoost / LightGBM pipeline selects settings using rolling validation before the holdout, freezes those settings, and generates monthly walk-forward predictions with Rank IC and feature-importance outputs. No factor panel has been supplied and no model-performance result has been generated. See [the ML instructions](docs/machine_learning.md).  
**机器学习：** 可选的 XGBoost / LightGBM 流程使用留出期之前的滚动验证选择参数，在留出期固定参数并逐月预测，输出 Rank IC 和特征重要性。目前尚无输入因子面板，也没有模型表现结果。运行说明见[机器学习文档](docs/machine_learning.md)。

## Setup and tests / 环境与测试

Install the base dependencies and run the unit tests:
安装基础依赖并运行单元测试：

```bash
pip install pandas numpy scipy
python -m unittest discover -s tests -t .
```

The current suite has 36 tests covering calculations and time-split behavior. Passing unit tests checks code behavior; it does not establish that a factor works in the market.  
当前共有 36 项单元测试，覆盖计算逻辑和时间切分。测试通过只说明代码行为符合预期，不代表因子在真实市场有效。

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

Run the model:
运行模型：

```bash
python scripts/run_ml_factor.py --model xgboost --split-date 2024-01-01
# Or use --model lightgbm
# 也可以将模型设为 lightgbm
```

The script reads from `data/manual_ml/` and writes predictions, monthly Rank IC, quintile returns, a costed portfolio, feature importance, parameter-selection scores, and run metadata to `reports/ml_factor/`.  
脚本从 `data/manual_ml/` 读取数据，并将预测值、月度 Rank IC、五分组收益、含成本组合、特征重要性、参数选择结果和运行信息写入 `reports/ml_factor/`。

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

The available data and scripts do not yet cover historical universe membership and delistings, ST/suspension/limit-price execution, point-in-time industry and size controls, or publication-date-safe accounting factors. The ML module is implemented but has not been run on real factor inputs. No unobserved strategy return or performance claim is made.  
现有数据和代码尚未覆盖历史股票池及退市处理、ST/停牌/涨跌停成交、点时行业和市值控制，以及按公告日期构建的财务因子。机器学习模块已实现，但还没有用真实因子数据运行。本文档不声称存在尚未验证的策略收益或业绩。
