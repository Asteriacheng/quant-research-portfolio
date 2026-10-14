# A股因子研究

这个仓库包含因子检验代码、A股行情因子案例，以及一个尚待真实分红数据输入的股息率研究框架。所有研究结果都应和对应的数据口径一起阅读。

## 仓库里有什么

- `factorkit/`：IC、换手、衰减、事件研究和数据完整性检查。
- `factors/`：价量因子、股息率因子和低波因子。
- `analysis/`：分组收益、Fama–MacBeth、成本回测及多因子诊断。
- `ml/`：XGBoost / LightGBM walk-forward 复合因子代码。需要单独安装 ML 依赖。
- `scripts/`：研究运行和图表导出脚本。
- `data/templates/`：离线股息研究输入格式和空白 CSV 表头。
- `docs/`：研究方法、数据口径和运行说明。
- `reports/real_data_v1/`、`reports/real_data_v2/`：价量因子研究的输出文件。

## 当前研究状态

**价量因子：** 仓库包含基于 A 股历史行情的单因子和组合分析输出。行情案例缺少历史退市、ST、停牌、涨跌停及点时行业/市值信息，因此结果不能直接理解成可成交的完整市场回测。

**股息率与低波：** 已实现公告版本时点约束、股息率计算、低波因子、等权秩组合、IC、五分组、Fama–MacBeth、成本情景和调仓频率敏感性。当前仓库没有点时分红版本数据和未复权价格，尚未生成这部分的实证结果。

**机器学习：** 已实现可选的 XGBoost / LightGBM walk-forward 流程，包括开发期滚动验证、留出期参数冻结、逐月训练预测、Rank IC 和特征重要性输出。目前没有训练输入数据，也没有生成模型表现结果。具体输入、时间切分和限制见 [`docs/machine_learning.md`](docs/machine_learning.md)。

## 环境与测试

基础因子检验使用 Python、pandas、numpy 和 scipy。运行单元测试：

```bash
pip install pandas numpy scipy
python -m unittest discover -s tests -t .
```

当前单元测试为 36 项，覆盖计算逻辑和时间切分；通过测试不代表因子在真实市场有效。

机器学习依赖单独安装，避免不用 ML 时安装额外包：

```bash
pip install -r requirements-ml.txt
```

## 运行机器学习因子

准备本地输入文件（不会提交到 Git）：

- `data/manual_ml/factors.csv`：`date,code` 加一列或多列因子值，例如 `dividend_yield,low_volatility,quality,momentum`。
- `data/manual_ml/forward_returns.csv`：`date,code,forward_return`。标签日期必须对应因子信号日期，收益从信号之后可成交的价格开始计算。

运行：

```bash
python scripts/run_ml_factor.py --model xgboost --split-date 2024-01-01
# 也可以使用 --model lightgbm
```

脚本默认使用 `data/manual_ml/`，输出到 `reports/ml_factor/`。它先在留出期之前的滚动验证窗口选择参数；进入留出期后固定参数，每个信号月只用当时已经有完整收益标签的历史样本重新训练。训练期的缩尾边界和缺失值填补值不会从验证或预测样本估计。

## 股息研究输入

股息研究不连接行情数据 API，输入为本地 CSV。字段、公告版本和价格要求见 [`data/templates/README.md`](data/templates/README.md)。输入准备好后运行：

```bash
python scripts/run_dividend_multifactor.py
python scripts/make_dividend_figures.py
```

月度信号区间为 2021–2025，最后 24 个信号月留作测试。为生成完整的季度调仓敏感性，价格数据需延伸到 2026 年 3 月。没有符合条件的分红版本和未复权价格时，脚本会停止并提示缺失文件，不会生成模拟市场结果。

## 研究边界

当前数据和程序尚未构成完整实盘回测：历史股票池和退市处理、ST/停牌/涨跌停成交限制、历史行业与市值中性，以及点时财务质量因子都需要对应的历史数据。机器学习模块也只提供模型研究流程；在真实输入通过质量检查并运行之前，不报告预测收益或策略业绩。

## English

Python research code for A-share factor diagnostics. The repository includes price/volume research outputs, an offline dividend-yield and low-volatility framework, and optional walk-forward XGBoost/LightGBM code. Dividend and ML empirical results are not available until point-in-time inputs are supplied and the pipelines are run. See the Chinese sections above for setup and data requirements.
