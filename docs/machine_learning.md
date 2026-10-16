# 机器学习与组合实验

本项目提供一组可选的月度横截面机器学习实验：随机森林二分类、Ridge 回归、XGBoost 与 LightGBM 回归、K-means 截面聚类，以及一个带长仓约束的均值—方差组合实验。代码借鉴 AlphaLab CN 的模块化验证方式，但不依赖其结果。所有模块都读取点时因子面板，不负责从行情原始字段自动推导股息公告、质量或估值因子。

## 已实现的步骤

1. XGBoost / LightGBM 在留出期之前用滚动 24 个月训练窗和 6 个月验证窗选参数；留出期内参数固定，每月使用当时已实现的最近 36 个月标签更新模型。
2. Ridge 使用固定 alpha=10，作为线性基准；随机森林的固定设置为 300 棵树、最大深度 5、最小叶节点 15，不在测试期调参。
3. 随机森林标签为“该股下一期收益是否高于该信号月横截面中位数”。它输出超越中位数的概率，不代表预测收益幅度。
4. 随机森林按预设顺序逐步加入特征组：股息、低波、基本面、价量、其他。特征名称按关键词归类；运行前检查 run_metadata.json 和输出中的实际分组，并在测试前固定新增顺序。
5. 固定等权秩基准是各输入因子当月百分位秩的等权平均。输入因子的方向必须事先统一为“越高越看好”，不能根据测试期调整方向。
6. K-means 每月仅按当月特征截面聚类，之后才汇总各簇收益。簇编号每月重估、没有自然顺序，不能当作持久类别或预测信号。
7. 有效前沿例程使用预测收益和截至调仓时已实现的历史收益估计协方差，采用 Ledoit-Wolf 收缩、长仓、单股权重上限及换手惩罚。它属于组合优化，不属于机器学习；短样本下只作为探索实验。
8. IC、Rank IC、分组收益、分类 AUC/Brier、回归检验和成本扣除属于统计评价，不是机器学习算法本身。
9. 可选 GARCH(1,1) 根据 date,market_return 日收益估计未来 21 个交易日市场波动率，可作为市场状态特征；它是传统时间序列模型，不是机器学习，也不替代横截面因子。需要额外安装 requirements-garch.txt，并通过 --market-returns 指定输入。
10. 缩尾边界和缺失值填补值只从每个模型当期训练样本计算。分类/回归模型、固定秩组合和优化组合在同一测试月份比较，并按现有回测函数扣除手续费、滑点和卖出印花税。

测试期内已实现的收益标签可以在之后的月份进入训练样本；它们不用于重新选择参数。这是按时间推进的 walk-forward 评估。`feature_importance` 只表示模型如何使用输入列，不代表因果贡献。

## 输入文件

将文件放在 `data/manual_ml/`，该目录里的 CSV 不会提交到 Git。

- `factors.csv`：每行对应一个 `date,code`，其他列是当日可获得的因子值，如 `dividend_yield,low_volatility,quality,momentum`。
- `forward_returns.csv`：每行对应一个 `date,code`，包含 `forward_return`。标签从信号之后可成交的价格开始计算。

两份文件的 `date,code` 不能重复。`forward_return` 只放在收益文件里，不得作为因子特征。财务质量或估值因子也必须使用当时已公告的数据。尽量保留历史退市股票和当时的股票池；当前成分股回填历史会带来幸存者偏差。

## 运行

ML 包是可选依赖，不需要运行机器学习时不用安装：

```bash
pip install -r requirements-ml.txt
python scripts/run_ml_experiments.py --regression-model both --split-date 2024-01-01
# Run only one boosting model if desired: --regression-model xgboost or lightgbm
# Optional GARCH market-volatility feature:
python scripts/run_ml_experiments.py --regression-model xgboost --split-date 2024-01-01 --market-returns data/manual_ml/index_returns.csv
```

The optional file must contain one row per date with columns `date,market_return`, where returns are decimals. The GARCH feature uses observations through each signal date and forecasts volatility over the following 21 sessions. Install its separate dependency with `pip install -r requirements-garch.txt`.
可选文件每个日期一行，包含 `date,market_return`，收益率使用小数格式。GARCH 特征使用截至各信号日的数据，预测之后 21 个交易日的波动率。使用前另行安装 `pip install -r requirements-garch.txt`。

脚本默认从 `data/manual_ml/` 读取输入，将文件写到 `reports/ml_experiments/`。默认同时运行 XGBoost 和 LightGBM；也可以只选其中一个。输出包括模型预测、逐月 Rank IC、随机森林分类指标、K-means 簇分布、等权秩/Ridge/随机森林/两种 boosting 模型的同月组合比较，以及基于 XGBoost 预测的有效前沿组合结果。组合使用佣金单边 3bp、滑点单边 10bp、卖出印花税 2023-08-28 前 10bp、此后 5bp 的假设。当前输入模板没有基准指数收益，策略比较是相对因子/模型基准的绝对收益比较，不是相对市场指数的超额收益。

## 当前状态与限制

运行器不会自动下载行情。仓库中的公开行情快照只包含价格和成交字段，且使用当前股票列表，不能单独满足点时分红收益率、历史股票池及退市处理要求。因此目前没有真实股息因子机器学习结果；运行缺少输入时会停止并提示，不会生成虚构表现。请先按下列格式准备真实面板，再启动研究。2021–2023 的开发期较短，参数网格已保持精简；测试期只用于一次最终比较。

这个流程不处理停牌、涨跌停成交、退市收益、历史 ST 状态或历史行业/市值中性。K-means 是描述性分析，Frontier 使用短历史估计，RF 的分类准确率也不是交易收益，GARCH 预测波动率也不代表收益预测。它是研究脚本，不是实盘交易系统。
