# Exploratory ML run on the pinned A-share price snapshot

This run checks that the research pipeline works end to end on public market data. It is an exploratory price/volume study, not a dividend-factor backtest or an investable performance claim. The source contains forward-adjusted prices and a current snapshot universe; it lacks reliable point-in-time membership, delisted names, historical ST/suspension status, and dividend announcement revisions.

本次运行用于验证研究流程能否在公开行情上端到端运行，是价量因子探索，不是股息因子回测或可投资业绩结论。来源使用前复权行情和当前快照股票池，缺少可靠的历史时点成分、退市股票、历史 ST/停牌状态及分红公告修订记录。

## Data and split

The source is the pinned public repository [`newbiestring-lang/astock`](https://github.com/newbiestring-lang/astock), commit `0babe4cf6c1a175c6b84add10803b4b1f9c6000c`. Local download time, hashes, and coverage are recorded in `data/processed/data_manifest.json`. The daily source spans 2020-06-01 to 2026-04-17 and 5,608 six-digit codes across its partitions. The monthly feature panel contains 272,937 stock-month rows, 4,902 codes, 60 signal months, and 272,926 next-month labels. The requested test period is 2024-01 to 2025-12; this final run evaluates all models on the 22 months where every model has a complete priced portfolio return.

数据来自固定版本的公开仓库 [`newbiestring-lang/astock`](https://github.com/newbiestring-lang/astock)，提交号为 `0babe4cf6c1a175c6b84add10803b4b1f9c6000c`。下载时间、哈希和覆盖范围记录在 `data/processed/data_manifest.json`。日线数据覆盖 2020-06-01 至 2026-04-17，六个分区合计 5,608 个六位代码。月度面板包含 272,937 条股票月记录、4,902 个代码、60 个信号月及 272,926 个次月收益标签。测试窗口固定为 2024-01 至 2025-12；为保证模型间可比，表中策略收益统一使用所有模型都有完整组合收益的 22 个月。

Signals are formed on the last common trading date of each month. Price/volume features use only observations through that date. The label is the next calendar month's first common-session open to last common-session open return from the source adjusted open series. The split is fixed at 2024-01-01. XGBoost and LightGBM hyperparameters are selected only from rolling validation folds ending before the split; the models are refit walk-forward using previously observed labels. The random forest uses 20 trees in this run to keep local computation bounded; its code default remains 300 trees. We did not choose settings by test-period results.

信号取每月最后一个共同交易日；价量特征只使用该日及此前数据。收益标签为下个自然月首个共同交易日开盘至月末共同交易日开盘的前复权价格收益。切分点固定为 2024-01-01。XGBoost 和 LightGBM 只用切分点前的滚动验证窗选择参数，之后按时间顺序使用已实现标签滚动重训。为控制本机计算时间，本次随机森林使用 20 棵树，代码默认仍为 300 棵；参数没有根据测试结果挑选。

## Results

The table below uses identical 22 test months and the runner's assumed commission, slippage, and sell stamp duty. It reports the model-generated top-quintile long-only portfolios and the fixed equal-weight rank baseline. These are snapshot diagnostics only: adjusted-price construction, survivorship bias, missing trading-status filters, and omitted benchmark-relative comparison prevent an investability claim.

下表采用完全相同的 22 个测试月及运行器设定的佣金、滑点和卖出印花税，比较模型生成的最高五分位多头组合与固定等权秩基准。它们只是快照诊断：前复权构造、幸存者偏差、缺少交易状态筛选以及未加入基准指数比较，都不支持可交易性结论。

| Model | Net annualized return | Net cumulative return | Annualized volatility | Max drawdown | Sharpe (0% RF) |
|---|---:|---:|---:|---:|---:|
| Random forest: low volatility | 9.66% | 18.43% | 16.45% | -14.58% | 0.64 |
| Random forest: low volatility + price/volume | 9.28% | 17.66% | 17.53% | -14.82% | 0.59 |
| XGBoost | 12.96% | 25.03% | 20.97% | -17.91% | 0.68 |
| LightGBM | 13.13% | 25.37% | 20.87% | -18.03% | 0.69 |
| Ridge | 10.21% | 19.50% | 21.46% | -19.33% | 0.55 |
| Equal-weight factor ranks | 6.89% | 13.00% | 17.02% | -14.44% | 0.47 |

In this run, adding price/volume features to the low-volatility random forest did not improve the classifier's average test AUC: it fell from 0.549 to 0.540; average accuracy was 53.4% versus 52.5%, and Brier score was 0.249 versus 0.250 (lower is better). XGBoost and LightGBM had mean monthly test Rank IC of 0.0556 and 0.0550 respectively, with positive IC in 17 of 24 months for each. The results are weak-to-moderate diagnostics, not evidence that the models add durable or tradable alpha. See `common_sample_comparison.csv`, `random_forest_feature_addition_metrics.csv`, and the monthly Rank IC files for the underlying summaries.

本次低波随机森林加入价量特征后，测试期平均 AUC 没有提高：从 0.549 降至 0.540；平均准确率从 53.4% 降至 52.5%，Brier 分数从 0.249 升至 0.250（越低越好）。XGBoost 与 LightGBM 的测试期月均 Rank IC 分别为 0.0556 和 0.0550，各有 17/24 个月 IC 为正。这些属于偏弱至中等的诊断信号，不能证明模型带来持久或可交易的超额收益。详情见 `common_sample_comparison.csv`、`random_forest_feature_addition_metrics.csv` 和月度 Rank IC 文件。

K-means is descriptive only; clusters are refit each month and their labels have no persistent ordering. The mean-variance portfolio uses XGBoost predictions and historical returns, with a 10% single-name cap; it is an exploratory optimizer, not an ML model. The optional GARCH feature was not requested in this run. The run did not include the Shanghai index in the ML comparison.

K-means 仅作描述性分析；聚类每月重估，编号没有持续顺序。均值—方差组合使用 XGBoost 预测及历史收益，单股上限 10%，属于探索性优化器，不是机器学习模型。本次未运行可选 GARCH 特征，也没有在 ML 比较中加入上证指数。

On the same 24 labelled signal months, the XGBoost-fed frontier diagnostic had 9.01% net annualized return, 34.31% annualized volatility, -19.82% maximum drawdown, 0.41 zero-rate Sharpe, and 86.5% mean one-way turnover. Its high estimated turnover and volatility are a reminder that optimizer weights can be unstable; this result is not part of the common 22-month model table.

同一组 24 个有标签的信号月中，使用 XGBoost 预测的有效前沿诊断净年化收益为 9.01%，年化波动率 34.31%，最大回撤 -19.82%，零利率夏普 0.41，平均单边换手 86.5%。较高换手和波动也说明优化权重容易不稳定；该结果不属于 22 个月共同样本的模型表。

## Reproduce

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

The raw Parquet files and generated per-stock CSV predictions stay local and are excluded from Git. Compact summaries, code, data manifests, and this report are versioned. Twelve input integrity checks are recorded in `data_quality_checks.csv`; all passed on this retrieved snapshot.

原始 Parquet 文件和逐股票预测 CSV 保留在本地并由 Git 忽略；代码、数据清单、精简汇总和本报告纳入版本控制。本次快照通过 12 项输入完整性检查，结果记录在 `data_quality_checks.csv`。

This does not fill the remaining dividend-study data gap. CNINFO's company snapshot reports realized dividends and their announcement/ex/record/payment dates, but its published page describes dividend coverage as the latest ten periods and does not provide version history for every historical proposal and amendment. A point-in-time dividend study still needs archived announcement versions and raw prices; do not backdate the current realized-dividend snapshot. See the [CNINFO company snapshot](https://www.cninfo.com.cn/new/snapshot/companyDetailEn?code=002407) and [`data/templates/README.md`](../../data/templates/README.md).

这仍未补齐股息研究所需的数据。巨潮公司概况页展示已实现分红及公告、除权、登记和派息日期，但页面说明分红数据覆盖最近十期，并未提供每项历史预案和修订的完整版本历史。因此正式点时股息研究仍需要历史公告版本与未复权价格，不能把当前实现分红快照回填到过去。参见[巨潮公司概况页](https://www.cninfo.com.cn/new/snapshot/companyDetailEn?code=002407)及[`data/templates/README.md`](../../data/templates/README.md)。
