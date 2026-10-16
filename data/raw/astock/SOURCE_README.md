# A股日线数据集

**固定版本**：GitHub commit `0babe4cf6c1a175c6b84add10803b4b1f9c6000c`
**股票数**：5,608 个六位代码（六个行情分区合并去重）
**时间**：2020-06-01 ~ 2026-04-17
**总行数**：6,413,882
**检索时间**：2026-10-16（UTC manifest 见 `data/processed/data_manifest.json`）

## 文件清单

| 文件 | 内容 |
|---|---|
| `kline_000.parquet` | 647 个代码, 203,755 行 |
| `kline_002.parquet` | 964 个代码, 1,154,188 行 |
| `kline_300.parquet` | 1,399 个代码, 1,686,097 行 |
| `kline_600.parquet` | 1,705 个代码, 2,308,467 行 |
| `kline_688.parquet` | 607 个代码, 653,539 行 |
| `kline_other.parquet` | 286 个代码, 407,836 行 |
| `stock_list.parquet` | 股票基本信息（code, name, sector） |
| `index_sh.parquet` | 上证指数日线 |
| `load.py` | 一键载入函数 |

## 字段说明（kline_*.parquet）

| 列 | 类型 | 说明 |
|---|---|---|
| code | str | 股票代码（6位数字） |
| date | datetime | 交易日 |
| open / high / low / close | float32 | 开/高/低/收盘价 |
| volume | Int64 | 成交量（停牌日为 NA） |
| amount | float64 | 成交额（元） |
| turn | float32 | 换手率（%） |
| pctChg | float32 | 涨跌幅（%） |

## 快速开始

```python
from load import load_kline, load_meta, load_index

df     = load_kline()                            # 全量（~580 万行）
single = load_kline(codes=["000001","600000"])   # 几只股票
recent = load_kline(start="2025-01-01")          # 指定日期起
meta   = load_meta()
idx    = load_index()
```

## 备注

- 价格为 baostock 默认（前复权）
- 约 685 只老股票的早期历史缺失，仅含 2026 年 1 月以来
- 约 504 只股票末期停牌，结束日为 2026-03-27
- 上游说明文件的行数和分区统计与该固定提交中的文件不完全一致；以上覆盖数字由下载文件重新计算。
- 该数据集未提供历史成分变更、退市证券、ST 状态或点时分红公告版本，不能据此构建无幸存者偏差的股息策略。
