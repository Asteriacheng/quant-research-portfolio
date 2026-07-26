"""Panel data container and synthetic market generator.

面板数据容器与合成行情生成器。

设计取舍
--------
1. 所有量都用「宽表」表示:index = 交易日, columns = 股票代码, 值 = 该日该股的量。
   截面研究天然是矩阵运算,宽表比长表快一个数量级,也更难写出跨股票串行的 bug。
2. 量纲在装载时归一(vol 一律为股, amount 一律为元),避免「手 vs 股」这类
   下游算出来才发现的错误。
3. 合成数据里**故意埋一个已知方向的反转 alpha**——工具包的自测必须能把它测出来,
   测不出来说明工具本身坏了。这是「先证明尺子是准的,再去量东西」。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["Panel", "make_synthetic_panel"]

_FIELDS = ("open", "high", "low", "close", "volume", "amount")


@dataclass(frozen=True)
class Panel:
    """A set of aligned T x N wide frames. / 一组对齐的 T×N 宽表。"""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    amount: pd.DataFrame

    def __post_init__(self) -> None:
        ref = self.close
        for name in _FIELDS:
            frame = getattr(self, name)
            if not frame.index.equals(ref.index) or not frame.columns.equals(ref.columns):
                raise ValueError(f"字段 {name} 与 close 未对齐 / field {name} is not aligned with close")

    # ---- 基本属性 ----------------------------------------------------------
    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def codes(self) -> pd.Index:
        return self.close.columns

    @property
    def shape(self) -> tuple[int, int]:
        return self.close.shape

    def slice_to(self, end) -> "Panel":
        """Truncate every field at `end` (inclusive). / 把所有字段截断到 end(含)。

        截断不变性探针靠这个方法制造「当时只看得到这么多数据」的世界。
        """
        end = pd.Timestamp(end)
        return Panel(**{f: getattr(self, f).loc[:end] for f in _FIELDS})

    # ---- 常用派生量 --------------------------------------------------------
    def returns(self) -> pd.DataFrame:
        """Close-to-close simple returns. / 收盘价日收益率。"""
        return self.close.pct_change()

    def prev_close(self) -> pd.DataFrame:
        return self.close.shift(1)

    def vwap(self) -> pd.DataFrame:
        """amount / volume, 缺量时退回收盘价。"""
        vwap = self.amount / self.volume.replace(0, np.nan)
        return vwap.where(vwap.notna(), self.close)

    # ---- 装载 --------------------------------------------------------------
    @classmethod
    def from_csv_dir(cls, directory: str | Path, date_col: str = "date") -> "Panel":
        """Load one CSV per stock (file stem = code). / 每只股票一个 CSV,文件名即代码。

        CSV 需含列: date, open, high, low, close, volume, amount。
        volume 单位为股;若你的数据源是「手」,请在这里乘 100,不要留到下游。
        """
        directory = Path(directory)
        files = sorted(directory.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"{directory} 下没有 CSV / no CSV found")
        frames: dict[str, dict[str, pd.Series]] = {f: {} for f in _FIELDS}
        for path in files:
            code = path.stem
            df = pd.read_csv(path, parse_dates=[date_col]).set_index(date_col).sort_index()
            missing = [c for c in _FIELDS if c not in df.columns]
            if missing:
                raise ValueError(f"{path.name} 缺列 {missing} / missing columns {missing}")
            for field in _FIELDS:
                frames[field][code] = df[field].astype("float64")
        built = {f: pd.DataFrame(frames[f]).sort_index() for f in _FIELDS}
        index = built["close"].index
        columns = built["close"].columns
        built = {f: v.reindex(index=index, columns=columns) for f, v in built.items()}
        return cls(**built)


def make_synthetic_panel(
    n_days: int = 600,
    n_stocks: int = 300,
    seed: int = 20260726,
    reversal_strength: float = 0.0004,
    limit_up_rate: float = 0.004,
) -> Panel:
    """Generate a synthetic A-share-like panel with a known reversal alpha.

    生成一段合成行情,内含一个**已知方向的 5 日反转 alpha**。

    Parameters
    ----------
    reversal_strength:
        每 1 个标准差的过去 5 日涨幅,对次日收益的负向贡献(单位:小数)。
        默认 0.0004 ≈ 4bp/sd,标定到 5 日反转因子 RankIC ≈ 0.05 —— 对应
        「很强但不离谱」的真实量级;调大它工具会更容易通过自测,那是自欺。
    limit_up_rate:
        随机制造「开盘一字涨停」的比例,用来演示 T+1 可执行性过滤。

    Notes
    -----
    这是**合成数据,不是真实行情**。它的用途只有一个:证明本工具包的指标算得对。
    任何策略结论都不能从这里得出。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-04", periods=n_days)
    codes = [f"S{i:04d}" for i in range(n_stocks)]

    beta = rng.uniform(0.6, 1.4, size=n_stocks)
    market = rng.normal(0.0002, 0.011, size=n_days)
    idio = rng.normal(0.0, 0.019, size=(n_days, n_stocks))

    ret = np.zeros((n_days, n_stocks))
    logp = np.zeros((n_days, n_stocks))
    logp[0] = np.log(rng.uniform(5, 60, size=n_stocks))

    for t in range(1, n_days):
        base = beta * market[t] + idio[t]
        if t >= 6:
            past5 = logp[t - 1] - logp[t - 6]          # t-1 收盘可观测的 5 日涨幅
            z = (past5 - past5.mean()) / (past5.std() + 1e-12)
            base = base - reversal_strength * z         # 涨多了的次日偏弱 —— 埋入的 alpha
        ret[t] = base
        logp[t] = logp[t - 1] + base

    close = pd.DataFrame(np.exp(logp), index=dates, columns=codes).round(2)
    prev_close = close.shift(1)

    gap = pd.DataFrame(rng.normal(0.0, 0.006, size=(n_days, n_stocks)), index=dates, columns=codes)
    open_ = (prev_close * (1 + gap)).round(2)
    open_.iloc[0] = close.iloc[0]

    span = pd.DataFrame(np.abs(rng.normal(0.012, 0.006, size=(n_days, n_stocks))), index=dates, columns=codes)
    high = (pd.concat([open_, close]).groupby(level=0).max() * (1 + span)).round(2)
    low = (pd.concat([open_, close]).groupby(level=0).min() * (1 - span)).round(2)

    turn = np.exp(rng.normal(0.0, 0.5, size=(n_days, n_stocks))) * (1 + 6 * np.abs(ret))
    volume = pd.DataFrame(turn * 1e6, index=dates, columns=codes).round(0)
    amount = (volume * close).round(0)

    # 随机制造开盘一字涨停:open = high = low = 昨收 x 1.1
    if limit_up_rate > 0:
        hit = rng.random((n_days, n_stocks)) < limit_up_rate
        hit[0] = False
        limit_price = (prev_close * 1.1).round(2)
        mask = pd.DataFrame(hit, index=dates, columns=codes) & limit_price.notna()
        open_ = open_.mask(mask, limit_price)
        high = high.mask(mask, limit_price)
        low = low.mask(mask, limit_price)
        close = close.mask(mask, limit_price)

    return Panel(open=open_, high=high, low=low, close=close, volume=volume, amount=amount)
