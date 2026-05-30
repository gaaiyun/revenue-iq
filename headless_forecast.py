"""无 Streamlit / 无 plotly 的销量预测。

trunk 自带的 ``forecast.py`` 把预测逻辑和 plotly 画图、``st.warning`` 绑在一起，
没法在 CLI / cron 里跑。这个模块用纯 numpy + pandas 实现同口径的两种基线预测：

- 移动平均（用最近 window 期均值外推）
- 线性趋势外推（最小二乘拟合 t→value 再外推）

输入是中文列销售流水（日期 / 销售额），与 ``headless_analytics`` 一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from headless_analytics import DEFAULT_COLS, _resolve_cols


@dataclass
class ForecastPoint:
    period: str
    value: float


@dataclass
class ForecastResult:
    method: str
    freq: str
    history_periods: int
    forecast: List[ForecastPoint] = field(default_factory=list)
    slope: Optional[float] = None        # 仅线性趋势

    def to_dict(self) -> dict:
        out = {
            "method": self.method,
            "freq": self.freq,
            "history_periods": self.history_periods,
            "forecast": [{"period": p.period, "value": float(p.value)}
                         for p in self.forecast],
        }
        if self.slope is not None:
            out["slope"] = float(self.slope)
        return out


def _series(df: pd.DataFrame, freq: str,
            column_map: Optional[Dict] = None) -> pd.Series:
    cols = _resolve_cols(df, column_map)
    freq_norm = {"M": "ME", "Q": "QE", "Y": "YE"}.get(freq, freq)
    d = df.copy()
    d[cols["date"]] = pd.to_datetime(d[cols["date"]])
    ts = d.set_index(cols["date"])[cols["amount"]].resample(freq_norm).sum()
    return ts.fillna(0.0)


def _future_index(ts: pd.Series, periods: int, freq: str) -> pd.DatetimeIndex:
    freq_norm = {"M": "ME", "Q": "QE", "Y": "YE"}.get(freq, freq)
    step = pd.tseries.frequencies.to_offset(freq_norm)
    start = ts.index[-1] + step
    return pd.date_range(start=start, periods=periods, freq=freq_norm)


def moving_average_forecast(df: pd.DataFrame, periods: int = 7,
                            window: int = 7, freq: str = "D",
                            column_map: Optional[Dict] = None) -> ForecastResult:
    """移动平均预测：用最近 window 期均值，平推 periods 期。"""
    ts = _series(df, freq, column_map)
    if len(ts) == 0:
        return ForecastResult(method="moving_average", freq=freq, history_periods=0)
    last_ma = float(ts.tail(window).mean())
    idx = _future_index(ts, periods, freq)
    pts = [ForecastPoint(period=str(d.date()), value=last_ma) for d in idx]
    return ForecastResult(method="moving_average", freq=freq,
                          history_periods=int(len(ts)), forecast=pts)


def trend_forecast(df: pd.DataFrame, periods: int = 7, freq: str = "D",
                   column_map: Optional[Dict] = None) -> ForecastResult:
    """线性趋势外推：最小二乘拟合 t→value，外推 periods 期（非负截断）。"""
    ts = _series(df, freq, column_map)
    n = len(ts)
    if n < 2:
        return ForecastResult(method="linear_trend", freq=freq, history_periods=n)
    x = np.arange(n, dtype=float)
    y = ts.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    future_x = np.arange(n, n + periods, dtype=float)
    future_y = np.clip(slope * future_x + intercept, 0.0, None)
    idx = _future_index(ts, periods, freq)
    pts = [ForecastPoint(period=str(d.date()), value=float(v))
           for d, v in zip(idx, future_y)]
    return ForecastResult(method="linear_trend", freq=freq,
                          history_periods=n, forecast=pts, slope=float(slope))
