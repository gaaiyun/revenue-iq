"""无 Streamlit / 无 plotly 的销售分析函数集合。

v1 的 ``SalesAnalyzer`` 和 ``SalesForecaster`` 在业务逻辑里调 ``st.error()`` /
``st.warning()``，把 Streamlit 当成必需依赖。这让 v1 没法在脚本 / CI / 报表 cron
任务里跑（streamlit 在没有 streamlit run 环境时调 ``st.error`` 是空操作但消息
丢了）。

v2 这个模块**纯 numpy + pandas**，不依赖 Streamlit / plotly / sklearn，给 CLI
和报表用。同样的指标公式，不靠交互渲染。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# 标准列名（v1 sample_data 用的中文）。允许用户传 ``column_map`` 重命名。
DEFAULT_COLS = {
    "date": "日期",
    "amount": "销售额",
    "qty": "销售数量",
    "category": "产品类别",
    "region": "地区",
    "customer": "客户 ID",
    "customer_tier": "客户等级",
}


@dataclass
class SalesSummary:
    n_records: int
    date_start: str
    date_end: str
    total_revenue: float
    total_orders: int
    avg_order_value: float
    top_category: Optional[str] = None
    top_category_revenue: float = 0.0
    top_region: Optional[str] = None
    top_region_revenue: float = 0.0
    n_customers: int = 0
    growth_first_to_last_month: Optional[float] = None     # %

    def to_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in self.__dict__.items()}


@dataclass
class TrendPoint:
    period: str
    revenue: float
    n_orders: int
    growth_pct: Optional[float] = None


@dataclass
class AnomalyAlert:
    period: str
    actual: float
    expected: float
    z_score: float
    direction: str    # "spike" / "drop"

    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "actual": float(self.actual),
            "expected": float(self.expected),
            "z_score": float(self.z_score),
            "direction": self.direction,
        }


def _resolve_cols(df: pd.DataFrame, column_map: Optional[Dict] = None) -> Dict[str, str]:
    """根据 column_map 或默认值，把内部 key 映射到实际 DataFrame 列名。"""
    cols = dict(DEFAULT_COLS)
    if column_map:
        cols.update(column_map)
    return cols


def compute_summary(df: pd.DataFrame, column_map: Optional[Dict] = None
                    ) -> SalesSummary:
    """整体统计。"""
    cols = _resolve_cols(df, column_map)
    if df is None or len(df) == 0:
        raise ValueError("DataFrame 为空")

    date_col = cols["date"]
    amount_col = cols["amount"]
    if date_col not in df.columns or amount_col not in df.columns:
        raise ValueError(f"缺必要列：{date_col} / {amount_col}")

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    revenue = float(df[amount_col].sum())
    n_orders = int(len(df))

    top_cat = None
    top_cat_rev = 0.0
    if cols["category"] in df.columns:
        cat_rev = df.groupby(cols["category"])[amount_col].sum()
        if len(cat_rev) > 0:
            top_cat = str(cat_rev.idxmax())
            top_cat_rev = float(cat_rev.max())

    top_region = None
    top_region_rev = 0.0
    if cols["region"] in df.columns:
        reg_rev = df.groupby(cols["region"])[amount_col].sum()
        if len(reg_rev) > 0:
            top_region = str(reg_rev.idxmax())
            top_region_rev = float(reg_rev.max())

    n_customers = 0
    if cols["customer"] in df.columns:
        n_customers = int(df[cols["customer"]].nunique())

    # 首月 vs 末月增长率
    growth_pct = None
    monthly = df.set_index(date_col)[amount_col].resample("ME").sum()
    if len(monthly) >= 2 and monthly.iloc[0] > 0:
        growth_pct = float((monthly.iloc[-1] / monthly.iloc[0] - 1) * 100)

    return SalesSummary(
        n_records=len(df),
        date_start=str(df[date_col].min().date()),
        date_end=str(df[date_col].max().date()),
        total_revenue=revenue,
        total_orders=n_orders,
        avg_order_value=(revenue / n_orders) if n_orders else 0.0,
        top_category=top_cat, top_category_revenue=top_cat_rev,
        top_region=top_region, top_region_revenue=top_region_rev,
        n_customers=n_customers,
        growth_first_to_last_month=growth_pct,
    )


def compute_trend(df: pd.DataFrame, period: str = "M",
                   column_map: Optional[Dict] = None) -> List[TrendPoint]:
    """按时间周期聚合收入 + 订单数 + 环比增长。

    period: "D" / "W" / "M" / "Q" / "Y"
    """
    cols = _resolve_cols(df, column_map)
    # pandas 2.x：M 已废弃，用 ME。给用户传 M 时自动转
    period_norm = {"M": "ME", "Q": "QE", "Y": "YE"}.get(period, period)
    df = df.copy()
    df[cols["date"]] = pd.to_datetime(df[cols["date"]])
    df = df.set_index(cols["date"]).sort_index()

    revenue = df[cols["amount"]].resample(period_norm).sum()
    orders = df[cols["amount"]].resample(period_norm).count()

    points = []
    prev_rev = None
    for ts, rev in revenue.items():
        growth = None
        if prev_rev is not None and prev_rev > 0:
            growth = float((rev / prev_rev - 1) * 100)
        points.append(TrendPoint(
            period=str(ts.date()),
            revenue=float(rev),
            n_orders=int(orders.loc[ts]),
            growth_pct=growth,
        ))
        prev_rev = rev
    return points


def detect_anomalies(df: pd.DataFrame, period: str = "D",
                     z_threshold: float = 2.0,
                     column_map: Optional[Dict] = None) -> List[AnomalyAlert]:
    """基于 z-score 检测异常销售（大涨 / 大跌）。

    用滚动 14 期窗口算 z-score，超过阈值标记。
    """
    cols = _resolve_cols(df, column_map)
    period_norm = {"M": "ME", "Q": "QE", "Y": "YE"}.get(period, period)
    df = df.copy()
    df[cols["date"]] = pd.to_datetime(df[cols["date"]])
    revenue = df.set_index(cols["date"])[cols["amount"]].resample(period_norm).sum()

    if len(revenue) < 5:
        return []

    rolling = revenue.rolling(window=14, min_periods=3)
    mean = rolling.mean()
    std = rolling.std()

    alerts = []
    for ts, val in revenue.items():
        mu = mean.loc[ts]
        sigma = std.loc[ts]
        if pd.isna(mu) or pd.isna(sigma) or sigma < 1e-9:
            continue
        z = (val - mu) / sigma
        if abs(z) >= z_threshold:
            alerts.append(AnomalyAlert(
                period=str(ts.date()),
                actual=float(val), expected=float(mu), z_score=float(z),
                direction="spike" if z > 0 else "drop",
            ))
    return alerts


def category_breakdown(df: pd.DataFrame,
                       column_map: Optional[Dict] = None) -> Dict[str, float]:
    """按品类的收入。"""
    cols = _resolve_cols(df, column_map)
    if cols["category"] not in df.columns:
        return {}
    grouped = df.groupby(cols["category"])[cols["amount"]].sum().sort_values(ascending=False)
    return {str(k): float(v) for k, v in grouped.items()}


def region_breakdown(df: pd.DataFrame,
                     column_map: Optional[Dict] = None) -> Dict[str, float]:
    cols = _resolve_cols(df, column_map)
    if cols["region"] not in df.columns:
        return {}
    grouped = df.groupby(cols["region"])[cols["amount"]].sum().sort_values(ascending=False)
    return {str(k): float(v) for k, v in grouped.items()}
