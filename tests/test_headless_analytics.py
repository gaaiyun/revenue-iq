"""headless_analytics.py 测试 —— 纯 pandas，无 Streamlit / plotly。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from headless_analytics import (
    AnomalyAlert,
    DEFAULT_COLS,
    SalesSummary,
    TrendPoint,
    category_breakdown,
    compute_summary,
    compute_trend,
    detect_anomalies,
    region_breakdown,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """120 天合成销售数据。"""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    rows = []
    cats = ["电子产品", "服装", "食品", "家居"]
    regs = ["华北", "华东", "华南", "西南"]
    for d in dates:
        for _ in range(rng.randint(1, 4)):
            rows.append({
                "日期": d,
                "销售额": float(rng.uniform(500, 5000)),
                "销售数量": int(rng.randint(1, 20)),
                "产品类别": rng.choice(cats),
                "地区": rng.choice(regs),
                "客户 ID": f"C{rng.randint(1000, 1050)}",
            })
    return pd.DataFrame(rows)


# --- compute_summary --------------------------------------------------------

def test_compute_summary_returns_object(sample_df):
    summary = compute_summary(sample_df)
    assert isinstance(summary, SalesSummary)
    assert summary.n_records == len(sample_df)


def test_summary_total_revenue_matches(sample_df):
    summary = compute_summary(sample_df)
    expected = float(sample_df["销售额"].sum())
    assert summary.total_revenue == pytest.approx(expected)


def test_summary_aov_is_revenue_over_orders(sample_df):
    summary = compute_summary(sample_df)
    assert summary.avg_order_value == pytest.approx(
        summary.total_revenue / summary.total_orders
    )


def test_summary_top_category_is_real_category(sample_df):
    summary = compute_summary(sample_df)
    assert summary.top_category in sample_df["产品类别"].unique()


def test_summary_top_region_is_real_region(sample_df):
    summary = compute_summary(sample_df)
    assert summary.top_region in sample_df["地区"].unique()


def test_summary_n_customers_unique(sample_df):
    summary = compute_summary(sample_df)
    assert summary.n_customers == sample_df["客户 ID"].nunique()


def test_summary_growth_computed_when_2plus_months(sample_df):
    summary = compute_summary(sample_df)
    # 4 个月数据 → 应该有增长率
    assert summary.growth_first_to_last_month is not None
    assert isinstance(summary.growth_first_to_last_month, float)


def test_summary_dates_returned_as_iso(sample_df):
    summary = compute_summary(sample_df)
    # 应该是 YYYY-MM-DD 格式
    assert "-" in summary.date_start
    pd.to_datetime(summary.date_start)  # 解析不抛 = 合法日期
    pd.to_datetime(summary.date_end)


def test_summary_empty_df_raises():
    with pytest.raises(ValueError, match="为空"):
        compute_summary(pd.DataFrame())


def test_summary_missing_required_cols_raises():
    df = pd.DataFrame({"日期": pd.date_range("2024-01-01", periods=5)})
    with pytest.raises(ValueError, match="缺必要列"):
        compute_summary(df)


def test_summary_to_dict_json_serializable(sample_df):
    import json
    summary = compute_summary(sample_df)
    json.dumps(summary.to_dict(), ensure_ascii=False)


def test_summary_with_custom_column_map(sample_df):
    """改列名映射也能工作。"""
    df2 = sample_df.rename(columns={"销售额": "amount", "日期": "date"})
    summary = compute_summary(df2, column_map={
        "date": "date", "amount": "amount",
    })
    assert summary.total_revenue == pytest.approx(float(sample_df["销售额"].sum()))


# --- compute_trend ----------------------------------------------------------

def test_trend_monthly_returns_points(sample_df):
    points = compute_trend(sample_df, period="M")
    # 4 个月 → 应该 4 个点
    assert len(points) == 4
    assert all(isinstance(p, TrendPoint) for p in points)


def test_trend_first_point_no_growth(sample_df):
    points = compute_trend(sample_df, period="M")
    assert points[0].growth_pct is None


def test_trend_subsequent_growth_is_float(sample_df):
    points = compute_trend(sample_df, period="M")
    for p in points[1:]:
        assert p.growth_pct is None or isinstance(p.growth_pct, float)


def test_trend_revenue_sums_match_total(sample_df):
    points = compute_trend(sample_df, period="M")
    total = sum(p.revenue for p in points)
    assert total == pytest.approx(float(sample_df["销售额"].sum()))


def test_trend_weekly_more_points_than_monthly(sample_df):
    monthly = compute_trend(sample_df, period="M")
    weekly = compute_trend(sample_df, period="W")
    assert len(weekly) > len(monthly)


def test_trend_period_M_normalized_to_ME(sample_df):
    """pandas 2.x：M 已废弃 → 模块内部自动换 ME，不应警告。"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # 不应产生 deprecation warning
        compute_trend(sample_df, period="M")


# --- detect_anomalies -------------------------------------------------------

def test_anomalies_returns_list(sample_df):
    alerts = detect_anomalies(sample_df, period="D")
    assert isinstance(alerts, list)


def test_anomalies_each_alert_has_direction(sample_df):
    alerts = detect_anomalies(sample_df, period="D", z_threshold=1.5)
    for a in alerts:
        assert a.direction in ("spike", "drop")


def test_anomalies_z_threshold_filters(sample_df):
    high = detect_anomalies(sample_df, period="D", z_threshold=3.0)
    low = detect_anomalies(sample_df, period="D", z_threshold=1.5)
    # 阈值越低，告警越多（或相等）
    assert len(low) >= len(high)


def test_anomalies_empty_when_data_too_short():
    df = pd.DataFrame({
        "日期": pd.date_range("2024-01-01", periods=3),
        "销售额": [100, 200, 300],
    })
    alerts = detect_anomalies(df, period="D")
    assert alerts == []


def test_anomalies_alert_to_dict():
    a = AnomalyAlert(period="2024-01-15", actual=10000.0,
                     expected=5000.0, z_score=3.5, direction="spike")
    d = a.to_dict()
    assert d["direction"] == "spike"
    assert d["actual"] == 10000.0


# --- breakdown -------------------------------------------------------------

def test_category_breakdown_sums_match(sample_df):
    cat = category_breakdown(sample_df)
    total = sum(cat.values())
    assert total == pytest.approx(float(sample_df["销售额"].sum()))


def test_category_breakdown_sorted_descending(sample_df):
    cat = category_breakdown(sample_df)
    values = list(cat.values())
    assert values == sorted(values, reverse=True)


def test_category_breakdown_empty_when_missing_column():
    df = pd.DataFrame({
        "日期": pd.date_range("2024-01-01", periods=5),
        "销售额": [100] * 5,
    })
    assert category_breakdown(df) == {}


def test_region_breakdown_sums_match(sample_df):
    reg = region_breakdown(sample_df)
    total = sum(reg.values())
    assert total == pytest.approx(float(sample_df["销售额"].sum()))


# --- 默认列名常量 ----------------------------------------------------------

def test_default_cols_has_all_keys():
    expected = {"date", "amount", "qty", "category", "region", "customer", "customer_tier"}
    assert expected.issubset(set(DEFAULT_COLS.keys()))
