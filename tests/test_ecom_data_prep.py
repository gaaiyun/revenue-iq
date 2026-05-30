"""ecom_data_prep.py 测试 —— 移植自 Ecommerce-Analytics 的数据预处理。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecom_data_prep import (
    load_campaigns, load_orders, load_products,
    overview_metrics, prepare_orders, repeat_purchase,
)

DATA = Path(__file__).resolve().parent.parent / "sample_data"


# --- prepare_orders -------------------------------------------------------

def test_prepare_orders_adds_revenue():
    df = pd.DataFrame({"quantity": [2, 3], "unit_price": [10.0, 20.0],
                       "cost_price": [5.0, 10.0],
                       "order_date": ["2024-01-01", "2024-01-02"]})
    out = prepare_orders(df)
    assert out["revenue"].tolist() == [20.0, 60.0]


def test_prepare_orders_adds_cost_and_profit():
    df = pd.DataFrame({"quantity": [2, 3], "unit_price": [10.0, 20.0],
                       "cost_price": [5.0, 10.0],
                       "order_date": ["2024-01-01", "2024-01-02"]})
    out = prepare_orders(df)
    assert out["cost"].tolist() == [10.0, 30.0]
    assert out["profit"].tolist() == [10.0, 30.0]


def test_prepare_orders_idempotent():
    df = pd.DataFrame({"quantity": [2], "unit_price": [10.0], "cost_price": [5.0],
                       "order_date": ["2024-01-01"]})
    once = prepare_orders(df)
    twice = prepare_orders(once)
    pd.testing.assert_frame_equal(once, twice)


def test_prepare_orders_no_cost_profit_equals_revenue():
    df = pd.DataFrame({"quantity": [2], "unit_price": [10.0],
                       "order_date": ["2024-01-01"]})
    out = prepare_orders(df)
    assert out["profit"].iloc[0] == 20.0


def test_prepare_orders_preserves_existing_revenue():
    df = pd.DataFrame({"quantity": [2], "unit_price": [10.0], "cost_price": [5.0],
                       "revenue": [99.0], "order_date": ["2024-01-01"]})
    out = prepare_orders(df)
    assert out["revenue"].iloc[0] == 99.0


def test_prepare_orders_parses_date():
    df = pd.DataFrame({"quantity": [1], "unit_price": [10.0],
                       "order_date": ["2024-01-15"]})
    out = prepare_orders(df)
    assert pd.api.types.is_datetime64_any_dtype(out["order_date"])


# --- overview_metrics -----------------------------------------------------

def test_overview_metrics_basic():
    df = prepare_orders(pd.DataFrame({
        "quantity": [2, 3], "unit_price": [10.0, 20.0], "cost_price": [5.0, 10.0],
        "order_date": ["2024-01-01", "2024-01-02"], "customer_id": ["C1", "C2"]}))
    m = overview_metrics(df)
    assert m["n_orders"] == 2
    assert m["total_revenue"] == 80.0
    assert m["total_profit"] == 40.0
    assert m["profit_margin_pct"] == 50.0
    assert m["unique_customers"] == 2


def test_overview_metrics_empty():
    m = overview_metrics(pd.DataFrame())
    assert m["n_orders"] == 0
    assert m["total_revenue"] == 0.0


def test_overview_metrics_aov():
    df = prepare_orders(pd.DataFrame({
        "quantity": [1, 1, 1], "unit_price": [100.0, 200.0, 300.0],
        "cost_price": [50.0] * 3, "order_date": ["2024-01-01"] * 3,
        "customer_id": ["C1", "C2", "C3"]}))
    assert overview_metrics(df)["avg_order_value"] == 200.0


# --- repeat_purchase ------------------------------------------------------

def test_repeat_purchase_basic():
    df = prepare_orders(pd.DataFrame({
        "quantity": [1] * 4, "unit_price": [10.0] * 4,
        "order_date": ["2024-01-01"] * 4,
        "customer_id": ["C1", "C1", "C2", "C3"]}))
    r = repeat_purchase(df)
    assert r["total_customers"] == 3
    assert r["repeat_customers"] == 1       # 只有 C1 下了 2 单
    assert r["repeat_rate_pct"] == pytest.approx(33.33, abs=0.01)


def test_repeat_purchase_empty():
    r = repeat_purchase(pd.DataFrame())
    assert r["total_customers"] == 0
    assert r["repeat_rate_pct"] == 0.0


# --- 样本文件加载 ---------------------------------------------------------

def test_load_orders_sample():
    df = load_orders(str(DATA / "orders.csv"))
    assert "revenue" in df.columns and "profit" in df.columns
    assert len(df) > 0


def test_load_products_sample():
    df = load_products(str(DATA / "products.csv"))
    assert {"stock_quantity", "reorder_level"}.issubset(df.columns)
    assert len(df) > 0


def test_load_campaigns_sample():
    df = load_campaigns(str(DATA / "campaigns.csv"))
    assert len(df) > 0
    assert pd.api.types.is_datetime64_any_dtype(df["start_date"])
