"""product_analyzer.py 测试 —— 移植自 Ecommerce-Analytics 的商品分析。

覆盖修过的 bug：CLI 原本调 get_top_sellers / get_low_stock_products（不存在），
现用真实方法 get_sales_ranking / get_inventory_status / get_low_stock。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecom_data_prep import prepare_orders
from product_analyzer import ProductAnalyzer


@pytest.fixture
def orders() -> pd.DataFrame:
    df = pd.DataFrame({
        "order_id": [f"O{i}" for i in range(8)],
        "product_id": ["P001", "P002", "P001", "P003", "P002", "P001", "P003", "P002"],
        "category": ["Electronics", "Electronics", "Electronics",
                     "Office", "Electronics", "Electronics", "Office", "Electronics"],
        "quantity": [2, 1, 3, 5, 1, 1, 2, 4],
        "unit_price": [100.0, 300.0, 100.0, 20.0, 300.0, 100.0, 20.0, 300.0],
        "cost_price": [60.0, 180.0, 60.0, 8.0, 180.0, 60.0, 8.0, 180.0],
        "order_date": ["2024-01-0%d" % (i + 1) for i in range(8)],
        "customer_id": ["C1", "C2", "C1", "C3", "C2", "C4", "C5", "C2"],
    })
    return prepare_orders(df)


@pytest.fixture
def products() -> pd.DataFrame:
    return pd.DataFrame({
        "product_id": ["P001", "P002", "P003"],
        "product_name": ["WirelessMouse", "MechanicalKeyboard", "MousePad"],
        "category": ["Electronics", "Electronics", "Office"],
        "stock_quantity": [150, 25, 300],     # P002 低库存
        "reorder_level": [50, 30, 100],
    })


# --- get_sales_ranking（修过的 bug：真实方法名）---------------------------

def test_sales_ranking_sorted_by_revenue(orders, products):
    pa = ProductAnalyzer(orders, products)
    ranking = pa.get_sales_ranking(top_n=5)
    assert len(ranking) <= 5
    assert {"product_id", "quantity", "revenue", "profit"}.issubset(ranking.columns)
    revs = ranking["revenue"].tolist()
    assert revs == sorted(revs, reverse=True)


def test_sales_ranking_merges_product_name(orders, products):
    pa = ProductAnalyzer(orders, products)
    ranking = pa.get_sales_ranking()
    assert "product_name" in ranking.columns
    assert ranking["product_name"].notna().all()


def test_sales_ranking_works_without_products(orders):
    pa = ProductAnalyzer(orders)     # products 为 None
    ranking = pa.get_sales_ranking(top_n=2)
    assert len(ranking) == 2
    assert "product_name" not in ranking.columns


def test_sales_ranking_empty_orders(products):
    empty = pd.DataFrame(columns=["order_id", "product_id", "category",
                                  "quantity", "revenue", "profit"])
    pa = ProductAnalyzer(empty, products)
    assert len(pa.get_sales_ranking()) == 0


# --- get_inventory_status / get_low_stock（修过的 bug：真实方法名）---------

def test_inventory_status_categories(orders, products):
    pa = ProductAnalyzer(orders, products)
    inv = pa.get_inventory_status()
    assert "stock_status" in inv.columns
    assert set(inv["stock_status"]).issubset({"低库存", "正常", "充足"})
    assert len(inv) == len(products)


def test_inventory_turnover_is_numeric(orders, products):
    pa = ProductAnalyzer(orders, products)
    inv = pa.get_inventory_status()
    assert pd.api.types.is_numeric_dtype(inv["turnover_rate"])


def test_low_stock_flags_p002(orders, products):
    pa = ProductAnalyzer(orders, products)
    low = pa.get_low_stock()
    assert "P002" in low["product_id"].astype(str).tolist()
    assert all(low["stock_status"] == "低库存")


def test_inventory_requires_products(orders):
    pa = ProductAnalyzer(orders)
    with pytest.raises(ValueError, match="需要 products"):
        pa.get_inventory_status()


# --- get_profit_analysis --------------------------------------------------

def test_profit_analysis_has_margin(orders, products):
    pa = ProductAnalyzer(orders, products)
    profit = pa.get_profit_analysis()
    assert "profit_margin" in profit.columns
    assert pd.api.types.is_numeric_dtype(profit["profit_margin"])
    assert len(profit) == 3


# --- get_category_performance ---------------------------------------------

def test_category_performance(orders, products):
    pa = ProductAnalyzer(orders, products)
    cat = pa.get_category_performance()
    assert {"category", "revenue", "avg_order_value", "order_count"}.issubset(cat.columns)
    assert cat["revenue"].tolist() == sorted(cat["revenue"].tolist(), reverse=True)


# --- get_restock_recommendations（含曾 crash 的 round 路径）---------------

def test_restock_recommendations_no_crash(orders, products):
    pa = ProductAnalyzer(orders, products)
    rec = pa.get_restock_recommendations()
    # 不应抛 TypeError: Expected numeric dtype
    if len(rec) > 0:
        assert "recommendation" in rec.columns
        assert "days_remaining" in rec.columns


def test_restock_includes_low_stock_item(orders, products):
    pa = ProductAnalyzer(orders, products)
    rec = pa.get_restock_recommendations()
    # P002 低库存 → 应在补货清单里且建议立即补货
    p002 = rec[rec["product_id"].astype(str) == "P002"]
    assert len(p002) == 1
    assert "立即补货" in p002.iloc[0]["recommendation"]


# --- get_price_optimization_suggestions -----------------------------------

def test_price_optimization(orders, products):
    pa = ProductAnalyzer(orders, products)
    sug = pa.get_price_optimization_suggestions()
    assert "price_suggestion" in sug.columns
    assert "profit_margin" in sug.columns
    assert len(sug) > 0
