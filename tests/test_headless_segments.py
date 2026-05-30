"""headless_segments.py 测试 —— 纯 pandas ABC + RFM，无 Streamlit/sklearn。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from headless_segments import (
    abc_analysis, abc_summary, rfm_segments, segment_counts,
)


@pytest.fixture
def sales_df() -> pd.DataFrame:
    rng = np.random.RandomState(11)
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    cats = ["电子产品", "服装", "食品", "家居", "图书"]
    weights = {"电子产品": 8, "服装": 4, "食品": 2, "家居": 3, "图书": 1}
    rows = []
    for d in dates:
        for _ in range(rng.randint(1, 4)):
            cat = rng.choice(cats)
            rows.append({
                "日期": d,
                "销售额": float(rng.uniform(100, 1000) * weights[cat]),
                "产品类别": cat,
                "客户 ID": f"C{rng.randint(1000, 1030)}",
            })
    return pd.DataFrame(rows)


# --- abc_analysis ---------------------------------------------------------

def test_abc_classes_valid(sales_df):
    rows = abc_analysis(sales_df)
    assert len(rows) > 0
    assert all(r["abc_class"] in ("A", "B", "C") for r in rows)


def test_abc_sorted_by_revenue(sales_df):
    rows = abc_analysis(sales_df)
    revs = [r["revenue"] for r in rows]
    assert revs == sorted(revs, reverse=True)


def test_abc_cum_pct_monotonic_increasing(sales_df):
    rows = abc_analysis(sales_df)
    cums = [r["cum_pct"] for r in rows]
    assert cums == sorted(cums)
    assert cums[-1] == pytest.approx(100.0, abs=0.5)


def test_abc_missing_column_returns_empty():
    df = pd.DataFrame({"日期": pd.date_range("2024-01-01", periods=3),
                       "销售额": [100, 200, 300]})
    assert abc_analysis(df) == []


# --- abc_summary ----------------------------------------------------------

def test_abc_summary_keys(sales_df):
    s = abc_summary(sales_df)
    assert set(s.keys()) == {"A", "B", "C"}
    for cls in ("A", "B", "C"):
        assert {"n_items", "revenue", "item_pct", "revenue_pct"}.issubset(s[cls])


def test_abc_summary_revenue_pct_sums_100(sales_df):
    s = abc_summary(sales_df)
    total_pct = sum(s[c]["revenue_pct"] for c in ("A", "B", "C"))
    assert total_pct == pytest.approx(100.0, abs=0.5)


# --- rfm_segments ---------------------------------------------------------

def test_rfm_segments_valid_labels(sales_df):
    rows = rfm_segments(sales_df)
    valid = {"重要价值客户", "重要发展客户", "一般客户", "低价值客户"}
    assert len(rows) > 0
    assert all(r["segment"] in valid for r in rows)


def test_rfm_score_in_range(sales_df):
    rows = rfm_segments(sales_df)
    for r in rows:
        assert 3 <= r["rfm_score"] <= 15


def test_rfm_missing_customer_col_returns_empty():
    df = pd.DataFrame({"日期": pd.date_range("2024-01-01", periods=3),
                       "销售额": [100, 200, 300]})
    assert rfm_segments(df) == []


# --- segment_counts -------------------------------------------------------

def test_segment_counts_sum_equals_customers(sales_df):
    counts = segment_counts(sales_df)
    n_customers = sales_df["客户 ID"].nunique()
    assert sum(counts.values()) == n_customers
