"""marketing_analyzer.py 测试 —— 移植自 Ecommerce-Analytics 的营销分析。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketing_analyzer import MarketingAnalyzer


@pytest.fixture
def campaigns() -> pd.DataFrame:
    return pd.DataFrame({
        "campaign_id": ["CMP001", "CMP002", "CMP003"],
        "campaign_name": ["A", "B", "C"],
        "channel": ["Online", "Offline", "Online"],
        "budget": [1000.0, 2000.0, 3000.0],
        "orders": [50, 40, 90],
        "revenue": [5000.0, 3000.0, 12000.0],
        "impressions": [100000, 80000, 150000],
        "clicks": [5000, 4000, 9000],
        "start_date": ["2024-01-01", "2024-01-08", "2024-01-15"],
        "end_date": ["2024-01-07", "2024-01-14", "2024-01-21"],
    })


# --- get_campaign_roi -----------------------------------------------------

def test_campaign_roi_values(campaigns):
    ma = MarketingAnalyzer(campaigns)
    roi = ma.get_campaign_roi()
    assert {"roi", "profit", "cost_per_order", "revenue_per_order"}.issubset(roi.columns)
    assert len(roi) == 3
    # CMP001: (5000-1000)/1000*100 = 400
    cmp1 = roi[roi["campaign_id"] == "CMP001"].iloc[0]
    assert cmp1["roi"] == pytest.approx(400.0)
    assert cmp1["profit"] == pytest.approx(4000.0)


def test_campaign_roi_sorted_desc(campaigns):
    ma = MarketingAnalyzer(campaigns)
    roi = ma.get_campaign_roi()
    assert roi["roi"].tolist() == sorted(roi["roi"].tolist(), reverse=True)


# --- get_conversion_metrics -----------------------------------------------

def test_conversion_metrics(campaigns):
    ma = MarketingAnalyzer(campaigns)
    conv = ma.get_conversion_metrics()
    assert {"ctr", "cvr", "cpc", "cpa", "overall_conversion"}.issubset(conv.columns)
    assert (conv["ctr"] >= 0).all()
    assert (conv["cvr"] >= 0).all()
    # CMP001 CTR = 5000/100000*100 = 5.0
    cmp1 = conv[conv["campaign_id"] == "CMP001"].iloc[0]
    assert cmp1["ctr"] == pytest.approx(5.0)
    assert cmp1["cvr"] == pytest.approx(1.0)   # 50/5000*100


# --- get_channel_performance ----------------------------------------------

def test_channel_performance_aggregates(campaigns):
    ma = MarketingAnalyzer(campaigns)
    ch = ma.get_channel_performance()
    assert {"channel", "roi", "ctr", "cvr"}.issubset(ch.columns)
    # 两个渠道：Online（CMP001+CMP003）、Offline（CMP002）
    assert len(ch) == 2
    online = ch[ch["channel"] == "Online"].iloc[0]
    assert online["budget"] == pytest.approx(4000.0)     # 1000 + 3000
    assert online["revenue"] == pytest.approx(17000.0)   # 5000 + 12000


# --- get_conversion_funnel ------------------------------------------------

def test_conversion_funnel_total(campaigns):
    ma = MarketingAnalyzer(campaigns)
    funnel = ma.get_conversion_funnel()
    assert funnel["impressions"] == 330000
    assert funnel["clicks"] == 18000
    assert funnel["orders"] == 180
    assert funnel["ctr"] > 0 and funnel["cvr"] > 0


def test_conversion_funnel_single_campaign(campaigns):
    ma = MarketingAnalyzer(campaigns)
    funnel = ma.get_conversion_funnel(campaign_id="CMP001")
    assert funnel["impressions"] == 100000
    assert funnel["orders"] == 50


def test_conversion_funnel_unknown_campaign(campaigns):
    ma = MarketingAnalyzer(campaigns)
    assert ma.get_conversion_funnel(campaign_id="NOPE") == {}


# --- get_campaign_efficiency_ranking（修过的潜在 bug：原引用未算的 roi）---

def test_efficiency_ranking_no_crash(campaigns):
    ma = MarketingAnalyzer(campaigns)
    eff = ma.get_campaign_efficiency_ranking()
    assert "efficiency_score" in eff.columns
    assert len(eff) == 3
    assert eff["efficiency_score"].tolist() == sorted(
        eff["efficiency_score"].tolist(), reverse=True)


# --- get_budget_allocation_suggestions ------------------------------------

def test_budget_allocation(campaigns):
    ma = MarketingAnalyzer(campaigns)
    alloc = ma.get_budget_allocation_suggestions()
    assert {"channel", "suggested_budget_percent", "recommendation"}.issubset(alloc.columns)
    assert len(alloc) == 2
