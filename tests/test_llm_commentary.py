"""llm_commentary.py 测试 —— mock LLM。"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_commentary import (
    CommentaryReport,
    LLMClient,
    LLMNotAvailable,
    _heuristic_commentary,
    _strip_fences,
    commentary,
)


# --- LLMClient -------------------------------------------------------------

def test_default_models():
    assert LLMClient(backend="openai", api_key="x").model == "gpt-4o-mini"
    assert LLMClient(backend="anthropic", api_key="x").model == "claude-3-5-haiku-20241022"
    assert LLMClient(backend="deepseek", api_key="x").model == "deepseek-chat"


def test_is_available_with_key():
    assert LLMClient(backend="deepseek", api_key="sk-test").is_available()


def test_chat_raises_without_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = LLMClient(backend="deepseek")
    with pytest.raises(LLMNotAvailable):
        c.chat("sys", "user")


# --- _strip_fences ---------------------------------------------------------

def test_strip_fences_json_block():
    assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_strip_fences_no_block():
    assert _strip_fences('{"a": 1}') == '{"a": 1}'


# --- _heuristic_commentary -------------------------------------------------

def _make_summary(**overrides) -> dict:
    base = {
        "total_revenue": 100000.0,
        "total_orders": 50,
        "avg_order_value": 2000.0,
        "n_customers": 30,
        "date_start": "2024-01-01",
        "date_end": "2024-03-31",
        "top_category": "电子产品",
        "top_category_revenue": 50000.0,
        "top_region": "华东",
        "top_region_revenue": 40000.0,
        "growth_first_to_last_month": 15.0,
    }
    base.update(overrides)
    return base


def test_heuristic_overview_mentions_revenue():
    report = _heuristic_commentary(_make_summary(), [], [], {}, {})
    assert "100,000" in report.overview or "100000" in report.overview


def test_heuristic_highlights_include_top_category():
    report = _heuristic_commentary(_make_summary(), [], [], {}, {})
    assert any("电子产品" in h for h in report.highlights)


def test_heuristic_highlights_growth_when_positive():
    report = _heuristic_commentary(_make_summary(growth_first_to_last_month=20),
                                   [], [], {}, {})
    assert any("扩张" in h or "增长" in h for h in report.highlights)


def test_heuristic_risk_when_negative_growth():
    report = _heuristic_commentary(_make_summary(growth_first_to_last_month=-15),
                                   [], [], {}, {})
    assert any("下降" in r for r in report.risks)


def test_heuristic_risk_when_anomaly_drop():
    drops = [{"period": "2024-03-15", "actual": 1000,
              "expected": 5000, "z_score": -3.5, "direction": "drop"}]
    report = _heuristic_commentary(_make_summary(), [], drops, {}, {})
    assert any("下跌" in r or "drop" in r for r in report.risks)


def test_heuristic_recommendation_when_anomaly_spike():
    spikes = [{"period": "2024-02-14", "actual": 15000,
               "expected": 5000, "z_score": 3.2, "direction": "spike"}]
    report = _heuristic_commentary(_make_summary(), [], spikes, {}, {})
    # 应该建议复盘 spike
    assert any("复盘" in r or "增长" in r for r in report.recommendations)


def test_heuristic_returns_report_object():
    report = _heuristic_commentary(_make_summary(), [], [], {}, {})
    assert isinstance(report, CommentaryReport)
    assert report.backend == "heuristic"


def test_heuristic_to_markdown_includes_all_sections():
    report = _heuristic_commentary(_make_summary(), [], [
        {"period": "2024-03-15", "actual": 1000, "expected": 5000,
         "z_score": -3, "direction": "drop"}
    ], {}, {})
    md = report.to_markdown()
    assert "## 整体概览" in md
    assert "## 亮点" in md
    assert "## 风险" in md
    assert "## 建议" in md


def test_heuristic_with_no_growth_data():
    s = _make_summary(growth_first_to_last_month=None)
    report = _heuristic_commentary(s, [], [], {}, {})
    # 不应崩
    assert isinstance(report, CommentaryReport)


# --- commentary（端到端 + LLM mock）--------------------------------------

def _make_mock_client(response: str) -> LLMClient:
    c = LLMClient(backend="deepseek", api_key="sk-test")
    c.chat = MagicMock(return_value=response)
    return c


def test_commentary_without_llm_uses_heuristic():
    report = commentary(_make_summary())
    assert report.backend == "heuristic"


def test_commentary_with_mocked_llm():
    client = _make_mock_client(
        '{"overview": "本期表现强劲", '
        '"highlights": ["收入 +20%", "客单提升"], '
        '"risks": ["客户复购率偏低"], '
        '"recommendations": ["加大投放"]}'
    )
    report = commentary(_make_summary(), llm_client=client)
    assert report.overview == "本期表现强劲"
    assert len(report.highlights) == 2
    assert "客户复购率偏低" in report.risks
    assert "llm" in report.backend


def test_commentary_llm_falls_back_on_bad_json():
    client = _make_mock_client("not json")
    report = commentary(_make_summary(), llm_client=client)
    assert report.backend == "heuristic"


def test_commentary_llm_handles_code_fence():
    client = _make_mock_client(
        '```json\n{"overview": "x", "highlights": [], '
        '"risks": [], "recommendations": []}\n```'
    )
    report = commentary(_make_summary(), llm_client=client)
    assert report.overview == "x"


def test_commentary_to_dict_serializable():
    import json
    report = commentary(_make_summary())
    json.dumps(report.to_dict(), ensure_ascii=False)


def test_commentary_filters_empty_list_items():
    """LLM 可能返回空字符串列表项 → 应过滤掉。"""
    client = _make_mock_client(
        '{"overview": "x", "highlights": ["good", "", null], '
        '"risks": [], "recommendations": ["ok"]}'
    )
    report = commentary(_make_summary(), llm_client=client)
    assert report.highlights == ["good"]
