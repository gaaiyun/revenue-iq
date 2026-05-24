"""把数字结果交给 LLM 写"销售经理看得懂的"自然语言点评。

v1 仪表板有图有指标但没有总结 —— 用户得自己把折线图看明白。v2 加这一层：
喂入 ``SalesSummary`` / 趋势点 / 异常点，LLM 生成几段中文 commentary：

- 整体表现总结（"本期 X 元，环比 +5%"）
- 区域 / 品类亮点
- 异常 spike / drop 解释建议
- 下期重点关注事项

LLM 缺 key 时退化为基于规则的模板生成器（无 LLM 也能用）。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional


LLMBackend = Literal["openai", "anthropic", "deepseek"]


class LLMNotAvailable(RuntimeError):
    pass


@dataclass
class CommentaryReport:
    overview: str
    highlights: List[str]
    risks: List[str]
    recommendations: List[str]
    backend: str

    def to_dict(self) -> dict:
        return {
            "overview": self.overview,
            "highlights": self.highlights,
            "risks": self.risks,
            "recommendations": self.recommendations,
            "backend": self.backend,
        }

    def to_markdown(self) -> str:
        lines = ["## 整体概览", "", self.overview, ""]
        if self.highlights:
            lines += ["## 亮点", ""] + [f"- {h}" for h in self.highlights] + [""]
        if self.risks:
            lines += ["## 风险", ""] + [f"- {r}" for r in self.risks] + [""]
        if self.recommendations:
            lines += ["## 建议", ""] + [f"- {r}" for r in self.recommendations]
        return "\n".join(lines)


class LLMClient:
    def __init__(self, backend: LLMBackend = "deepseek",
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 timeout: float = 60.0):
        self.backend = backend
        self.timeout = timeout
        self.api_key = api_key or self._default_key(backend)
        self.base_url = base_url or self._default_base_url(backend)
        self.model = model or self._default_model(backend)

    @staticmethod
    def _default_key(backend: LLMBackend) -> Optional[str]:
        return {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "deepseek": os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
        }.get(backend)

    @staticmethod
    def _default_base_url(backend: LLMBackend) -> Optional[str]:
        return {"deepseek": "https://api.deepseek.com/v1"}.get(backend)

    @staticmethod
    def _default_model(backend: LLMBackend) -> str:
        return {"openai": "gpt-4o-mini",
                "anthropic": "claude-3-5-haiku-20241022",
                "deepseek": "deepseek-chat"}.get(backend, "gpt-4o-mini")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, user: str, temperature: float = 0.3) -> str:
        if not self.is_available():
            raise LLMNotAvailable(
                f"{self.backend} backend 缺 API key（环境变量 "
                f"{self.backend.upper()}_API_KEY）"
            )
        if self.backend == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=self.api_key, timeout=self.timeout)
            resp = client.messages.create(
                model=self.model, max_tokens=2048, temperature=temperature,
                system=system, messages=[{"role": "user", "content": user}])
            return resp.content[0].text if resp.content else ""
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                        timeout=self.timeout)
        resp = client.chat.completions.create(
            model=self.model, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}])
        return resp.choices[0].message.content or ""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


# --- 规则 fallback ---------------------------------------------------------

def _heuristic_commentary(summary: Dict, trend: List[Dict],
                          anomalies: List[Dict],
                          category: Dict, region: Dict) -> CommentaryReport:
    revenue = summary.get("total_revenue", 0)
    growth = summary.get("growth_first_to_last_month")
    n_orders = summary.get("total_orders", 0)
    aov = summary.get("avg_order_value", 0)
    n_customers = summary.get("n_customers", 0)
    period_start = summary.get("date_start", "未知")
    period_end = summary.get("date_end", "未知")

    growth_str = f"，环比首月增长 {growth:+.1f}%" if growth is not None else ""
    overview = (
        f"{period_start} 至 {period_end} 期间，共 {n_orders} 笔订单、"
        f"{n_customers} 名客户，总收入 CNY{revenue:,.0f}，"
        f"平均客单 CNY{aov:,.0f}{growth_str}。"
    )

    highlights = []
    if summary.get("top_category"):
        highlights.append(
            f"品类 Top1：{summary['top_category']} "
            f"贡献 CNY{summary['top_category_revenue']:,.0f}"
        )
    if summary.get("top_region"):
        highlights.append(
            f"区域 Top1：{summary['top_region']} "
            f"贡献 CNY{summary['top_region_revenue']:,.0f}"
        )
    if growth is not None and growth > 10:
        highlights.append(f"末月较首月增长 {growth:+.1f}%，业务在扩张")

    risks = []
    drops = [a for a in anomalies if a.get("direction") == "drop"]
    if drops:
        first_drop = drops[0]
        risks.append(
            f"{first_drop['period']} 出现异常下跌（实际 CNY{first_drop['actual']:,.0f}，"
            f"预期 CNY{first_drop['expected']:,.0f}，z={first_drop['z_score']:.2f}）"
        )
    if growth is not None and growth < -10:
        risks.append(f"末月较首月下降 {growth:.1f}%，需排查原因")
    if n_customers > 0 and n_orders / n_customers < 1.2:
        risks.append("人均订单数低，复购率有提升空间")

    recommendations = []
    if summary.get("top_category"):
        recommendations.append(
            f"加大 {summary['top_category']} 类目库存 / 营销投放"
        )
    if drops:
        recommendations.append("排查异常下跌时段的因素（促销 / 库存 / 渠道）")
    spikes = [a for a in anomalies if a.get("direction") == "spike"]
    if spikes:
        recommendations.append(
            f"复盘 {spikes[0]['period']} 异常增长动因，看能否复制"
        )
    if not recommendations:
        recommendations.append("继续维持现有运营节奏，关注下期数据")

    return CommentaryReport(
        overview=overview,
        highlights=highlights,
        risks=risks,
        recommendations=recommendations,
        backend="heuristic",
    )


# --- 主入口 -----------------------------------------------------------------

def commentary(
    summary: Dict,
    trend: Optional[List[Dict]] = None,
    anomalies: Optional[List[Dict]] = None,
    category_breakdown: Optional[Dict] = None,
    region_breakdown: Optional[Dict] = None,
    llm_client: Optional[LLMClient] = None,
    backend: Optional[str] = None,
) -> CommentaryReport:
    """喂入指标 → 出自然语言报告。"""
    trend = trend or []
    anomalies = anomalies or []
    category = category_breakdown or {}
    region = region_breakdown or {}

    client = llm_client
    if client is None and backend:
        client = LLMClient(backend=backend)

    if client and client.is_available():
        try:
            return _llm_commentary(summary, trend, anomalies, category, region, client)
        except (LLMNotAvailable, ValueError, json.JSONDecodeError):
            pass

    return _heuristic_commentary(summary, trend, anomalies, category, region)


def _llm_commentary(summary, trend, anomalies, category, region,
                    client: LLMClient) -> CommentaryReport:
    system = (
        "你是一名零售销售分析师。基于给定的销售指标，写一份中文 commentary："
        "整体概览（≤120 字）+ 3-5 条亮点 + 0-3 条风险 + 1-3 条下期建议。"
        "只输出 JSON，字段：overview (str), highlights (list[str]), "
        "risks (list[str]), recommendations (list[str])。"
    )
    payload = {
        "summary": summary,
        "trend_last_6": trend[-6:] if trend else [],
        "anomalies_top_3": anomalies[:3] if anomalies else [],
        "category_breakdown_top5": dict(list(category.items())[:5]),
        "region_breakdown_top5": dict(list(region.items())[:5]),
    }
    user = (
        "销售指标：\n" + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\n按 system 要求输出 JSON。"
    )
    raw = _strip_fences(client.chat(system, user, temperature=0.3))
    data = json.loads(raw)
    return CommentaryReport(
        overview=str(data.get("overview", "")),
        highlights=[str(h) for h in data.get("highlights", []) if h],
        risks=[str(r) for r in data.get("risks", []) if r],
        recommendations=[str(r) for r in data.get("recommendations", []) if r],
        backend=f"llm:{client.backend}",
    )
