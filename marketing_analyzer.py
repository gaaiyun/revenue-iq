"""营销分析（英文列 schema）。

面向 campaigns 表：活动 ROI、转化率（CTR/CVR/CPC/CPA）、渠道表现、
转化漏斗、效率排名、预算分配建议。纯 pandas，不依赖 Streamlit / plotly。

campaigns 表预期列：campaign_id, campaign_name, channel, budget, orders,
revenue, impressions, clicks, start_date, end_date。
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd


class MarketingAnalyzer:
    """营销分析器。

    orders_df 可选，目前指标全部基于 campaigns 表（含汇总好的展示 / 点击 /
    订单 / 营收），保留 orders 参数以便后续做活动→订单归因。
    """

    def __init__(self, campaigns_df: pd.DataFrame,
                 orders_df: Optional[pd.DataFrame] = None):
        self.campaigns_df = campaigns_df
        self.orders_df = orders_df

    def get_campaign_roi(self) -> pd.DataFrame:
        """活动 ROI = (营收 - 预算) / 预算。"""
        roi = self.campaigns_df.copy()
        roi["roi"] = ((roi["revenue"] - roi["budget"]) / roi["budget"] * 100).round(2)
        roi["profit"] = (roi["revenue"] - roi["budget"]).round(2)
        roi["cost_per_order"] = (roi["budget"] / roi["orders"]).round(2)
        roi["revenue_per_order"] = (roi["revenue"] / roi["orders"]).round(2)
        return roi.sort_values("roi", ascending=False)

    def get_conversion_metrics(self) -> pd.DataFrame:
        """转化指标：CTR / CVR / 总转化 / CPC / CPA。"""
        conv = self.campaigns_df.copy()
        conv["ctr"] = (conv["clicks"] / conv["impressions"] * 100).round(2)
        conv["cvr"] = (conv["orders"] / conv["clicks"] * 100).round(2)
        conv["overall_conversion"] = (conv["orders"] / conv["impressions"] * 100).round(2)
        conv["cpc"] = (conv["budget"] / conv["clicks"]).round(2)
        conv["cpa"] = (conv["budget"] / conv["orders"]).round(2)
        return conv.sort_values("cvr", ascending=False)

    def get_channel_performance(self) -> pd.DataFrame:
        """渠道表现：按 channel 聚合后算 ROI / CTR / CVR / CPC / CPA。"""
        channel = self.campaigns_df.groupby("channel").agg(
            budget=("budget", "sum"),
            orders=("orders", "sum"),
            revenue=("revenue", "sum"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
        ).reset_index()
        channel["roi"] = ((channel["revenue"] - channel["budget"]) /
                          channel["budget"] * 100).round(2)
        channel["ctr"] = (channel["clicks"] / channel["impressions"] * 100).round(2)
        channel["cvr"] = (channel["orders"] / channel["clicks"] * 100).round(2)
        channel["cpc"] = (channel["budget"] / channel["clicks"]).round(2)
        channel["cpa"] = (channel["budget"] / channel["orders"]).round(2)
        return channel.sort_values("revenue", ascending=False)

    def get_conversion_funnel(self, campaign_id: Optional[str] = None) -> Dict:
        """转化漏斗：展示 → 点击 → 订单（单活动或全量汇总）。"""
        if campaign_id:
            data = self.campaigns_df[self.campaigns_df["campaign_id"] == campaign_id]
            if data.empty:
                return {}
            data = data.iloc[0]
        else:
            data = self.campaigns_df[["impressions", "clicks", "orders"]].sum()

        impressions = int(data["impressions"])
        clicks = int(data["clicks"])
        orders = int(data["orders"])
        return {
            "impressions": impressions,
            "clicks": clicks,
            "orders": orders,
            "ctr": round(clicks / impressions * 100, 2) if impressions else 0,
            "cvr": round(orders / clicks * 100, 2) if clicks else 0,
            "overall_conversion": round(orders / impressions * 100, 4) if impressions else 0,
        }

    def get_campaign_efficiency_ranking(self) -> pd.DataFrame:
        """活动效率综合得分：ROI 40% + CVR 30% + CTR 30%。"""
        eff = self.get_conversion_metrics().copy()
        roi = self.get_campaign_roi()[["campaign_id", "roi"]]
        eff = eff.merge(roi, on="campaign_id", how="left")

        def _norm(col: pd.Series, weight: float) -> pd.Series:
            top = col.max()
            return (col / top * weight) if top and top > 0 else col * 0

        eff["roi_score"] = _norm(eff["roi"], 40)
        eff["cvr_score"] = _norm(eff["cvr"], 30)
        eff["ctr_score"] = _norm(eff["ctr"], 30)
        eff["efficiency_score"] = (
            eff["roi_score"] + eff["cvr_score"] + eff["ctr_score"]
        ).round(2)
        return eff.sort_values("efficiency_score", ascending=False)

    def get_budget_allocation_suggestions(self) -> pd.DataFrame:
        """预算分配建议：按渠道 ROI 给增 / 保 / 减建议。"""
        channel_perf = self.get_channel_performance()
        total_budget = float(self.campaigns_df["budget"].sum())
        max_roi = channel_perf["roi"].max()

        rows = []
        for _, row in channel_perf.iterrows():
            if row["roi"] > 100:
                pct = min(40, row["roi"] / max_roi * 35) if max_roi else 35
                reco = "增加预算 - 高 ROI 渠道"
            elif row["roi"] > 50:
                pct = 25
                reco = "保持预算 - 表现良好"
            else:
                pct = 15
                reco = "优化或减少预算 - ROI 偏低"
            rows.append({
                "channel": row["channel"],
                "current_roi": row["roi"],
                "suggested_budget_percent": round(pct, 1),
                "suggested_budget_amount": round(total_budget * pct / 100, 2),
                "recommendation": reco,
            })
        return pd.DataFrame(rows)
