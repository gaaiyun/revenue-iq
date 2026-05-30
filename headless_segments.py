"""无 Streamlit / 无 sklearn 的 ABC 分析 + RFM 客户分群。

trunk 的 ``sales_analyzer.py`` 把 ABC / RFM 和 plotly、``st.error``、sklearn
KMeans 绑在一起，没法在 CLI 里跑。这个模块用纯 pandas 复刻同口径：

- ABC（帕累托）：按销售额降序累计，≤70% 为 A，≤90% 为 B，其余 C
- RFM：Recency / Frequency / Monetary 各分 5 档，按总分给 4 类客户分群

输入是中文列销售流水（日期 / 销售额 / 客户 ID / 产品类别 等）。
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from headless_analytics import _resolve_cols


def abc_analysis(df: pd.DataFrame, by: Optional[str] = None,
                 column_map: Optional[Dict] = None) -> List[Dict]:
    """ABC 帕累托分析。

    by: 聚合维度列名，默认用产品类别列；列不存在时回退到品类。
    返回每个项目的销售额 / 累计占比 / ABC 分类。
    """
    cols = _resolve_cols(df, column_map)
    amount = cols["amount"]
    group_col = by or cols["category"]
    if group_col not in df.columns or amount not in df.columns:
        return []

    grouped = (df.groupby(group_col)[amount].sum()
               .sort_values(ascending=False).reset_index())
    grouped.columns = [group_col, "revenue"]
    total = float(grouped["revenue"].sum())
    if total <= 0:
        return []
    grouped["cum_revenue"] = grouped["revenue"].cumsum()
    grouped["cum_pct"] = (grouped["cum_revenue"] / total * 100).round(2)

    def _abc(pct: float) -> str:
        if pct <= 70:
            return "A"
        if pct <= 90:
            return "B"
        return "C"

    grouped["abc_class"] = grouped["cum_pct"].apply(_abc)
    return [{"item": str(r[group_col]), "revenue": float(r["revenue"]),
             "cum_pct": float(r["cum_pct"]), "abc_class": r["abc_class"]}
            for _, r in grouped.iterrows()]


def abc_summary(df: pd.DataFrame, by: Optional[str] = None,
                column_map: Optional[Dict] = None) -> Dict[str, Dict]:
    """ABC 汇总：每类的项目数 / 销售额 / 占比。"""
    rows = abc_analysis(df, by=by, column_map=column_map)
    if not rows:
        return {}
    total_rev = sum(r["revenue"] for r in rows)
    n_items = len(rows)
    out: Dict[str, Dict] = {}
    for cls in ("A", "B", "C"):
        items = [r for r in rows if r["abc_class"] == cls]
        rev = sum(r["revenue"] for r in items)
        out[cls] = {
            "n_items": len(items),
            "revenue": round(rev, 2),
            "item_pct": round(len(items) / n_items * 100, 2) if n_items else 0.0,
            "revenue_pct": round(rev / total_rev * 100, 2) if total_rev else 0.0,
        }
    return out


def rfm_segments(df: pd.DataFrame,
                 column_map: Optional[Dict] = None) -> List[Dict]:
    """RFM 客户分群（纯 pandas，5 分位分箱）。"""
    cols = _resolve_cols(df, column_map)
    date_col, amount_col, cust_col = cols["date"], cols["amount"], cols["customer"]
    if not all(c in df.columns for c in (date_col, amount_col, cust_col)):
        return []

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    now = d[date_col].max()
    rfm = d.groupby(cust_col).agg(
        R=(date_col, lambda x: (now - x.max()).days),
        F=(date_col, "count"),
        M=(amount_col, "sum"),
    ).reset_index()

    def _score(series: pd.Series, reverse: bool = False) -> pd.Series:
        ranks = series.rank(method="first")
        try:
            labels = [5, 4, 3, 2, 1] if reverse else [1, 2, 3, 4, 5]
            return pd.qcut(ranks, q=5, labels=labels).astype(int)
        except ValueError:
            # 唯一值太少分不了 5 档 → 退化为等距分箱
            labels = [5, 4, 3, 2, 1] if reverse else [1, 2, 3, 4, 5]
            return pd.cut(series, bins=min(5, max(1, series.nunique())),
                          labels=labels[:min(5, max(1, series.nunique()))],
                          include_lowest=True, duplicates="drop").astype(int)

    rfm["R_score"] = _score(rfm["R"], reverse=True)   # R 越小越好
    rfm["F_score"] = _score(rfm["F"])
    rfm["M_score"] = _score(rfm["M"])
    rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]

    def _segment(total: int) -> str:
        if total >= 12:
            return "重要价值客户"
        if total >= 9:
            return "重要发展客户"
        if total >= 6:
            return "一般客户"
        return "低价值客户"

    rfm["segment"] = rfm["RFM_score"].apply(_segment)
    return [{"customer": str(r[cust_col]), "R": int(r["R"]), "F": int(r["F"]),
             "M": float(r["M"]), "rfm_score": int(r["RFM_score"]),
             "segment": r["segment"]}
            for _, r in rfm.iterrows()]


def segment_counts(df: pd.DataFrame,
                   column_map: Optional[Dict] = None) -> Dict[str, int]:
    """各客户分群的人数。"""
    rows = rfm_segments(df, column_map=column_map)
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["segment"]] = counts.get(r["segment"], 0) + 1
    return counts
