"""电商订单数据预处理（英文列 schema）。

与 ``headless_analytics`` 面向的中文列销售流水不同，这套面向电商三件套：
orders / products / campaigns（英文列）。把原本散在 UI 加载逻辑里的
revenue / cost / profit 计算抽成幂等纯函数，供 CLI 的 products / marketing /
retention 子命令复用。

orders 必需列：quantity, unit_price；可选 cost_price / cost / order_date /
customer_id。revenue = quantity * unit_price；profit = revenue - cost。
"""
from __future__ import annotations

import pandas as pd


def load_orders(path: str) -> pd.DataFrame:
    """读订单 CSV 并补 revenue / cost / profit。"""
    return prepare_orders(pd.read_csv(path))


def prepare_orders(df: pd.DataFrame) -> pd.DataFrame:
    """补 revenue / cost / profit 列；幂等（已存在不覆盖）。"""
    df = df.copy()
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    if "revenue" not in df.columns:
        df["revenue"] = df["quantity"] * df["unit_price"]
    if "cost" not in df.columns and "cost_price" in df.columns:
        df["cost"] = df["quantity"] * df["cost_price"]
    if "profit" not in df.columns:
        if "cost" in df.columns:
            df["profit"] = df["revenue"] - df["cost"]
        elif "cost_price" in df.columns:
            df["profit"] = df["revenue"] - df["quantity"] * df["cost_price"]
        else:
            df["profit"] = df["revenue"]    # 没有成本列 → profit = revenue
    return df


def load_products(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "last_restock_date" in df.columns:
        df["last_restock_date"] = pd.to_datetime(df["last_restock_date"],
                                                 errors="coerce")
    return df


def load_campaigns(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ("start_date", "end_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def overview_metrics(orders_df: pd.DataFrame) -> dict:
    """整体 KPI：单数 / 营收 / 利润 / 利润率 / 客单价 / 客户数。"""
    if len(orders_df) == 0:
        return {"n_orders": 0, "total_revenue": 0.0, "total_profit": 0.0,
                "profit_margin_pct": 0.0, "avg_order_value": 0.0,
                "unique_customers": 0}
    n_orders = int(len(orders_df))
    total_revenue = float(orders_df["revenue"].sum())
    total_profit = float(orders_df["profit"].sum()) if "profit" in orders_df.columns else 0.0
    margin_pct = (total_profit / total_revenue * 100) if total_revenue else 0.0
    aov = total_revenue / n_orders if n_orders else 0.0
    n_customers = (int(orders_df["customer_id"].nunique())
                   if "customer_id" in orders_df.columns else 0)
    return {
        "n_orders": n_orders,
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "profit_margin_pct": float(margin_pct),
        "avg_order_value": float(aov),
        "unique_customers": n_customers,
    }


def repeat_purchase(orders_df: pd.DataFrame) -> dict:
    """复购 / 留存：复购客户占比 + 人均订单数。"""
    if len(orders_df) == 0 or "customer_id" not in orders_df.columns:
        return {"total_customers": 0, "repeat_customers": 0,
                "repeat_rate_pct": 0.0, "avg_orders_per_customer": 0.0}
    per_customer = orders_df.groupby("customer_id").size()
    total = int(len(per_customer))
    repeat = int((per_customer > 1).sum())
    return {
        "total_customers": total,
        "repeat_customers": repeat,
        "repeat_rate_pct": round(repeat / total * 100, 2) if total else 0.0,
        "avg_orders_per_customer": round(float(per_customer.mean()), 2),
    }
