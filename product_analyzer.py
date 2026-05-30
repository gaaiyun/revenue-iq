"""商品分析（英文列 schema）。

面向 orders + products 两张表：销量排名、利润率、品类表现、库存状态、
补货建议、价格建议、竞品对照。纯 pandas，不依赖 Streamlit / plotly。

订单表预期列：order_id, product_id, category, quantity, revenue, profit；
产品表预期列：product_id, product_name, category, stock_quantity, reorder_level。
revenue / profit 由 ``ecom_data_prep.prepare_orders`` 预先补好。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class ProductAnalyzer:
    """商品分析器。

    products_df 可选：只传 orders 时，库存 / 补货 / 价格类方法不可用，
    但销量排名 / 品类表现仍可算。
    """

    def __init__(self, orders_df: pd.DataFrame,
                 products_df: Optional[pd.DataFrame] = None):
        self.orders_df = orders_df
        self.products_df = products_df

    def _require_products(self, method: str) -> pd.DataFrame:
        if self.products_df is None:
            raise ValueError(f"{method} 需要 products 数据（含库存 / 价格列）")
        return self.products_df

    def get_sales_ranking(self, top_n: int = 10) -> pd.DataFrame:
        """销量 / 营收排名（按营收降序）。"""
        if len(self.orders_df) == 0:
            return self.orders_df.iloc[0:0]
        sales = self.orders_df.groupby("product_id").agg(
            quantity=("quantity", "sum"),
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
        ).reset_index()
        sales["product_id"] = sales["product_id"].astype(str)

        if self.products_df is not None:
            products_copy = self.products_df.copy()
            products_copy["product_id"] = products_copy["product_id"].astype(str)
            keep = [c for c in ("product_id", "product_name", "category")
                    if c in products_copy.columns]
            sales = sales.merge(products_copy[keep], on="product_id", how="left")

        return sales.sort_values("revenue", ascending=False).head(top_n)

    def get_profit_analysis(self) -> pd.DataFrame:
        """利润分析：按产品聚合利润 + 利润率。"""
        profit = self.orders_df.groupby("product_id").agg(
            profit=("profit", "sum"),
            revenue=("revenue", "sum"),
            quantity=("quantity", "sum"),
        ).reset_index()
        profit["profit_margin"] = (
            profit["profit"] / profit["revenue"].replace(0, np.nan) * 100
        ).round(2)
        profit["product_id"] = profit["product_id"].astype(str)

        if self.products_df is not None:
            products_copy = self.products_df.copy()
            products_copy["product_id"] = products_copy["product_id"].astype(str)
            keep = [c for c in ("product_id", "product_name", "category")
                    if c in products_copy.columns]
            profit = profit.merge(products_copy[keep], on="product_id", how="left")

        return profit.sort_values("profit", ascending=False)

    def get_category_performance(self) -> pd.DataFrame:
        """品类表现：按 category 聚合销量 / 营收 / 利润 / 订单数 / 客单价。"""
        category_perf = self.orders_df.groupby("category").agg(
            quantity=("quantity", "sum"),
            revenue=("revenue", "sum"),
            profit=("profit", "sum"),
            order_count=("order_id", "count"),
        ).reset_index()
        category_perf["avg_order_value"] = (
            category_perf["revenue"] / category_perf["order_count"]
        ).round(2)
        return category_perf.sort_values("revenue", ascending=False)

    def get_inventory_status(self) -> pd.DataFrame:
        """库存状态：低库存 / 正常 / 充足 + 周转率。"""
        inventory = self._require_products("get_inventory_status").copy()
        inventory["stock_status"] = inventory.apply(
            lambda row: "低库存" if row["stock_quantity"] <= row["reorder_level"]
            else ("充足" if row["stock_quantity"] > row["reorder_level"] * 2
                  else "正常"),
            axis=1,
        )

        sales_qty = (self.orders_df.groupby("product_id")["quantity"].sum()
                     .reset_index().rename(columns={"quantity": "sold_qty"}))
        inventory["product_id"] = inventory["product_id"].astype(str)
        sales_qty["product_id"] = sales_qty["product_id"].astype(str)

        inventory = inventory.merge(sales_qty, on="product_id", how="left")
        inventory["sold_qty"] = inventory["sold_qty"].fillna(0)
        inventory["turnover_rate"] = (
            inventory["sold_qty"] / inventory["stock_quantity"].replace(0, np.nan) * 100
        ).round(2)
        return inventory

    def get_low_stock(self) -> pd.DataFrame:
        """只返回低库存（stock <= reorder_level）的商品。"""
        inv = self.get_inventory_status()
        return inv[inv["stock_status"] == "低库存"]

    def get_restock_recommendations(self, days_span: int = 25) -> pd.DataFrame:
        """补货建议：基于日均销量估算剩余天数 + 建议补货量。"""
        inventory = self.get_inventory_status()
        span = max(days_span, 1)
        inventory["daily_sales"] = (inventory["sold_qty"] / span).round(1)
        daily = inventory["daily_sales"].replace(0, np.nan)
        inventory["days_remaining"] = (
            inventory["stock_quantity"] / daily
        ).round(1)

        def _reco(row) -> str:
            if row["stock_quantity"] <= row["reorder_level"]:
                return f"立即补货，建议补货量 {int(row['daily_sales'] * 15)}"
            if pd.notna(row["days_remaining"]) and row["days_remaining"] < 10:
                return f"准备补货，预计 {int(row['days_remaining'])} 天后库存不足"
            return "库存充足"

        inventory["recommendation"] = inventory.apply(_reco, axis=1)
        return inventory[inventory["stock_quantity"] <= inventory["reorder_level"] * 1.5]

    def get_price_optimization_suggestions(self) -> pd.DataFrame:
        """价格优化建议：按利润率给提价 / 促销 / 维持建议。"""
        self._require_products("get_price_optimization_suggestions")
        profit_analysis = self.get_profit_analysis()

        def _suggest(row) -> str:
            margin = row["profit_margin"]
            if pd.isna(margin):
                return "数据不足"
            if margin < 30:
                return f"建议提价 {int((35 - margin) / 10 * 5)}% 或优化成本"
            if margin > 60:
                return "利润率优秀，可考虑促销扩大销量"
            return "价格策略合理"

        profit_analysis["price_suggestion"] = profit_analysis.apply(_suggest, axis=1)
        cols = [c for c in ("product_id", "product_name", "profit_margin",
                            "price_suggestion") if c in profit_analysis.columns]
        return profit_analysis[cols]
