"""revenue-iq 统一 CLI。

把订单 CSV 一条龙跑成经营洞察。两套数据 schema、两组子命令：

销售流水（中文列：日期 / 销售额 / 产品类别 / 地区 / 客户 ID）
    summary       整体统计：营收 / 订单 / 客户 / 首末月增长
    trend         按周期聚合：D / W / M / Q / Y + 环比
    anomalies     z-score 异常 spike / drop 检测
    breakdown     品类 / 区域分解
    segments      ABC 帕累托 + RFM 客户分群
    forecast      移动平均 / 线性趋势销量预测
    commentary    LLM 或规则中文经营月报（综合以上）

电商三件套（英文列：orders / products / campaigns）
    overview      整体 KPI：营收 / 利润 / 利润率 / 客单价
    products      畅销榜 + 品类表现 + 库存告警 + 补货建议
    marketing     渠道 ROI + 转化率 + 预算分配建议
    retention     复购率 / 人均订单数

    list-models   列 LLM backend 配置状态
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from headless_analytics import (
    category_breakdown, compute_summary, compute_trend,
    detect_anomalies, region_breakdown,
)
from headless_forecast import moving_average_forecast, trend_forecast
from headless_segments import abc_analysis, abc_summary, rfm_segments, segment_counts
from llm_commentary import LLMClient, commentary as gen_commentary
from ecom_data_prep import (
    load_campaigns, load_orders, load_products, overview_metrics, repeat_purchase,
)
from product_analyzer import ProductAnalyzer
from marketing_analyzer import MarketingAnalyzer


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _emit(payload, args, *, is_json: bool = True) -> None:
    """打印 + 可选写文件。payload 已经是 str（文本）或可序列化对象。"""
    text = payload if isinstance(payload, str) else json.dumps(
        payload, ensure_ascii=False, indent=2)
    print(text)
    if getattr(args, "output", None):
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")


def _df_records(df: pd.DataFrame) -> list:
    """DataFrame → JSON 友好 list（datetime 转字符串，NaN 转 None）。"""
    if df is None or len(df) == 0:
        return []
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d")
    return json.loads(df.to_json(orient="records", force_ascii=False))


# --- 销售流水（中文列）子命令 ------------------------------------------------

def cmd_summary(args) -> int:
    summary = compute_summary(_load_csv(args.csv))
    _emit(summary.to_dict(), args)
    return 0


def cmd_trend(args) -> int:
    points = compute_trend(_load_csv(args.csv), period=args.period)
    lines = [f"{'period':<12} {'revenue':>14} {'orders':>8} {'growth%':>9}",
             "-" * 50]
    for p in points:
        g = f"{p.growth_pct:+.1f}%" if p.growth_pct is not None else "-"
        lines.append(f"{p.period:<12} {p.revenue:>14,.0f} {p.n_orders:>8} {g:>9}")
    print("\n".join(lines))
    if args.output:
        payload = [{"period": p.period, "revenue": p.revenue,
                    "n_orders": p.n_orders, "growth_pct": p.growth_pct}
                   for p in points]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def cmd_anomalies(args) -> int:
    alerts = detect_anomalies(_load_csv(args.csv), period=args.period,
                              z_threshold=args.z)
    if not alerts:
        print("没检测到异常 spike/drop")
    else:
        lines = [f"{'period':<12} {'actual':>14} {'expected':>14} {'z':>8} dir",
                 "-" * 60]
        for a in alerts:
            lines.append(f"{a.period:<12} {a.actual:>14,.0f} {a.expected:>14,.0f} "
                         f"{a.z_score:>+8.2f} {a.direction}")
        print("\n".join(lines))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps([a.to_dict() for a in alerts], ensure_ascii=False, indent=2),
            encoding="utf-8")
    return 0


def cmd_breakdown(args) -> int:
    df = _load_csv(args.csv)
    cat = category_breakdown(df)
    reg = region_breakdown(df)
    lines = ["【品类分解】"]
    lines += [f"  {k:<12} CNY {v:>14,.0f}" for k, v in cat.items()]
    lines += ["", "【区域分解】"]
    lines += [f"  {k:<12} CNY {v:>14,.0f}" for k, v in reg.items()]
    print("\n".join(lines))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps({"category": cat, "region": reg}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    return 0


def cmd_segments(args) -> int:
    df = _load_csv(args.csv)
    payload = {
        "abc_summary": abc_summary(df),
        "abc_detail": abc_analysis(df),
        "rfm_segment_counts": segment_counts(df),
        "rfm_detail": rfm_segments(df),
    }
    _emit(payload, args)
    return 0


def cmd_forecast(args) -> int:
    df = _load_csv(args.csv)
    if args.method == "trend":
        result = trend_forecast(df, periods=args.periods, freq=args.freq)
    else:
        result = moving_average_forecast(df, periods=args.periods,
                                         window=args.window, freq=args.freq)
    _emit(result.to_dict(), args)
    return 0


def cmd_commentary(args) -> int:
    df = _load_csv(args.csv)
    summary = compute_summary(df)
    trend = compute_trend(df, period="M")
    alerts = detect_anomalies(df, period="D", z_threshold=2.0)
    cat = category_breakdown(df)
    reg = region_breakdown(df)

    client = LLMClient(backend=args.backend) if args.use_llm else None
    if args.use_llm and client and not client.is_available():
        sys.stderr.write(
            f"[warn] --use-llm 但 {args.backend.upper()}_API_KEY 未配，"
            "退化为规则启发式\n")

    report = gen_commentary(
        summary=summary.to_dict(),
        trend=[{"period": p.period, "revenue": p.revenue,
                "n_orders": p.n_orders, "growth_pct": p.growth_pct}
               for p in trend],
        anomalies=[a.to_dict() for a in alerts],
        category_breakdown=cat,
        region_breakdown=reg,
        llm_client=client,
    )
    _emit(report.to_dict() if args.format == "json" else report.to_markdown(),
          args, is_json=(args.format == "json"))
    return 0


# --- 电商三件套（英文列）子命令 ----------------------------------------------

def cmd_overview(args) -> int:
    _emit(overview_metrics(load_orders(args.orders)), args)
    return 0


def cmd_products(args) -> int:
    """畅销榜 + 品类表现 + 库存告警 + 补货建议。

    修复了原 Ecommerce CLI 调不存在方法的 bug：
    get_top_sellers / get_low_stock_products → get_sales_ranking /
    get_inventory_status / get_low_stock。
    """
    orders = load_orders(args.orders)
    products = load_products(args.products) if args.products else None
    analyzer = ProductAnalyzer(orders, products)

    payload = {
        "category_performance": _df_records(analyzer.get_category_performance()),
        "top_sellers": _df_records(analyzer.get_sales_ranking(top_n=args.top_n)),
    }
    if products is not None:
        payload["low_stock"] = _df_records(analyzer.get_low_stock())
        payload["restock_recommendations"] = _df_records(
            analyzer.get_restock_recommendations())
    _emit(payload, args)
    return 0


def cmd_marketing(args) -> int:
    orders = load_orders(args.orders) if args.orders else None
    campaigns = load_campaigns(args.campaigns)
    analyzer = MarketingAnalyzer(campaigns, orders)
    payload = {
        "campaign_roi": _df_records(analyzer.get_campaign_roi()),
        "conversion": _df_records(analyzer.get_conversion_metrics()),
        "channel_performance": _df_records(analyzer.get_channel_performance()),
        "funnel": analyzer.get_conversion_funnel(),
        "budget_allocation": _df_records(analyzer.get_budget_allocation_suggestions()),
    }
    _emit(payload, args)
    return 0


def cmd_retention(args) -> int:
    _emit(repeat_purchase(load_orders(args.orders)), args)
    return 0


def cmd_list_models(args) -> int:
    import os
    rows = [
        ("openai", "gpt-4o-mini", "OPENAI_API_KEY"),
        ("anthropic", "claude-3-5-haiku-20241022", "ANTHROPIC_API_KEY"),
        ("deepseek", "deepseek-chat", "DEEPSEEK_API_KEY"),
    ]
    lines = [f"{'backend':<12} {'default model':<32} configured", "-" * 60]
    for b, m, e in rows:
        lines.append(f"{b:<12} {m:<32} {'yes' if os.getenv(e) else 'no'}")
    print("\n".join(lines))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="revenue-iq",
        description="订单 CSV → KPI / 品类 ABC / 库存告警 / 渠道 ROI / 销量预测 / 经营月报",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # 销售流水（中文列）
    for name, help_text, fn in [
        ("summary", "整体统计", cmd_summary),
        ("breakdown", "品类 + 区域分解", cmd_breakdown),
        ("segments", "ABC 帕累托 + RFM 客户分群", cmd_segments),
    ]:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("csv")
        sp.add_argument("-o", "--output")
        sp.set_defaults(func=fn)

    sp = sub.add_parser("trend", help="按周期聚合（D/W/M/Q/Y）")
    sp.add_argument("csv")
    sp.add_argument("--period", default="M", choices=["D", "W", "M", "Q", "Y"])
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_trend)

    sp = sub.add_parser("anomalies", help="z-score 异常检测")
    sp.add_argument("csv")
    sp.add_argument("--period", default="D", choices=["D", "W", "M"])
    sp.add_argument("--z", type=float, default=2.0)
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_anomalies)

    sp = sub.add_parser("forecast", help="移动平均 / 线性趋势销量预测")
    sp.add_argument("csv")
    sp.add_argument("--method", default="moving_average",
                    choices=["moving_average", "trend"])
    sp.add_argument("--periods", type=int, default=7)
    sp.add_argument("--window", type=int, default=7)
    sp.add_argument("--freq", default="D", choices=["D", "W", "M", "Q", "Y"])
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_forecast)

    sp = sub.add_parser("commentary", help="LLM/规则中文经营月报")
    sp.add_argument("csv")
    sp.add_argument("--use-llm", action="store_true")
    sp.add_argument("--backend", default="deepseek",
                    choices=["openai", "anthropic", "deepseek"])
    sp.add_argument("--format", default="markdown", choices=["markdown", "json"])
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_commentary)

    # 电商三件套（英文列）
    sp = sub.add_parser("overview", help="电商整体 KPI（orders）")
    sp.add_argument("--orders", required=True)
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_overview)

    sp = sub.add_parser("products", help="畅销榜 / 品类 / 库存告警 / 补货")
    sp.add_argument("--orders", required=True)
    sp.add_argument("--products", help="products CSV（含库存列才能查库存告警）")
    sp.add_argument("--top-n", type=int, default=10)
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_products)

    sp = sub.add_parser("marketing", help="渠道 ROI / 转化率 / 预算分配")
    sp.add_argument("--campaigns", required=True)
    sp.add_argument("--orders", help="orders CSV（可选，用于后续归因）")
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_marketing)

    sp = sub.add_parser("retention", help="复购率 / 人均订单数")
    sp.add_argument("--orders", required=True)
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_retention)

    sp = sub.add_parser("list-models", help="列 LLM backend 配置状态")
    sp.set_defaults(func=cmd_list_models)
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
