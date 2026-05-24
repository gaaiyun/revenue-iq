"""Sales-Dashboard CLI（v2）。

子命令：
    summary <csv>      整体统计：总收入 / 订单数 / 客户数 / 首末月增长
    trend <csv>        按周期聚合：D / W / M / Q / Y
    anomalies <csv>    异常 spike / drop 检测
    breakdown <csv>    品类 / 区域分解
    commentary <csv>   LLM 或规则自然语言报告（综合以上）
    list-models        列 LLM backend
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
from llm_commentary import LLMClient, commentary as gen_commentary


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def cmd_summary(args) -> int:
    df = _load_csv(args.csv)
    summary = compute_summary(df)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
    return 0


def cmd_trend(args) -> int:
    df = _load_csv(args.csv)
    points = compute_trend(df, period=args.period)
    # 渲染表
    print(f"{'period':<12} {'revenue':>14} {'orders':>8} {'growth%':>9}")
    print("-" * 50)
    for p in points:
        g_str = f"{p.growth_pct:+.1f}%" if p.growth_pct is not None else "-"
        print(f"{p.period:<12} {p.revenue:>14,.0f} {p.n_orders:>8} {g_str:>9}")

    if args.output:
        payload = [{"period": p.period, "revenue": p.revenue,
                    "n_orders": p.n_orders, "growth_pct": p.growth_pct}
                   for p in points]
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def cmd_anomalies(args) -> int:
    df = _load_csv(args.csv)
    alerts = detect_anomalies(df, period=args.period, z_threshold=args.z)
    if not alerts:
        print("没检测到异常 spike/drop")
        return 0
    print(f"{'period':<12} {'actual':>14} {'expected':>14} {'z':>8} dir")
    print("-" * 60)
    for a in alerts:
        print(f"{a.period:<12} {a.actual:>14,.0f} {a.expected:>14,.0f} "
              f"{a.z_score:>+8.2f} {a.direction}")
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
    print("【品类分解】")
    for k, v in cat.items():
        print(f"  {k:<12} CNY {v:>14,.0f}")
    print()
    print("【区域分解】")
    for k, v in reg.items():
        print(f"  {k:<12} CNY {v:>14,.0f}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps({"category": cat, "region": reg},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
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
            "退化为规则启发式\n"
        )

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

    if args.format == "json":
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(report.to_markdown())

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        content = (json.dumps(report.to_dict(), ensure_ascii=False, indent=2)
                   if args.format == "json" else report.to_markdown())
        Path(args.output).write_text(content, encoding="utf-8")
    return 0


def cmd_list_models(args) -> int:
    import os
    rows = [
        ("openai", "gpt-4o-mini", "OPENAI_API_KEY"),
        ("anthropic", "claude-3-5-haiku-20241022", "ANTHROPIC_API_KEY"),
        ("deepseek", "deepseek-chat", "DEEPSEEK_API_KEY"),
    ]
    print(f"{'backend':<12} {'default model':<32} configured")
    print("-" * 60)
    for b, m, e in rows:
        print(f"{b:<12} {m:<32} {'yes' if os.getenv(e) else 'no'}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sales", description="销售数据 headless 分析 + LLM commentary"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, help_text, fn in [
        ("summary", "整体统计", cmd_summary),
        ("breakdown", "品类 + 区域分解", cmd_breakdown),
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

    sp = sub.add_parser("commentary", help="LLM/规则自然语言报告")
    sp.add_argument("csv")
    sp.add_argument("--use-llm", action="store_true")
    sp.add_argument("--backend", default="deepseek",
                    choices=["openai", "anthropic", "deepseek"])
    sp.add_argument("--format", default="markdown", choices=["markdown", "json"])
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_commentary)

    sp = sub.add_parser("list-models", help="列 LLM backend 配置状态")
    sp.set_defaults(func=cmd_list_models)
    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
