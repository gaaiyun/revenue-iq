"""CLI 端到端冒烟测试 —— 每个子命令都要真跑通（exit 0 + 有输出）。

重点回归：products 子命令曾因调用不存在的 get_top_sellers /
get_low_stock_products 而崩溃，现在用真实方法 get_sales_ranking /
get_inventory_status / get_low_stock，必须正常返回。
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# __main__.py 不是普通可 import 的模块名，单独按路径加载
_spec = importlib.util.spec_from_file_location("revenue_iq_cli", ROOT / "__main__.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)

SALES = str(ROOT / "sample_data" / "sample_sales.csv")
ORDERS = str(ROOT / "sample_data" / "orders.csv")
PRODUCTS = str(ROOT / "sample_data" / "products.csv")
CAMPAIGNS = str(ROOT / "sample_data" / "campaigns.csv")


def _run(capsys, argv) -> str:
    rc = cli.main(argv)
    assert rc == 0, f"{argv} 退出码非 0：{rc}"
    out = capsys.readouterr().out
    assert out.strip(), f"{argv} 没有任何输出"
    return out


# --- 销售流水（中文列）子命令 --------------------------------------------

def test_cli_summary(capsys):
    out = _run(capsys, ["summary", SALES])
    data = json.loads(out)
    assert data["total_orders"] > 0
    assert "total_revenue" in data


def test_cli_trend(capsys):
    out = _run(capsys, ["trend", SALES, "--period", "M"])
    assert "growth%" in out


def test_cli_anomalies(capsys):
    out = _run(capsys, ["anomalies", SALES, "--z", "1.5"])
    assert "spike" in out or "drop" in out or "没检测到" in out


def test_cli_breakdown(capsys):
    out = _run(capsys, ["breakdown", SALES])
    assert "品类分解" in out and "区域分解" in out


def test_cli_segments(capsys):
    out = _run(capsys, ["segments", SALES])
    data = json.loads(out)
    assert set(data["abc_summary"].keys()) == {"A", "B", "C"}
    assert sum(data["rfm_segment_counts"].values()) > 0


def test_cli_forecast_moving_average(capsys):
    out = _run(capsys, ["forecast", SALES, "--method", "moving_average",
                        "--periods", "5"])
    data = json.loads(out)
    assert data["method"] == "moving_average"
    assert len(data["forecast"]) == 5


def test_cli_forecast_trend(capsys):
    out = _run(capsys, ["forecast", SALES, "--method", "trend",
                        "--freq", "M", "--periods", "3"])
    data = json.loads(out)
    assert data["method"] == "linear_trend"
    assert "slope" in data


def test_cli_commentary_markdown(capsys):
    out = _run(capsys, ["commentary", SALES])
    assert "## 整体概览" in out


def test_cli_commentary_json(capsys):
    out = _run(capsys, ["commentary", SALES, "--format", "json"])
    data = json.loads(out)
    assert data["backend"] == "heuristic"
    assert "overview" in data


# --- 电商三件套（英文列）子命令 ------------------------------------------

def test_cli_overview(capsys):
    out = _run(capsys, ["overview", "--orders", ORDERS])
    data = json.loads(out)
    assert data["n_orders"] == 50
    assert data["total_revenue"] > 0


def test_cli_products_regression(capsys):
    """曾经崩溃的 products 子命令 —— 现在必须跑通。"""
    out = _run(capsys, ["products", "--orders", ORDERS,
                        "--products", PRODUCTS, "--top-n", "5"])
    data = json.loads(out)
    assert "top_sellers" in data
    assert len(data["top_sellers"]) <= 5
    assert "category_performance" in data
    assert "low_stock" in data            # 传了 products → 有库存告警字段
    assert "restock_recommendations" in data


def test_cli_products_without_products_csv(capsys):
    """不传 --products 时只算畅销 / 品类，不应崩。"""
    out = _run(capsys, ["products", "--orders", ORDERS])
    data = json.loads(out)
    assert "top_sellers" in data
    assert "low_stock" not in data        # 没 products → 无库存字段


def test_cli_marketing(capsys):
    out = _run(capsys, ["marketing", "--campaigns", CAMPAIGNS])
    data = json.loads(out)
    assert "campaign_roi" in data
    assert "channel_performance" in data
    assert data["funnel"]["impressions"] > 0


def test_cli_retention(capsys):
    out = _run(capsys, ["retention", "--orders", ORDERS])
    data = json.loads(out)
    assert data["total_customers"] > 0
    assert "repeat_rate_pct" in data


def test_cli_list_models(capsys):
    out = _run(capsys, ["list-models"])
    assert "backend" in out and "deepseek" in out


# --- 导出文件 -------------------------------------------------------------

def test_cli_output_file(capsys, tmp_path):
    out_file = tmp_path / "kpi.json"
    _run(capsys, ["overview", "--orders", ORDERS, "-o", str(out_file)])
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["n_orders"] == 50
