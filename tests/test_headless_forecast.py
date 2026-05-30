"""headless_forecast.py 测试 —— 纯 numpy/pandas 预测，无 Streamlit/plotly。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from headless_forecast import (
    ForecastResult, moving_average_forecast, trend_forecast,
)


@pytest.fixture
def sales_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    rng = np.random.RandomState(7)
    return pd.DataFrame({
        "日期": dates,
        "销售额": np.linspace(1000, 3000, 60) + rng.normal(0, 50, 60),
    })


# --- moving_average_forecast ----------------------------------------------

def test_moving_average_count(sales_df):
    res = moving_average_forecast(sales_df, periods=7, window=7, freq="D")
    assert isinstance(res, ForecastResult)
    assert res.method == "moving_average"
    assert len(res.forecast) == 7


def test_moving_average_flat(sales_df):
    res = moving_average_forecast(sales_df, periods=5, window=7)
    values = [p.value for p in res.forecast]
    # 移动平均外推应是常数
    assert len(set(round(v, 6) for v in values)) == 1


def test_moving_average_future_dates_after_history(sales_df):
    res = moving_average_forecast(sales_df, periods=3, freq="D")
    first = pd.to_datetime(res.forecast[0].period)
    assert first > pd.Timestamp("2024-02-29")    # 60 天后 = 3/1 起


def test_moving_average_empty():
    res = moving_average_forecast(pd.DataFrame({"日期": [], "销售额": []}))
    assert res.history_periods == 0
    assert res.forecast == []


# --- trend_forecast -------------------------------------------------------

def test_trend_slope_positive(sales_df):
    res = trend_forecast(sales_df, periods=7, freq="D")
    assert res.method == "linear_trend"
    assert res.slope is not None and res.slope > 0


def test_trend_forecast_count(sales_df):
    res = trend_forecast(sales_df, periods=10, freq="D")
    assert len(res.forecast) == 10


def test_trend_values_non_negative(sales_df):
    res = trend_forecast(sales_df, periods=7)
    assert all(p.value >= 0 for p in res.forecast)


def test_trend_too_short():
    df = pd.DataFrame({"日期": ["2024-01-01"], "销售额": [100.0]})
    res = trend_forecast(df, periods=5)
    assert res.forecast == []


# --- to_dict --------------------------------------------------------------

def test_forecast_to_dict_serializable(sales_df):
    import json
    res = trend_forecast(sales_df, periods=3)
    payload = res.to_dict()
    json.dumps(payload, ensure_ascii=False)
    assert "slope" in payload
    assert len(payload["forecast"]) == 3


def test_monthly_frequency(sales_df):
    res = trend_forecast(sales_df, periods=2, freq="M")
    assert res.freq == "M"
    assert len(res.forecast) == 2
