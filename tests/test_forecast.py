"""
预测模块单元测试
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecast import SalesForecaster, compare_forecasts
from data_loader import create_sample_sales_data


class TestSalesForecaster:
    """销售预测器测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        return create_sample_sales_data()
    
    @pytest.fixture
    def time_series_data(self):
        """创建时间序列测试数据"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        values = 100 + np.random.randn(100).cumsum()
        return pd.DataFrame({
            '日期': dates,
            '销售额': values
        })
    
    def test_init(self, time_series_data):
        """测试初始化"""
        forecaster = SalesForecaster(time_series_data)
        
        assert forecaster.data is not None
        assert forecaster.date_col == '日期'
        assert forecaster.value_col == '销售额'
    
    def test_prepare_time_series(self, time_series_data):
        """测试时间序列准备"""
        forecaster = SalesForecaster(time_series_data)
        ts = forecaster.prepare_time_series(freq='D')
        
        assert isinstance(ts, pd.Series)
        assert len(ts) == 100
        assert not ts.isnull().any()
    
    def test_moving_average_forecast(self, time_series_data):
        """测试移动平均预测"""
        forecaster = SalesForecaster(time_series_data)
        result = forecaster.moving_average_forecast(window=7, periods=30)
        
        assert 'method' in result
        assert result['method'] == '移动平均'
        assert 'forecast_dates' in result
        assert 'forecast_values' in result
        assert 'figure' in result
        assert len(result['forecast_values']) == 30
    
    def test_moving_average_window(self, time_series_data):
        """测试不同移动平均窗口"""
        forecaster = SalesForecaster(time_series_data)
        
        result_7 = forecaster.moving_average_forecast(window=7, periods=10)
        result_14 = forecaster.moving_average_forecast(window=14, periods=10)
        
        assert result_7['window'] == 7
        assert result_14['window'] == 14
    
    def test_exponential_smoothing_forecast(self, time_series_data):
        """测试指数平滑预测"""
        forecaster = SalesForecaster(time_series_data)
        result = forecaster.exponential_smoothing_forecast(alpha=0.3, periods=30)
        
        assert 'method' in result
        assert result['method'] == '指数平滑'
        assert result['alpha'] == 0.3
        assert len(result['forecast_values']) == 30
    
    def test_exponential_smoothing_alpha(self, time_series_data):
        """测试不同平滑系数"""
        forecaster = SalesForecaster(time_series_data)
        
        result_01 = forecaster.exponential_smoothing_forecast(alpha=0.1, periods=10)
        result_09 = forecaster.exponential_smoothing_forecast(alpha=0.9, periods=10)
        
        assert result_01['alpha'] == 0.1
        assert result_09['alpha'] == 0.9
    
    def test_trend_projection(self, time_series_data):
        """测试趋势外推预测"""
        forecaster = SalesForecaster(time_series_data)
        result = forecaster.trend_projection(periods=30)
        
        assert 'method' in result
        assert result['method'] == '线性趋势'
        assert 'forecast_dates' in result
        assert 'forecast_values' in result
        assert 'slope' in result
        assert 'intercept' in result
        assert len(result['forecast_values']) == 30
    
    def test_trend_slope(self, time_series_data):
        """测试趋势斜率"""
        forecaster = SalesForecaster(time_series_data)
        result = forecaster.trend_projection(periods=30)
        
        # 斜率应该是数值
        assert isinstance(result['slope'], (int, float, np.number))
    
    def test_arima_forecast_missing_statsmodels(self, time_series_data):
        """测试 ARIMA 预测（无 statsmodels）"""
        forecaster = SalesForecaster(time_series_data)
        result = forecaster.arima_forecast(periods=30)
        
        # 如果没有 statsmodels，应该返回空字典或警告
        # 这个测试主要验证函数不会崩溃
        assert isinstance(result, dict)
    
    def test_evaluate_forecast_trend(self, time_series_data):
        """测试趋势预测评估"""
        forecaster = SalesForecaster(time_series_data)
        metrics = forecaster.evaluate_forecast(method='trend', train_ratio=0.8)
        
        if metrics:  # 如果有返回结果
            assert 'MAE' in metrics
            assert 'RMSE' in metrics
            assert 'MAPE' in metrics
            assert 'R2' in metrics
    
    def test_evaluate_forecast_ma(self, time_series_data):
        """测试移动平均预测评估"""
        forecaster = SalesForecaster(time_series_data)
        metrics = forecaster.evaluate_forecast(method='ma', train_ratio=0.8)
        
        if metrics:
            assert 'MAE' in metrics
            assert 'RMSE' in metrics
    
    def test_get_forecast_summary(self, time_series_data):
        """测试预测汇总"""
        forecaster = SalesForecaster(time_series_data)
        
        # 先运行一些预测
        forecaster.moving_average_forecast(periods=10)
        forecaster.trend_projection(periods=10)
        
        summary = forecaster.get_forecast_summary()
        
        assert 'methods_available' in summary
        assert 'data_range' in summary
        assert 'total_records' in summary
        assert len(summary['methods_available']) == 2
    
    def test_compare_forecasts(self, time_series_data):
        """测试多方法对比"""
        fig = compare_forecasts(time_series_data, periods=30)
        
        assert fig is not None
        # Plotly 图表应该有数据轨迹
        assert len(fig.data) > 0
    
    def test_forecast_with_missing_dates(self):
        """测试含缺失日期的数据"""
        # 创建含日期缺失的数据
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        # 删除一些日期
        dates = dates.delete([10, 20, 30])
        
        df = pd.DataFrame({
            '日期': dates,
            '销售额': np.random.randn(len(dates)).cumsum() + 100
        })
        
        forecaster = SalesForecaster(df)
        ts = forecaster.prepare_time_series(freq='D')
        
        # 应该填充了缺失的日期
        assert len(ts) > len(df)
    
    def test_forecast_empty_data(self):
        """测试空数据预测"""
        df = pd.DataFrame(columns=['日期', '销售额'])
        forecaster = SalesForecaster(df)
        
        # 应该能处理空数据而不崩溃
        result = forecaster.moving_average_forecast(periods=10)
        # 结果可能为空或包含空值
    
    def test_forecast_missing_columns(self):
        """测试缺少必要列的预测"""
        df = pd.DataFrame({'A': [1, 2, 3]})
        forecaster = SalesForecaster(df, date_col='日期', value_col='销售额')
        
        # 应该能处理缺少列的情况
        try:
            ts = forecaster.prepare_time_series()
        except (KeyError, Exception):
            # 预期会抛出异常
            pass
    
    def test_forecast_results_storage(self, time_series_data):
        """测试预测结果存储"""
        forecaster = SalesForecaster(time_series_data)
        
        # 执行多个预测
        forecaster.moving_average_forecast(periods=10)
        forecaster.exponential_smoothing_forecast(periods=10)
        forecaster.trend_projection(periods=10)
        
        # 检查结果存储
        assert 'ma' in forecaster.forecast_results
        assert 'es' in forecaster.forecast_results
        assert 'trend' in forecaster.forecast_results
    
    def test_different_frequencies(self, time_series_data):
        """测试不同频率"""
        forecaster = SalesForecaster(time_series_data)
        
        # 测试不同频率
        ts_daily = forecaster.prepare_time_series(freq='D')
        ts_weekly = forecaster.prepare_time_series(freq='W')
        ts_monthly = forecaster.prepare_time_series(freq='M')
        
        assert len(ts_daily) >= len(ts_weekly)
        assert len(ts_weekly) >= len(ts_monthly)
    
    def test_forecast_values_type(self, time_series_data):
        """测试预测值类型"""
        forecaster = SalesForecaster(time_series_data)
        result = forecaster.moving_average_forecast(periods=10)
        
        # 预测值应该是数值数组
        assert isinstance(result['forecast_values'], (np.ndarray, list))
        assert all(isinstance(v, (int, float, np.number)) for v in result['forecast_values'])
    
    def test_forecast_dates_type(self, time_series_data):
        """测试预测日期类型"""
        forecaster = SalesForecaster(time_series_data)
        result = forecaster.moving_average_forecast(periods=10)
        
        # 预测日期应该是 DatetimeIndex
        assert isinstance(result['forecast_dates'], pd.DatetimeIndex)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
