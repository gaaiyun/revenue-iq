"""
销售预测模块
基于历史数据的时间序列预测
支持多种预测模型：移动平均、指数平滑、ARIMA、Prophet
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# 尝试导入 statsmodels，如果不存在则使用简单方法
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False
    st.warning("statsmodels 未安装，将使用简单预测方法")


class SalesForecaster:
    """销售预测器类"""
    
    def __init__(self, data: pd.DataFrame, date_col: str = '日期', value_col: str = '销售额'):
        self.data = data.copy()
        self.date_col = date_col
        self.value_col = value_col
        self.model = None
        self.forecast_results = {}
        
        # 预处理
        if date_col in self.data.columns:
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            self.data = self.data.sort_values(date_col)
    
    def prepare_time_series(self, freq: str = 'D') -> pd.Series:
        """
        准备时间序列数据
        
        参数:
            freq: 频率 'D' 日，'W' 周，'M' 月
        """
        df = self.data.copy()
        df.set_index(self.date_col, inplace=True)
        
        # 按频率重采样
        ts = df[self.value_col].resample(freq).sum()
        
        # 填充缺失值
        ts = ts.fillna(method='ffill').fillna(0)
        
        return ts
    
    def moving_average_forecast(self, window: int = 7, periods: int = 30) -> Dict[str, Any]:
        """
        移动平均预测
        
        参数:
            window: 移动平均窗口大小
            periods: 预测期数
        """
        ts = self.prepare_time_series()
        
        # 计算移动平均
        ma = ts.rolling(window=window).mean()
        
        # 预测 (使用最后一个移动平均值)
        if len(ma) == 0 or len(ts) == 0 or pd.isna(ma.iloc[-1]):
            last_ma = ts.mean() if len(ts) > 0 else 0  # 回退到整体均值
            # 如果数据为空，使用当前日期
            start_date = pd.Timestamp.now()
        else:
            last_ma = ma.iloc[-1]
            start_date = ts.index[-1] + pd.Timedelta(days=1)
        
        forecast_dates = pd.date_range(
            start=start_date,
            periods=periods,
            freq='D'
        )
        forecast_values = np.full(periods, last_ma)
        
        # 可视化
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='实际值'))
        fig.add_trace(go.Scatter(x=ma.index, y=ma.values, mode='lines', name=f'{window}日移动平均'))
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines',
            name='预测值',
            line=dict(dash='dash', color='red')
        ))
        
        fig.update_layout(title='移动平均预测', height=500)
        
        self.forecast_results['ma'] = {
            'method': '移动平均',
            'window': window,
            'forecast_dates': forecast_dates,
            'forecast_values': forecast_values,
            'figure': fig
        }
        
        return self.forecast_results['ma']
    
    def exponential_smoothing_forecast(self, alpha: float = 0.3, periods: int = 30) -> Dict[str, Any]:
        """
        指数平滑预测
        
        参数:
            alpha: 平滑系数 (0-1)
            periods: 预测期数
        """
        ts = self.prepare_time_series()
        
        if STATS_AVAILABLE:
            # 使用 statsmodels 的指数平滑
            model = ExponentialSmoothing(ts, trend='add', seasonal=None)
            fitted = model.fit(smoothing_level=alpha)
            forecast = fitted.forecast(periods)
            
            forecast_dates = forecast.index
            forecast_values = forecast.values
        else:
            # 简单指数平滑
            forecast_values = [ts.iloc[0]]
            for i in range(1, len(ts)):
                forecast_values.append(alpha * ts.iloc[i] + (1 - alpha) * forecast_values[-1])
            
            # 外推预测
            last_value = forecast_values[-1]
            forecast_dates = pd.date_range(
                start=ts.index[-1] + pd.Timedelta(days=1),
                periods=periods,
                freq='D'
            )
            forecast_values = np.full(periods, last_value)
        
        # 可视化
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='实际值'))
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines',
            name='预测值',
            line=dict(dash='dash', color='green')
        ))
        
        fig.update_layout(title='指数平滑预测', height=500)
        
        self.forecast_results['es'] = {
            'method': '指数平滑',
            'alpha': alpha,
            'forecast_dates': forecast_dates,
            'forecast_values': forecast_values,
            'figure': fig
        }
        
        return self.forecast_results['es']
    
    def arima_forecast(self, order: Tuple[int, int, int] = (1, 1, 1), periods: int = 30) -> Dict[str, Any]:
        """
        ARIMA 预测
        
        参数:
            order: (p, d, q) 参数
            periods: 预测期数
        """
        if not STATS_AVAILABLE:
            st.warning("statsmodels 未安装，无法使用 ARIMA")
            return {}
        
        ts = self.prepare_time_series()
        
        try:
            # 拟合 ARIMA 模型
            model = ARIMA(ts, order=order)
            fitted = model.fit()
            forecast = fitted.get_forecast(steps=periods)
            forecast_mean = forecast.predicted_mean
            forecast_ci = forecast.conf_int()
            
            forecast_dates = forecast_mean.index
            forecast_values = forecast_mean.values
            
            # 可视化
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='实际值'))
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values,
                mode='lines',
                name='预测值',
                line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=np.concatenate([forecast_dates, forecast_dates[::-1]]),
                y=np.concatenate([forecast_ci.iloc[:, 1], forecast_ci.iloc[:, 0][::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% 置信区间'
            ))
            
            fig.update_layout(title='ARIMA 预测', height=500)
            
            self.forecast_results['arima'] = {
                'method': 'ARIMA',
                'order': order,
                'forecast_dates': forecast_dates,
                'forecast_values': forecast_values,
                'confidence_interval': forecast_ci,
                'figure': fig,
                'model_summary': fitted.summary()
            }
            
            return self.forecast_results['arima']
            
        except Exception as e:
            st.error(f"ARIMA 模型拟合失败：{str(e)}")
            return {}
    
    def trend_projection(self, periods: int = 30) -> Dict[str, Any]:
        """
        趋势外推预测 (线性回归)
        """
        ts = self.prepare_time_series()
        
        # 创建时间索引
        X = np.arange(len(ts)).reshape(-1, 1)
        y = ts.values
        
        # 简单线性回归
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        # 预测
        future_X = np.arange(len(ts), len(ts) + periods).reshape(-1, 1)
        forecast_values = model.predict(future_X)
        forecast_dates = pd.date_range(
            start=ts.index[-1] + pd.Timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        # 计算拟合值
        fitted_values = model.predict(X)
        
        # 可视化
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='实际值'))
        fig.add_trace(go.Scatter(
            x=ts.index,
            y=fitted_values,
            mode='lines',
            name='趋势线',
            line=dict(color='orange')
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines',
            name='预测值',
            line=dict(dash='dash', color='purple')
        ))
        
        fig.update_layout(title='趋势外推预测', height=500)
        
        self.forecast_results['trend'] = {
            'method': '线性趋势',
            'forecast_dates': forecast_dates,
            'forecast_values': forecast_values,
            'figure': fig,
            'slope': model.coef_[0],
            'intercept': model.intercept_
        }
        
        return self.forecast_results['trend']
    
    def evaluate_forecast(self, method: str = 'ma', train_ratio: float = 0.8) -> Dict[str, float]:
        """
        评估预测模型
        
        参数:
            method: 'ma', 'es', 'arima', 'trend'
            train_ratio: 训练集比例
        """
        ts = self.prepare_time_series()
        
        # 划分训练集和测试集
        split_idx = int(len(ts) * train_ratio)
        train = ts.iloc[:split_idx]
        test = ts.iloc[split_idx:]
        
        # 根据方法预测
        if method == 'ma':
            # 简单移动平均
            window = 7
            predictions = []
            for i in range(len(test)):
                if i < window:
                    pred = train.mean()
                else:
                    pred = np.mean(list(train[-window:]) + predictions[:i])
                predictions.append(pred)
        
        elif method == 'trend':
            from sklearn.linear_model import LinearRegression
            X_train = np.arange(len(train)).reshape(-1, 1)
            model = LinearRegression()
            model.fit(X_train, train.values)
            X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
            predictions = model.predict(X_test)
        
        else:
            st.warning(f"方法 {method} 的评估暂未实现")
            return {}
        
        # 计算评估指标
        mae = mean_absolute_error(test.values, predictions)
        rmse = np.sqrt(mean_squared_error(test.values, predictions))
        mape = np.mean(np.abs((test.values - predictions) / test.values)) * 100
        r2 = r2_score(test.values, predictions)
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': f"{mape:.2f}%",
            'R2': r2
        }
        
        return metrics
    
    def get_forecast_summary(self) -> Dict[str, Any]:
        """获取预测汇总"""
        summary = {
            'methods_available': list(self.forecast_results.keys()),
            'data_range': f"{self.data[self.date_col].min()} 至 {self.data[self.date_col].max()}",
            'total_records': len(self.data)
        }
        
        return summary


def compare_forecasts(data: pd.DataFrame, periods: int = 30) -> go.Figure:
    """
    比较多种预测方法
    """
    forecaster = SalesForecaster(data)
    
    # 运行多种预测方法
    ma_result = forecaster.moving_average_forecast(periods=periods)
    es_result = forecaster.exponential_smoothing_forecast(periods=periods)
    trend_result = forecaster.trend_projection(periods=periods)
    
    # 创建对比图
    fig = go.Figure()
    
    ts = forecaster.prepare_time_series()
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='实际值', line=dict(color='black')))
    
    if ma_result:
        fig.add_trace(go.Scatter(
            x=ma_result['forecast_dates'],
            y=ma_result['forecast_values'],
            mode='lines',
            name='移动平均',
            line=dict(dash='dash')
        ))
    
    if es_result:
        fig.add_trace(go.Scatter(
            x=es_result['forecast_dates'],
            y=es_result['forecast_values'],
            mode='lines',
            name='指数平滑',
            line=dict(dash='dot')
        ))
    
    if trend_result:
        fig.add_trace(go.Scatter(
            x=trend_result['forecast_dates'],
            y=trend_result['forecast_values'],
            mode='lines',
            name='趋势外推',
            line=dict(dash='dashdot')
        ))
    
    fig.update_layout(title='多方法预测对比', height=600, hovermode='x unified')
    
    return fig


if __name__ == "__main__":
    # 测试
    from data_loader import create_sample_sales_data
    
    df = create_sample_sales_data()
    forecaster = SalesForecaster(df)
    
    print("预测汇总:")
    print(forecaster.get_forecast_summary())
    
    print("\n移动平均预测:")
    ma = forecaster.moving_average_forecast()
    
    print("\n趋势预测:")
    trend = forecaster.trend_projection()
    print(f"斜率：{trend['slope']:.2f}")
