"""
销售分析模块
包含：销售趋势分析、客户分析 (RFM 模型)、产品分析 (ABC 分析)、客户分群
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple
import streamlit as st


class SalesAnalyzer:
    """销售分析器类"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.analysis_results = {}
    
    def sales_trend_analysis(self, period: str = 'M') -> Dict[str, Any]:
        """
        销售趋势分析
        
        参数:
            period: 'D' 日，'W' 周，'M' 月，'Q' 季，'Y' 年
        """
        if '日期' not in self.data.columns:
            st.error("数据中缺少'日期'列")
            return {}
        
        df = self.data.copy()
        df['日期'] = pd.to_datetime(df['日期'])
        df.set_index('日期', inplace=True)
        
        # 按周期聚合
        if '销售额' in df.columns:
            sales_resampled = df['销售额'].resample(period).sum()
        else:
            st.error("数据中缺少'销售额'列")
            return {}
        
        # 计算增长率
        sales_growth = sales_resampled.pct_change() * 100
        
        # 创建趋势图
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sales_resampled.index,
            y=sales_resampled.values,
            mode='lines+markers',
            name='销售额',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=f'销售趋势 ({period})',
            xaxis_title='时间',
            yaxis_title='销售额',
            hovermode='x unified',
            height=500
        )
        
        self.analysis_results['sales_trend'] = {
            'data': sales_resampled,
            'growth': sales_growth,
            'figure': fig
        }
        
        return self.analysis_results['sales_trend']
    
    def regional_analysis(self) -> Dict[str, Any]:
        """地区分布分析"""
        if '地区' not in self.data.columns or '销售额' not in self.data.columns:
            st.error("数据中缺少'地区'或'销售额'列")
            return {}
        
        # 按地区聚合
        regional_sales = self.data.groupby('地区')['销售额'].agg(['sum', 'mean', 'count']).reset_index()
        regional_sales.columns = ['地区', '总销售额', '平均销售额', '订单数']
        
        # 创建地图可视化
        fig = px.choropleth(
            regional_sales,
            locations='地区',
            locationmode='country names',
            color='总销售额',
            title='地区销售分布',
            color_continuous_scale='Viridis'
        )
        
        # 创建条形图
        bar_fig = px.bar(
            regional_sales.sort_values('总销售额', ascending=False),
            x='地区',
            y='总销售额',
            title='各地区销售额对比',
            labels={'地区': '地区', '总销售额': '销售额'}
        )
        
        self.analysis_results['regional'] = {
            'data': regional_sales,
            'map_figure': fig,
            'bar_figure': bar_fig
        }
        
        return self.analysis_results['regional']
    
    def product_analysis(self) -> Dict[str, Any]:
        """产品分析"""
        required_cols = ['产品类别', '产品名称', '销售额', '销售数量']
        if not all(col in self.data.columns for col in required_cols):
            st.error("数据中缺少产品相关列")
            return {}
        
        # 按类别分析
        category_sales = self.data.groupby('产品类别').agg({
            '销售额': 'sum',
            '销售数量': 'sum'
        }).reset_index()
        
        # 按产品分析
        product_sales = self.data.groupby('产品名称').agg({
            '销售额': 'sum',
            '销售数量': 'sum'
        }).reset_index()
        
        # 创建饼图
        pie_fig = px.pie(
            category_sales,
            values='销售额',
            names='产品类别',
            title='各类别销售占比'
        )
        
        # 创建产品排名图
        top_products = product_sales.nlargest(10, '销售额')
        bar_fig = px.bar(
            top_products,
            x='销售额',
            y='产品名称',
            orientation='h',
            title='Top 10 产品销售额'
        )
        
        self.analysis_results['product'] = {
            'category_data': category_sales,
            'product_data': product_sales,
            'pie_figure': pie_fig,
            'bar_figure': bar_fig
        }
        
        return self.analysis_results['product']
    
    def rfm_analysis(self) -> Dict[str, Any]:
        """
        RFM 模型分析
        Recency: 最近一次消费时间
        Frequency: 消费频率
        Monetary: 消费金额
        """
        required_cols = ['客户 ID', '日期', '销售额']
        if not all(col in self.data.columns for col in required_cols):
            st.error("数据中缺少 RFM 分析所需列")
            return {}
        
        df = self.data.copy()
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 计算 RFM 指标
        now = df['日期'].max()
        rfm = df.groupby('客户 ID').agg({
            '日期': lambda x: (now - x.max()).days,  # Recency
            '订单 ID': 'count',  # Frequency
            '销售额': 'sum'  # Monetary
        }).reset_index()
        
        rfm.columns = ['客户 ID', 'R', 'F', 'M']
        
        # RFM 分箱 (1-5 分)
        for col in ['R', 'F', 'M']:
            try:
                if col == 'R':  # R 值越小越好，需要反转
                    rfm[f'{col}_score'] = pd.qcut(rfm[col].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
                else:
                    rfm[f'{col}_score'] = pd.qcut(rfm[col].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
            except ValueError:
                # 如果分箱失败，使用简单分位
                rfm[f'{col}_score'] = pd.cut(rfm[col], bins=5, labels=[1, 2, 3, 4, 5])
        
        # 计算 RFM 总分
        rfm['RFM_score'] = rfm['R_score'].astype(int) + rfm['F_score'].astype(int) + rfm['M_score'].astype(int)
        
        # 客户分群
        def segment_customer(row):
            if row['RFM_score'] >= 12:
                return '重要价值客户'
            elif row['RFM_score'] >= 9:
                return '重要发展客户'
            elif row['RFM_score'] >= 6:
                return '一般客户'
            else:
                return '低价值客户'
        
        rfm['客户分群'] = rfm.apply(segment_customer, axis=1)
        
        # 可视化
        segment_counts = rfm['客户分群'].value_counts().reset_index()
        segment_counts.columns = ['客户分群', '客户数']
        
        pie_fig = px.pie(
            segment_counts,
            values='客户数',
            names='客户分群',
            title='客户分群分布'
        )
        
        # 3D 散点图
        scatter_3d = px.scatter_3d(
            rfm,
            x='R',
            y='F',
            z='M',
            color='客户分群',
            title='RFM 3D 分布',
            labels={'R': '最近消费天数', 'F': '消费频次', 'M': '消费金额'}
        )
        
        self.analysis_results['rfm'] = {
            'data': rfm,
            'segment_counts': segment_counts,
            'pie_figure': pie_fig,
            'scatter_3d': scatter_3d
        }
        
        return self.analysis_results['rfm']
    
    def abc_analysis(self) -> Dict[str, Any]:
        """
        ABC 分析 (帕累托分析)
        A 类：累计 70% 销售额的产品
        B 类：累计 70-90% 销售额的产品
        C 类：累计 90-100% 销售额的产品
        """
        if '产品名称' not in self.data.columns or '销售额' not in self.data.columns:
            st.error("数据中缺少 ABC 分析所需列")
            return {}
        
        # 按产品聚合
        product_sales = self.data.groupby('产品名称')['销售额'].sum().reset_index()
        product_sales = product_sales.sort_values('销售额', ascending=False)
        
        # 计算累计百分比
        product_sales['累计销售额'] = product_sales['销售额'].cumsum()
        total_sales = product_sales['销售额'].sum()
        product_sales['累计百分比'] = (product_sales['累计销售额'] / total_sales * 100).round(2)
        
        # ABC 分类
        def classify_abc(row):
            if row['累计百分比'] <= 70:
                return 'A'
            elif row['累计百分比'] <= 90:
                return 'B'
            else:
                return 'C'
        
        product_sales['ABC 分类'] = product_sales.apply(classify_abc, axis=1)
        
        # 统计各类别
        abc_summary = product_sales.groupby('ABC 分类').agg({
            '产品名称': 'count',
            '销售额': 'sum'
        }).reset_index()
        abc_summary.columns = ['分类', '产品数', '销售额']
        abc_summary['产品占比'] = (abc_summary['产品数'] / abc_summary['产品数'].sum() * 100).round(2)
        abc_summary['销售占比'] = (abc_summary['销售额'] / abc_summary['销售额'].sum() * 100).round(2)
        
        # 可视化
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=product_sales['产品名称'].head(50),  # 只显示前 50 个产品
            y=product_sales['销售额'].head(50),
            name='销售额'
        ))
        fig.add_trace(go.Scatter(
            x=product_sales['产品名称'].head(50),
            y=product_sales['累计百分比'].head(50),
            name='累计百分比',
            yaxis='y2',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='ABC 分析 (帕累托图)',
            yaxis=dict(title='销售额'),
            yaxis2=dict(title='累计百分比%', overlaying='y', anchor='x', range=[0, 100]),
            height=500
        )
        
        self.analysis_results['abc'] = {
            'data': product_sales,
            'summary': abc_summary,
            'figure': fig
        }
        
        return self.analysis_results['abc']
    
    def customer_segmentation(self, n_clusters: int = 4) -> Dict[str, Any]:
        """
        客户分群 (K-Means 聚类)
        """
        required_cols = ['客户 ID', '日期', '销售额', '销售数量']
        if not all(col in self.data.columns for col in required_cols):
            st.error("数据中缺少客户分群所需列")
            return {}
        
        df = self.data.copy()
        df['日期'] = pd.to_datetime(df['日期'])
        
        # 构建特征
        now = df['日期'].max()
        features = df.groupby('客户 ID').agg({
            '日期': lambda x: (now - x.max()).days,  # 最近消费
            '订单 ID': 'count',  # 消费频次
            '销售额': 'sum',  # 总消费
            '销售数量': 'sum'  # 总数量
        }).reset_index()
        
        features.columns = ['客户 ID', 'Recency', 'Frequency', 'Monetary', 'Quantity']
        
        # 标准化
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features[['Recency', 'Frequency', 'Monetary', 'Quantity']])
        
        # K-Means 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        features['Cluster'] = kmeans.fit_predict(features_scaled)
        
        # 分析各集群特征
        cluster_profile = features.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'Quantity': 'mean',
            '客户 ID': 'count'
        }).reset_index()
        cluster_profile.columns = ['集群', '平均 Recency', '平均 Frequency', '平均 Monetary', '平均 Quantity', '客户数']
        
        # 可视化
        scatter_fig = px.scatter(
            features,
            x='Frequency',
            y='Monetary',
            color='Cluster',
            title='客户分群 (Frequency vs Monetary)',
            hover_data=['客户 ID', 'Recency']
        )
        
        # 雷达图数据
        radar_data = cluster_profile.copy()
        # 标准化到 0-1 范围用于雷达图
        for col in ['平均 Recency', '平均 Frequency', '平均 Monetary', '平均 Quantity']:
            min_val = radar_data[col].min()
            max_val = radar_data[col].max()
            if max_val > min_val:
                radar_data[col + '_norm'] = (radar_data[col] - min_val) / (max_val - min_val)
            else:
                radar_data[col + '_norm'] = 0.5
        
        self.analysis_results['segmentation'] = {
            'features': features,
            'cluster_profile': cluster_profile,
            'scatter_figure': scatter_fig,
            'labels': kmeans.labels_
        }
        
        return self.analysis_results['segmentation']
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """获取汇总统计"""
        summary = {}
        
        if '销售额' in self.data.columns:
            summary['总销售额'] = self.data['销售额'].sum()
            summary['平均销售额'] = self.data['销售额'].mean()
            summary['最大销售额'] = self.data['销售额'].max()
            summary['最小销售额'] = self.data['销售额'].min()
        
        if '销售数量' in self.data.columns:
            summary['总销售数量'] = self.data['销售数量'].sum()
            summary['平均销售数量'] = self.data['销售数量'].mean()
        
        if '客户 ID' in self.data.columns:
            summary['客户总数'] = self.data['客户 ID'].nunique()
        
        if '订单 ID' in self.data.columns:
            summary['订单总数'] = self.data['订单 ID'].nunique()
        
        if '日期' in self.data.columns:
            df = self.data.copy()
            df['日期'] = pd.to_datetime(df['日期'])
            summary['时间范围'] = f"{df['日期'].min().strftime('%Y-%m-%d')} 至 {df['日期'].max().strftime('%Y-%m-%d')}"
        
        return summary


if __name__ == "__main__":
    # 测试
    from data_loader import create_sample_sales_data
    
    df = create_sample_sales_data()
    analyzer = SalesAnalyzer(df)
    
    print("销售汇总统计:")
    print(analyzer.get_summary_statistics())
    
    print("\nRFM 分析:")
    rfm = analyzer.rfm_analysis()
    if rfm:
        print(rfm['segment_counts'])
