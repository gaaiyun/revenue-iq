"""
销售分析模块单元测试
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sales_analyzer import SalesAnalyzer
from data_loader import create_sample_sales_data


class TestSalesAnalyzer:
    """销售分析器测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        return create_sample_sales_data()
    
    @pytest.fixture
    def minimal_data(self):
        """创建最小测试数据"""
        return pd.DataFrame({
            '日期': pd.date_range('2024-01-01', periods=10),
            '销售额': [100, 150, 200, 180, 220, 250, 230, 280, 300, 320],
            '销售数量': [10, 15, 20, 18, 22, 25, 23, 28, 30, 32],
            '产品类别': ['A'] * 5 + ['B'] * 5,
            '产品名称': ['P1', 'P2'] * 5,
            '地区': ['华北'] * 5 + ['华东'] * 5,
            '客户 ID': ['C1', 'C2', 'C3', 'C4', 'C5'] * 2,
            '订单 ID': [f'O{i}' for i in range(10)]
        })
    
    def test_init(self, sample_data):
        """测试初始化"""
        analyzer = SalesAnalyzer(sample_data)
        
        assert analyzer.data is not None
        assert len(analyzer.data) == len(sample_data)
    
    def test_sales_trend_analysis(self, minimal_data):
        """测试销售趋势分析"""
        analyzer = SalesAnalyzer(minimal_data)
        result = analyzer.sales_trend_analysis(period='D')
        
        assert 'data' in result
        assert 'figure' in result
        assert len(result['data']) > 0
    
    def test_sales_trend_missing_columns(self):
        """测试缺少必要列的情况"""
        df = pd.DataFrame({'A': [1, 2, 3]})
        analyzer = SalesAnalyzer(df)
        result = analyzer.sales_trend_analysis()
        
        assert result == {}
    
    def test_regional_analysis(self, minimal_data):
        """测试地区分析"""
        analyzer = SalesAnalyzer(minimal_data)
        result = analyzer.regional_analysis()
        
        assert 'data' in result
        assert 'bar_figure' in result
        assert len(result['data']) == 2  # 华北和华东
    
    def test_regional_missing_columns(self):
        """测试地区分析缺少列"""
        df = pd.DataFrame({'A': [1, 2, 3]})
        analyzer = SalesAnalyzer(df)
        result = analyzer.regional_analysis()
        
        assert result == {}
    
    def test_product_analysis(self, minimal_data):
        """测试产品分析"""
        analyzer = SalesAnalyzer(minimal_data)
        result = analyzer.product_analysis()
        
        assert 'category_data' in result
        assert 'product_data' in result
        assert 'pie_figure' in result
    
    def test_product_missing_columns(self):
        """测试产品分析缺少列"""
        df = pd.DataFrame({'A': [1, 2, 3]})
        analyzer = SalesAnalyzer(df)
        result = analyzer.product_analysis()
        
        assert result == {}
    
    def test_rfm_analysis(self, minimal_data):
        """测试 RFM 分析"""
        analyzer = SalesAnalyzer(minimal_data)
        result = analyzer.rfm_analysis()
        
        assert 'data' in result
        assert 'segment_counts' in result
        assert 'pie_figure' in result
        
        # 检查 RFM 分数列是否存在
        rfm_data = result['data']
        assert 'R_score' in rfm_data.columns
        assert 'F_score' in rfm_data.columns
        assert 'M_score' in rfm_data.columns
        assert 'RFM_score' in rfm_data.columns
        assert '客户分群' in rfm_data.columns
    
    def test_rfm_missing_columns(self):
        """测试 RFM 分析缺少列"""
        df = pd.DataFrame({'A': [1, 2, 3]})
        analyzer = SalesAnalyzer(df)
        result = analyzer.rfm_analysis()
        
        assert result == {}
    
    def test_abc_analysis(self, minimal_data):
        """测试 ABC 分析"""
        analyzer = SalesAnalyzer(minimal_data)
        result = analyzer.abc_analysis()
        
        assert 'data' in result
        assert 'summary' in result
        assert 'figure' in result
        
        # 检查 ABC 分类
        abc_data = result['data']
        assert 'ABC 分类' in abc_data.columns
        assert '累计百分比' in abc_data.columns
    
    def test_abc_missing_columns(self):
        """测试 ABC 分析缺少列"""
        df = pd.DataFrame({'A': [1, 2, 3]})
        analyzer = SalesAnalyzer(df)
        result = analyzer.abc_analysis()
        
        assert result == {}
    
    def test_customer_segmentation(self, minimal_data):
        """测试客户分群"""
        analyzer = SalesAnalyzer(minimal_data)
        result = analyzer.customer_segmentation(n_clusters=3)
        
        assert 'features' in result
        assert 'cluster_profile' in result
        assert 'scatter_figure' in result
        
        # 检查聚类结果
        features = result['features']
        assert 'Cluster' in features.columns
        assert features['Cluster'].nunique() <= 3
    
    def test_segmentation_missing_columns(self):
        """测试客户分群缺少列"""
        df = pd.DataFrame({'A': [1, 2, 3]})
        analyzer = SalesAnalyzer(df)
        result = analyzer.customer_segmentation()
        
        assert result == {}
    
    def test_summary_statistics(self, minimal_data):
        """测试汇总统计"""
        analyzer = SalesAnalyzer(minimal_data)
        summary = analyzer.get_summary_statistics()
        
        assert '总销售额' in summary
        assert '平均销售额' in summary
        assert '总销售数量' in summary
        assert '客户总数' in summary
        assert '订单总数' in summary
        assert '时间范围' in summary
        
        # 验证数值
        assert summary['总销售额'] == minimal_data['销售额'].sum()
        assert summary['客户总数'] == minimal_data['客户 ID'].nunique()
    
    def test_summary_statistics_partial_columns(self):
        """测试部分列的汇总统计"""
        df = pd.DataFrame({
            '销售额': [100, 200, 300]
        })
        analyzer = SalesAnalyzer(df)
        summary = analyzer.get_summary_statistics()
        
        assert '总销售额' in summary
        assert '客户总数' not in summary
    
    def test_analysis_results_storage(self, minimal_data):
        """测试分析结果存储"""
        analyzer = SalesAnalyzer(minimal_data)
        
        # 执行多个分析
        analyzer.sales_trend_analysis()
        analyzer.regional_analysis()
        analyzer.product_analysis()
        
        # 检查结果是否存储
        assert 'sales_trend' in analyzer.analysis_results
        assert 'regional' in analyzer.analysis_results
        assert 'product' in analyzer.analysis_results
    
    def test_rfm_segmentation_logic(self, minimal_data):
        """测试 RFM 分群逻辑"""
        analyzer = SalesAnalyzer(minimal_data)
        result = analyzer.rfm_analysis()
        
        rfm_data = result['data']
        
        # 检查分群是否合理
        segments = rfm_data['客户分群'].unique()
        valid_segments = ['重要价值客户', '重要发展客户', '一般客户', '低价值客户']
        
        for segment in segments:
            assert segment in valid_segments
    
    def test_abc_classification_logic(self, minimal_data):
        """测试 ABC 分类逻辑"""
        analyzer = SalesAnalyzer(minimal_data)
        result = analyzer.abc_analysis()
        
        abc_data = result['data']
        
        # 检查分类是否正确
        classes = abc_data['ABC 分类'].unique()
        valid_classes = ['A', 'B', 'C']
        
        for cls in classes:
            assert cls in valid_classes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
