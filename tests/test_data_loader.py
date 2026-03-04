"""
数据加载模块单元测试
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import DataLoader, create_sample_sales_data


class TestDataLoader:
    """数据加载器测试类"""
    
    def test_create_sample_data(self):
        """测试示例数据生成"""
        df = create_sample_sales_data()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1000
        assert '日期' in df.columns
        assert '销售额' in df.columns
        assert '客户 ID' in df.columns
        assert '产品类别' in df.columns
    
    def test_load_csv(self, tmp_path):
        """测试 CSV 加载"""
        # 创建测试 CSV 文件
        test_data = pd.DataFrame({
            '日期': ['2024-01-01', '2024-01-02'],
            '销售额': [100, 200]
        })
        csv_file = tmp_path / "test.csv"
        test_data.to_csv(csv_file, index=False)
        
        # 测试加载
        loader = DataLoader()
        result = loader.load_csv(str(csv_file))
        
        assert len(result) == 2
        assert '销售额' in result.columns
        assert loader.file_path == str(csv_file)
    
    def test_load_excel(self, tmp_path):
        """测试 Excel 加载"""
        # 创建测试 Excel 文件
        test_data = pd.DataFrame({
            '日期': ['2024-01-01', '2024-01-02'],
            '销售额': [100, 200]
        })
        excel_file = tmp_path / "test.xlsx"
        test_data.to_excel(excel_file, index=False)
        
        # 测试加载
        loader = DataLoader()
        result = loader.load_excel(str(excel_file))
        
        assert len(result) == 2
        assert '销售额' in result.columns
    
    def test_clean_data_drop_missing(self):
        """测试删除缺失值"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, 7, 8]
        })
        
        loader = DataLoader()
        loader.data = df
        cleaned = loader.clean_data(handle_missing='drop')
        
        assert len(cleaned) == 2  # 只剩第 1 行和第 4 行
    
    def test_clean_data_fill_missing(self):
        """测试填充缺失值"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [5, np.nan, 7, 8, 9]
        })
        
        loader = DataLoader()
        loader.data = df
        cleaned = loader.clean_data(handle_missing='fill')
        
        assert cleaned.isnull().sum().sum() == 0
        # 检查 A 列是否用中位数填充
        assert cleaned.loc[2, 'A'] == df['A'].median()
    
    def test_clean_data_remove_duplicates(self):
        """测试删除重复值"""
        df = pd.DataFrame({
            'A': [1, 1, 2, 3],
            'B': [4, 4, 5, 6]
        })
        
        loader = DataLoader()
        loader.data = df
        cleaned = loader.clean_data(remove_duplicates=True)
        
        assert len(cleaned) == 3
    
    def test_clean_data_iqr_outliers(self):
        """测试 IQR 方法处理异常值"""
        # 创建含异常值的数据
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 100],  # 100 是异常值
        })
        
        loader = DataLoader()
        loader.data = df
        cleaned = loader.clean_data(outlier_method='iqr')
        
        # 异常值应该被移除
        assert 100 not in cleaned['A'].values
    
    def test_get_info(self):
        """测试获取数据信息"""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        
        loader = DataLoader()
        loader.data = df
        info = loader.get_info()
        
        assert 'shape' in info
        assert info['shape'] == (3, 2)
        assert 'columns' in info
        assert len(info['columns']) == 2
    
    def test_get_sample_data(self):
        """测试获取样本数据"""
        df = pd.DataFrame({'A': range(10)})
        
        loader = DataLoader()
        loader.data = df
        sample = loader.get_sample_data(n=5)
        
        assert len(sample) == 5
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        loader = DataLoader()
        result = loader.load_csv("nonexistent_file.csv")
        
        assert len(result) == 0
    
    def test_clean_empty_data(self):
        """测试清洗空数据"""
        loader = DataLoader()
        loader.data = None
        cleaned = loader.clean_data()
        
        assert len(cleaned) == 0
    
    def test_get_info_empty(self):
        """测试获取空数据信息"""
        loader = DataLoader()
        loader.data = None
        info = loader.get_info()
        
        assert info == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
