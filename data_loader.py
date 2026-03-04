"""
数据加载模块
支持多种数据格式：Excel, CSV, 数据库
自动处理缺失值、异常值
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import streamlit as st


class DataLoader:
    """数据加载器类"""
    
    def __init__(self):
        self.data = None
        self.file_path = None
    
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """加载 CSV 文件"""
        try:
            self.data = pd.read_csv(file_path, **kwargs)
            self.file_path = file_path
            return self.data
        except Exception as e:
            st.error(f"加载 CSV 失败：{str(e)}")
            return pd.DataFrame()
    
    def load_excel(self, file_path: str, sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """加载 Excel 文件"""
        try:
            if sheet_name:
                self.data = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            else:
                self.data = pd.read_excel(file_path, **kwargs)
            self.file_path = file_path
            return self.data
        except Exception as e:
            st.error(f"加载 Excel 失败：{str(e)}")
            return pd.DataFrame()
    
    def load_from_session(self, uploaded_file) -> pd.DataFrame:
        """从 Streamlit 上传的文件加载"""
        try:
            file_name = uploaded_file.name
            if file_name.endswith('.csv'):
                self.data = pd.read_csv(uploaded_file)
            elif file_name.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(uploaded_file)
            else:
                st.error("不支持的文件格式")
                return pd.DataFrame()
            
            self.file_path = file_name
            return self.data
        except Exception as e:
            st.error(f"加载文件失败：{str(e)}")
            return pd.DataFrame()
    
    def clean_data(self, 
                   handle_missing: str = 'drop',
                   remove_duplicates: bool = True,
                   outlier_method: Optional[str] = None) -> pd.DataFrame:
        """
        数据清洗
        
        参数:
            handle_missing: 'drop' 删除缺失值，'fill' 填充缺失值
            remove_duplicates: 是否删除重复值
            outlier_method: None, 'iqr' (四分位距), 'zscore' (Z 分数)
        """
        if self.data is None:
            st.warning("没有数据可清洗")
            return pd.DataFrame()
        
        df = self.data.copy()
        initial_rows = len(df)
        
        # 处理缺失值
        if handle_missing == 'drop':
            df = df.dropna()
        elif handle_missing == 'fill':
            # 数值列用中位数填充，分类列用众数填充
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype in ['int64', 'float64']:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        # 删除重复值
        if remove_duplicates:
            df = df.drop_duplicates()
        
        # 处理异常值
        if outlier_method == 'iqr':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
        
        elif outlier_method == 'zscore':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < 3]
        
        rows_removed = initial_rows - len(df)
        if rows_removed > 0:
            st.info(f"清洗后移除了 {rows_removed} 行数据")
        
        self.data = df
        return df
    
    def get_info(self) -> Dict[str, Any]:
        """获取数据基本信息"""
        if self.data is None:
            return {}
        
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum() / 1024 ** 2  # MB
        }
    
    def get_sample_data(self, n: int = 5) -> pd.DataFrame:
        """获取样本数据"""
        if self.data is None:
            return pd.DataFrame()
        return self.data.head(n)


def create_sample_sales_data() -> pd.DataFrame:
    """创建示例销售数据"""
    np.random.seed(42)
    
    n_records = 1000
    
    # 生成日期
    dates = pd.date_range('2023-01-01', periods=n_records, freq='D')
    
    # 生成产品类别
    categories = ['电子产品', '服装', '食品', '家居', '图书']
    products = {
        '电子产品': ['手机', '电脑', '平板', '耳机'],
        '服装': ['T 恤', '裤子', '外套', '鞋子'],
        '食品': ['零食', '饮料', '生鲜', '粮油'],
        '家居': ['家具', '装饰', '厨具', '床品'],
        '图书': ['小说', '教材', '杂志', '漫画']
    }
    
    # 生成地区
    regions = ['华北', '华东', '华南', '西南', '西北', '东北']
    cities = {
        '华北': ['北京', '天津', '石家庄'],
        '华东': ['上海', '杭州', '南京'],
        '华南': ['广州', '深圳', '厦门'],
        '西南': ['成都', '重庆', '昆明'],
        '西北': ['西安', '兰州', '乌鲁木齐'],
        '东北': ['沈阳', '长春', '哈尔滨']
    }
    
    data = []
    for i in range(n_records):
        category = np.random.choice(categories)
        product = np.random.choice(products[category])
        region = np.random.choice(regions)
        city = np.random.choice(cities[region])
        
        # 生成销售数据
        quantity = np.random.randint(1, 50)
        unit_price = np.random.uniform(10, 5000)
        discount = np.random.choice([0, 0.05, 0.1, 0.15, 0.2], p=[0.5, 0.2, 0.15, 0.1, 0.05])
        revenue = quantity * unit_price * (1 - discount)
        
        # 生成客户信息
        customer_id = f"C{np.random.randint(1000, 9999)}"
        customer_segment = np.random.choice(['普通', '白银', '黄金', '钻石'], p=[0.5, 0.3, 0.15, 0.05])
        
        data.append({
            '日期': dates[i],
            '订单 ID': f"ORD{np.random.randint(100000, 999999)}",
            '产品类别': category,
            '产品名称': product,
            '地区': region,
            '城市': city,
            '销售数量': quantity,
            '单价': round(unit_price, 2),
            '折扣率': discount,
            '销售额': round(revenue, 2),
            '客户 ID': customer_id,
            '客户等级': customer_segment
        })
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # 测试数据加载器
    df = create_sample_sales_data()
    print(f"创建示例数据：{df.shape}")
    print(df.head())
    
    # 测试数据加载器
    loader = DataLoader()
    loader.data = df
    info = loader.get_info()
    print(f"\n数据信息：{info['shape']}")
