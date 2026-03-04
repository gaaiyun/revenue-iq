"""
销售数据分析仪表板
主界面 - Streamlit 应用
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os

# 导入自定义模块
from data_loader import DataLoader, create_sample_sales_data
from sales_analyzer import SalesAnalyzer
from forecast import SalesForecaster, compare_forecasts

# 页面配置
st.set_page_config(
    page_title="销售数据分析仪表板",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    padding: 1rem 0;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


def main():
    """主函数"""
    
    # 标题
    st.markdown('<p class="main-header">📊 销售数据分析仪表板</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # 初始化 session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.data = None
    
    # 侧边栏
    with st.sidebar:
        st.header("🎯 控制面板")
        
        # 数据导入
        st.subheader("1️⃣ 数据导入")
        upload_method = st.radio(
            "选择数据源",
            ["上传文件", "使用示例数据"],
            index=1
        )
        
        if upload_method == "上传文件":
            uploaded_file = st.file_uploader(
                "上传 Excel 或 CSV 文件",
                type=['csv', 'xlsx', 'xls'],
                help="支持 CSV、Excel 格式"
            )
            
            if uploaded_file is not None:
                loader = DataLoader()
                data = loader.load_from_session(uploaded_file)
                if not data.empty:
                    st.session_state.data = data
                    st.session_state.data_loaded = True
                    st.success(f"✅ 成功加载 {len(data)} 条记录")
        else:
            if st.button("📁 加载示例数据", use_container_width=True):
                with st.spinner("正在生成示例数据..."):
                    data = create_sample_sales_data()
                    st.session_state.data = data
                    st.session_state.data_loaded = True
                    st.success(f"✅ 生成 {len(data)} 条示例记录")
        
        # 数据清洗选项
        if st.session_state.data_loaded:
            st.subheader("2️⃣ 数据清洗")
            handle_missing = st.selectbox(
                "缺失值处理",
                ["drop", "fill"],
                help="drop: 删除缺失值，fill: 填充缺失值"
            )
            remove_outliers = st.checkbox("处理异常值 (IQR 方法)", value=False)
            
            if st.button("🧹 执行数据清洗", use_container_width=True):
                loader = DataLoader()
                loader.data = st.session_state.data
                outlier_method = 'iqr' if remove_outliers else None
                cleaned_data = loader.clean_data(
                    handle_missing=handle_missing,
                    outlier_method=outlier_method
                )
                st.session_state.data = cleaned_data
                st.success(f"✅ 清洗完成，剩余 {len(cleaned_data)} 条记录")
        
        # 分析选项
        if st.session_state.data_loaded:
            st.subheader("3️⃣ 分析选项")
            show_trend = st.checkbox("📈 销售趋势", value=True)
            show_regional = st.checkbox("🗺️ 地区分析", value=True)
            show_product = st.checkbox("🏷️ 产品分析", value=True)
            show_rfm = st.checkbox("👥 RFM 客户分析", value=True)
            show_abc = st.checkbox("📊 ABC 分析", value=True)
            show_segmentation = st.checkbox("🎯 客户分群", value=True)
            show_forecast = st.checkbox("🔮 销售预测", value=True)
            
            st.divider()
            
            # 导出选项
            st.subheader("💾 导出报告")
            if st.button("📥 导出 Excel 报告", use_container_width=True):
                export_excel_report(st.session_state.data)
            
            if st.button("📄 导出 PDF 报告", use_container_width=True):
                st.info("PDF 导出功能开发中...")
    
    # 主内容区
    if st.session_state.data_loaded:
        df = st.session_state.data
        
        # 显示数据基本信息
        with st.expander("📋 数据概览", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            if '销售额' in df.columns:
                total_sales = df['销售额'].sum()
                avg_sales = df['销售额'].mean()
                col1.metric("💰 总销售额", f"¥{total_sales:,.2f}")
                col2.metric("📊 平均销售额", f"¥{avg_sales:,.2f}")
            
            if '销售数量' in df.columns:
                total_qty = df['销售数量'].sum()
                col3.metric("📦 总销售数量", f"{total_qty:,}")
            
            if '客户 ID' in df.columns:
                total_customers = df['客户 ID'].nunique()
                col4.metric("👥 客户总数", f"{total_customers:,}")
            
            st.dataframe(df.head(10), use_container_width=True)
            st.write(f"**数据维度**: {df.shape[0]} 行 × {df.shape[1]} 列")
        
        # 销售趋势分析
        if show_trend and '日期' in df.columns and '销售额' in df.columns:
            st.markdown("### 📈 销售趋势分析")
            analyzer = SalesAnalyzer(df)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                period = st.selectbox("选择时间粒度", ["D", "W", "M", "Q"], index=2)
                trend_result = analyzer.sales_trend_analysis(period=period)
                if trend_result and 'figure' in trend_result:
                    st.plotly_chart(trend_result['figure'], use_container_width=True)
            
            with col2:
                if trend_result and 'data' in trend_result:
                    st.subheader("关键指标")
                    sales_data = trend_result['data']
                    st.metric("最高销售额", f"¥{sales_data.max():,.2f}")
                    st.metric("最低销售额", f"¥{sales_data.min():,.2f}")
                    if len(sales_data) > 1:
                        growth = ((sales_data.iloc[-1] - sales_data.iloc[0]) / sales_data.iloc[0]) * 100
                        st.metric("期间增长率", f"{growth:.2f}%")
        
        # 地区分析
        if show_regional and '地区' in df.columns:
            st.markdown("### 🗺️ 地区分布分析")
            analyzer = SalesAnalyzer(df)
            regional_result = analyzer.regional_analysis()
            
            if regional_result:
                col1, col2 = st.columns(2)
                with col1:
                    if 'bar_figure' in regional_result:
                        st.plotly_chart(regional_result['bar_figure'], use_container_width=True)
                with col2:
                    if 'data' in regional_result:
                        st.dataframe(regional_result['data'], use_container_width=True)
        
        # 产品分析
        if show_product and '产品类别' in df.columns:
            st.markdown("### 🏷️ 产品分析")
            analyzer = SalesAnalyzer(df)
            product_result = analyzer.product_analysis()
            
            if product_result:
                col1, col2 = st.columns(2)
                with col1:
                    if 'pie_figure' in product_result:
                        st.plotly_chart(product_result['pie_figure'], use_container_width=True)
                with col2:
                    if 'bar_figure' in product_result:
                        st.plotly_chart(product_result['bar_figure'], use_container_width=True)
        
        # RFM 分析
        if show_rfm and all(col in df.columns for col in ['客户 ID', '日期', '销售额']):
            st.markdown("### 👥 RFM 客户分析")
            analyzer = SalesAnalyzer(df)
            rfm_result = analyzer.rfm_analysis()
            
            if rfm_result:
                col1, col2 = st.columns(2)
                with col1:
                    if 'pie_figure' in rfm_result:
                        st.plotly_chart(rfm_result['pie_figure'], use_container_width=True)
                with col2:
                    if 'segment_counts' in rfm_result:
                        st.dataframe(rfm_result['segment_counts'], use_container_width=True)
                
                # 显示 RFM 详细数据
                with st.expander("查看 RFM 详细数据"):
                    if 'data' in rfm_result:
                        st.dataframe(rfm_result['data'], use_container_width=True)
        
        # ABC 分析
        if show_abc and '产品名称' in df.columns:
            st.markdown("### 📊 ABC 分析 (帕累托)")
            analyzer = SalesAnalyzer(df)
            abc_result = analyzer.abc_analysis()
            
            if abc_result:
                if 'figure' in abc_result:
                    st.plotly_chart(abc_result['figure'], use_container_width=True)
                
                if 'summary' in abc_result:
                    st.subheader("ABC 分类汇总")
                    st.dataframe(abc_result['summary'], use_container_width=True)
        
        # 客户分群
        if show_segmentation and all(col in df.columns for col in ['客户 ID', '日期', '销售额', '销售数量']):
            st.markdown("### 🎯 客户分群 (K-Means)")
            analyzer = SalesAnalyzer(df)
            
            n_clusters = st.slider("选择聚类数量", 2, 6, 4)
            seg_result = analyzer.customer_segmentation(n_clusters=n_clusters)
            
            if seg_result:
                if 'scatter_figure' in seg_result:
                    st.plotly_chart(seg_result['scatter_figure'], use_container_width=True)
                
                if 'cluster_profile' in seg_result:
                    st.subheader("集群特征")
                    st.dataframe(seg_result['cluster_profile'], use_container_width=True)
        
        # 销售预测
        if show_forecast and '日期' in df.columns and '销售额' in df.columns:
            st.markdown("### 🔮 销售预测")
            
            forecast_method = st.selectbox(
                "选择预测方法",
                ["移动平均", "指数平滑", "趋势外推", "多方法对比"]
            )
            
            forecast_periods = st.slider("预测期数 (天)", 7, 90, 30)
            
            forecaster = SalesForecaster(df)
            
            if forecast_method == "移动平均":
                window = st.slider("移动平均窗口", 3, 30, 7)
                result = forecaster.moving_average_forecast(window=window, periods=forecast_periods)
                if result and 'figure' in result:
                    st.plotly_chart(result['figure'], use_container_width=True)
            
            elif forecast_method == "指数平滑":
                alpha = st.slider("平滑系数 α", 0.1, 0.9, 0.3, 0.1)
                result = forecaster.exponential_smoothing_forecast(alpha=alpha, periods=forecast_periods)
                if result and 'figure' in result:
                    st.plotly_chart(result['figure'], use_container_width=True)
            
            elif forecast_method == "趋势外推":
                result = forecaster.trend_projection(periods=forecast_periods)
                if result and 'figure' in result:
                    st.plotly_chart(result['figure'], use_container_width=True)
            
            elif forecast_method == "多方法对比":
                fig = compare_forecasts(df, periods=forecast_periods)
                st.plotly_chart(fig, use_container_width=True)
            
            # 模型评估
            with st.expander("📊 模型评估"):
                st.write("使用 80% 数据训练，20% 数据测试")
                metrics = forecaster.evaluate_forecast(method='trend')
                if metrics:
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("MAE", f"{metrics.get('MAE', 0):.2f}")
                    col2.metric("RMSE", f"{metrics.get('RMSE', 0):.2f}")
                    col3.metric("MAPE", metrics.get('MAPE', 'N/A'))
                    col4.metric("R²", f"{metrics.get('R2', 0):.3f}")
    
    else:
        # 未加载数据时的欢迎界面
        st.markdown("""
        ### 👋 欢迎使用销售数据分析仪表板！
        
        **功能亮点**:
        - 📁 **多数据源支持**: Excel, CSV 文件导入
        - 🧹 **智能数据清洗**: 自动处理缺失值和异常值
        - 📈 **销售趋势分析**: 多维度时间序列分析
        - 🗺️ **地区分布**: 地理可视化分析
        - 🏷️ **产品分析**: 类别和产品排名
        - 👥 **RFM 模型**: 客户价值分析
        - 📊 **ABC 分析**: 帕累托优先级分析
        - 🎯 **客户分群**: K-Means 聚类分析
        - 🔮 **销售预测**: 多种预测模型对比
        
        **开始使用**:
        1. 在左侧面板选择"使用示例数据"或上传您的数据文件
        2. 配置数据清洗选项
        3. 选择需要展示的分析模块
        4. 查看交互式图表和洞察
        
        💡 **提示**: 首次使用建议先加载示例数据熟悉功能！
        """)
        
        # 显示示例图表
        if st.button("🎲 预览示例数据分析", use_container_width=True):
            st.session_state.data = create_sample_sales_data()
            st.session_state.data_loaded = True
            st.rerun()


def export_excel_report(data: pd.DataFrame):
    """导出 Excel 报告"""
    from io import BytesIO
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # 原始数据
        data.to_excel(writer, sheet_name='原始数据', index=False)
        
        # 汇总统计
        if '销售额' in data.columns:
            summary = data.groupby('产品类别')['销售额'].sum().reset_index()
            summary.to_excel(writer, sheet_name='类别汇总', index=False)
        
        if '地区' in data.columns and '销售额' in data.columns:
            regional = data.groupby('地区')['销售额'].sum().reset_index()
            regional.to_excel(writer, sheet_name='地区汇总', index=False)
    
    output.seek(0)
    
    st.download_button(
        label="📥 点击下载 Excel 报告",
        data=output.getvalue(),
        file_name=f"销售分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


if __name__ == "__main__":
    main()
