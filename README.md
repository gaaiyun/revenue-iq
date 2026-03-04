# 📊 销售数据分析仪表板

专业的销售数据分析和可视化平台，基于 Streamlit 构建。

## ✨ 功能特性

### 🎯 核心功能

1. **多数据源导入**
   - ✅ 支持 CSV 文件格式
   - ✅ 支持 Excel 文件格式 (.xlsx, .xls)
   - ✅ 内置示例数据生成

2. **智能数据清洗**
   - ✅ 缺失值处理（删除/填充）
   - ✅ 重复值检测与删除
   - ✅ 异常值处理（IQR 方法、Z-Score 方法）

3. **销售趋势分析**
   - ✅ 多时间粒度（日/周/月/季）
   - ✅ 增长率计算
   - ✅ 交互式可视化

4. **地区分布分析**
   - ✅ 地区销售对比
   - ✅ 地理可视化
   - ✅ 多维度统计

5. **产品分析**
   - ✅ 产品类别占比
   - ✅ 产品排名 Top10
   - ✅ 销售数量分析

6. **客户分析**
   - ✅ RFM 模型（最近消费/消费频次/消费金额）
   - ✅ 客户价值分群
   - ✅ K-Means 聚类分析

7. **ABC 分析**
   - ✅ 帕累托分析
   - ✅ 产品优先级分类
   - ✅ 累计百分比曲线

8. **销售预测**
   - ✅ 移动平均预测
   - ✅ 指数平滑预测
   - ✅ 线性趋势外推
   - ✅ 多方法对比
   - ✅ 模型评估（MAE/RMSE/MAPE/R²）

9. **报告导出**
   - ✅ Excel 报告导出
   - 🔄 PDF 报告（开发中）

## 🚀 快速开始

### 1. 安装依赖

```bash
cd sales-dashboard
pip install -r requirements.txt
```

### 2. 运行应用

```bash
streamlit run dashboard.py
```

### 3. 访问应用

浏览器打开：`http://localhost:8501`

## 📁 项目结构

```
sales-dashboard/
├── dashboard.py          # 主界面 (Streamlit 应用)
├── data_loader.py        # 数据加载模块
├── sales_analyzer.py     # 销售分析模块
├── forecast.py           # 预测模块
├── requirements.txt      # 依赖包列表
├── README.md            # 项目说明
├── tests/               # 单元测试
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_analyzer.py
│   └── test_forecast.py
└── sample_data/         # 示例数据
    └── sample_sales.csv
```

## 📖 使用指南

### 数据导入

1. **使用示例数据**
   - 点击侧边栏"加载示例数据"按钮
   - 系统自动生成 1000 条模拟销售记录

2. **上传自有数据**
   - 选择"上传文件"选项
   - 支持 CSV、Excel 格式
   - 数据需包含以下列（部分可选）：
     - `日期` (必需)
     - `销售额` (必需)
     - `销售数量`
     - `产品类别`
     - `产品名称`
     - `地区`
     - `客户 ID`
     - `订单 ID`

### 数据清洗

- **缺失值处理**
  - `drop`: 直接删除含缺失值的行
  - `fill`: 数值列用中位数填充，分类列用众数填充

- **异常值处理**
  - IQR 方法：移除超出 1.5 倍四分位距的数据
  - Z-Score 方法：移除 Z 分数>3 的数据

### 分析模块

所有分析模块可在侧边栏自由勾选，支持：

- 交互式图表（Plotly）
- 数据表格展示
- 关键指标卡片
- 下钻分析

### 销售预测

1. 选择预测方法
2. 设置预测期数（7-90 天）
3. 查看预测结果和置信区间
4. 查看模型评估指标

## 🧪 测试

### 运行所有测试

```bash
cd sales-dashboard
pytest tests/ -v --cov=.
```

### 测试覆盖率

```bash
pytest tests/ --cov-report=html
# 打开 htmlcov/index.html 查看详细报告
```

### 测试要求

- ✅ 测试覆盖率 > 70%
- ✅ 所有核心功能单元测试
- ✅ 集成测试验证

## 📊 示例数据说明

示例数据包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| 日期 | datetime | 销售日期 |
| 订单 ID | string | 订单编号 |
| 产品类别 | string | 电子产品/服装/食品/家居/图书 |
| 产品名称 | string | 具体产品名称 |
| 地区 | string | 华北/华东/华南/西南/西北/东北 |
| 城市 | string | 具体城市 |
| 销售数量 | int | 销售件数 |
| 单价 | float | 商品单价 |
| 折扣率 | float | 折扣比例 |
| 销售额 | float | 实际销售金额 |
| 客户 ID | string | 客户编号 |
| 客户等级 | string | 普通/白银/黄金/钻石 |

## 🛠️ 技术栈

- **前端框架**: Streamlit 1.32.0
- **数据处理**: Pandas 2.2.0, NumPy 1.26.4
- **可视化**: Plotly 5.19.0
- **机器学习**: Scikit-learn 1.4.0
- **统计分析**: Statsmodels (可选)
- **文件处理**: OpenPyXL, XlsxWriter

## 🔧 配置选项

### 环境变量

```bash
# 可选：设置 Streamlit 端口
STREAMLIT_SERVER_PORT=8501

# 可选：设置主题
STREAMLIT_THEME=light
```

### Streamlit 配置

创建 `.streamlit/config.toml`:

```toml
[server]
port = 8501
headless = true

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

## 📈 性能优化建议

1. **大数据集处理**
   - 建议使用数据采样
   - 启用数据缓存

2. **预测模型**
   - 大数据集建议使用简单模型（移动平均、趋势外推）
   - ARIMA 模型计算较慢，适合中小数据集

3. **可视化**
   - 大量数据点时启用降采样
   - 使用 Plotly 的 WebGL 渲染

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发环境设置

```bash
# 克隆项目
git clone <repo-url>
cd sales-dashboard

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装开发依赖
pip install -r requirements.txt
pip install pytest pytest-cov

# 运行测试
pytest tests/ -v
```

## 📝 更新日志

### v1.0.0 (2026-03-04)
- ✅ 初始版本发布
- ✅ 多数据源导入
- ✅ 完整分析模块
- ✅ 销售预测功能
- ✅ 单元测试覆盖

## 📄 许可证

MIT License

## 👥 作者

Created with ❤️ by OpenClaw Agent

## 📞 支持

如有问题，请提交 Issue 或联系开发团队。

---

**开始您的数据分析之旅吧！** 🚀
