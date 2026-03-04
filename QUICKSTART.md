# 🚀 快速启动指南

## 安装依赖

```bash
cd sales-dashboard
pip install -r requirements.txt
```

## 运行应用

```bash
streamlit run dashboard.py
```

浏览器会自动打开 `http://localhost:8501`

## 快速测试

### 1. 使用示例数据
- 点击左侧"使用示例数据"
- 点击"加载示例数据"按钮
- 系统自动生成 1000 条销售记录

### 2. 查看所有功能
- 在侧边栏勾选所有分析选项
- 滚动查看各个分析模块
- 尝试不同的预测方法

### 3. 上传自有数据
数据文件需包含以下列（推荐）：
- `日期` (必需) - 销售日期
- `销售额` (必需) - 销售金额
- `销售数量` - 销售件数
- `产品类别` - 产品分类
- `产品名称` - 具体产品
- `地区` - 销售地区
- `客户 ID` - 客户编号
- `订单 ID` - 订单编号

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 查看测试覆盖率
pytest tests/ --cov=. --cov-report=html
# 打开 htmlcov/index.html 查看详细报告
```

## 常见问题

### Q: 端口被占用怎么办？
```bash
streamlit run dashboard.py --server.port 8502
```

### Q: 如何修改主题？
创建 `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
```

### Q: 数据量太大怎么办？
- 使用数据采样
- 选择更大的时间粒度（月/季）
- 启用数据缓存

## 技术支持

如遇问题，请查看：
- README.md - 完整文档
- htmlcov/index.html - 测试覆盖率报告

---

**祝您使用愉快！** 📊✨
