# revenue-iq

把一份订单 CSV 跑成一套经营洞察：KPI、品类 ABC、库存告警、渠道 ROI、销量预测、
中文经营月报。一条命令出结果，无需服务器、无需联网。

面向电商 / 零售运营：手头有订单流水，想知道卖得好不好、哪些品类该补货、哪个渠道
值得加预算、下个月大概能卖多少、以及一份能直接发群的中文总结。`revenue-iq` 用一个
CLI 把这些串起来，每个子命令都能单独跑、可导出 JSON。

## 能做什么

| 子命令 | 回答的问题 | 输入 |
|---|---|---|
| `summary` | 这段时间总营收 / 订单 / 客户 / 首末月增长是多少 | 销售流水 |
| `trend` | 按日 / 周 / 月 / 季 / 年看营收走势和环比 | 销售流水 |
| `anomalies` | 哪几天销量异常暴涨 / 暴跌 | 销售流水 |
| `breakdown` | 营收在品类 / 区域怎么分布 | 销售流水 |
| `segments` | 品类 ABC 帕累托 + RFM 客户分群 | 销售流水 |
| `forecast` | 未来几期销量大概多少（移动平均 / 线性趋势） | 销售流水 |
| `commentary` | 一份中文经营月报（综合以上，可选接 LLM） | 销售流水 |
| `overview` | 电商订单整体 KPI（含利润 / 利润率） | 订单表 |
| `products` | 畅销榜 + 品类表现 + 库存告警 + 补货建议 | 订单 + 产品表 |
| `marketing` | 各活动 / 渠道的 ROI、转化率、预算分配建议 | 活动表 |
| `retention` | 复购率 / 人均订单数 | 订单表 |
| `list-models` | 当前 LLM backend 配置状态 | 无 |

两套数据 schema 各管一摊：销售流水用中文列（日期 / 销售额 / 产品类别 / 地区 /
客户 ID），电商三件套用英文列（orders / products / campaigns）。两套都自带样本数据，
clone 下来直接能跑。

## 安装

```bash
pip install -r requirements.txt
# 可选：commentary 接真实 LLM 时才需要
pip install openai       # openai / deepseek 共用
pip install anthropic
```

只跑 CLI 的话，核心依赖就是 pandas + numpy。Streamlit / plotly / scikit-learn /
statsmodels 是给可选的 v1 交互式仪表板用的。

## 快速开始

仓库自带样本数据，下面的命令直接复制就能跑：

```bash
# 销售流水（中文列样本：sample_data/sample_sales.csv，4 个月 443 笔）
python __main__.py summary   sample_data/sample_sales.csv
python __main__.py trend     sample_data/sample_sales.csv --period M
python __main__.py anomalies sample_data/sample_sales.csv --z 2.0
python __main__.py breakdown sample_data/sample_sales.csv
python __main__.py segments  sample_data/sample_sales.csv
python __main__.py forecast  sample_data/sample_sales.csv --method trend --freq M --periods 3
python __main__.py commentary sample_data/sample_sales.csv          # 无 key 走规则模板

# 电商三件套（英文列样本：sample_data/{orders,products,campaigns}.csv）
python __main__.py overview  --orders sample_data/orders.csv
python __main__.py products  --orders sample_data/orders.csv --products sample_data/products.csv --top-n 5
python __main__.py marketing --campaigns sample_data/campaigns.csv
python __main__.py retention --orders sample_data/orders.csv

# 任意命令加 -o 导出 JSON / Markdown
python __main__.py overview --orders sample_data/orders.csv -o kpi.json
```

Windows 终端如遇中文乱码，命令前加 `set PYTHONIOENCODING=utf-8`（PowerShell：
`$env:PYTHONIOENCODING="utf-8"`）。

### 真实输出示例

```
$ python __main__.py trend sample_data/sample_sales.csv --period M
period              revenue   orders   growth%
--------------------------------------------------
2024-01-31          727,353      121         -
2024-02-29          805,090       99    +10.7%
2024-03-31        1,131,856      116    +40.6%
2024-04-30        1,047,041      107     -7.5%

$ python __main__.py overview --orders sample_data/orders.csv
{
  "n_orders": 50,
  "total_revenue": 11538.0,
  "total_profit": 5527.0,
  "profit_margin_pct": 47.9,
  "avg_order_value": 230.76,
  "unique_customers": 43
}
```

## 经营月报（commentary）

`commentary` 把 summary / trend / anomalies / breakdown 的结果喂给 LLM，生成四段
中文报告：整体概览、亮点、风险、下期建议。没配 API key 时自动退化为规则模板，
同样产出这四段，离线可用。

```bash
# 接 DeepSeek（需要 DEEPSEEK_API_KEY 环境变量）
python __main__.py commentary sample_data/sample_sales.csv --use-llm --backend deepseek -o report.md
```

输出结构：

```json
{
  "overview": "2024-01-01 至 2024-04-29 期间，共 443 笔订单...",
  "highlights": ["品类 Top1：电子产品 贡献 CNY3,024,701", "..."],
  "risks": ["人均订单数低，复购率有提升空间"],
  "recommendations": ["加大 电子产品 类目库存 / 营销投放"],
  "backend": "llm:deepseek"
}
```

支持三个 backend：`openai`（gpt-4o-mini）、`anthropic`（claude-3-5-haiku）、
`deepseek`（deepseek-chat）。缺 key 时 `backend` 为 `heuristic`。

## 库调用

每个能力都是普通 Python 模块，可以脱离 CLI 直接 import：

```python
import pandas as pd
from headless_analytics import compute_summary, compute_trend, detect_anomalies
from headless_segments import abc_summary, rfm_segments
from headless_forecast import trend_forecast
from llm_commentary import commentary

df = pd.read_csv("sample_data/sample_sales.csv")
print(compute_summary(df).to_dict())
print(abc_summary(df))
print(trend_forecast(df, periods=3, freq="M").to_dict())
```

```python
from ecom_data_prep import load_orders, load_products, load_campaigns, overview_metrics
from product_analyzer import ProductAnalyzer
from marketing_analyzer import MarketingAnalyzer

orders = load_orders("sample_data/orders.csv")        # 自动补 revenue / cost / profit
products = load_products("sample_data/products.csv")
campaigns = load_campaigns("sample_data/campaigns.csv")

print(overview_metrics(orders))
print(ProductAnalyzer(orders, products).get_sales_ranking(top_n=5))
print(ProductAnalyzer(orders, products).get_low_stock())
print(MarketingAnalyzer(campaigns).get_campaign_roi())
```

## 数据 schema

### 销售流水（中文列）

| 列 | 类型 | 必需 |
|---|---|---|
| 日期 | 可解析日期 | 是 |
| 销售额 | float | 是 |
| 销售数量 | int | 否 |
| 产品类别 | str | 否（segments / breakdown 用）|
| 地区 | str | 否（breakdown 用）|
| 客户 ID | str | 否（RFM 分群用）|

换成英文列名时传 `column_map`：

```python
compute_summary(df, column_map={"date": "transaction_date", "amount": "revenue"})
```

### 电商三件套（英文列）

`orders.csv` 必需 `quantity` + `unit_price`，可选 `cost_price`（影响利润）、
`customer_id`（影响客户数 / 复购）、`order_date`、`category`、`region`、`city`。
`revenue = quantity * unit_price`，`cost = quantity * cost_price`，
`profit = revenue - cost`；已存在的列不覆盖（幂等）。

`products.csv` 需含 `product_id` / `stock_quantity` / `reorder_level` 才能做库存告警；
`campaigns.csv` 需含 `channel` / `budget` / `revenue` / `impressions` / `clicks` /
`orders` 才能算 ROI 和转化率。

## 异常检测口径

`anomalies` 用 14 期滚动窗口的 z-score：某期实际销售偏离滚动均值 / 滚动标准差
达到阈值（默认 2.0）即标记 spike / drop。至少需要 5 期数据。

```
z = (actual - rolling_mean_14) / rolling_std_14
|z| >= z_threshold  →  异常
```

## 设计取舍

- **无头优先**：核心分析（`headless_*` + 两个 analyzer + `ecom_data_prep`）只依赖
  pandas + numpy，能在脚本 / CI / cron 里跑。Streamlit / plotly 只属于可选的
  v1 仪表板（`dashboard.py`），不影响 CLI。
- **两套 schema 不强行合并**：销售流水（中文列、按金额聚合）和电商订单（英文列、
  按 product_id / 成本聚合）是两种真实场景，硬塞进一张表会丢信息。CLI 用
  不同子命令分别接，互不干扰。
- **commentary 只做翻译，不预测不画图**：把数字讲成运营听得懂的话。预测交给
  `forecast`，分布交给 `breakdown` / `segments`。
- **货币用 ASCII `CNY`**：避免 Windows GBK 终端渲染 `¥` 乱码或 CI 崩溃。
- **pandas 2.x 兼容**：用户传 `M` / `Q` / `Y` 时内部自动换成 `ME` / `QE` / `YE`。

## 项目结构

```
revenue-iq/
├── __main__.py                  # 统一 CLI（12 子命令）
├── headless_analytics.py        # summary / trend / anomalies / breakdown（纯 pandas）
├── headless_segments.py         # ABC 帕累托 + RFM 客户分群（纯 pandas）
├── headless_forecast.py         # 移动平均 / 线性趋势预测（纯 numpy/pandas）
├── llm_commentary.py            # 中文经营月报（LLM 或规则 fallback）
├── ecom_data_prep.py            # 电商订单预处理 + KPI + 复购
├── product_analyzer.py          # 畅销榜 / 品类 / 库存告警 / 补货 / 价格建议
├── marketing_analyzer.py        # 渠道 ROI / 转化率 / 预算分配
├── dashboard.py                 # 可选：v1 Streamlit 仪表板
├── sales_analyzer.py            # 可选：v1 RFM/ABC/分群（Streamlit 版）
├── forecast.py                  # 可选：v1 ARIMA/指数平滑（Streamlit 版）
├── data_loader.py               # 可选：v1 CSV/Excel 加载
├── sample_data/
│   ├── sample_sales.csv         # 销售流水样本（中文列，4 个月）
│   ├── orders.csv               # 电商订单样本（英文列）
│   ├── products.csv             # 产品 / 库存样本
│   └── campaigns.csv            # 营销活动样本
├── tests/                       # 170 测试
└── requirements.txt
```

## 测试

```bash
python -m pytest tests/ -q -o addopts=""
```

170 个测试，约 10 秒跑完。LLM、文件 IO 全部 mock 或用临时文件 / 自带样本，无网络依赖。
覆盖六个无头分析模块、两个移植进来的 analyzer、以及 CLI 每个子命令的端到端冒烟。

## 可选：Streamlit 仪表板

老版交互式仪表板仍然保留，需要图表 / 鼠标筛选时用：

```bash
streamlit run dashboard.py
```

## 许可

MIT
