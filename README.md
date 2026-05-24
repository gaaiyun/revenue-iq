# Sales-Dashboard

销售数据分析平台：Streamlit 仪表板（v1）+ headless CLI + LLM 自然语言报告（v2）。

v1 提供 Streamlit 交互式仪表板（图表 / 筛选 / 预测）+ 50 个单元测试。v2 在不动
v1 代码的前提下补三件：

1. **Headless 分析模块** — v1 的 `SalesAnalyzer` 在业务逻辑里调 `st.error()`，
   导致脚本 / cron / CI 跑不了。v2 加 `headless_analytics.py` 纯 pandas 实现
   同样的指标：summary / trend / breakdown / anomalies，不依赖 Streamlit / plotly /
   sklearn。
2. **CLI 入口** — `__main__.py` 6 个子命令让数据团队能脚本化生成日报。
3. **LLM 自然语言报告** — 喂入指标 → LLM 生成 overview / 亮点 / 风险 / 建议四段
   commentary。缺 API key 时退化为规则模板生成器。

## v2 新增模块

| 文件 | 干什么 |
|---|---|
| `headless_analytics.py` | `compute_summary` + `compute_trend` + `detect_anomalies` + `category_breakdown` + `region_breakdown`，纯 pandas |
| `llm_commentary.py` | `commentary(summary, trend, anomalies, ...)` → `CommentaryReport`（overview/亮点/风险/建议）+ markdown 渲染 |
| `__main__.py` | CLI：summary / trend / breakdown / anomalies / commentary / list-models |
| `tests/test_headless_analytics.py` | 31 测试：summary / trend / anomalies / breakdown |
| `tests/test_llm_commentary.py` | 17 测试：规则 fallback + LLM mock |

总测试 98 个（50 v1 + 48 v2），6 秒跑完。

## v1 仍保留

| 模块 | 干什么 |
|---|---|
| `dashboard.py` | Streamlit 交互式主界面 |
| `sales_analyzer.py` | RFM / ABC / 客户分群（用 sklearn KMeans） |
| `forecast.py` | ARIMA / 指数平滑 / Prophet 预测 |
| `data_loader.py` | CSV / Excel 上传加载 |
| `sample_data/sample_sales.csv` | 示例销售数据（v2 顺手修了混合全角 / 半角逗号导致 pandas 无法解析的 bug） |

## 安装

```bash
pip install -r requirements.txt
# 可选：v2 LLM commentary
pip install openai      # openai / deepseek
pip install anthropic
```

## 快速开始

### v2 headless CLI

```bash
# 总体统计
python __main__.py summary sample_data/sample_sales.csv

# 月度趋势 + 环比
python __main__.py trend sample_data/sample_sales.csv --period M

# 日度 z-score 异常检测
python __main__.py anomalies sample_data/sample_sales.csv --z 2.0

# 品类 + 区域分解
python __main__.py breakdown sample_data/sample_sales.csv

# 综合 LLM 报告（缺 key 退规则）
python __main__.py commentary sample_data/sample_sales.csv \
    --use-llm --backend deepseek -o report.md

# 看 LLM backend 配置
python __main__.py list-models
```

### v1 Streamlit（仍能跑）

```bash
streamlit run dashboard.py
```

### 库调用

```python
import pandas as pd
from headless_analytics import (
    compute_summary, compute_trend, detect_anomalies,
    category_breakdown, region_breakdown,
)
from llm_commentary import commentary, LLMClient

df = pd.read_csv("sales.csv")
summary = compute_summary(df)
trend = compute_trend(df, period="M")
alerts = detect_anomalies(df, period="D", z_threshold=2.0)
cat = category_breakdown(df)
reg = region_breakdown(df)

# LLM 报告
report = commentary(
    summary=summary.to_dict(),
    trend=[{"period": p.period, "revenue": p.revenue,
            "n_orders": p.n_orders, "growth_pct": p.growth_pct}
           for p in trend],
    anomalies=[a.to_dict() for a in alerts],
    category_breakdown=cat,
    region_breakdown=reg,
    llm_client=LLMClient(backend="deepseek"),
)
print(report.to_markdown())
```

## 数据 schema

默认接 v1 sample_data 的中文列名：

| 列 | 类型 | 必需 |
|---|---|---|
| 日期 | datetime-parseable | 是 |
| 销售额 | float | 是 |
| 销售数量 | int | 否 |
| 产品类别 | str | 否 |
| 地区 | str | 否 |
| 客户 ID | str | 否 |
| 客户等级 | str | 否 |

要换列名（比如 English 列）：

```python
compute_summary(df, column_map={
    "date": "transaction_date",
    "amount": "revenue",
    "category": "product_category",
    "region": "geo",
})
```

## 异常检测口径

`detect_anomalies` 用 14 期滚动窗口的 z-score：当某天的实际销售偏离滚动均值 / 滚动标
准差 ≥ z 阈值（默认 2.0）时标记为 spike / drop。最少要有 5 期数据才检测。

```
z = (actual - rolling_mean_14) / rolling_std_14
|z| >= z_threshold → 异常
```

## LLM commentary 输出结构

```json
{
  "overview": "本期表现稳健，月环比增长 12%...",
  "highlights": ["电子产品贡献 50% 收入", "..."],
  "risks": ["3 月 15 日出现异常下跌", "..."],
  "recommendations": ["加大热销品类投放", "..."],
  "backend": "llm:deepseek"
}
```

缺 key 时 `backend="heuristic"`，规则模板生成同样字段。

## 设计取舍

- **headless_analytics 与 v1 SalesAnalyzer 共存**：原版 Streamlit 强依赖没动；v2
  另起一个纯 pandas 模块。两边算同样的指标（销售额 / 客户数 / 增长率），公式
  对齐但**入口不同**。要 Streamlit UI 用 v1，要脚本化用 v2。
- **commentary 不预测 / 不画图**：只做"把数字翻译成销售经理读得懂的话"。预测走
  v1 的 `forecast.py`（ARIMA/Prophet），画图走 Streamlit。
- **货币单位用 ASCII `CNY`**：Windows GBK 终端不能渲染 `¥`，CI 跑会崩；写文件
  / 写 JSON 都用 CNY，避免乱码。
- **pandas 2.x 兼容**：`M` 已废弃 → 内部自动转 `ME`，用户传 `M` 也能跑。

## 项目结构

```
Sales-Dashboard/
├── __main__.py                  # v2 CLI
├── headless_analytics.py        # v2 纯 pandas 分析
├── llm_commentary.py            # v2 LLM 报告
├── dashboard.py                 # v1 Streamlit 主入口
├── sales_analyzer.py            # v1 RFM/ABC/客户分群
├── forecast.py                  # v1 时间序列预测
├── data_loader.py               # v1 CSV/Excel 加载
├── tests/                       # 98 测试
│   ├── test_analyzer.py
│   ├── test_data_loader.py
│   ├── test_forecast.py
│   ├── test_headless_analytics.py   # v2 新增
│   └── test_llm_commentary.py       # v2 新增
├── sample_data/sample_sales.csv
└── requirements.txt
```

## 测试

```bash
pytest tests/ --no-cov
```

98 个测试，6 秒跑完。LLM / pandas / 文件 IO 全部 mock 或临时文件，无网络依赖。

## 已知限制

- `anomalies` 用固定 14 期滚动窗口，不能调；想用 7/30 期窗口需要改 `headless_analytics.py`。
- `commentary` 输出最多 4 段 + 5 条 highlights，超出会被 LLM 自动裁；想要长报告
  自己改 system prompt。
- v1 dashboard.py 还没接 v2 commentary —— Streamlit 用户暂时看不到 LLM 总结，
  下一版考虑加。

## 许可

MIT
