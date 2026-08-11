# MindMemOS 评测报告（LifeBench 全量）

> 日期：2026-08-09 | 数据集：lifebench（10 人 × 365 天，3380 QA）| 算法：schema | 模型：deepseek-v4-flash

## 1. 如何运行 MindMemOS 评测

### 前置环境

1. **Docker 依赖栈**（qdrant / neo4j / kafka）：
   ```bash
   docker compose -f systems/MindMemOS/docker-compose.yml up -d
   ```
2. **MindMemOS API 服务器**（端口 8000）：
   ```bash
   cd systems/MindMemOS
   uv run uvicorn mindmemos.api.app:app --host 127.0.0.1 --port 8000
   ```
3. **LLM 代理**（端口 18443，用于服务器端 add/search LLM 调用记账，转发到真实 dashscope）：
   ```bash
   python src/utils/llm_proxy.py
   ```
   服务器配置 `systems/MindMemOS/config/mindmemos/dev.yaml` 的 `chat_model_router.api_base` 指向 `http://localhost:18443`。
4. **API key**：`systems/MindMemOS/config/mindmemos/api_keys.yaml`，每个 key 对应独立 project（实验隔离）。lifebench 用 `key_lifebench_schema_lifebench_20260807`。

### 运行命令

```bash
LLM_BASE_URL='http://localhost:18443' \
OUTPUT_DIR='results/lifebench-mindmemos-schema' \
MINDMEMOS_API_KEY='dev-api-key-lifebench-schema-lifebench-20260807' \
PYTHONPATH='src;systems/MindMemOS/src/mindmemos_sdk' \
python cli.py --dataset lifebench --system mindmemos --run-name schema \
  --output-dir results/lifebench-mindmemos-schema \
  --enable-tracker --tracker-interval 60
```

- SDK 需 Python ≥3.11，评测环境是 3.10，故用 `PYTHONPATH` 直接指向 SDK 源码而非 pip 安装
- 支持中断 + checkpoint 续跑（`checkpoint_default.json`，按日期粒度记录，日期内不中断保护）
- `--enable-tracker` 输出资源时间线到 `results/.../tracker/global_resource_timeline.json`

## 2. 系统简要流程

```
adapter (src/adapters/mindmemos_adapter.py)
  → mindmemos_sdk → FastAPI (127.0.0.1:8000)
  → add:  schema 算法
        chunker（rule 切分，非 llm）→ 每会话生成 episode
        → 实体/属性抽取（LLM）→ 实体合并 → 高阶记忆 → Kafka drain worker 入库
  → search: agentic 循环（max_rounds=1，round 1 后直接 break，跳过充分性判断）
        时间抽取 → schema search（实体召回 800 / 属性 45 / 双路扩展 / multi-hop 2）→ rerank
  → answer: 直接调用 LLM（deepseek-v4-flash，经代理记账）
  → evaluate: LLM judge 逐题打分（3380 条）
```

关键配置（`systems/MindMemOS/config/mindmemos/dev.yaml` + `api_keys.yaml`）：
- `chunker.split_mode: rule`、`max_episode_length: 200`、`split_on_user_speaker: false` —— 避免 llm 切分产生大量 episode（13 条消息 1 个 episode 而非 13 个）
- `agentic.max_rounds: 1` + `api/schemas.py` 请求默认 `max_rounds=1` —— 单轮检索、跳过充分性判断，单次搜索 20-40s（空闲）→ 负载下 3-6min

## 3. 结果

| 指标 | 结果 |
|---|---|
| **Accuracy** | **67.37%**（2277/3380） |
| **Weighted score** | **57.98%** |
| 总耗时 | ~29.4h（ answer+evaluate 8742s ≈ 2.4h +  add/search 27h） |
| LLM token（代理记账） | 9229 请求，73.2M tokens（prompt 63.9M + completion 9.3M） |
| 资源峰值 | 内存 785MB / CPU 77.7% |

### 分题型准确率

| 题型 | 准确率 |
|---|---|
| Conflict | 84.7% |
| Unanswerable | 84.1% |
| Single_hop | 79.8% |
| Knowledge_update | 69.7% |
| Temporal | 42.5% |
| Multi_hop | 42.0% |
| Causal | 38.1% |
| Pattern_recognition | 35.4% |
| Hidden_info | 26.8% |

### 输出文件（results/lifebench-mindmemos-schema/）

- `eval_results.json`：逐题 is_correct + reasoning
- `answer_results.json` / `search_results.json`（3380 条）/ `add_latency.json` / `search_latency.json`
- `tracker/timeline_merged.json`：三阶段合并资源时间线（timeline_1 → timeline_2 → global）+ 9229 条 token 记录
- `report.txt`：汇总（Total Time 已含此前 add/search 27h）

## 4. 结论

- **强项**：单跳检索（79.8%）、冲突检测（84.7%）、不可回答问题拒绝回答（84.1%）——schema 算法的实体/属性结构化记忆对直接事实查询有效
- **弱项**：需要跨记忆推理的题型全部偏低——Multi_hop 42.0%、Causal 38.1%、Pattern_recognition 35.4%、Hidden_info 26.8%、Temporal 42.5%
- **推断**：搜索召回中实体属性碎片化（同实体多条独立属性记忆）、跨实体关联弱（multi-hop 只 2 跳、属性扩展带出的关联有限），导致"组合多个记忆片段才能回答"的问题召回不足
- **优化方向**：
  1. 增强多跳/图扩展：提高 multi_hop 上限、提高 episode_edge / 实体邻居召回
  2. 属性合并：开启 `use_property_merge` 减少碎片化
  3. Hidden_info 类：提升 search_fields 扩展（`search_fields_max`、`episode_search_fields_augment`）
  4. 大日期（如 12-31 单日 ~190 QA）串行搜索耗时 2.2h，可考虑日期内并发
