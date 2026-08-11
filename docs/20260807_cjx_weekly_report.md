# LifeBench Evaluation 综合报告

> 日期：2026-08-07
> 系统：cognee | hindsight | MemosCloud | MemuCloud | EverMemOS
> 数据集：LifeBench（10 人、3,380 QA）、LoCoMo
---

## 一、LifeBench 分类别正确率（3,380 题）

| Category | cognee | hindsight | MemosCloud | MemuCloud | EverMemOS |
|---|---|---|---|---|---|
| Single_hop | 71.69% | 80.63% | 72.94% | 60.34% | 72.00% |
| Multi_hop | 37.07% | 45.84% | 32.91% | 23.87% | 44.94% |
| Causal | 43.16% | 49.06% | 40.21% | 30.29% | 50.40% |
| Temporal | 27.36% | 45.85% | 26.98% | 17.92% | 40.94% |
| Knowledge_update | 72.49% | 79.18% | 71.47% | 47.30% | 73.01% |
| Conflict | 68.63% | 76.89% | 68.63% | 62.26% | 73.82% |
| Hidden_info | 30.93% | 37.63% | 35.57% | 23.20% | 30.93% |
| Pattern_recognition (Non-declarative) | 45.44% | 46.39% | 33.65% | 23.19% | 45.06% |
| Unanswerable | 92.14% | 75.38% | 88.89% | 90.77% | 84.96% |
| **Overall (Weighted)** | **63.17%** | **67.66%** | **61.51%** | **52.75%** | **64.73%** |
| **Macro Average** | **54.32%** | **59.65%** | **52.36%** | **42.13%** | **57.34%** |
| **Geometric Mean** | **50.19%** | **57.30%** | **48.03%** | **36.41%** | **54.53%** |

> **指标说明**：
> - **Overall (Weighted)**：按各 category 题目数加权（即总体正确率）。
> - **Macro Average**：9 个 category 的算术平均，等权看待每个类别。
> - **Geometric Mean**：9 个 category 的几何平均，对低分类别更敏感，惩罚表现不均衡。

**排名（Overall）：hindsight (67.66%) > EverMemOS (64.73%) > cognee (63.17%) > MemosCloud (61.51%) > MemuCloud (52.75%)**

**结论**：
- **hindsight 综合指标最高**：三种指标均领先；Geometric 仅比 Macro 低 2.35pp，表现最均衡。
- **MemuCloud 最不均衡**：Geometric 比 Macro 低 5.72pp，Temporal（17.92%）和 Pattern_recognition（23.19%）是明显短板。
- **Temporal 是所有系统的共同短板**：最高仅 45.85%（hindsight），时序推理仍是最大挑战。
- **Unanswerable 识别两极分化**：cognee（92.14%）保守准确，hindsight（75.38%）最激进。
- **EverMemOS 在 Causal（50.40%）和 Multi_hop（44.94%）上表现最佳**，与 hindsight 并列推理类第一梯队。

---

## 二、LifeBench search + add 总时延

> 数据来源：各系统 `add_latency.json` / `search_latency.json`

| 指标 | cognee | hindsight | MemosCloud | MemuCloud | EverMemOS |
|---|---|---|---|---|---|
| add sessions 数 | 3,650 | 3,650 | 3,650 | 3650 | 3,650 |
| search QAs 数 | 3,380 | 3,380 | 3,380 | 3380 | 3,380 |
| 单次 add 均值时延 | 182.6s | 111.2s | 90.9s | 7.19s | 224.0s |
| 单次 search 均值时延 | 84.3s | 48.1s | 3.2s | 5.18s | 15.4s |


**结论**：
- **MemuCloud add 最快（7.19s）**，输入的文本进行切分，分类存储，没有大规模使用llm进行深度处理。
- **EverMemOS add 最慢（224.0s）**，但按天批量处理；cognee search 最慢（84.3s）。
- **hindsight 居中**：add 111.2s、search 48.1s，在准确率最高（67.66%）的前提下时延处于中游。

---

## 三、LifeBench 总 token 消耗量

| 指标 | cognee | hindsight | MemosCloud | MemuCloud | EverMemOS |
|---|---|---|---|---|---|
| llm_total_tokens | 114,817,842 (114.8M) | 31,280,543 (31.3M) | — | — | 116,324,757 (116.3M) |
| llm_request_count | 25,756 | 3,795 | — | — | 23,863 |
| 平均 token/请求 | 4,458 | 8,243 | — | — | 4,875 |
| 最终存储占用 | 1,149 MB | 1,152 MB | — | — | 9,661 MB |

> MemosCloud / MemuCloud 的 LLM 消耗发生在远端服务器，API 响应不含 usage 等字段，客户端无法记录。

**结论**：
- **hindsight 是 token 效率最高**：仅 31.3M token（EverMemOS/cognee 的 27%），请求次数最少（3,795）但单次最重（8,243 token/请求）。
- **EverMemOS 与 cognee 消耗相当**：116.3M vs 114.8M，请求次数也接近（23,863 vs 25,756）。
- **存储占用两档分化**：本地系统 1.1~9.7GB（EverMemOS 多库架构最重），远端系统不可观测。

---

## 四、EverMemOS 完整 LifeBench 实验

> 数据来源：`results/lifebench-evermemos/`。准确率（64.73%）、时延、token省略。

### 4.1 总体结果

| 指标 | 值 |
|------|-----|
| **总时长** | **97,416.57s（27.1h）** |
| Add Search | 75,939.80s |
| Answer | 17,412.94s（3,380 题）|
| Evaluate | 4,057.78s（3,380 题）|

### 4.2 资源消耗（tracker_merged）

| 指标 | 值 | 备注 |
|------|-----|------|
| 内存峰值 | 525.95 MB | 内存和cpu均为本地评测程序的占用 |
| 内存均值 | 271.73 MB | |
| CPU 峰值 | 99.9% |  |
| CPU 均值 | 7.85% | |
| 最终存储 | 9,661 MB | docker 卷明细见下 |
| 存储 delta | 108.5 MB | 索引建立阶段 delta 达 3,423.5 MB |
| 采样时长 | 71,439.47s（19.8h）| 覆盖 Add+Search 阶段 |

**存储 breakdown（docker 卷，MB）**：

| 卷 | 大小 |
|----|-----:|
| mongodb_data | 4,603 |
| milvus_minio_data | 3,323 |
| milvus_data | 1,336 |
| elasticsearch_data | 276 |
| milvus_etcd_data | 123 |
| redis_data | 0.009 |
| **合计** | **9,661** |

**结论**：
- **27.1h 总时长中 Add Search 占 78%**（75,940s）——记忆提取 + 多库同步（Mongo/ES/Milvus）是主要瓶颈。
- **docker容器的虚拟环境开销较大**：实际运行时，docker容器的内存占用平均7G，cpu占用平均430%。如果不进行虚拟环境占用的限制会出现异常运行

---

## 五、三系统 Locomo 串行实验对比


### 5.1 实验配置

| 项 | 值 |
|----|-----|
| 数据集 | LoCoMo|
| 执行参数 | `--serial` + `--enable-tracker --tracker-interval 1.0` |

### 5.2 准确率与总时长

| 系统 | 总时长 | Add Search | Answer  | Evaluate  | **准确率** |
|------|:------:|:----------:|:--------------:|:----------------:|:---------:|
| **EverMemOS** | 19,948.8s（5.5h）| 13,428.1s | 6,351.1s | 169.5s | **85.41%**（199/233）|
| MemosCloud | 3,183.8s（53min）| 2,065.3s | 971.9s | 146.3s | 79.83%（186/233）|
| MemuCloud | 1,412.8s（23.5min）| 255.2s | 1,029.0s | 128.6s | 79.83%（186/233）|

### 5.3 单次操作时延（per-op 均值）

| 系统 | add 单次 | search 单次 | answer 单次 |
|------|:--------:|:-----------:|:-----------:|
| EverMemOS | 280.12s | 11.95s | 27.26s |
| MemosCloud | 49.17s | 0.84s | 4.16s |
| MemuCloud | 0.70s | 0.97s | 4.40s |


### 5.4 LLM Token 消耗

| 系统 | total_tokens | request_count | 平均 token/请求 |
|------|:---:|:---:|:---:|
| EverMemOS | 4,250,846（4.25M）| 1,211 | 3,511 |
| MemosCloud | — | — | — |
| MemuCloud | — | — | — |

> MemosCloud/MemuCloud 的 add/search LLM 消耗在远端服务器，API 响应不含 usage 等字段，无法记录。

### 5.5 本地负载（tracker timeline）

| 系统 | 内存峰值 | 内存均值 | CPU 峰值 | CPU 均值 | 采样点 |
|------|:-------:|:-------:|:-------:|:-------:|:-----:|
| EverMemOS | 146.61 MB | 105.40 MB | 95.4% | 29.32% | 14,155 |
| MemosCloud | 75.67 MB | 60.91 MB | 29.2% | 1.96% | 1,630 |
| MemuCloud | 69.85 MB | 63.21 MB | 25.9% | 3.30% | 1,284 |

> 本地负载反映测试进程：EverMemOS 本地跑完整 docker 服务栈（Mongo/ES/Milvus/Redis），内存/CPU 显著高于远端 API 客户端（memos/memu 仅为 HTTP 客户端）。

### 5.6 存储占用

| 系统 | 最终存储 | 存储增量 |
|------|:-------:|:-------:|
| EverMemOS | 1,116 MB | 213 MB |
| MemosCloud | —（远端）| — |
| MemuCloud | —（远端）| — |

### 5.7 综合对比

| 维度 | MemosCloud | MemuCloud | EverMemOS | 
|------|:---:|:---:|:---:|
| 准确率 | 79.83% | 79.83% | **85.41%** | **EverMemOS (+5.6pp)** |
| 总时长 | 53min | **23.5min** | 5.5h | **MemuCloud** |
| add 单次 | 49.2s | **0.70s** | 280.1s | **MemuCloud**（远端无 LLM 提取）|
| search 单次 | **0.84s** | 0.97s | 12.0s | **MemosCloud** |
| answer 单次 | **4.2s** | 4.4s | 27.3s | **MemosCloud** |
| token 消耗 | — | — | 4.25M | — |
| 内存峰值 | 75.7MB | 69.9MB | 146.6MB | **MemuCloud** |
| CPU 峰值 | 29.2% | 25.9% | 95.4% | **MemuCloud** |

**结论**：
- **EverMemOS 准确率最高（85.41%）**，但代价是 5.5h 总耗时与最高资源占用。
- **MemosCloud 与 MemuCloud 准确率完全一致（186/233）**，但 MemosCloud 多花 2.3 倍时间——其服务器端 LLM 记忆提取带来的质量提升在简单数据集上不明显。
- **架构本质差异**：EverMemOS 本地服务栈（内存 146MB、CPU 峰值 95%），memos/memu 仅远端 API 客户端（~70MB、CPU <30%）。

