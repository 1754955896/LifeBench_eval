
## 1. 实验设计与结果对比

### 1.1 实验环境

- **评估框架**：LifeBench_eval（通用评测流水线）
- **数据集**：
  - **LoCoMo**
  - **LifeBench**（完整3380QA数据集）
- **记忆系统**：Memos Cloud API（`https://memos.memtensor.cn/api/openmem/v1`）
- **LLM**：DeepSeek-v4-flash

### 1.2 各方法准确率对比

| 方法 / 系统 | 数据集 | 准确率 | 备注 |
|-------------|--------|--------|------|
| **OmniMemEval** (官方) | LoCoMo  | 88.0% (GPT-4o-mini) <br> 76.0% (DeepSeek) | 官方基线 |
| **LifeBench_eval + Hindsight** | LoCoMo | 85% | 说明cli正确实现 |
| **LifeBench_eval + Memos (旧版)** | LoCoMo | 33.48% | 旧adapter，存在时间戳错误和检索割裂 |
| **LifeBench_eval + Memos (新版)** | LoCoMo  | 79.4% | 修复adapter |
| **LifeBench_eval + Memos (新版)** | LifeBench | 61.51% | 使用 DeepSeek-v4-flash |

### 1.3 LLM 开销统计（完整 LifeBench 评估）

| 指标 | 数值 |
|------|------|
| API 调用次数 | 6,760 次 |
| 总 Tokens 量 | 23,133,400 Tokens |
| 总耗时 | 45,382 秒（≈12.6 小时） |

> 该开销为使用 LifeBench_eval 框架在 LifeBench 数据集上运行完整流水线（ADD + SEARCH + ANSWER + EVALUATE）所产生的 LLM 消耗，包含答案生成和 LLM Judge 评估两个环节。

## 2. adapter修改总结

新版adapter在**存储策略、检索策略、API 参数**三方面进行了重构，参考了 OmniMemEval 官方实现（`memos_client.py`）。

### 2.1 存储（ADD）策略变更

| 方面 | 旧版 (old) | 新版 (new) | 说明 |
|------|------------|------------|------|
| **user_id** | `conversation_id`（单用户） | 每个说话人独立：`{conv_id}_speaker_{name}` | 为每位说话人建立独立的记忆空间 |
| **角色分配** | 根据 `speaker_name` 是否含 "assistant" 简单判断 | 从当前说话人视角：自己的话 → `user`，对方的话 → `assistant` | 模拟个人记忆库，使系统更聚焦于该说话人的事实 |
| **消息内容** | 原始 `msg.content` | 添加说话人前缀：`f"{speaker_name}: {msg.content}"` | 增强记忆中的身份信息，便于 LLM 识别来源 |
| **写入次数** | 每 chunk 1 次 | 每 chunk 按说话人数分别写入 | 存储量翻倍，但可提升检索覆盖度 |

### 2.2 检索（SEARCH）策略变更

| 方面 | 旧版 (old) | 新版 (new) | 说明 |
|------|------------|------------|------|
| **搜索目标** | 仅 `conversation_id` 作为 `user_id` | 遍历所有已注册说话人的 `user_id`，分别搜索并合并结果 | 确保检索到所有说话人的记忆 |
| **去重合并** | 无 | 按内容前 100 字符去重，并按 `score` 排序截断 | 避免重复记忆，保留最相关片段 |

---

## 3. 关键问题分析与修复

1. **修复时间戳**：从 `chunk.session_time_str` 精确解析实际对话时间（如 `"1:56 pm on 8 May, 2023"`），并回退到 `chat_time` 字段。
2. **多视角存储与检索**：为每个说话人建立独立记忆库，并在检索时合并所有说话人的结果，确保双方信息都被考虑。
3. **角色与内容增强**：按说话人视角分配 `role`，并在内容中添加说话人前缀，使记忆条目更明确。


---

## 4. 结论

通过重构 Memos Cloud adapter，解决了时间戳、检索范围、角色分配等核心问题，，使其更接近 OmniMemEval 官方基线。后续验证其他记忆系统在lifebench_eval流水线和lifebench数据集的具体效果