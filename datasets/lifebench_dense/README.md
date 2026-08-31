# lifebench_dense

单人生命日志（`孙雨薇`）的 **dense 子集**构建目录。包含四类数据文件、以及生成它们的构建脚本与中间产物。

- **token 口径**：`tiktoken` 的 `cl100k_base`，与 `src/utils/recall_evaluator.py` 的 `_count_tokens` 一致。
- **evidence 定义**：以 `datasets/lifebench_raw/question_id_to_evidence_mapping.json` 中 `question_id → evidence` 的映射为准，证据条目的唯一键为 `dia_id = {session_date}_{source}{phone_id}`。

---

## 顶层数据文件（4 个最终数据集）

| 文件 | 含义 | QA | 条数 | token | session | 月份 |
|---|---|---|---|---|---|---|
| `lifebench_raw.json` | 原始完整数据 | 328 | 7466 | 715168 | 365 | 01~12 |
| `lifebench_evidence.json` | 纯净 evidence | 155 | 1060 | 100863 | 148 | 04~09 |
| `lifebench_dense.json` | dense（完整上下文） | 155 | 3856 | 368416 | 183 | 04~09 |
| `lifebench_sparse.json` | sparse（均匀稀疏） | 155 | 3876 | 368919 | 364 | 01~12 |

### 三者关系

`dense` 与 `sparse` 的 **evidence 完全相同**（均严格包含 `lifebench_evidence.json` 的 155 条 QA / 1060 条证据，0 缺失），token 总量也基本一致（≈36.8 万）。唯一区别在于**上下文如何填充**：

- **dense**：只保留 4~9 月的**完整**对话上下文 —— 数据集中在目标窗口。
- **sparse**：保留 evidence 后，从**全年 12 个月**均匀采样非 evidence 数据（distractor）补齐到与 dense 相同的 token 总量 —— 数据全年均匀、不集中在目标月份。

### 月份分布（token）

| 月 | dense | sparse |
|---|---:|---:|
| 01 | 0 | 30319 |
| 02 | 0 | 30354 |
| 03 | 0 | 30305 |
| 04 | 72148 | 30248 |
| 05 | 66442 | 35331 |
| 06 | 61049 | 30391 |
| 07 | 59811 | 30430 |
| 08 | 54250 | 30269 |
| 09 | 54716 | 30324 |
| 10 | 0 | 30303 |
| 11 | 0 | 30280 |
| 12 | 0 | 30365 |

> 05 月 sparse 为 35331，是 evidence 硬约束：该月 evidence 本身即有 35331 token（4~9 月 evidence 的大头），不可删减，故高于其他月的 ~30.3k 水位。

### QA 类别分布（多标签，计数口径：类别数组含该类即 +1）

| 类别 | raw (328) | evidence / dense / sparse (155) |
|---|---:|---:|
| Single_hop | 160 | 82 |
| Multi_hop | 96 | 42 |
| Temporal | 55 | 22 |
| Conflict | 44 | 24 |
| Knowledge_update | 37 | 20 |
| Pattern_recognition(Non-declarative) | 40 | 14 |
| Causal | 30 | 16 |
| Unanswerable | 59 | 28 |
| Hidden_info | 11 | 9 |

---

## 构建链路（lineage）

```
lifebench_raw.json
 │
 ├─ build_dense_evidence_only.py ───────► files/lifebench_locomo_1people_evidence_only.json
 │                                        （剔除每个 session 中非 evidence 数据，保留全部 328 QA 的证据）
 │
 ├─ build_dense_apr_sep.py ─────────────► files/lifebench_locomo_1people_dense_apr_sep.json
 │   （evidence ≥90% 落在 4~9 月的可答题，127 QA）
 │
 ├─ build_dense_apr_sep_with_unanswerable.py
 │        └─► files/lifebench_locomo_1people_dense_apr_sep_v2.json  ──►  lifebench_evidence.json
 │             （127 可答题 + 28 Unanswerable；Unanswerable 按题面事件日期，缺日期退回 ask_time）
 │
 ├─ build_dense_full_sessions.py ───────► files/lifebench_locomo_1people_dense_full_sessions.json
 │   （v2 的 148 个非空 session 取原始完整数据，155 QA / 3205 条）
 │
 ├─ build_dense_apr_sep_full.py ────────► lifebench_dense.json
 │   （4~9 月全部 183 个 session 的完整数据，155 QA / 3856 条）
 │
 └─ build_dense_sparsity.py ────────────► lifebench_sparse.json
     （evidence + 各月均匀采样 distractor，token 总量对齐 dense，155 QA / 3876 条）
```

> 箭头 `──►` 表示最终被采用并重命名放入顶层；`──► files/` 表示中间产物存于 `files/`。

---

## files/ 目录

构建脚本与中间数据（历史产物，未删除以便回溯）。

**构建脚本**：

- `build_lifebench_datasets.py` — **通用入口**：从 `lifebench_raw.json` 一次生成 evidence / dense / sparse 三份，参数化（`--window` / `--threshold` / `--seed` / `--prefix` / `--raw` / `--mapping` / `--out-dir`）。默认参数即可复现顶层三个文件。其余脚本为单步骤的历史实现，其逻辑已合并至此。

  ```bash
  python build_lifebench_datasets.py \
      --raw ../lifebench_raw.json \
      --mapping ../../lifebench_raw/question_id_to_evidence_mapping.json \
      --out-dir .. --prefix lifebench
  ```

- `build_dense_evidence_only.py` — 剔除非 evidence 数据。
- `build_dense_apr_sep.py` — 按 evidence 时间集中度（≥90% 在 4~9 月）筛 QA。
- `build_dense_apr_sep_with_unanswerable.py` — 加入 Unanswerable（题面日期）。
- `build_dense_full_sessions.py` — v2 非空 session 取完整数据。
- `build_dense_apr_sep_full.py` — 4~9 月全部 session 完整数据。
- `build_dense_sparsity.py` — 均匀稀疏采样，对齐 dense token 总量。
- `explore_sparsity.py` — 每月条数 / token 分布探索。

**中间数据**：

| 文件 | QA | 条数 | token | session | 月份 |
|---|---|---|---|---|---|
| `lifebench_locomo_1people_evidence_only.json` | 328 | 2469 | 237276 | 312 | 01~12 |
| `lifebench_locomo_1people_dense_apr_sep.json` | 127 | 1060 | 100863 | 148 | 04~09 |
| `lifebench_locomo_1people_dense_full_sessions.json` | 155 | 3205 | 304657 | 148 | 04~09 |

---

## 数据格式

每个文件是一个长度为 1 的列表，元素为 `{sample_id, conversation, qa}`：

- `conversation`：字典，含 `speaker_a` / `speaker_b`，以及 `session_N` + `session_N_date_time` 键对（一年 365 个 session = 每天一个）。session 值为 `[{speaker, dia_id, text}]` 列表。
- `qa`：`{question, answer, evidence, category, question_type, question_id, ask_time, score_points}` 列表。