# lifebench_dense

三人生命日志（`孙雨薇` / `于晓薇` / `冯浩然`）的 **dense 子集**构建目录。包含四类数据文件、以及生成它们的构建脚本与中间产物。

- **token 口径**：`tiktoken` 的 `cl100k_base`，与 `src/utils/recall_evaluator.py` 的 `_count_tokens` 一致。
- **evidence 定义**：以 `datasets/lifebench_raw/question_id_to_evidence_mapping.json` 中 `question_id → evidence` 的映射为准，证据条目的唯一键为 `dia_id = {session_date}_{source}{phone_id}`。

---

## 顶层数据文件（4 个最终数据集，每个含 3 个 sample）

| 文件 | 含义 | sample | QA | 条数 | token | session | 月份 |
|---|---|---|---|---|---|---|---|
| `lifebench_raw.json` | 原始完整数据 | 3 | 1008 | 22251 | 2089963 | 1095 | 01~12 |
| `lifebench_evidence.json` | 纯净 evidence | 3 | 453 | 3965 | 371479 | 482 | 04~09 |
| `lifebench_dense.json` | dense（完整上下文） | 3 | 453 | 11404 | 1070078 | 549 | 04~09 |
| `lifebench_sparse.json` | sparse（均匀稀疏） | 3 | 453 | 11401 | 1070127 | 1093 | 01~12 |

### 人物构成（3 个 sample）

| 人物 | raw QA | evidence/dense/sparse QA | dense 条数 | dense token |
|---|---:|---:|---:|---:|
| 孙雨薇 | 328 | 155 | 3856 | 368416 |
| 于晓薇 | 332 | 136 | 3631 | 338378 |
| 冯浩然 | 348 | 162 | 3917 | 363284 |

> `lifebench_raw.json` 由 `files/build_raw_3people.py` 从 `lifebench_locomo_format/lifebench_locomo_conversation_format_v2.0_3380QA.json` 抽取上述 3 人构成。

### 三者关系

`dense` 与 `sparse` 的 **evidence 完全相同**（均严格包含 `lifebench_evidence.json` 的 453 条 QA / 3965 条证据，0 缺失），token 总量也基本一致（≈107 万）。唯一区别在于**上下文如何填充**（对每个 sample 分别处理）：

- **dense**：只保留 4~9 月的**完整**对话上下文 —— 数据集中在目标窗口。
- **sparse**：保留 evidence 后，从**全年 12 个月**均匀采样非 evidence 数据（distractor）补齐到与该 sample 的 dense 相同的 token 总量 —— 数据全年均匀、不集中在目标月份。

### 月份分布（token，3 人合计）

| 月 | dense | sparse |
|---|---:|---:|
| 01 | 0 | 87452 |
| 02 | 0 | 87493 |
| 03 | 0 | 87638 |
| 04 | 205163 | 94187 |
| 05 | 190544 | 100800 |
| 06 | 172004 | 87562 |
| 07 | 170549 | 87553 |
| 08 | 167360 | 87490 |
| 09 | 164458 | 87479 |
| 10 | 0 | 87224 |
| 11 | 0 | 87519 |
| 12 | 0 | 87730 |

> sparse 各月基本对齐 ~87.5k 的水位；04、05 月偏高，是该两月 evidence 本身较大、不可删减所致。

### QA 类别分布（多标签，计数口径：类别数组含该类即 +1）

| 类别 | raw (1008) | evidence / dense / sparse (453) |
|---|---:|---:|
| Single_hop | 480 | 214 |
| Multi_hop | 326 | 147 |
| Temporal | 161 | 53 |
| Conflict | 134 | 70 |
| Knowledge_update | 118 | 54 |
| Pattern_recognition(Non-declarative) | 134 | 59 |
| Causal | 113 | 65 |
| Unanswerable | 178 | 81 |
| Hidden_info | 38 | 25 |

---

## 构建链路（lineage）

```
lifebench_locomo_conversation_format_v2.0_3380QA.json（10 人 / 3380 QA）
 │
 ├─ build_raw_3people.py ────────────────► lifebench_raw.json
 │   （抽取 孙雨薇 / 于晓薇 / 冯浩然 3 人，1008 QA）
 │
 └─ build_lifebench_datasets.py ─────────► lifebench_evidence.json / lifebench_dense.json / lifebench_sparse.json
     （对每个 sample：剔除非 evidence 数据 → evidence；4~9 月完整上下文 → dense；全年均匀采样 → sparse）
```

> 箭头 `──►` 表示最终被采用并重命名放入顶层；`──► files/` 表示中间产物存于 `files/`。

---

## files/ 目录

构建脚本与中间数据（历史产物，未删除以便回溯）。

**构建脚本**：

- `build_raw_3people.py` — 从 3380QA 抽取指定人物生成多人 `lifebench_raw.json`。
- `build_lifebench_datasets.py` — **通用入口**：从 `lifebench_raw.json` 一次生成 evidence / dense / sparse 三份，参数化（`--window` / `--threshold` / `--seed` / `--prefix` / `--raw` / `--mapping` / `--out-dir`）。raw 含多人时，每份输出为多 sample 的列表。默认参数即可复现顶层三个文件。其余脚本为单步骤的历史实现，其逻辑已合并至此。

  ```bash
  python files/build_raw_3people.py
  python files/build_lifebench_datasets.py \
      --raw lifebench_raw.json \
      --mapping ../lifebench_raw/question_id_to_evidence_mapping.json \
      --out-dir . --prefix lifebench
  ```

- `build_dense_evidence_only.py` — 剔除非 evidence 数据。
- `build_dense_apr_sep.py` — 按 evidence 时间集中度（≥90% 在 4~9 月）筛 QA。
- `build_dense_apr_sep_with_unanswerable.py` — 加入 Unanswerable（题面日期）。
- `build_dense_full_sessions.py` — v2 非空 session 取完整数据。
- `build_dense_apr_sep_full.py` — 4~9 月全部 session 完整数据。
- `build_dense_sparsity.py` — 均匀稀疏采样，对齐 dense token 总量。
- `explore_sparsity.py` — 每月条数 / token 分布探索。

**中间数据**（单人 `孙雨薇` 历史产物）：

| 文件 | QA | 条数 | token | session | 月份 |
|---|---|---|---|---|---|
| `lifebench_locomo_1people_evidence_only.json` | 328 | 2469 | 237276 | 312 | 01~12 |
| `lifebench_locomo_1people_dense_apr_sep.json` | 127 | 1060 | 100863 | 148 | 04~09 |
| `lifebench_locomo_1people_dense_full_sessions.json` | 155 | 3205 | 304657 | 148 | 04~09 |

---

## 数据格式

每个文件是一个长度为 3 的列表，元素为 `{sample_id, conversation, qa}`：

- `conversation`：字典，含 `speaker_a` / `speaker_b`，以及 `session_N` + `session_N_date_time` 键对（一年 365 个 session = 每天一个）。session 值为 `[{speaker, dia_id, text}]` 列表。
- `qa`：`{question, answer, evidence, category, question_type, question_id, ask_time, score_points}` 列表。