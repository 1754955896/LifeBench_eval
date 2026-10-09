# lifebench_dense

The **dense-subset** build directory for three people's life logs (`孙雨薇` / `于晓薇` / `冯浩然`). Contains four kinds of data files, plus the build scripts and intermediate artifacts that produce them.

- **token convention**: `tiktoken`'s `cl100k_base`, consistent with `src/utils/recall_evaluator.py`'s `_count_tokens`.
- **evidence definition**: governed by the `question_id → evidence` mapping in `datasets/lifebench_raw/question_id_to_evidence_mapping.json`; the unique key of an evidence entry is `dia_id = {session_date}_{source}{phone_id}`.

---

## Top-level data files (4 final datasets, each with 3 samples)

| File | Meaning | sample | QA | entries | token | session | month |
|---|---|---|---|---|---|---|---|
| `lifebench_raw.json` | raw complete data | 3 | 1008 | 22251 | 2089963 | 1095 | 01~12 |
| `lifebench_evidence.json` | pure evidence | 3 | 453 | 3965 | 371479 | 482 | 04~09 |
| `lifebench_dense.json` | dense (full context) | 3 | 453 | 11404 | 1070078 | 549 | 04~09 |
| `lifebench_sparse.json` | sparse (uniformly sparse) | 3 | 453 | 11401 | 1070127 | 1093 | 01~12 |

### Person composition (3 samples)

| Person | raw QA | evidence/dense/sparse QA | dense entries | dense token |
|---|---:|---:|---:|---:|
| 孙雨薇 | 328 | 155 | 3856 | 368416 |
| 于晓薇 | 332 | 136 | 3631 | 338378 |
| 冯浩然 | 348 | 162 | 3917 | 363284 |

> `lifebench_raw.json` is produced by `files/build_raw_3people.py`, extracting the above 3 people from `lifebench_locomo_format/lifebench_locomo_conversation_format_v2.0_3380QA.json`.

### Relationship among the three

`dense` and `sparse` have **identical evidence** (both strictly contain all 453 QAs / 3965 evidence entries of `lifebench_evidence.json`, with 0 missing), and their total token counts are basically equal (≈1.07M). The only difference is **how the context is filled** (handled per sample):

- **dense**: keeps only the **complete** Apr–Sep conversation context — data concentrated in the target window.
- **sparse**: after keeping the evidence, uniformly samples non-evidence data (distractors) from the **full 12 months** to reach the same total token count as that sample's dense — data uniformly spread across the year, not concentrated in the target months.

### Month distribution (token, 3 people combined)

| Month | dense | sparse |
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

> sparse months basically align at the ~87.5k watermark; months 04 and 05 are higher because those two months' evidence is itself larger and cannot be trimmed.

### QA category distribution (multi-label; counting convention: a category is +1 if the category array contains it)

| Category | raw (1008) | evidence / dense / sparse (453) |
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

## Build chain (lineage)

```
lifebench_locomo_conversation_format_v2.0_3380QA.json (10 people / 3380 QA)
 │
 ├─ build_raw_3people.py ────────────────► lifebench_raw.json
 │   (extract 孙雨薇 / 于晓薇 / 冯浩然, 1008 QA)
 │
 └─ build_lifebench_datasets.py ─────────► lifebench_evidence.json / lifebench_dense.json / lifebench_sparse.json
     (per sample: strip non-evidence data → evidence; Apr–Sep full context → dense; uniform full-year sampling → sparse)
```

> The arrow `──►` means finally adopted and renamed into the top level; `──► files/` means intermediate artifacts are stored in `files/`.

---

## files/ directory

Build scripts and intermediate data (historical artifacts, not deleted for retroactive tracing).

**Build scripts**:

- `build_raw_3people.py` — extracts specified people from 3380QA to produce the multi-person `lifebench_raw.json`.
- `build_lifebench_datasets.py` — **general entry**: generates evidence / dense / sparse at once from `lifebench_raw.json`, parameterized (`--window` / `--threshold` / `--seed` / `--prefix` / `--raw` / `--mapping` / `--out-dir`). When raw contains multiple people, each output is a list of multiple samples. The default parameters reproduce the top-level three files. The remaining scripts are single-step historical implementations, whose logic has been merged here.

  ```bash
  python files/build_raw_3people.py
  python files/build_lifebench_datasets.py \
      --raw lifebench_raw.json \
      --mapping ../lifebench_raw/question_id_to_evidence_mapping.json \
      --out-dir . --prefix lifebench
  ```

- `build_dense_evidence_only.py` — strips non-evidence data.
- `build_dense_apr_sep.py` — filters QAs by evidence time concentration (≥90% in Apr–Sep).
- `build_dense_apr_sep_with_unanswerable.py` — adds Unanswerable (by question-surface date).
- `build_dense_full_sessions.py` — v2 non-empty sessions take complete data.
- `build_dense_apr_sep_full.py` — complete data for all Apr–Sep sessions.
- `build_dense_sparsity.py` — uniform sparse sampling, aligned to the dense token total.
- `explore_sparsity.py` — explores per-month entry / token distribution.

**Intermediate data** (single-person `孙雨薇` historical artifacts):

| File | QA | entries | token | session | month |
|---|---|---|---|---|---|
| `lifebench_locomo_1people_evidence_only.json` | 328 | 2469 | 237276 | 312 | 01~12 |
| `lifebench_locomo_1people_dense_apr_sep.json` | 127 | 1060 | 100863 | 148 | 04~09 |
| `lifebench_locomo_1people_dense_full_sessions.json` | 155 | 3205 | 304657 | 148 | 04~09 |

---

## Data format

Each file is a list of length 3, whose elements are `{sample_id, conversation, qa}`:

- `conversation`: a dict containing `speaker_a` / `speaker_b`, plus `session_N` + `session_N_date_time` key pairs (365 sessions a year = one per day). A session's value is a list of `[{speaker, dia_id, text}]`.
- `qa`: a list of `{question, answer, evidence, category, question_type, question_id, ask_time, score_points}`.
