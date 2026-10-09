# results_clean/ evaluation results directory description

This directory stores the **cleaned final results** of LifeBench memory evaluation; each subdirectory corresponds to one independent experiment.

The results directory has been cleaned up: deleted `.bak` backups, `report.txt`, and intermediate artifacts `*_fix.json` / `*_fixed.json` / `recall_checkpoint*.json`, keeping only the final (post-fix) deliverable files.

## Naming convention

Directory name format: `lifebench[-_]<variant>-<system>[-<model>]`

| Component | Meaning | Values |
|---|---|---|
| Dataset | LifeBench memory evaluation benchmark | `lifebench` (standard) / `lifebench_offline` (offline variant) / `lifebench_locomo_1people` (LOCOMO single-person variant) |
| System | memory system under test | `cognee` / `graphiti_local` / `hindsight` / `mem0` / `direct_evidence` / `evermemos` / `evermemos_native` / `graphrag` / `memos_cloud` / `memu_cloud` / `mindmemos` |
| Model | model variant within the system (ablation) | `8b` / `14b` / `32b` / `qwen3.8MAX` / `glm5.2` (hindsight series); `qwen8` / `qwen14` / `qwen32` / `qwen3.8MAX` / `glm5.2` (evermemos series, glm5.2 corresponds to the directory's `glm52`) |

Examples:

- `lifebench-hindsight-8b` = LifeBench dataset × hindsight system × 8b model variant
- `lifebench_offline-hindsight` = offline dataset × hindsight system

## Experiment directories

### lifebench-direct_evidence
- **System**: direct-evidence baseline (Direct Evidence)
- **Description**: bypasses any retrieval system, directly pastes the question's golden evidence into the prompt for the LLM to answer, serving as the **upper bound** reference. Therefore it only has `answer_results.json` and `eval_results.json`, without retrieval / recall / latency files.

### lifebench-cognee
- **System**: cognee (graph memory system)
- **Retrieval**: `retriever_type = graph_completion`

### lifebench-graphiti_local
- **System**: graphiti local graph memory system
- **Retrieval**: layered retrieval `layers = {entity, edge, episode}`

### lifebench-mem0
- **System**: mem0 (`mode = oss`)

### lifebench-hindsight
- **System**: hindsight memory system (default config)

### lifebench-hindsight-{8b,14b,32b,qwen3.8MAX,glm5.2}
- **System**: hindsight
- **Model variant**: the suffix denotes different model configs, used for model-ablation comparison.

### lifebench_offline-hindsight
- **Dataset**: offline dataset variant
- **System**: hindsight

### lifebench-evermemos
- **System**: evermemos memory system (default config)

### lifebench-evermemos-{qwen8,qwen14,qwen32,qwen3.8MAX,glm52}
- **System**: evermemos
- **Model variant**: the suffix denotes different in-system model configs (the qwen series named by parameter scale, glm52 = GLM-5.2), used for model-ablation comparison.

### lifebench-graphrag
- **System**: graphrag graph memory system

### lifebench-memos_cloud
- **System**: memos (cloud deployment)

### lifebench-memu_cloud
- **System**: memu (cloud deployment)
- **Recall evaluation**: uses the text-match approach; the recall result file is `recall_results_text_match.json`

### lifebench-mindmemos
- **System**: mindmemos

### lifebench_offline-evermemos
- **Dataset**: offline dataset variant
- **System**: evermemos

### lifebench_offline-graphrag
- **Dataset**: offline dataset variant
- **System**: graphrag

### lifebench_locomo_1people-{cognee,graphiti_local,hindsight,mem0,graphrag,mindmemos,evermemos}
- **Dataset**: LOCOMO single-person (`locomo_1people`) variant
- **System**: cognee / graphiti_local / hindsight / mem0 / graphrag / mindmemos / evermemos
- **Description**: seven-system comparison on the single-person LOCOMO dataset; differs from the standard `lifebench` dataset only in question source; this variant did not run recall evaluation, so there is no `recall_results.json`.


## File descriptions

File naming within each experiment directory is uniform, with the following meanings:

| File | Meaning |
|---|---|
| `search_results.json` | retrieval results: the context retrieved from the memory system for each question, with `retrieval_metadata` (adapter, total_results, etc.) |
| `answer_results.json` | QA results: the `answer` generated from the retrieved context, `golden_answer`, and question metadata (`question_type`, `score_points`, `conversation_id`, etc.) |
| `eval_results.json` | QA evaluation: LLM judge scoring of answers (`accuracy`, `weighted_score`), metadata contains the judge model and fix info |
| `recall_results.json` | recall evaluation: evidence coverage / answerability / precision (`recall`, `recall@k`, `precision`, `answerable_rate`, etc.) |
| `add_latency.json` | write latency: time for each session to be imported into the memory system (`latency_seconds`) |
| `search_latency.json` | retrieval latency: retrieval time per question (`latency_seconds`) |
| `checkpoint_default.json` | pipeline stage checkpoint (`run_name`, `completed_stages`, etc.) |
| `llm_token_stats.json` | LLM token usage statistics (`prompt_tokens` / `completion_tokens` / `total_tokens` / `request_count` / `records`), only present in newer experiment directories |

### Per-file details

**`search_results.json`** (list, 3380 entries)

One retrieval record per question, fields: `question_id` / `query` / `conversation_id` / `results` (the retrieved top-k context) / `retrieval_metadata` (`adapter`, `total_results`, and system-specific fields such as cognee's `retriever_type`, graphiti's `layers`, hindsight's `budget`/`entities`, mem0's `mode`).

**`answer_results.json`** (list, 3380 entries)

One QA record per question, fields: `question_id` / `question` / `answer` (generated answer) / `golden_answer` / `category` / `conversation_id` / `formatted_context` / `metadata` (`ask_time`, `question_type`, `score_points`, etc.).

**`eval_results.json`** (object)

- Top level: `total_questions` / `correct` / `accuracy` / `weighted_score` / `detailed_results` / `metadata`
- `metadata`: judge model `deepseek-v4-flash`, `num_runs=3` + `aggregation=majority_vote` (three judgments take the majority), plus `fix_info` / `rejudged` fields — recording the fix situation for questions where the judge returned an empty response (accuracy is higher after fixes).

**`recall_results.json`** (object)

- Top-level scalar metrics: `recall` (= macro coverage), `recall_at_5` / `recall_at_20`, `precision` / `precision_at_5` / `precision_at_20`, `answerable_rate`, `covered_and_answerable_rate`, `coverage_rate_micro`, `redundancy`, `avg_results_per_question`, `avg_tokens_per_*`, etc.
- `per_question`: per-question detail (2795 questions with evidence mapping)
- `by_source`: coverage grouped by evidence source (sms/note/calendar/photo/call/push/agent_chat)
- `rejudge_metadata`: fix statistics (targeted/recovered/still_failed) for recall-judge (`glm-5.2`) failed questions

**`add_latency.json`** (list, ~3650 sessions)

Fields: `date` / `session_id` / `num_chunks` / `num_messages` / `latency_seconds` / `added` / `failed` / `metadata`.

**`search_latency.json`** (list, 3380 entries)

Fields: `question_id` / `conversation_id` / `latency_seconds`.

**`checkpoint_default.json`** (object)

Pipeline execution progress, fields: `run_name` / `completed_stages` / `answered_qa_ids` / `sample_add_completed` / `sample_search_completed` / `date_add_completed` / `date_search_completed` / `last_updated`.

## Shared model config (global, not a per-directory differentiator)

These configs are consistent across all experiment directories, controlled uniformly by `.env` / `config/systems/*.yaml`, and are not used as directory-name differentiators:

| Purpose | Model |
|---|---|
| Answer-generation LLM | `deepseek-v4-flash` |
| QA evaluation judge | `deepseek-v4-flash` |
| Recall-evaluation judge | `glm-5.2` |
| Default embedding | `Qwen/Qwen3-Embedding-4B` |
| Default reranker | `Qwen/Qwen3-Reranker-4B` |

The hindsight series' `-8b/-14b/-32b/-qwen3.8MAX/-glm5.2` suffixes are exactly the ablation variants of the hindsight in-system models (embedding / retrieval model).

## Compressed archive description

The raw data subdirectories (`results_clean/*/`) are too large (a single `search_results.json` can reach 400+ MB, exceeding GitHub's 100 MB single-file limit), so they have been gzip/zip compressed into `.zip` archives before committing, and the raw subdirectories are ignored in `.gitignore` (`results_clean/*/`).

The archive files sit alongside this README, with this naming rule:

- One `.zip` per experiment: `<directory name>.zip` (after extraction restores to `results_clean/<directory name>/`)
- `lifebench-cognee`, being the largest, is split into two packages:
  - `lifebench-cognee_search_results.zip` (only `search_results.json`)
  - `lifebench-cognee_rest.zip` (the remaining files)
  - extracting both into the same directory restores the complete `lifebench-cognee/`

Compression is **lossless** (zip DEFLATE / LZMA), byte-for-byte identical to the original files after extraction.

Note: the individual JSONs of `lifebench-evermemos.zip`, `lifebench-evermemos-glm52.zip`, `lifebench-evermemos-qwen14.zip`, and `lifebench-mindmemos.zip` still exceed the 100 MB limit even after DEFLATE, so they use **LZMA compression** (sizes ~13 MB / 13 MB / 12 MB / 47 MB respectively). LZMA archives can only be extracted via **Method 1 (Python zipfile)**; `unzip` and PowerShell `Expand-Archive` do not support them.

### Extraction and restoration

All archives embed the top-level directory name (`<directory name>/<file>`); extracting all `.zip`s into `results_clean/` automatically restores the original directory structure; cognee's two packages extracted to the same location auto-merge back into the complete `lifebench-cognee/`.

From the **project root** (`LifeBench_eval/`), choose one method:

**Method 1: Python (cross-platform, recommended, always installed in this repo's environment)**

```bash
python -c "import zipfile, glob; [zipfile.ZipFile(f).extractall('results_clean') for f in glob.glob('results_clean/*.zip')]"
```

**Method 2: Git Bash / Linux / macOS (`unzip`)**

```bash
cd results_clean && unzip -o '*.zip' && cd ..
```

**Method 3: Windows PowerShell (`Expand-Archive`)**

```powershell
Get-ChildItem results_clean\*.zip | ForEach-Object { Expand-Archive $_.FullName -DestinationPath results_clean -Force }
```

After extraction, `results_clean/<directory name>/` will reappear (these subdirectories are still ignored by `.gitignore` and do not enter version control).
