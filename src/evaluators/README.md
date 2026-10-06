# evaluators/ — 答案评估器

对 ANSWER 阶段的输出打分，产出 `eval_results.json`。

| 文件 | 作用 |
|------|------|
| `base.py` | `BaseEvaluator` 抽象接口：`evaluate(answer_results) -> EvaluationResult` |
| `llm_judge.py` | 默认评估器：一套统一的类型感知 prompt 覆盖所有题目类型（Single_hop / Temporal / Unanswerable 等），支持 `num_runs` 多次判定取多数 |
| `exact_match.py` | 精确匹配：生成答案与标准答案直接比对 |
| `hybrid.py` | 先精确匹配，不中再做简单相似度判断，不调用 LLM |
| `registry.py` | 评估器注册表（懒加载），`create_evaluator()` 按名创建 |

约定：

- 由 `config/datasets/*.yaml` 的 `evaluation.type` 指定（值为 `llm_judge` / `exact_match` / `hybrid`）
- 题目类型共用同一套判分口径，避免多套 prompt 导致分数不可比
