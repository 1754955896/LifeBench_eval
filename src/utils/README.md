# utils/ — 工具函数与服务

| 文件 | 作用 |
|------|------|
| `config.py` | YAML 配置加载（支持 `${ENV_VAR:default}` 环境变量替换）、`normalize_system_config()` 校验与补全 |
| `logging.py` | `setup_logger()`：日志初始化 |
| `retry.py` | `retry_with_backoff()`：异步重试 |
| `llm_proxy.py` | LLM 代理服务：转发请求到真实上游并记账 token（`python -m src.utils.llm_proxy`） |
| `recall_evaluator.py` | 召回评测（LLM judge）：证据覆盖率 + 可回答性 → `recall_results.json` |
| `recall_text_match.py` | 召回评测（纯文本匹配、零 LLM）：适合存原文的系统 → `recall_results_text_match.json` |
| `direct_evidence_baseline.py` | 直接证据基线：跳过检索、把 golden 证据直接喂给 LLM，给出准确率上界（oracle） |
| `split_cognee_results.py` | 把 cognee 的图节点检索结果拆成单项，便于计算 recall@K |

注意：本目录的 `logging.py` 会遮蔽标准库 `logging`，因此本目录脚本要用 `python -m` 启动（例如 `python -m src.utils.llm_proxy`），直接 `python src/utils/llm_proxy.py` 会循环导入报错。
