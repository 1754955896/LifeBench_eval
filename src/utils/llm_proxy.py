"""
LLM Proxy Service - forwards requests to real LLM and tracks token usage.

Usage:
    # Start with default passthrough (no request transformation)
    python -m src.utils.llm_proxy

    # Start with cognee tracker's request transform (handles liteLLM quirks)
    python -m src.utils.llm_proxy --system cognee
    python -m src.utils.llm_proxy -s default
"""
import argparse
import logging
import os
import json
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from flask import Flask, request, Response
from dotenv import load_dotenv

# Load .env from project root
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Real LLM configuration
REAL_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
REAL_API_KEY = os.getenv("LLM_API_KEY", "")
REAL_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# Request / response transformation — set via --system CLI flag.
# Default: identity passthrough (forwards request/response JSON unchanged).
_transform_llm_request: Callable[[dict], dict] = staticmethod(lambda data: dict(data))
_wrap_llm_response: Callable[[dict, dict | None], dict] = staticmethod(
    lambda resp, ti: resp
)

# Token statistics
class TokenStats:
    def __init__(self):
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.request_count = 0
        self._records = []
        self._save_path = project_root / "results" / "llm_token_stats.json"

    def add(self, usage, prompt_preview: str = ""):
        with self._lock:
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
            self.request_count += 1

            self._records.append({
                "request_id": self.request_count,
                "timestamp": time.time(),
                "prompt_preview": prompt_preview[:100] if prompt_preview else "",
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            })

            self._save()

    def _save(self):
        try:
            self._save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._save_path, "w", encoding="utf-8") as f:
                json.dump({
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                    "total_tokens": self.total_tokens,
                    "request_count": self.request_count,
                    "records": self._records,
                }, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def get_stats(self):
        with self._lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "request_count": self.request_count,
            }

    def get_records(self):
        with self._lock:
            return list(self._records)

    def reset(self):
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.request_count = 0
            self._records = []


token_stats = TokenStats()


@app.route("/v1/chat/completions", methods=["POST"])
@app.route("/chat/completions", methods=["POST"])  # Support both paths
def chat_completions():
    """Proxy endpoint for chat completions."""
    try:
        from openai import OpenAI
    except ImportError:
        return Response(
            json.dumps({"error": "openai package not installed"}),
            status=500,
            mimetype="application/json"
        )

    client = OpenAI(api_key=REAL_API_KEY, base_url=REAL_BASE_URL)

    raw_data = request.json or {}
    # Let the system-specific tracker sanitize the request (e.g. fold
    # liteLLM-expanded extra_body keys back into extra_body).
    data = _transform_llm_request(raw_data)

    # Extract prompt preview from messages for token tracking
    messages = data.get("messages", [])
    prompt_preview = ""
    if messages:
        last_msg = messages[-1] if messages else {}
        prompt_preview = last_msg.get("content", "")

    try:
        # Separate standard OpenAI SDK parameters from provider-specific
        # extra_body.  Standard params are passed as top-level kwargs;
        # everything else goes into extra_body so the SDK doesn't choke.
        _SDK_PARAMS = frozenset({
            "model", "messages",
            "max_tokens", "max_completion_tokens",
            "temperature", "top_p", "top_k",
            "stream", "stream_options",
            "stop", "n",
            "frequency_penalty", "presence_penalty",
            "logprobs", "top_logprobs", "logit_bias",
            "user", "seed",
            "response_format",
            "tools", "tool_choice",
            "parallel_tool_calls",
            "functions", "function_call",
            "metadata", "store",
            "reasoning_effort",
            "service_tier",
            "modalities", "audio",
        })

        extra_body: dict = {}
        if "extra_body" in data:
            eb = data["extra_body"]
            if isinstance(eb, dict):
                extra_body.update(eb)
            elif isinstance(eb, str):
                try:
                    extra_body.update(json.loads(eb))
                except (TypeError, ValueError):
                    pass

        # Strip internal metadata before forwarding to the real LLM
        tool_info = data.pop("_proxy_tool_info", None)

        sdk_kwargs: dict = {}
        for key, value in data.items():
            if key == "extra_body":
                continue
            if key in _SDK_PARAMS:
                sdk_kwargs[key] = value
            else:
                extra_body[key] = value

        response = client.chat.completions.create(
            **sdk_kwargs,
            extra_body=extra_body if extra_body else None,
        )

        # Track token usage
        if response.usage:
            usage_dict = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            token_stats.add(usage_dict, prompt_preview)
            print(f"[LLM Proxy] tokens: prompt={usage_dict['prompt_tokens']}, "
                  f"completion={usage_dict['completion_tokens']}, "
                  f"total={usage_dict['total_tokens']}")

        # Let the system tracker wrap the response (e.g. JSON→tool_call)
        response_dict = json.loads(response.model_dump_json())
        response_dict = _wrap_llm_response(response_dict, tool_info)

        return Response(
            json.dumps(response_dict),
            mimetype="application/json"
        )

    except Exception as e:
        print(f"[LLM Proxy] Error: {e}", file=sys.stderr)
        return Response(
            json.dumps({"error": str(e)}),
            status=500,
            mimetype="application/json"
        )


@app.route("/v1/models", methods=["GET"])
def models():
    """Proxy endpoint for models list."""
    return Response(
        json.dumps({
            "object": "list",
            "data": [{"id": REAL_MODEL, "object": "model"}]
        }),
        mimetype="application/json"
    )


@app.route("/token-stats", methods=["GET"])
@app.route("/v1/token-stats", methods=["GET"])
def get_token_stats():
    """Get current token statistics with records."""
    stats = token_stats.get_stats()
    stats["records"] = token_stats.get_records()
    return Response(
        json.dumps(stats),
        mimetype="application/json"
    )


@app.route("/token-stats", methods=["DELETE"])
@app.route("/v1/token-stats", methods=["DELETE"])
def reset_token_stats():
    """Reset token statistics."""
    token_stats.reset()
    return Response(
        json.dumps({"status": "reset"}),
        mimetype="application/json"
    )


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return Response(
        json.dumps({"status": "ok", "proxy": "llm"}),
        mimetype="application/json"
    )


def _load_transforms(system_name: str | None) -> tuple[
    Callable[[dict], dict],
    Callable[[dict, dict | None], dict],
]:
    """Resolve *system_name* to its transform and wrap callables.

    Returns ``(transform, wrap)`` — both identity passthroughs when
    *system_name* is ``None`` / ``"default"``, or when the named tracker
    cannot be imported.
    """
    _identity_transform = staticmethod(lambda data: dict(data))
    _identity_wrap = staticmethod(lambda resp, ti: resp)

    if system_name is None or system_name == "default":
        from src.trackers.system_trackers.default import DefaultTracker
        return (
            getattr(DefaultTracker, "transform_llm_request", _identity_transform),
            getattr(DefaultTracker, "wrap_llm_response", _identity_wrap),
        )

    # Add LifeBench_eval to sys.path so tracker imports resolve
    _proxy_dir = Path(__file__).resolve().parent.parent
    if str(_proxy_dir) not in sys.path:
        sys.path.insert(0, str(_proxy_dir))

    try:
        from src.trackers.system_trackers import get_tracker

        tracker_cls = type(get_tracker(system_name))
        if tracker_cls is None:
            logger.warning(
                "Unknown system '%s' — falling back to default passthrough",
                system_name,
            )
            return (_identity_transform, _identity_wrap)

        transform = getattr(tracker_cls, "transform_llm_request", _identity_transform)
        wrap = getattr(tracker_cls, "wrap_llm_response", _identity_wrap)
        return (transform, wrap)
    except Exception as exc:
        logger.warning(
            "Failed to load tracker '%s': %s — falling back to default passthrough",
            system_name, exc,
        )
        return (_identity_transform, _identity_wrap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM Proxy — forward requests and track token usage",
    )
    parser.add_argument(
        "-s", "--system",
        default=os.getenv("LLM_PROXY_SYSTEM", "default"),
        help="Memory system name (default, cognee, ...). "
             "Loads the corresponding tracker's transform_llm_request "
             "to normalize incoming requests before forwarding. "
             "Use 'default' for identity passthrough.",
    )
    args, _ = parser.parse_known_args()

    _transform_llm_request, _wrap_llm_response = _load_transforms(
        args.system if args.system != "default" else None
    )

    port = int(os.getenv("LLM_PROXY_PORT", "18443"))
    print(f"[LLM Proxy] System:     {args.system}")
    _tf_name = getattr(_transform_llm_request, "__qualname__", None) \
               or getattr(_transform_llm_request, "__name__", "passthrough")
    _wf_name = getattr(_wrap_llm_response, "__qualname__", None) \
               or getattr(_wrap_llm_response, "__name__", "passthrough")
    print(f"[LLM Proxy] Transform:  {_tf_name}")
    print(f"[LLM Proxy] Wrap:       {_wf_name}")
    print(f"[LLM Proxy] Starting on http://0.0.0.0:{port}")
    print(f"[LLM Proxy] Forwarding to {REAL_BASE_URL}")
    print(f"[LLM Proxy] Model: {REAL_MODEL}")
    print(f"[LLM Proxy] Token stats saved to: {token_stats._save_path}")
    app.run(host="0.0.0.0", port=port, threaded=True)
