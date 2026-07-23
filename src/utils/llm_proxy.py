"""
LLM Proxy Service - forwards requests to real LLM and tracks token usage.

Usage:
    # Terminal 1: Start the proxy
    python -m src.utils.llm_proxy

    # Terminal 2: Run evaluation with proxy URL
    # Set LLM_BASE_URL=http://localhost:8000 in your .env
"""
import os
import json
import threading
import time
from pathlib import Path
from flask import Flask, request, Response
from dotenv import load_dotenv

# Load .env from project root
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

app = Flask(__name__)

# Real LLM configuration
REAL_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
REAL_API_KEY = os.getenv("LLM_API_KEY", "")
REAL_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

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

    # Create client with real LLM config
    client = OpenAI(api_key=REAL_API_KEY, base_url=REAL_BASE_URL)

    data = request.json
    model = data.get("model", REAL_MODEL)
    messages = data.get("messages", [])

    # Extract prompt preview from messages
    prompt_preview = ""
    if messages:
        last_msg = messages[-1] if messages else {}
        prompt_preview = last_msg.get("content", "")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            stream=data.get("stream", False),
            # Pass through other params
            **{k: v for k, v in data.items()
               if k not in ["model", "messages", "max_tokens", "temperature", "top_p", "stream"]}
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

        return Response(
            response.model_dump_json(),
            mimetype="application/json"
        )

    except Exception as e:
        print(f"[LLM Proxy] Error: {e}")
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
def get_token_stats():
    """Get current token statistics with records."""
    stats = token_stats.get_stats()
    stats["records"] = token_stats.get_records()
    return Response(
        json.dumps(stats),
        mimetype="application/json"
    )


@app.route("/token-stats", methods=["DELETE"])
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


if __name__ == "__main__":
    port = int(os.getenv("LLM_PROXY_PORT", "18443"))
    print(f"[LLM Proxy] Starting on http://0.0.0.0:{port}")
    print(f"[LLM Proxy] Forwarding to {REAL_BASE_URL}")
    print(f"[LLM Proxy] Model: {REAL_MODEL}")
    print(f"[LLM Proxy] Token stats saved to: {token_stats._save_path}")
    app.run(host="0.0.0.0", port=port, threaded=True)
