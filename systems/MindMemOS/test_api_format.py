#!/usr/bin/env python3
"""
Test script to verify SiliconFlow API model formats.
"""
import httpx
import json

# SiliconFlow API settings from .env
API_KEY = "sk-fcllchwlfsczntekleqnnptkcfkqhlcfgltzptckjqyaaoqq"
EMBED_BASE_URL = "https://api.siliconflow.cn/v1"
RERANK_BASE_URL = "https://api.siliconflow.cn/v1/rerank"

# Test embedding models
EMBED_MODELS = [
    "Qwen/Qwen3-Embedding-4B",
    "openai/Qwen/Qwen3-Embedding-4B",
    "text-embedding-3-large",
]

# Test rerank models
RERANK_MODELS = [
    "Qwen/Qwen3-Reranker-4B",
    "openai/Qwen/Qwen3-Reranker-4B",
    "cohere/qwen3-reranker-4b",
]


def test_embedding(model: str):
    """Test embedding API"""
    print(f"\nTesting embedding model: {model}")
    url = f"{EMBED_BASE_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "input": "Hello world",
        "dimensions": 2560,
    }

    try:
        response = httpx.post(url, json=data, headers=headers, timeout=30)
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"  Success! Output length: {len(result.get('data', [{}])[0].get('embedding', []))}")
            return True
        else:
            print(f"  Error: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False


def test_rerank(model: str):
    """Test rerank API"""
    print(f"\nTesting rerank model: {model}")
    url = RERANK_BASE_URL
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "query": "Hello",
        "documents": ["Hello world", "Goodbye world"],
    }

    try:
        response = httpx.post(url, json=data, headers=headers, timeout=30)
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"  Success! Results: {len(result.get('results', []))}")
            return True
        else:
            print(f"  Error: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"  Exception: {e}")
        return False


def main():
    print("=" * 70)
    print("Testing SiliconFlow API Model Formats")
    print("=" * 70)

    print("\n--- Embedding Tests ---")
    embed_results = {}
    for model in EMBED_MODELS:
        embed_results[model] = test_embedding(model)

    print("\n--- Rerank Tests ---")
    rerank_results = {}
    for model in RERANK_MODELS:
        rerank_results[model] = test_rerank(model)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    print("\nEmbedding models that work:")
    for model, success in embed_results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {model}")

    print("\nRerank models that work:")
    for model, success in rerank_results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {model}")


if __name__ == "__main__":
    main()