"""
Local cognee smoke test: add → cognify → search.

Prerequisites:
    .env configured with LLM_API_KEY (DeepSeek) and EMBEDDING_* (SiliconFlow).
    No external databases needed — uses SQLite + LanceDB + KuzuDB defaults.

Usage:
    cd LifeBench_eval/systems/cognee
    python test_local_cognee.py
"""

import asyncio
import time

import cognee
from cognee import SearchType


TEST_DATA = [
    "The Apollo 11 mission landed on the Moon on July 20, 1969. "
    "Neil Armstrong was the first person to walk on the lunar surface, "
    "followed by Buzz Aldrin. Michael Collins remained in orbit aboard the Command Module.",

    "The Great Barrier Reef is the world's largest coral reef system, "
    "stretching over 2,300 kilometers along the northeast coast of Australia. "
    "It is composed of over 2,900 individual reef systems and is home to "
    "a vast diversity of marine life including 1,500 species of fish.",

    "Python is a high-level programming language created by Guido van Rossum "
    "and first released in 1991. It emphasizes code readability with its "
    "use of significant indentation. Python is dynamically typed and "
    "garbage-collected, supporting multiple programming paradigms.",

    "Quantum computing uses quantum bits (qubits) that can exist in "
    "superposition states, unlike classical bits which are binary. "
    "This allows quantum computers to solve certain problems, like factoring "
    "large numbers, exponentially faster than classical computers.",
]


async def main():
    print("=" * 60)
    print("  Cognee Local Smoke Test")
    print("=" * 60)

    # ---- Step 1: Clean start ----
    print("\n[1/4] Cleaning up previous data...")
    t0 = time.time()
    try:
        await cognee.forget(everything=True)
    except Exception as e:
        print(f"  Cleanup failed: {e}")
        print("  Try deleting .cognee_system/ and .cognee_data/ directories manually.")
        raise
    print(f"  Done ({time.time() - t0:.1f}s)")

    # ---- Step 2: Add data ----
    print(f"\n[2/4] Adding {len(TEST_DATA)} documents...")
    t0 = time.time()
    for i, doc in enumerate(TEST_DATA):
        await cognee.add(doc, dataset_name="smoke_test")
        print(f"  [{i+1}/{len(TEST_DATA)}] added: {doc[:60]}...")
    print(f"  Done ({time.time() - t0:.1f}s)")

    # ---- Step 3: Cognify (build knowledge graph) ----
    print("\n[3/4] Cognifying — building knowledge graph...")
    print("  (extracting entities, relationships, embeddings...)")
    t0 = time.time()
    await cognee.cognify(datasets=["smoke_test"])
    print(f"  Done ({time.time() - t0:.1f}s)")

    # ---- Step 4: Search ----
    print("\n[4/4] Searching...")

    queries = [
        ("Who walked on the Moon first?", SearchType.GRAPH_COMPLETION),
        ("What is the Great Barrier Reef?", SearchType.GRAPH_COMPLETION),
        ("Who created Python?", SearchType.CHUNKS),
    ]

    for query, search_type in queries:
        print(f"\n  Query: '{query}' (type={search_type.value})")
        print(f"  {'-' * 50}")
        t0 = time.time()
        try:
            results = await cognee.search(
                query,
                query_type=search_type,
                datasets=["smoke_test"],
            )
            elapsed = time.time() - t0
            for r in results:
                text = str(r)[:300]
                print(f"  -> {text}")
            print(f"  ({len(results)} result(s), {elapsed:.1f}s)")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("  Smoke test complete.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
