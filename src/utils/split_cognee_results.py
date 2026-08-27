"""
Split cognee graph-node search results into individual result items.

Cognee's search_results.json stores one result per question containing
the full knowledge graph text with Node markers. This script splits each
Node into its own result item so that recall@K evaluation can produce
meaningful per-rank metrics.

The split is lossless: nodes and edges both become ``results`` items (nodes
first, then edges tagged ``metadata.kind == "edge"``), and the top-level
``retrieval_metadata`` field is carried over unchanged.

Usage:
    cd LifeBench_eval
    python src/utils/split_cognee_results.py \\
        results/lifebench-cognee/search_results.json

    # Custom output path:
    python src/utils/split_cognee_results.py \\
        results/lifebench-cognee/search_results.json \\
        --output results/lifebench-cognee/search_results_split.json
"""
import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple


def split_node_content(raw_content: str) -> List[str]:
    """Split a cognee graph text into per-node content strings.

    Format::

        Nodes:
        Node: <title>
        __node_content_start__
        <body>
        __node_content_end__
        Node: <title2>
        ...

    Returns one string per node, combining the title and body.
    """
    nodes: List[str] = []

    # Normalize line endings
    text = raw_content.replace("\r\n", "\n").replace("\r", "\n")

    # Strip leading "Nodes:\n" if present
    text = re.sub(r"^Nodes:\n?", "", text)

    # Split on "\nNode: " — each chunk (except possibly the first) is a node
    parts = text.split("\nNode: ")
    for part in parts:
        part = part.strip()
        # Strip leading "Node: " from the first chunk
        if part.startswith("Node: "):
            part = part[6:]
        part = part.strip()
        if not part:
            continue

        # Extract title (everything before __node_content_start__)
        title_end = part.find("__node_content_start__")
        if title_end < 0:
            # No content markers — treat the whole thing as one node
            nodes.append(part.strip())
            continue

        title = part[:title_end].strip()
        rest = part[title_end + len("__node_content_start__"):]

        # Extract body (between markers)
        body_end = rest.find("__node_content_end__")
        if body_end < 0:
            body = rest.strip()
        else:
            body = rest[:body_end].strip()

        if title or body:
            combined = f"{title}\n{body}" if title and body else (title or body)
            nodes.append(combined)

    return nodes


def split_graph_content(raw_content: str) -> Tuple[List[str], List[str]]:
    """Split a cognee graph text into (node_strings, edge_strings).

    Cognee's GRAPH_COMPLETION context has two sections::

        Nodes:
        Node: <title>
        __node_content_start__
        <body>
        __node_content_end__
        ...
        Connections:
        <source> --[<relation>]--> <target>  (<description>)
        ...

    Returns the per-node strings (title + body) and the raw connection lines.
    Keeping the connections (rather than dropping them as a node-only split
    does) preserves the graph's edge/relationship information.
    """
    text = raw_content.replace("\r\n", "\n").replace("\r", "\n")

    node_part = text
    conn_part = ""
    m = re.search(r"\nConnections:\n", text)
    if m:
        node_part = text[: m.start()]
        conn_part = text[m.end():]

    nodes = split_node_content(node_part)
    edges = [ln.strip() for ln in conn_part.split("\n") if ln.strip()]
    return nodes, edges


def split_results(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Transform search_results: split each entry's graph nodes into individual results.

    Nodes and edges both become result items (nodes first, then edges), so a
    consumer that reads ``results`` sees the full graph context. Edge items are
    tagged ``metadata.kind == "edge"``; the top-level ``retrieval_metadata``
    field is carried over unchanged.
    """
    transformed = []
    total_before = 0
    total_after = 0
    total_edges = 0

    for sr in data:
        results = sr.get("results", [])
        total_before += len(results)

        new_results = []
        for r in results:
            content = r.get("content", "")
            score = r.get("score", 0)
            base_meta = dict(r.get("metadata", {}) or {})
            if "Node:" in content and "__node_content_start__" in content:
                nodes, edges = split_graph_content(content)
                for node_text in nodes:
                    new_results.append({
                        "content": node_text,
                        "score": score,
                        "metadata": dict(base_meta),
                    })
                for edge in edges:
                    new_results.append({
                        "content": edge,
                        "score": score,
                        "metadata": {**base_meta, "kind": "edge"},
                    })
                    total_edges += 1
            else:
                # Keep as-is (no graph markers)
                new_results.append(r)

        total_after += len(new_results)

        entry = {
            "question_id": sr["question_id"],
            "query": sr.get("query", ""),
            "conversation_id": sr.get("conversation_id", ""),
            "results": new_results,
        }
        if "retrieval_metadata" in sr:
            entry["retrieval_metadata"] = sr["retrieval_metadata"]

        transformed.append(entry)

    print(f"Results: {total_before} → {total_after} (avg {total_after/len(data):.1f} per question)")
    print(f"Edges folded into results: {total_edges}")
    return transformed


def main():
    parser = argparse.ArgumentParser(
        description="Split cognee graph-node search results into individual items"
    )
    parser.add_argument("input", help="Path to search_results.json")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output path (default: overwrite input)"
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"ERROR: file not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output).resolve() if args.output else input_path

    # ---- Step 1: backup ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = str(input_path) + f".bak_{ts}"
    shutil.copy2(input_path, backup_path)
    print(f"Backup: {backup_path}")

    # ---- Step 2: load ----
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} questions")

    # ---- Step 3: split ----
    transformed = split_results(data)

    # ---- Step 4: save ----
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transformed, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
