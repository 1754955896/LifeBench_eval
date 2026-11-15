#!/usr/bin/env python3
"""Plot tracker metrics from LifeBench evaluation results.

Generates 3 figures:
  1. Global timed metrics (storage, memory, CPU, llm_total_tokens, llm_request_count)
  2. Per-operation metrics (elapsed_seconds, num_messages, llm_total_tokens consumed)
  3. Sampled operation-internal timed metrics (incl. llm_total_tokens)
"""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── config ──────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print(f"Usage: python {Path(__file__).name} <results_dir>")
    print(f"Example: python {Path(__file__).name} results/lifebench_locomo_1people-cognee")
    sys.exit(1)

RESULT_DIR = Path(sys.argv[1]).resolve()
TRACKER_DIR = RESULT_DIR / "tracker"
OUT_DIR = RESULT_DIR / "plots"
N_SAMPLED = 8  # number of operations to sample for figure 3

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load data ───────────────────────────────────────────────────────────────
with open(TRACKER_DIR / "global_resource_timeline.json", encoding="utf-8") as f:
    timeline = json.load(f)

with open(TRACKER_DIR / "tracker_records.json", encoding="utf-8") as f:
    records = json.load(f)

all_samples = timeline["samples"]

# op_start / op_end are sentinel markers with all-zero metrics (545 each).
# Remove them to avoid artificial drops-to-zero in the plots.
samples = [s for s in all_samples if not s.get("entry_type")]
sentinel_count = len(all_samples) - len(samples)
print(f"Loaded {len(all_samples)} timeline samples, {len(records)} operation records")
print(f"  Filtered out {sentinel_count} sentinel markers (op_start/op_end with zero values)")

# Forward-fill anomalous zero drops in cumulative / always-positive metrics.
# Monitoring gaps produce sudden 0 values that distort the plots.
_cleaned = 0
_last_sto = 0
_last_rss = 0
_last_llm = 0
for _s in samples:
    if _s["storage_mb"] > 0:
        _last_sto = _s["storage_mb"]
    elif _last_sto > 0:
        _s["storage_mb"] = _last_sto
        _cleaned += 1
    if _s["memory_rss_mb"] > 0:
        _last_rss = _s["memory_rss_mb"]
    elif _last_rss > 0:
        _s["memory_rss_mb"] = _last_rss
        _cleaned += 1
    _tok = _s.get("extra", {}).get("llm_total_tokens", 0)
    if _tok > 0:
        _last_llm = _tok
    elif _last_llm > 0:
        _s.setdefault("extra", {})["llm_total_tokens"] = _last_llm
        _cleaned += 1
print(f"  Forward-filled {_cleaned} anomalous zero values")

# ── helpers ─────────────────────────────────────────────────────────────────
def smooth(y, window=5):
    """Simple moving average."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


# ── figure 1: global timed metrics ──────────────────────────────────────────
print("Generating Figure 1: global timed metrics...")

ts = [s["t"] for s in samples]
t0 = ts[0]
elapsed_min = [(t - t0) / 60.0 for t in ts]

storage = [s["storage_mb"] for s in samples]
rss = [s["memory_rss_mb"] for s in samples]
cpu = [s["cpu_percent"] for s in samples]
llm_total = [s.get("extra", {}).get("llm_total_tokens", 0) for s in samples]
llm_requests = [s.get("extra", {}).get("llm_request_count", 0) for s in samples]

# Baseline to start from 0 so initial offset doesn't skew the view
base_storage = storage[0]
base_llm = llm_total[0]
storage = [v - base_storage for v in storage]
llm_total = [v - base_llm for v in llm_total]

fig1, axes1 = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

# storage
axes1[0].plot(elapsed_min, storage, linewidth=0.6, color="#2196F3", alpha=0.8)
axes1[0].plot(elapsed_min, smooth(storage, 10), linewidth=1.2, color="#0D47A1")
axes1[0].set_ylabel("Storage Δ (MB)")
axes1[0].legend(["raw", "smoothed"], fontsize=7, loc="upper left")
axes1[0].grid(True, alpha=0.3)

# memory RSS
axes1[1].plot(elapsed_min, rss, linewidth=0.6, color="#4CAF50", alpha=0.8)
axes1[1].plot(elapsed_min, smooth(rss, 10), linewidth=1.2, color="#1B5E20")
axes1[1].set_ylabel("Memory RSS (MB)")
axes1[1].grid(True, alpha=0.3)

# CPU
axes1[2].plot(elapsed_min, cpu, linewidth=0.6, color="#FF9800", alpha=0.7)
axes1[2].set_ylabel("CPU (%)")
axes1[2].grid(True, alpha=0.3)

# llm_total_tokens + llm_request_count
ax = axes1[3]
ax.plot(elapsed_min, llm_total, linewidth=1.0, color="#9C27B0")
ax.set_ylabel("llm_total_tokens", color="#9C27B0")
ax.tick_params(axis="y", labelcolor="#9C27B0")
ax.set_xlabel("Elapsed (min)")
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K"))

# twin for request count
ax_r = ax.twinx()
ax_r.plot(elapsed_min, llm_requests, linewidth=1.0, color="#E91E63", linestyle="--")
ax_r.set_ylabel("llm_request_count", color="#E91E63")
ax_r.tick_params(axis="y", labelcolor="#E91E63")

fig1.suptitle(f"Global Timed Metrics ({RESULT_DIR.name})", fontsize=13, fontweight="bold")
fig1.tight_layout(rect=[0, 0, 1, 0.96])
fig1.savefig(OUT_DIR / "01_global_timed_metrics.png", dpi=150)
plt.close(fig1)
print("  -> saved 01_global_timed_metrics.png")


# ── figure 2: per-operation metrics ─────────────────────────────────────────
print("Generating Figure 2: per-operation metrics...")

# Build op -> sample mapping and time windows from sentinel markers.
# op_start / op_end are paired: op_start marks the beginning, op_end marks the end.
# Real measurements (no entry_type) between them belong to that operation.
op_windows = []    # [(t_start, t_end)] per operation
op_idx_map = []    # (sample_index_in_all, operation_index)
current_op = -1
current_t_start = None
for i, s in enumerate(all_samples):
    et = s.get("entry_type", "")
    if et == "op_start":
        current_op += 1
        current_t_start = s["t"]
    if current_op >= 0:
        op_idx_map.append((i, current_op))
    if et == "op_end" and current_t_start is not None:
        # ensure op_windows list is padded to the right length
        while len(op_windows) <= current_op:
            op_windows.append(None)
        op_windows[current_op] = (current_t_start, s["t"])

total_ops = current_op + 1
# Fill any missing windows (shouldn't happen, but be safe)
while len(op_windows) < total_ops:
    op_windows.append(None)

# Compute per-operation token deltas: collect all real samples
# (no entry_type) whose t falls within the operation's [t_start, t_end] window.
op_token_deltas = {}
op_storage_deltas = {}
op_peak_rss = {}
op_avg_cpu = {}
for op_idx in range(len(records)):
    if op_idx >= len(op_windows) or op_windows[op_idx] is None:
        op_token_deltas[op_idx] = 0
        op_storage_deltas[op_idx] = 0
        op_peak_rss[op_idx] = 0
        op_avg_cpu[op_idx] = 0
        continue
    t_start, t_end = op_windows[op_idx]
    window_samples = [s for s in all_samples
                      if not s.get("entry_type") and t_start <= s["t"] <= t_end]
    if len(window_samples) >= 2:
        first_tok = window_samples[0].get("extra", {}).get("llm_total_tokens", 0)
        last_tok = window_samples[-1].get("extra", {}).get("llm_total_tokens", 0)
        op_token_deltas[op_idx] = last_tok - first_tok
        op_storage_deltas[op_idx] = window_samples[-1]["storage_mb"] - window_samples[0]["storage_mb"]
        op_peak_rss[op_idx] = max(s["memory_rss_mb"] for s in window_samples)
        op_avg_cpu[op_idx] = sum(s["cpu_percent"] for s in window_samples) / len(window_samples)
    else:
        op_token_deltas[op_idx] = 0
        op_storage_deltas[op_idx] = 0
        op_peak_rss[op_idx] = 0
        op_avg_cpu[op_idx] = 0

adds = [r for r in records if r["operation"] == "add"]
searches = [r for r in records if r["operation"] == "search"]
add_indices = [i for i, r in enumerate(records) if r["operation"] == "add"]
search_indices = [i for i, r in enumerate(records) if r["operation"] == "search"]

add_elapsed = [r["elapsed_seconds"] for r in adds]
search_elapsed = [r["elapsed_seconds"] for r in searches]
add_num_msg = [r["result_data"].get("num_messages", 0) for r in adds]
add_num_chunks = [r["result_data"].get("num_chunks", 0) for r in adds]
add_tokens = [op_token_deltas[i] for i in add_indices]
search_tokens = [op_token_deltas[i] for i in search_indices]
add_storage_deltas = [op_storage_deltas[i] for i in add_indices]
add_peak_rss = [op_peak_rss[i] for i in add_indices]
search_peak_rss = [op_peak_rss[i] for i in search_indices]
add_avg_cpu = [op_avg_cpu[i] for i in add_indices]
search_avg_cpu = [op_avg_cpu[i] for i in search_indices]

fig2, axes2 = plt.subplots(6, 1, figsize=(14, 19))

# elapsed per operation
ax = axes2[0]
add_x = list(range(1, len(adds) + 1))
search_x = list(range(1, len(searches) + 1))
ax.bar(add_x, add_elapsed, width=0.7, color="#2196F3", alpha=0.7, label=f"add (n={len(adds)})")
ax.bar(search_x, search_elapsed, width=0.7, color="#FF5722", alpha=0.7, label=f"search (n={len(searches)})")
ax.set_ylabel("Elapsed (seconds)")
ax.set_title("Per-Operation Elapsed Time")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# add detail: num_messages
ax = axes2[1]
ax.bar(add_x, add_num_msg, width=0.7, color="#4CAF50", alpha=0.7)
ax.set_ylabel("num_messages")
ax.set_title("Add Operations: num_messages per session")
ax.grid(True, alpha=0.3, axis="y")

# llm_total_tokens consumed per operation
ax = axes2[2]
ax.bar(add_x, add_tokens, width=0.7, color="#2196F3", alpha=0.7, label=f"add")
ax.bar(search_x, search_tokens, width=0.7, color="#FF5722", alpha=0.7, label=f"search")
ax.set_ylabel("llm_total_tokens delta")
ax.set_title("Per-Operation LLM Token Consumption")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K" if v >= 1000 else f"{v:.0f}"))

# add storage delta per operation
ax = axes2[3]
ax.bar(add_x, add_storage_deltas, width=0.7, color="#2196F3", alpha=0.7)
ax.set_ylabel("storage delta (MB)")
ax.set_title("Add Operations: storage growth per session")
ax.grid(True, alpha=0.3, axis="y")
# highlight negative deltas in red
neg_indices = [i for i, d in enumerate(add_storage_deltas) if d < 0]
if neg_indices:
    neg_x = [add_x[i] for i in neg_indices]
    neg_d = [add_storage_deltas[i] for i in neg_indices]
    ax.bar(neg_x, neg_d, width=0.7, color="#FF5722", alpha=0.7)

# peak RSS per operation
ax = axes2[4]
ax.bar(add_x, add_peak_rss, width=0.7, color="#2196F3", alpha=0.7, label=f"add")
ax.bar(search_x, search_peak_rss, width=0.7, color="#FF5722", alpha=0.7, label=f"search")
ax.set_ylabel("peak RSS (MB)")
ax.set_title("Per-Operation Peak Memory (RSS)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# average CPU per operation
ax = axes2[5]
ax.bar(add_x, add_avg_cpu, width=0.7, color="#2196F3", alpha=0.7, label=f"add")
ax.bar(search_x, search_avg_cpu, width=0.7, color="#FF5722", alpha=0.7, label=f"search")
ax.set_ylabel("avg CPU (%)")
ax.set_xlabel("Operation #")
ax.set_title("Per-Operation Average CPU Usage")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

fig2.suptitle("Per-Operation Metrics", fontsize=13, fontweight="bold")
fig2.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig(OUT_DIR / "02_per_operation_metrics.png", dpi=150)
plt.close(fig2)
print("  -> saved 02_per_operation_metrics.png")


# ── figure 3: sampled operation-internal metrics ────────────────────────────
print("Generating Figure 3: sampled operation-internal metrics...")

print(f"  Mapped {len(op_idx_map)} timeline indices to {total_ops} operations, {len(op_windows)} time windows")


def sample_evenly(indices, n):
    """Pick n evenly-spaced indices from a list."""
    if len(indices) <= n:
        return sorted(indices)
    step = len(indices) / n
    return sorted(indices[int(i * step)] for i in range(n))


def get_op_timeline_samples(op_idx):
    """Return real (non-sentinel) samples around the operation's [t_start, t_end]
    time window, including one sample before t_start and one after t_end for
    context. Time is relative to the first real sample in the window."""
    if op_idx >= len(op_windows) or op_windows[op_idx] is None:
        return [], [], [], [], [], []
    t_start, t_end = op_windows[op_idx]

    # All real samples, already sorted by t
    real = [s for s in all_samples if not s.get("entry_type")]

    # Find: last sample with t < t_start  (pre-context)
    pre = None
    for s in reversed(real):
        if s["t"] < t_start:
            pre = s
            break

    # Find: in-window samples
    inside = [s for s in real if t_start <= s["t"] <= t_end]

    # Find: first sample with t > t_end  (post-context)
    post = None
    for s in real:
        if s["t"] > t_end:
            post = s
            break

    subset = []
    if pre is not None:
        subset.append(pre)
    subset.extend(inside)
    if post is not None:
        subset.append(post)
    if not subset:
        return [], [], [], [], [], []
    t_start = subset[0]["t"]
    x = [(s["t"] - t_start) for s in subset]
    raw_storage = [s["storage_mb"] for s in subset]
    base_sto = raw_storage[0]
    storage = [v - base_sto for v in raw_storage]  # baseline to 0
    rss = [s["memory_rss_mb"] for s in subset]
    cpu = [s["cpu_percent"] for s in subset]
    raw_tokens = [s.get("extra", {}).get("llm_total_tokens", 0) for s in subset]
    base_token = raw_tokens[0] if raw_tokens else 0
    tokens = [t - base_token for t in raw_tokens]  # relative to first sample
    return x, storage, rss, cpu, tokens


sampled_adds = sample_evenly(add_indices, N_SAMPLED)
sampled_searches = sample_evenly(search_indices, N_SAMPLED)

n_total = len(sampled_adds) + len(sampled_searches)
n_cols = 4
n_rows = math.ceil(n_total / n_cols)

def norm(vals):
    """Normalize to [0,1]; returns original if constant."""
    mn, mx = min(vals), max(vals)
    return vals if mx == mn else [(v - mn) / (mx - mn) for v in vals]


def plot_one_op(ax, x, sto, rss_, cpu_, tokens, rec, op_idx, include_storage=True):
    """Plot metrics normalized to [0,1] for intra-op variation analysis."""
    if not x:
        return
    has_tok = tokens and max(tokens) > 0

    lines = []
    if include_storage:
        sto_n = norm(sto)
        lines += ax.plot(x, sto_n, linewidth=1.2, color="#2196F3", label="storage")

    rss_n = norm(rss_)
    cpu_n = norm(cpu_)
    tok_n = norm(tokens)
    lines += ax.plot(x, rss_n, linewidth=1.2, color="#4CAF50", label="memory RSS")
    lines += ax.plot(x, cpu_n, linewidth=1.0, color="#FF9800", alpha=0.7, label="CPU")
    if has_tok:
        lines += ax.plot(x, tok_n, linewidth=1.0, color="#9C27B0", alpha=0.8, label="Δ llm_tokens")

    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel("normalized", fontsize=6)
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=5, loc="upper left")

    # Build title with actual value ranges
    parts = []
    if include_storage:
        parts.append(f"sto[{min(sto):.1f}~{max(sto):.1f}]")
    parts.append(f"rss[{min(rss_):.0f}~{max(rss_):.0f}]")
    parts.append(f"cpu[{min(cpu_):.0f}~{max(cpu_):.0f}]")
    tok_val = op_token_deltas.get(op_idx, 0)
    date = rec["result_data"].get("date", "?")
    el = rec["elapsed_seconds"]
    prefix = "Add" if include_storage else "Search"
    title = (f"{prefix} #{op_idx+1} | {date} | {el:.1f}s\n"
             f"{' '.join(parts)} Δtok={tok_val}")
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xlabel("relative time (s)", fontsize=6)
    ax.grid(True, alpha=0.3)


fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(18, 3.8 * n_rows))
axes3 = axes3.flatten()

plot_idx = 0

for op_idx in sampled_adds:
    ax = axes3[plot_idx]
    x, sto, rss_, cpu_, tokens = get_op_timeline_samples(op_idx)
    rec = records[op_idx]
    plot_one_op(ax, x, sto, rss_, cpu_, tokens, rec, op_idx)
    plot_idx += 1

for op_idx in sampled_searches:
    ax = axes3[plot_idx]
    x, sto, rss_, cpu_, tokens = get_op_timeline_samples(op_idx)
    rec = records[op_idx]
    plot_one_op(ax, x, sto, rss_, cpu_, tokens, rec, op_idx, include_storage=False)
    ax.set_title(ax.get_title(), fontsize=7, color="#D84315")
    plot_idx += 1

# hide unused axes
for j in range(plot_idx, len(axes3)):
    axes3[j].set_visible(False)

fig3.suptitle(
    f"Sampled Operation-Internal Timed Metrics "
    f"({len(sampled_adds)} adds, {len(sampled_searches)} searches)",
    fontsize=13, fontweight="bold",
)
fig3.tight_layout(rect=[0, 0, 1, 0.96])
fig3.savefig(OUT_DIR / "03_sampled_operation_internal.png", dpi=150)
plt.close(fig3)
print("  -> saved 03_sampled_operation_internal.png")

print(f"\nDone. Plots saved to: {OUT_DIR}")