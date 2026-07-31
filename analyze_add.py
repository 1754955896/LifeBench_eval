import json

# ===== CLoud-omni (有LLM) =====
with open('results/lifebench-memos_cloud-omni/add_latency.json', encoding='utf-8') as f:
    omni_data = json.load(f)

omni_latencies = [d['latency_seconds'] for d in omni_data]
omni_latencies.sort()
n_omni = len(omni_data)
total_omni = sum(omni_latencies)

# ===== No-LLM (纯RAG, 无LLM调用) =====
with open('results/lifebench-memu_cloud-nollm/add_latency.json', encoding='utf-8') as f:
    nollm_data = json.load(f)

nollm_latencies = [d['latency_seconds'] for d in nollm_data]
nollm_latencies.sort()
n_nollm = len(nollm_data)
total_nollm = sum(nollm_latencies)

print('===========================================')
print('ADD 阶段对比: Cloud-omni (有LLM) vs No-LLM (纯RAG)')
print('===========================================\n')

print('--- 整体指标 ---')
print(f'{"":30s} {"Cloud-omni":>15s} {"No-LLM":>15s}')
print(f'{"操作次数":30s} {n_omni:>15,} {n_nollm:>15,}')
print(f'{"延迟总和 (累计)":30s} {total_omni:>15,.0f}s {total_nollm:>15,.0f}s')
print(f'{"延迟总和 (小时)":30s} {total_omni/3600:>15.1f}h {total_nollm/3600:>15.1f}h')
print(f'{"平均延迟":30s} {total_omni/n_omni:>15.2f}s {total_nollm/n_nollm:>15.2f}s')
print(f'{"中位延迟":30s} {omni_latencies[n_omni//2]:>15.2f}s {nollm_latencies[n_nollm//2]:>15.2f}s')
print(f'{"95分位延迟":30s} {omni_latencies[int(n_omni*0.95)]:>15.2f}s {nollm_latencies[int(n_nollm*0.95)]:>15.2f}s')
print(f'{"最小延迟":30s} {omni_latencies[0]:>15.2f}s {nollm_latencies[0]:>15.2f}s')
print(f'{"最大延迟":30s} {omni_latencies[-1]:>15.2f}s {nollm_latencies[-1]:>15.2f}s')
print()

avg_omni = total_omni / n_omni
avg_nollm = total_nollm / n_nollm
print('--- LLM 消耗估算 (Cloud-omni 的额外延迟 = LLM推理开销) ---')
print(f'  每次操作 LLM 推理平均耗时: {avg_omni - avg_nollm:.2f}s')
print(f'  LLM 推理累计总耗时: {(total_omni - total_nollm):.0f}s = {(total_omni - total_nollm)/3600:.1f}h')
print()

# By num_messages comparison
print('--- 按消息数分组对比 ---')
print(f'{"消息数":>6s} {"次数O":>7s} {"O均值":>8s} {"次数N":>7s} {"N均值":>8s} {"LLM推理":>8s} {"O/N比值":>8s}')
omni_by_msgs = {}
for d in omni_data:
    omni_by_msgs.setdefault(d['num_messages'], []).append(d['latency_seconds'])

nollm_by_msgs = {}
for d in nollm_data:
    nollm_by_msgs.setdefault(d['num_messages'], []).append(d['latency_seconds'])

for n in sorted(set(list(omni_by_msgs.keys()) + list(nollm_by_msgs.keys()))):
    if n > 30:
        continue
    o_t = omni_by_msgs.get(n, [])
    n_t = nollm_by_msgs.get(n, [])
    o_avg = sum(o_t)/len(o_t) if o_t else 0
    n_avg = sum(n_t)/len(n_t) if n_t else 0
    ratio = o_avg/n_avg if n_avg else 0
    llm_part = o_avg - n_avg
    print(f'{n:6d} {len(o_t):>7,} {o_avg:>7.2f}s {len(n_t):>7,} {n_avg:>7.2f}s {llm_part:>7.2f}s {ratio:>7.1f}x')
