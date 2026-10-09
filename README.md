# LifeBench_eval Memory System Evaluation Framework

[![🤗 Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange?style=flat-square)](https://huggingface.co/datasets/C1754955896/Lifebenchv2.0)
[![arXiv](https://img.shields.io/badge/arXiv-2603.03781-b31b1b?style=flat-square)](https://arxiv.org/abs/2603.03781)
[![GitHub](https://img.shields.io/badge/GitHub-LifeBench__eval-181717?style=flat-square&logo=github)](https://github.com/1754955896/LifeBench_eval)

Used to evaluate the performance of memory systems such as Mem0 in real-life scenarios.

## Introduction

LifeBench_eval is a general-purpose memory-system evaluation framework, supporting:

- **Multi-system evaluation**: Mem0, Cognee, Graphiti, Hindsight, EverMemos, MindMemos, and more
- **Three-stage pipeline**: ADD + SEARCH → ANSWER → EVALUATE
- **Resumable runs**: supports checkpoint recovery at Sample, Date, and QA granularity
- **LLM Judge evaluation**: scores answer quality via a large language model
- **Dockerized deployment**: the systems under test run as Docker containers

## Quick Start

### 1. Configure the environment

```bash
# copy the environment variable template
cp env.template .env

# edit .env and fill in your API Key
vim .env
```

### 2. Run an evaluation

```bash
# full evaluation
python cli.py --dataset lifebench_locomo_format --system mem0

# specify an output directory
python cli.py --dataset lifebench_locomo_format --system mem0 --output results/my-run

# resume from a checkpoint
python cli.py --dataset lifebench_locomo_format --system mem0 --resume

# Debug mode (produces detailed logs)
python cli.py --dataset lifebench_locomo_format --system mem0 --debug
```

### 3. View results

Evaluation results are saved in the `results/{dataset}-{system}/` directory:

```
results/lifebench_locomo_format-mem0/
├── checkpoint_default.json     # checkpoint records
├── add_latency.json            # ADD latency statistics
├── search_latency.json         # SEARCH latency statistics
├── search_results.json         # retrieval results
├── answer_results.json         # answer results
├── eval_results.json           # evaluation results
└── report.txt                  # text report
```

## CLI parameters

```
--dataset TEXT         dataset name (required)
--system TEXT          system name (required)
--output PATH          output directory (default: results/{dataset}-{system})
--resume               resume from a checkpoint
--debug                enable Debug mode
--max-workers N        number of concurrent threads (default: 10)
--rerank-model MODEL   rerank model name
--rerank-provider PROVIDER  rerank provider
```

## Leader Board

Accuracy (%) on LifeBench (3380 QA) and LoCoMo. **Micro** is the accuracy over all questions; **Macro** is the arithmetic mean over the nine question types. **Gold Evidence** feeds the annotated evidence directly to the answer model, serving as the upper-bound reference for "perfect retrieval", and is not a memory system. LoCoMo results do not include adversarial questions.

| Memory system | Base model | Micro | Macro | LoCoMo |
|---|---:|---:|---:|---:|
| *Gold Evidence* † | DeepSeek-V4-Flash | 97.28 | 95.92 | — |
| Hindsight | GLM-5.2 | **74.41** | 66.49 | — |
| Hindsight | Qwen-3.8-MAX | 72.31 | 65.36 | — |
| Hindsight | DeepSeek-V4-Flash | 71.98 | **66.57** | 82.83 |
| EverMemOS | GLM-5.2 | 70.95 | 65.53 | — |
| EverMemOS | Qwen-3.8-MAX | 69.32 | 64.75 | — |
| MindMemOS | DeepSeek-V4-Flash | 67.37 | 55.89 | 86.70 |
| EverMemOS | DeepSeek-V4-Flash | 66.04 | 60.07 | 80.69 |
| Zep | DeepSeek-V4-Flash | 63.85 | 58.06 | **88.84** |
| Cognee | DeepSeek-V4-Flash | 63.17 | 54.32 | 81.12 |
| Mem0 | DeepSeek-V4-Flash | 61.54 | 55.01 | 85.32 |
| MemOS | DeepSeek-V4-Flash | 61.51 | 52.36 | 79.40 |
| MemU | DeepSeek-V4-Flash | 52.75 | 42.13 | 80.26 |
| GraphRAG | DeepSeek-V4-Flash | 30.50 | 23.26 | 82.72 |

**Bold** marks the best result among memory systems in that column (Gold Evidence is not compared). † denotes Gold Evidence (the gold-evidence upper-bound reference).

The full per-question-type breakdown (Single-hop / Multi-hop / Temporal / Non-declarative / Knowledge update / Causal / Conflict / Hidden info / Unanswerable) is below, and can also be explored and sorted in the [interactive Leaderboard](https://huggingface.co/spaces/C1754955896/Lifebench-Leaderboard).

<details>
<summary>Full results by base model × memory system × question type</summary>

**SH** Single-hop; **MH** Multi-hop; **TR** Temporal; **ND** Non-declarative; **KU** Knowledge update; **CR** Causal; **CD** Conflict detection; **HI** Hidden information; **UA** Unanswerable. **Bold** marks the best memory-system result in that column (Gold Evidence is not compared). "—" denotes not reported.

<table>
  <thead>
    <tr>
      <th>Base model</th>
      <th>Memory system</th>
      <th>SH</th><th>MH</th><th>TR</th><th>ND</th><th>KU</th><th>CR</th><th>CD</th><th>HI</th><th>UA</th><th>Micro</th><th>Macro</th><th>LoCoMo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="10">DeepSeek-V4-Flash</td>
      <td>Mem0</td>
      <td>78.09</td><td>41.41</td><td>32.45</td><td>48.48</td><td>75.84</td><td>52.55</td><td>78.07</td><td>32.47</td><td>55.73</td><td>61.54</td><td>55.01</td><td>85.32</td>
    </tr>
    <tr>
      <td>Cognee</td>
      <td>71.69</td><td>37.07</td><td>27.36</td><td>45.44</td><td>72.49</td><td>43.16</td><td>68.63</td><td>30.93</td><td><b>92.14</b></td><td>63.17</td><td>54.32</td><td>81.12</td>
    </tr>
    <tr>
      <td>Hindsight</td>
      <td>83.43</td><td>53.98</td><td>44.34</td><td><b>58.75</b></td><td>85.09</td><td><b>67.83</b></td><td>80.66</td><td><b>49.48</b></td><td>75.56</td><td>71.98</td><td><b>66.57</b></td><td>82.83</td>
    </tr>
    <tr>
      <td>MemU</td>
      <td>60.34</td><td>23.87</td><td>17.92</td><td>23.19</td><td>47.30</td><td>30.29</td><td>62.26</td><td>23.20</td><td>90.77</td><td>52.75</td><td>42.13</td><td>80.26</td>
    </tr>
    <tr>
      <td>MemOS</td>
      <td>72.94</td><td>32.91</td><td>26.98</td><td>33.65</td><td>71.47</td><td>40.21</td><td>68.63</td><td>35.57</td><td>88.89</td><td>61.51</td><td>52.36</td><td>79.40</td>
    </tr>
    <tr>
      <td>EverMemOS</td>
      <td>71.76</td><td>48.55</td><td>38.30</td><td>54.75</td><td>81.23</td><td>59.25</td><td>67.45</td><td>34.02</td><td>85.30</td><td>66.04</td><td>60.07</td><td>80.69</td>
    </tr>
    <tr>
      <td>MindMemOS</td>
      <td>79.83</td><td>42.04</td><td>42.45</td><td>35.36</td><td>69.67</td><td>38.07</td><td><b>84.67</b></td><td>26.80</td><td>84.10</td><td>67.37</td><td>55.89</td><td>86.70</td>
    </tr>
    <tr>
      <td>Zep</td>
      <td>77.78</td><td>49.46</td><td>46.23</td><td>43.92</td><td>73.78</td><td>53.89</td><td>83.25</td><td>39.69</td><td>54.53</td><td>63.85</td><td>58.06</td><td><b>88.84</b></td>
    </tr>
    <tr>
      <td>GraphRAG</td>
      <td>22.10</td><td>12.93</td><td>12.83</td><td>19.77</td><td>16.97</td><td>15.01</td><td>11.08</td><td>10.31</td><td>88.38</td><td>30.50</td><td>23.26</td><td>82.72</td>
    </tr>
    <tr>
      <td><i>Gold Evidence†</i></td>
      <td>98.14</td><td>94.85</td><td>94.91</td><td>93.73</td><td>96.14</td><td>94.10</td><td>98.11</td><td>93.30</td><td>100.00</td><td>97.28</td><td>95.92</td><td>—</td>
    </tr>
    <tr>
      <td rowspan="2">GLM-5.2</td>
      <td>Hindsight</td>
      <td><b>85.60</b></td><td>53.62</td><td>45.85</td><td>56.65</td><td>83.55</td><td>66.76</td><td>80.90</td><td>40.72</td><td>84.79</td><td><b>74.41</b></td><td>66.49</td><td>—</td>
    </tr>
    <tr>
      <td>EverMemOS</td>
      <td>79.64</td><td><b>58.95</b></td><td><b>59.25</b></td><td><b>58.75</b></td><td>84.83</td><td>57.64</td><td>78.54</td><td>40.72</td><td>71.45</td><td>70.95</td><td>65.53</td><td>—</td>
    </tr>
    <tr>
      <td rowspan="2">Qwen-3.8-MAX</td>
      <td>Hindsight</td>
      <td>83.30</td><td>50.72</td><td>38.87</td><td>54.94</td><td>83.29</td><td>64.88</td><td>82.08</td><td>44.85</td><td>85.30</td><td>72.31</td><td>65.36</td><td>—</td>
    </tr>
    <tr>
      <td>EverMemOS</td>
      <td>73.49</td><td>55.06</td><td>55.09</td><td>54.37</td><td><b>86.38</b></td><td>55.50</td><td>72.17</td><td>43.81</td><td>86.84</td><td>69.32</td><td>64.75</td><td>—</td>
    </tr>
  </tbody>
</table>

</details>

## Citation

- 📄 **Paper**: [LifeBench: A Benchmark for Long-Horizon Multi-Source Memory](https://arxiv.org/abs/2603.03781) (arXiv:2603.03781)
- 🤗 **Dataset**: [LifeBench v2.0](https://huggingface.co/datasets/C1754955896/Lifebenchv2.0) (Hugging Face)
- 🐙 **Dataset repository**: [LifeBench](https://github.com/1754955896/LifeBench) (GitHub)

## Appendix

### Core concepts

#### Builder

Responsible for starting the runtime environment (Docker) of the memory system under test. Each system has a corresponding Builder:

| Builder | System | Startup method |
|---------|------|---------|
| `Mem0Builder` | Mem0 | docker-compose (PostgreSQL + Mem0 Server) |
| `CogneeBuilder` | Cognee | docker-compose |
| `GraphitiBuilder` | Graphiti | docker-compose (Neo4j) |
| `HindsightBuilder` | Hindsight | docker-compose (AlloyDB) |
| ... | ... | ... |

#### Adapter

Defines the unified interface between the framework and the memory systems:

```python
class BaseAdapter:
    async def add_chunks(self, chunks: List[MessageChunk]) -> None
    async def search(self, query: str, user_id: str, top_k: int = 5) -> List[SearchResult]
    async def answer(self, query: str, context: List[str]) -> str
    async def cleanup(self) -> None
```

#### Pipeline

Four-stage evaluation flow:

```
ADD + SEARCH → ANSWER → EVALUATE
   (interleaved by date)    (generates answers)  (LLM Judge)
```

**ADD + SEARCH**: interleaves data ingestion and retrieval in date order
**ANSWER**: generates answers based on the retrieved memories
**EVALUATE**: uses an LLM Judge to evaluate answer quality, supporting the following types:

- Single_hop, Multi_hop, Temporal, Conflict
- Unanswerable, Pattern_recognition, Causal
- Knowledge_update, Hidden_info

### Adding a new system

1. **Implement a Builder**: create `{system}_builder.py` in `src/builders/`
2. **Implement an Adapter**: create `{system}_adapter.py` in `src/adapters/`
3. **Add config**: create `{system}.yaml` in `config/systems/`
4. **Register the system**: register it in the Builder and Adapter registries

### Difference from LifeMem/evaluation

| Aspect | LifeMem/evaluation | LifeBench_eval |
|------|-------------------|----------------|
| Project positioning | internal evaluation tool, deeply coupled with LifeMem | general evaluation framework, multi-system |
| Adapter interface | adapter add receives the whole conversation | add receives pre-split message chunks |
| Run mode | unified ADD then SEARCH | interleaved ADD/SEARCH, by date |
| Memory system deployment | code-based | Docker-based |
| Resumable runs | stage-level | Date + QA level (finer granularity) |
| Result saving | saved once at the end | incremental save |
| Debug mode | none | produces debug log files |
