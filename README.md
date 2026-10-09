# LifeBench_eval 记忆系统评估框架

用于评估 Mem0 等记忆系统在实际生活场景中的表现。

## 简介

LifeBench_eval 是一个通用的记忆系统评测框架，支持：

- **多系统评测**：Mem0、Cognee、Graphiti、Hindsight、EverMemos、MindMemos 等
- **三阶段流水线**：ADD + SEARCH → ANSWER → EVALUATE
- **断点续跑**：支持按 Sample、Date、QA 粒度的断点恢复
- **LLM Judge 评估**：基于大模型对答案质量进行评分
- **Docker 化部署**：被测系统以 Docker 容器方式运行

## 快速开始

### 1. 配置环境

```bash
# 复制环境变量模板
cp env.template .env

# 编辑 .env，填入你的 API Key
vim .env
```

### 2. 运行评测

```bash
# 完整评测
python cli.py --dataset lifebench_locomo_format --system mem0

# 指定输出目录
python cli.py --dataset lifebench_locomo_format --system mem0 --output results/my-run

# 断点续跑
python cli.py --dataset lifebench_locomo_format --system mem0 --resume

# Debug 模式（生成详细日志）
python cli.py --dataset lifebench_locomo_format --system mem0 --debug
```

### 3. 查看结果

评测结果保存在 `results/{dataset}-{system}/` 目录：

```
results/lifebench_locomo_format-mem0/
├── checkpoint_default.json     # 断点记录
├── add_latency.json            # ADD 延迟统计
├── search_latency.json         # SEARCH 延迟统计
├── search_results.json         # 检索结果
├── answer_results.json         # 回答结果
├── eval_results.json           # 评估结果
└── report.txt                  # 文本报告
```

## CLI 参数

```
--dataset TEXT         数据集名称（必需）
--system TEXT         系统名称（必需）
--output PATH         输出目录（默认: results/{dataset}-{system}）
--resume              从断点恢复运行
--debug               开启 Debug 模式
--max-workers N       并发线程数（默认: 10）
--rerank-model MODEL  Rerank 模型名称
--rerank-provider PROVIDER  Rerank 提供商
```

## Leader Board

在 LifeBench（3380 QA）与 LoCoMo 上的准确率（%）。**Micro** 为全部问题的准确率；**Macro** 为九类题型的算术平均。**Gold Evidence**（金标准证据）把标注证据直接喂给答案模型，作为「完美检索」的上界参考，并非记忆系统。LoCoMo 结果不含对抗性问题。

| 记忆系统 | 基础模型 | Micro | Macro | LoCoMo |
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

**粗体**标记该列中记忆系统的最优结果（Gold Evidence 不参与比较）。† 表示 Gold Evidence（金标准证据上界参考）。

完整的分题型（Single-hop / Multi-hop / Temporal / Non-declarative / Knowledge update / Causal / Conflict / Hidden info / Unanswerable）细分见下，也可在[交互式 Leaderboard](https://huggingface.co/spaces/C1754955896/Lifebench-Leaderboard) 中探索与排序。

<details>
<summary>按基础模型 × 记忆系统 × 题型的完整结果</summary>

**SH** Single-hop；**MH** Multi-hop；**TR** Temporal；**ND** Non-declarative；**KU** Knowledge update；**CR** Causal；**CD** Conflict detection；**HI** Hidden information；**UA** Unanswerable。**粗体**为该列记忆系统最优结果（Gold Evidence 不参与比较）。「—」表示未报告。

<table>
  <thead>
    <tr>
      <th>基础模型</th>
      <th>记忆系统</th>
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

## 引用

- 📄 **论文**：[LifeBench: A Benchmark for Long-Horizon Multi-Source Memory](https://arxiv.org/abs/2603.03781)（arXiv:2603.03781）
- 🤗 **数据集**：[LifeBench v2.0](https://huggingface.co/datasets/C1754955896/Lifebenchv2.0)（Hugging Face）
- 🐙 **数据集仓库**：[LifeBench](https://github.com/1754955896/LifeBench)（GitHub）

## 附录

### 核心概念

#### Builder（构建器）

负责启动被测记忆系统的运行环境（Docker）。每个系统对应一个 Builder：

| Builder | 系统 | 启动方式 |
|---------|------|---------|
| `Mem0Builder` | Mem0 | docker-compose (PostgreSQL + Mem0 Server) |
| `CogneeBuilder` | Cognee | docker-compose |
| `GraphitiBuilder` | Graphiti | docker-compose (Neo4j) |
| `HindsightBuilder` | Hindsight | docker-compose (AlloyDB) |
| ... | ... | ... |

#### Adapter（适配器）

定义框架与记忆系统的统一接口：

```python
class BaseAdapter:
    async def add_chunks(self, chunks: List[MessageChunk]) -> None
    async def search(self, query: str, user_id: str, top_k: int = 5) -> List[SearchResult]
    async def answer(self, query: str, context: List[str]) -> str
    async def cleanup(self) -> None
```

#### Pipeline（流水线）

四阶段评测流程：

```
ADD + SEARCH → ANSWER → EVALUATE
   (按日期交叉)    (生成答案)  (LLM Judge)
```

**ADD + SEARCH**：按日期顺序，交叉执行数据摄入与检索
**ANSWER**：基于检索到的记忆生成答案
**EVALUATE**：使用 LLM Judge 评估答案质量，支持以下类型：

- Single_hop、Multi_hop、Temporal、Conflict
- Unanswerable、Pattern_recognition、Causal
- Knowledge_update、Hidden_info

### 添加新系统

1. **实现 Builder**：在 `src/builders/` 中创建 `{system}_builder.py`
2. **实现 Adapter**：在 `src/adapters/` 中创建 `{system}_adapter.py`
3. **添加配置**：在 `config/systems/` 中创建 `{system}.yaml`
4. **注册系统**：在 Builder 和 Adapter 的 registry 中注册

### 与 LifeMem/evaluation 的区别

| 方面 | LifeMem/evaluation | LifeBench_eval |
|------|-------------------|----------------|
| 项目定位 | 内部评估工具，深度耦合 LifeMem | 通用评估框架，支持多系统 |
| 适配器接口 | adapter add 接收整个 conversation | add 接收分割好的 message chunk |
| 运行方式 | 统一 ADD 后再 SEARCH | 边 ADD 边 SEARCH，按日期交叉 |
| 记忆系统部署 | 基于代码 | 基于 Docker |
| 断点续跑 | Stage 级别 | Date + QA 级别（更细粒度） |
| 结果保存 | 最后一次性保存 | 增量保存 |
| Debug 模式 | 无 | 生成 debug 日志文件 |
