# 统一评测入口

当前统一支持 zero-shot、random ICL、BM25 ICL、KATE ICL、CEIL ICL 和 GENICL。

在项目根目录运行（需使用装有 requests、datasets、tqdm 的 Python 环境）：

```bash
python scripts/evaluate.py --method zero_shot --models qwen --datasets banking77
```

先用少量样本检查服务，并使用单独的输出目录：

```bash
python scripts/evaluate.py --method zero_shot --models qwen --datasets banking77 --max-samples 10 --output-dir results/zero_shot_smoke
```

使用 Qwen2.5-1.5B 在五个数据集上做小样本检查：

```bash
python scripts/evaluate.py \
  --method zero_shot \
  --models qwen1_5b \
  --datasets sst5 paws banking77 boolq cmsqa \
  --max-samples 10 \
  --output-dir results/qwen1_5b_zero_shot_smoke
```

随机选择 4 条训练示例进行 ICL：

```bash
python scripts/evaluate.py \
  --method random \
  --models qwen \
  --datasets banking77 \
  --num-shots 4 \
  --seed 42 \
  --output-dir results/random_k4_qwen_banking77
```

`--seed` 固定后，同一数据集和同一测试样本会选择相同的演示示例，因而不同
模型之间也能公平对比。随机选择是无放回抽样，示例只来自训练集；每条 JSONL
结果会保存所选示例的 ID、输入和答案。

使用 BM25 为每道测试题选择词项最相关的 4 条训练示例：

```bash
python scripts/evaluate.py \
  --method bm25 \
  --models qwen \
  --datasets banking77 \
  --num-shots 4 \
  --output-dir results/bm25_k4_qwen_banking77
```

BM25 默认使用 `k1=1.5`、`b=0.75`，可通过 `--bm25-k1` 和 `--bm25-b`
覆盖。BM25 是确定性方法，不使用 `--seed`。结果中的每条 demonstration 额外
保存 `selection_score`，便于检查检索排序。

使用原始 KATE encoder 按语义相似度选择 4 条训练示例：

```bash
python scripts/evaluate.py \
  --method kate \
  --models qwen llama \
  --datasets banking77 \
  --num-shots 4 \
  --kate-device cuda:0 \
  --output-dir results/kate_k4_banking77
```

KATE 默认使用 `sentence-transformers/roberta-large-nli-stsb-mean-tokens`、
128 token 截断、mean pooling 和 cosine similarity。训练池 embedding 会按数据、
encoder 配置和文本内容生成指纹并缓存在 `data/kate_cache/`。所有测试样本的
top-k 会在模型循环前一次性确定，因此 Qwen 与 Llama 使用完全相同的示例。
默认按相似度从高到低放入 prompt；可用
`--kate-order most_similar_last` 反转同一组 top-k 的 prompt 顺序。

CEIL 需要先离线生成目标 LLM 监督并训练 EPR/conditional-DPP encoder。完整的低成本
SST-5 流程见 `scripts/ceil_training/README.md`。已有 checkpoint 后，通过统一入口评测：

```bash
python scripts/evaluate.py \
  --method ceil \
  --models qwen1_5b \
  --datasets sst5 \
  --num-shots 4 \
  --ceil-checkpoint checkpoints/ceil/sst5_qwen1_5b_k4 \
  --ceil-device cuda:0 \
  --output-dir results/ceil_k4_qwen1_5b_sst5
```

GENICL 严格保留论文的 E5 候选缩减、LLM ground-truth likelihood feedback、
answer-/demonstration-level KTO、LoRA 潜变量学习和公式 (11) 生成概率重排流程。
训练与评测命令见 `scripts/genicl_training/README.md`。已有 checkpoint 后可运行：

```bash
python scripts/evaluate.py \
  --method genicl \
  --models qwen \
  --datasets sst5 \
  --genicl-checkpoint checkpoints/genicl/qwen/sst5 \
  --genicl-e5-device cuda:0 \
  --genicl-device cuda:0 \
  --output-dir results/genicl_qwen_sst5
```

可用模型名称和默认服务如下：

| 命令行名称 | 模型 | 默认端口 |
| --- | --- | --- |
| `qwen` | Qwen2.5-7B-Instruct | 8001 |
| `qwen1_5b` | Qwen2.5-1.5B-Instruct | 8003 |
| `llama` | Llama-3.1-8B-Instruct | 8002 |

不指定模型和数据集时，仍依次运行 `qwen`、`llama` 在五个数据集上的零样本
评测；`qwen1_5b` 需要通过 `--models qwen1_5b` 显式选择。脚本不会启动
vLLM 服务。也支持 `python -m scripts.evaluate ...`。

评测客户端访问 `127.0.0.1`、`localhost` 或 `::1` 上的 vLLM 时会强制绕过
`HTTP_PROXY`、`HTTPS_PROXY` 和 `ALL_PROXY`，保证 prompt/response 只经过服务器
回环接口，不进入 SSH RemoteForward 或外部网络。远程模型地址仍保留系统代理。

## 模块职责

| 文件 | 职责 |
| --- | --- |
| evaluate.py | 统一启动入口 |
| evaluation/config.py | 模型地址、项目路径和标签常量 |
| evaluation/task_data.py | 统一任务数据结构 |
| evaluation/data_utils.py | 数据加载、转换和缓存 |
| evaluation/prompts.py | 各数据集提示词 |
| evaluation/model_client.py | vLLM 请求、模型 ID 匹配和重试 |
| evaluation/metrics.py | 分类标签解析和精确匹配评分 |
| evaluation/runner.py | 选择/拼接示例、逐样本评测、JSONL 输出和统计 |
| evaluation/cli.py | 命令行参数、实验调度和 summary.json 汇总 |
| example_selectors/random_selector.py | 可复现的随机训练示例选择器 |
| example_selectors/bm25_selector.py | 基于倒排索引的 BM25 训练示例选择器 |
| example_selectors/kate_selector.py | KATE 句向量编码、缓存和 cosine top-k 选择器 |
| example_selectors/ceil_selector.py | 加载 CEIL checkpoint 并执行 conditional-DPP MAP 选择 |
| example_selectors/genicl_selector.py | E5 缩池并按 `log P(z | demo, query)` 执行 GENICL 选择 |
| ceil_training/ | EPR 初始化、LLM 监督缓存和 CEIL 集合级训练流水线 |
| genicl_training/ | GENICL LLM feedback、KTO/LoRA 训练与独立依赖 |

当前按 SST-5、PAWS、Banking77、BoolQ、CommonsenseQA 的顺序支持五个数据集，
并统一提供 zero-shot、Random、BM25、KATE、CEIL 和 GENICL 六种评测方法。
后续选择器应复用公共模块，并明确传递方法和示例元数据。选择器目录命名为
`example_selectors`，避免遮蔽 Python 标准库 `selectors`。

## 保留的评测行为与限制

- 默认方法是 zero-shot；random/BM25/KATE/CEIL 未指定 `--num-shots` 时默认 4-shot。
- 默认全量评测；`--max-samples` 仍是取前 N 条测试题，不是 ICL 示例数量。
- `--num-shots` 才是每个测试问题使用的训练示例数量。
- 五个数据集都使用固定标签输出，默认通过 structured output 约束模型响应。
- 请求失败最多重试两次；失败计入 invalid，也计入准确率分母。
- 模型与数据集均串行评测，未加入断点续跑。
- 同名 JSONL 和 summary.json 会覆盖，请为不同实验设置独立输出目录。
- 服务只有一个模型时，模型 ID 不匹配仍会警告后继续；请核对启动的实际模型。
- 公共模块统一维护提示词、数据加载、模型请求和评分规则。

离线回归测试（不连接真实模型服务）：

```bash
python -m unittest discover -s tests -v
```
