# GENICL：严格论文逻辑的训练与统一评测接入

实现对应 ACL 2025 论文 *Learning to Select In-Context Demonstration Preferred
by Large Language Model* 与作者代码：

1. `intfloat/e5-base-v2` 从训练示例池召回候选子集；
2. 使用目标因果语言模型计算论文公式 (5) 的
   `log P(y | demonstration, query)`；
3. 最高分候选作为 preferred demonstrations，最低分候选作为
   non-preferred demonstrations；
4. 构造论文公式 (6) 的 demonstration-level KTO 数据并执行一次
   `theta <- theta - eta1 * grad Ld(theta)`；
5. 使用正确/错误答案构造公式 (8) 的 answer-level KTO 数据，紧接着执行
   `theta <- theta - eta2 * grad La(theta)`；
6. 对每个 query mini-batch 严格重复 `Ld -> La` 两次独立 optimizer step，
   从执行顺序上实现公式 (10) 与 Algorithm 1；
7. 推理时先用 E5 缩小候选池，再按公式 (11)
   `log P(z | demonstration, query)` 排序并选择前 K 个示例；
8. 冻结选择结果，交给现有 vLLM Qwen/Llama 服务完成 ICL 回答和准确率统计。

算法阶段没有替换为 embedding 排序或其他代理目标。E5 只负责论文规定的候选空间
缩减，最终排名一定来自生成式选择模型对任务潜变量 `z` 的条件概率。训练 loader
禁止混合或随机穿插两类目标，并强制 `gradient_accumulation_steps=1`，防止一次更新
同时累积 `Ld` 与 `La` 梯度。两个阶段都生成完整的 32-record 物理 batch；较少的
answer-level records 仅作等比例重复，故其平均 loss/梯度不变，并确保 TRL 的 KL
参考分布配对始终留在同一阶段内。

## 环境

KTO/LoRA 训练依赖与 vLLM 环境冲突，必须使用独立环境：

```bash
python -m venv envs/genicl
envs/genicl/bin/pip install -r scripts/genicl_training/requirements.txt
```

## 每个目标模型分别训练

GenICL 学习的是特定 LLM 的 demonstration preference。若要分别报告 Qwen 和
Llama 的原生 GenICL 结果，应各训练一个 checkpoint。下面以 SST-5 为例：

```bash
envs/genicl/bin/python scripts/genicl_training/train_genicl.py \
  --dataset sst5 \
  --base-model models/Qwen2.5-7B-Instruct \
  --output-dir checkpoints/genicl/qwen/sst5

envs/genicl/bin/python scripts/genicl_training/train_genicl.py \
  --dataset sst5 \
  --base-model models/Llama-3.1-8B-Instruct \
  --output-dir checkpoints/genicl/llama/sst5
```

默认训练超参数来自论文 Appendix A Table 6：AdamW、3000 warmup steps、20000
training steps、`5e-6` learning rate、batch size 32、最大输入 1024、最大输出
64、`beta=0.1`、`lambda_w=lambda_l=1`。LoRA 使用作者代码的 rank 32、alpha
64，并作用于 attention 与 MLP projection。

论文 Algorithm 1 写有两个学习率 `eta1`、`eta2`，但 Appendix A 只报告一个
learning rate，因此实现按论文可复现信息设置 `eta1 = eta2 = 5e-6`。训练步数必须
为偶数，防止最后只完成 `Ld` 更新而缺少与之配对的 `La` 更新。checkpoint 会额外
保存 `training_schedule.json`，记录实际执行顺序和每轮 optimizer batch 数。

训练会保存 `preferences.json`，中断后可加 `--resume-preferences` 跳过昂贵的
LLM feedback 重新生成。不要在完整训练中设置 `--max-train-samples`；该参数只用于
smoke test。

多 GPU 时必须分成两条命令。先用单进程和 `--feedback-only` 生成公式 (5) 的
preference；再用 `accelerate launch` 加 `--resume-preferences` 训练。全局 batch 32
会通过 `split_batches=True` 切成同一个阶段的各 rank 子 batch，不会让一个 rank
计算 `Ld`、另一个 rank 计算 `La`。

## 分别评测 Qwen 与 Llama

```bash
envs/genicl/bin/python scripts/evaluate.py \
  --method genicl \
  --models qwen \
  --datasets sst5 \
  --genicl-checkpoint checkpoints/genicl/qwen/sst5 \
  --genicl-e5-device cuda:0 \
  --genicl-device cuda:0 \
  --output-dir results/genicl_qwen_sst5

envs/genicl/bin/python scripts/evaluate.py \
  --method genicl \
  --models llama \
  --datasets sst5 \
  --genicl-checkpoint checkpoints/genicl/llama/sst5 \
  --genicl-e5-device cuda:0 \
  --genicl-device cuda:0 \
  --output-dir results/genicl_llama_sst5
```

`--num-shots` 对 GenICL 默认是论文使用的 8。统一评测仍通过现有服务器地址调用
Qwen/Llama；selector 本身离线加载对应 LoRA checkpoint。每条结果保存
`selection_score`（公式 11）、`e5_score`、`e5_rank` 和完整 demonstration。

论文 Table 4 也验证了一个模型选择的 demonstrations 可以迁移给其他冻结 LLM。
如果研究目标是迁移实验，可以用同一 checkpoint 同时指定
`--models qwen llama`；但这不等同于两个模型分别训练各自 GenICL selector 的主实验。
