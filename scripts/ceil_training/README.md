# 官方 CEIL 逻辑的 Qwen 适配

本实现以 `third_party/icl-ceil/` 的官方代码为规范，只把冻结打分模型替换为
Qwen2.5-7B-Instruct，并接入本项目数据与评测入口。Qwen 使用 chat template，监督值是
gold assistant tokens 的平均 NLL（越低越好）。

对齐项：EPR top-5 正例/bottom-5 难负例、batch 64 与批内负例；EPR top-100 后按官方
`cand_random` 生成 50 个 16-shot 集合；全部 50 个集合参与 conditional-DPP pairwise
margin ranking；scale factor 0.1、margin 0.02、warmup；90/10 train/validation 划分，
按 validation loss 保存最佳模型；推理使用官方 DPP kernel，并按 top-100 的相关性顺序排列选中示例。

旧版 singleton/set 缓存和 checkpoint 属于 `legacy-simplified-v1`，评分方向、prompt 协议和
训练目标均不同，不能续训。新脚本会主动拒绝这些旧文件，但旧 checkpoint 仍可用于复现旧结果。

## SST-5 正式训练（1000 query / 8000 pool / 100 candidates）

六个离线阶段也可以用总控脚本顺序运行（默认 SST-5、Llama-3.1-8B、物理 6 号 GPU）：

```bash
bash scripts/run_ceil_training_pipeline.sh
```

运行 `bash scripts/run_ceil_training_pipeline.sh --help` 可查看数据集、教师模型、GPU、
批大小及断点续跑参数。脚本不会启动 vLLM，也不会执行最终评测。

```bash
python -m scripts.ceil_training.build_candidates \
  --dataset sst5 --max-queries 1000 --max-pool-size 8000 \
  --candidate-count 100 \
  --output data/ceil_cache/official_sst5_qwen7b/candidates.jsonl

python -m scripts.ceil_training.score_singletons \
  --input data/ceil_cache/official_sst5_qwen7b/candidates.jsonl \
  --output data/ceil_cache/official_sst5_qwen7b/singleton_scores.jsonl \
  --teacher-model models/Qwen2.5-7B-Instruct --device cuda:6

python -m scripts.ceil_training.train_epr \
  --input data/ceil_cache/official_sst5_qwen7b/singleton_scores.jsonl \
  --output-dir checkpoints/ceil/official_sst5_qwen7b/epr \
  --epochs 30 --batch-size 64 --learning-rate 1e-5 \
  --warmup-ratio 0.1 --device cuda:6

python -m scripts.ceil_training.build_sets \
  --input data/ceil_cache/official_sst5_qwen7b/singleton_scores.jsonl \
  --output data/ceil_cache/official_sst5_qwen7b/candidate_sets.jsonl \
  --epr-checkpoint checkpoints/ceil/official_sst5_qwen7b/epr \
  --dpp-topk 100 --num-shots 16 --sets-per-query 50 --device cuda:6

python -m scripts.ceil_training.score_sets \
  --input data/ceil_cache/official_sst5_qwen7b/candidate_sets.jsonl \
  --output data/ceil_cache/official_sst5_qwen7b/set_scores.jsonl \
  --teacher-model models/Qwen2.5-7B-Instruct --device cuda:6

python -m scripts.ceil_training.train_ceil \
  --input data/ceil_cache/official_sst5_qwen7b/set_scores.jsonl \
  --epr-checkpoint checkpoints/ceil/official_sst5_qwen7b/epr \
  --output-dir checkpoints/ceil/official_sst5_qwen7b/ceil_k16_seed42 \
  --dataset sst5 --teacher-model models/Qwen2.5-7B-Instruct \
  --epochs 30 --batch-size 64 --learning-rate 1e-5 \
  --warmup-ratio 0.1 --scale-factor 0.1 --margin 0.02 \
  --candidate-pool-size 100 --device cuda:6
```

训练集合固定为 16-shot，推理可直接使用同一个 checkpoint 跑 `k=1,2,4,8,16`，无需重训：

```bash
python scripts/evaluate.py --method ceil --models qwen --datasets sst5 \
  --num-shots 4 \
  --ceil-checkpoint checkpoints/ceil/official_sst5_qwen7b/ceil_k16_seed42 \
  --ceil-device cuda:6 --output-dir results/official_ceil_qwen7b_sst5_k4
```
