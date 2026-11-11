#!/usr/bin/env bash
# PAWS + Llama-3.1-8B 的 CEIL 离线训练入口（阶段 1-6，不含最终推理）。

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 默认使用项目已有的 vllm Python；仍可在命令前用 PYTHON_BIN=... 覆盖。
export PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/../envs/vllm/bin/python}"

exec bash scripts/run_ceil_training_pipeline.sh \
  --dataset paws \
  --teacher-model models/Llama-3.1-8B-Instruct \
  --teacher-tag llama8b \
  --gpu 6 \
  --seed 42 \
  --max-queries 1000 \
  --max-pool-size 8000 \
  --candidate-count 100 \
  --num-shots 16 \
  --sets-per-query 50 \
  --epochs 30 \
  --singleton-batch-size 8 \
  --set-batch-size 2 \
  "$@"
