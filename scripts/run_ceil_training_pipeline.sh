#!/usr/bin/env bash
# 顺序运行 CEIL 的六个离线阶段；不启动 vLLM，也不执行最终评测。

set -Eeuo pipefail

DATASET="sst5"
TEACHER_MODEL="models/Llama-3.1-8B-Instruct"
TEACHER_TAG="llama8b"
GPU="6"
SEED="42"
SET_SEED="43"
MAX_QUERIES="1000"
MAX_POOL_SIZE="8000"
CANDIDATE_COUNT="100"
NUM_SHOTS="16"
SETS_PER_QUERY="50"
EPOCHS="30"
SINGLETON_BATCH_SIZE="8"
SET_BATCH_SIZE="2"
TRAIN_BATCH_SIZE="64"
CONTEXT_BATCH_SIZE="128"
ENCODER="bert-base-uncased"
MAX_LENGTH="128"
FROM_STAGE="1"
FORCE="0"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
用法：
  bash scripts/run_ceil_training_pipeline.sh [参数]

默认运行 SST-5 + Llama-3.1-8B，并使用物理 6 号 GPU。

主要参数：
  --dataset NAME                 数据集，默认 sst5
  --teacher-model PATH           教师模型路径
  --teacher-tag NAME             输出目录标签，默认 llama8b
  --gpu ID                       物理 GPU 编号，默认 6
  --seed N                       训练随机种子，默认 42
  --max-queries N                监督 query 数，默认 1000
  --max-pool-size N              demonstration pool 大小，默认 8000
  --candidate-count N            每题单例候选数，默认 100
  --num-shots N                  每个候选集合的示例数，默认 16
  --sets-per-query N             每题候选集合数，默认 50
  --epochs N                     EPR/CEIL 训练轮数，默认 30
  --singleton-batch-size N       单例教师评分 batch，默认 8
  --set-batch-size N             集合教师评分 batch，默认 2
  --from-stage N                 从第 N 阶段开始检查/运行，范围 1-6
  --force                        即使产物完整也重新运行阶段
  -h, --help                     显示帮助

阶段：
  1 构造候选    2 单示例评分    3 训练 EPR
  4 构造集合    5 集合评分      6 训练 CEIL
EOF
}

need_value() {
  if [[ $# -lt 2 || -z "${2:-}" ]]; then
    echo "参数 $1 缺少取值" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) need_value "$@"; DATASET="$2"; shift 2 ;;
    --teacher-model) need_value "$@"; TEACHER_MODEL="$2"; shift 2 ;;
    --teacher-tag) need_value "$@"; TEACHER_TAG="$2"; shift 2 ;;
    --gpu) need_value "$@"; GPU="$2"; shift 2 ;;
    --seed) need_value "$@"; SEED="$2"; shift 2 ;;
    --max-queries) need_value "$@"; MAX_QUERIES="$2"; shift 2 ;;
    --max-pool-size) need_value "$@"; MAX_POOL_SIZE="$2"; shift 2 ;;
    --candidate-count) need_value "$@"; CANDIDATE_COUNT="$2"; shift 2 ;;
    --num-shots) need_value "$@"; NUM_SHOTS="$2"; shift 2 ;;
    --sets-per-query) need_value "$@"; SETS_PER_QUERY="$2"; shift 2 ;;
    --epochs) need_value "$@"; EPOCHS="$2"; shift 2 ;;
    --singleton-batch-size) need_value "$@"; SINGLETON_BATCH_SIZE="$2"; shift 2 ;;
    --set-batch-size) need_value "$@"; SET_BATCH_SIZE="$2"; shift 2 ;;
    --from-stage) need_value "$@"; FROM_STAGE="$2"; shift 2 ;;
    --force) FORCE="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$FROM_STAGE" =~ ^[1-6]$ ]]; then
  echo "--from-stage 必须是 1 到 6" >&2
  exit 2
fi

# 无论从哪个目录调用，都切换到项目根目录。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_NAME="official_${DATASET}_${TEACHER_TAG}"
CACHE_DIR="data/ceil_cache/${RUN_NAME}"
CHECKPOINT_DIR="checkpoints/ceil/${RUN_NAME}"
LOG_DIR="logs/ceil_${RUN_NAME}"
EPR_DIR="${CHECKPOINT_DIR}/epr"
CEIL_DIR="${CHECKPOINT_DIR}/ceil_k${NUM_SHOTS}_seed${SEED}"

mkdir -p "$CACHE_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

# 数据和模型都已在本地缓存；禁止不稳定的 Hub 请求拖住流水线。
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

if ! "$PYTHON_BIN" -c 'import torch, datasets, transformers' >/dev/null 2>&1; then
  echo "当前 Python 缺少 torch/datasets/transformers：$PYTHON_BIN" >&2
  echo "请先激活项目 vllm 环境，或设置 PYTHON_BIN=/path/to/python。" >&2
  exit 1
fi

jsonl_complete() {
  local path="$1"
  [[ -s "$path" ]] && [[ "$(wc -l < "$path")" -eq "$MAX_QUERIES" ]]
}

checkpoint_complete() {
  local path="$1"
  [[ -s "$path/metadata.json" ]] && [[ -s "$path/encoder/model.safetensors" ]]
}

should_run() {
  local stage="$1"
  local complete="$2"
  [[ "$stage" -ge "$FROM_STAGE" ]] && { [[ "$FORCE" == "1" ]] || [[ "$complete" == "0" ]]; }
}

announce_skip() {
  echo "[跳过] $1：检测到完整产物；使用 --force 可重新运行。"
}

echo "CEIL 离线训练流水线"
echo "dataset=$DATASET teacher=$TEACHER_MODEL gpu=$GPU run=$RUN_NAME"
echo "内部 CUDA 设备固定为 cuda:0（映射到物理 GPU $GPU）"

complete=0; jsonl_complete "$CACHE_DIR/candidates.jsonl" && complete=1
if should_run 1 "$complete"; then
  echo "[1/6] 构造候选"
  "$PYTHON_BIN" -u -m scripts.ceil_training.build_candidates \
    --dataset "$DATASET" --max-queries "$MAX_QUERIES" \
    --max-pool-size "$MAX_POOL_SIZE" --candidate-count "$CANDIDATE_COUNT" \
    --seed "$SEED" --cache-dir data/hf_cache \
    --output "$CACHE_DIR/candidates.jsonl" \
    2>&1 | tee "$LOG_DIR/01_build_candidates.log"
else
  announce_skip "1/6 构造候选"
fi

complete=0; jsonl_complete "$CACHE_DIR/singleton_scores.jsonl" && complete=1
if should_run 2 "$complete"; then
  echo "[2/6] 教师模型单示例评分"
  env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u -m scripts.ceil_training.score_singletons \
    --input "$CACHE_DIR/candidates.jsonl" \
    --output "$CACHE_DIR/singleton_scores.jsonl" \
    --teacher-model "$TEACHER_MODEL" --device cuda:0 \
    --batch-size "$SINGLETON_BATCH_SIZE" \
    2>&1 | tee "$LOG_DIR/02_score_singletons.log"
else
  announce_skip "2/6 单示例评分"
fi

complete=0; checkpoint_complete "$EPR_DIR" && complete=1
if should_run 3 "$complete"; then
  echo "[3/6] 训练 EPR"
  env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u -m scripts.ceil_training.train_epr \
    --input "$CACHE_DIR/singleton_scores.jsonl" --output-dir "$EPR_DIR" \
    --encoder "$ENCODER" --cache-dir data/hf_cache/models --device cuda:0 \
    --epochs "$EPOCHS" --batch-size "$TRAIN_BATCH_SIZE" --learning-rate 1e-5 \
    --warmup-ratio 0.1 --validation-fraction 0.1 \
    --positive-topk 5 --negative-bottomk 5 --max-length "$MAX_LENGTH" --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/03_train_epr.log"
else
  announce_skip "3/6 训练 EPR"
fi

complete=0; jsonl_complete "$CACHE_DIR/candidate_sets.jsonl" && complete=1
if should_run 4 "$complete"; then
  echo "[4/6] 构造候选集合"
  env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u -m scripts.ceil_training.build_sets \
    --input "$CACHE_DIR/singleton_scores.jsonl" \
    --output "$CACHE_DIR/candidate_sets.jsonl" --epr-checkpoint "$EPR_DIR" \
    --device cuda:0 --num-shots "$NUM_SHOTS" --sets-per-query "$SETS_PER_QUERY" \
    --dpp-topk "$CANDIDATE_COUNT" --max-length "$MAX_LENGTH" --seed "$SET_SEED" \
    2>&1 | tee "$LOG_DIR/04_build_sets.log"
else
  announce_skip "4/6 构造候选集合"
fi

complete=0; jsonl_complete "$CACHE_DIR/set_scores.jsonl" && complete=1
if should_run 5 "$complete"; then
  echo "[5/6] 教师模型集合评分"
  env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u -m scripts.ceil_training.score_sets \
    --input "$CACHE_DIR/candidate_sets.jsonl" --output "$CACHE_DIR/set_scores.jsonl" \
    --teacher-model "$TEACHER_MODEL" --device cuda:0 --batch-size "$SET_BATCH_SIZE" \
    2>&1 | tee "$LOG_DIR/05_score_sets.log"
else
  announce_skip "5/6 集合评分"
fi

complete=0; checkpoint_complete "$CEIL_DIR" && complete=1
if should_run 6 "$complete"; then
  echo "[6/6] 训练最终 CEIL 选择器"
  env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u -m scripts.ceil_training.train_ceil \
    --input "$CACHE_DIR/set_scores.jsonl" --epr-checkpoint "$EPR_DIR" \
    --output-dir "$CEIL_DIR" --dataset "$DATASET" --teacher-model "$TEACHER_MODEL" \
    --device cuda:0 --epochs "$EPOCHS" --batch-size "$TRAIN_BATCH_SIZE" \
    --context-batch-size "$CONTEXT_BATCH_SIZE" --learning-rate 1e-5 \
    --warmup-ratio 0.1 --validation-fraction 0.1 --max-length "$MAX_LENGTH" \
    --scale-factor 0.1 --margin 0.02 --candidate-pool-size "$CANDIDATE_COUNT" \
    --seed "$SEED" \
    2>&1 | tee "$LOG_DIR/06_train_ceil.log"
else
  announce_skip "6/6 训练最终 CEIL"
fi

echo "完成。最终 checkpoint：$CEIL_DIR"
echo "本脚本不执行最终评测。"
