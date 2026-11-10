"""集中管理模型服务、数据集名称和标签；命令行可覆盖服务配置。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 将本地模型及其服务地址直接记录在脚本中，确保实验配置清晰且可复现，
# 避免关键配置只存在于临时启动命令中。
MODEL_CONFIGS = {
    "qwen": {
        "model_id": str(PROJECT_ROOT / "models" / "Qwen2.5-7B-Instruct"),
        "base_url": "http://127.0.0.1:8001",
    },
    "qwen1_5b": {
        "model_id": str(PROJECT_ROOT / "models" / "Qwen2.5-1.5B-Instruct"),
        "base_url": "http://127.0.0.1:8003",
    },
    "llama": {
        "model_id": str(PROJECT_ROOT / "models" / "Llama-3.1-8B-Instruct"),
        "base_url": "http://127.0.0.1:8002",
    },
}

DATASET_NAMES = ("sst5", "paws", "banking77", "boolq", "cmsqa")

# SST-5 的五级情感标签，顺序与数据集中的整数标签一致。
SST5_LABELS = (
    "very negative",
    "negative",
    "neutral",
    "positive",
    "very positive",
)

PAWS_LABELS = ("not paraphrase", "paraphrase")
BOOLQ_LABELS = ("yes", "no")
CMSQA_LABELS = ("A", "B", "C", "D", "E")
