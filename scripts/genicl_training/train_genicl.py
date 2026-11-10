"""为项目中的单个数据集训练严格遵循论文的 GENICL 选择器。

生成的 PEFT 检查点由 ``GenICLSelector`` 加载。本脚本实现论文公式（5）、
（6）、（8）和（10），包括 E5 候选子集构造、目标 LLM 的真实答案似然反馈、
偏好与非偏好演示样本构造，以及答案级和演示级 KTO 交替训练。
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.config import DATASET_NAMES
from scripts.evaluation.data_utils import DATASET_LOADERS
from scripts.example_selectors.genicl_selector import (
    DEFAULT_GENICL_E5_MODEL,
    GenICLE5Encoder,
    candidate_text,
    latent_description,
)
from scripts.genicl_training.alternating import (
    build_alternating_plan,
    make_alternating_kto_trainer_class,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GENICL with the ACL 2025 objective")
    parser.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "hf_cache")
    parser.add_argument("--banking77-dir", type=Path, default=PROJECT_ROOT / "data" / "banking77")
    parser.add_argument("--e5-model", default=DEFAULT_GENICL_E5_MODEL)
    parser.add_argument("--e5-device", default="cuda:0")
    parser.add_argument("--e5-batch-size", type=int, default=128)
    parser.add_argument("--feedback-device", default="cuda:0")
    parser.add_argument("--feedback-batch-size", type=int, default=8)
    parser.add_argument("--e5-pool-size", type=int, default=300)
    parser.add_argument("--feedback-candidates", type=int, default=32)
    parser.add_argument("--preferred-count", type=int, default=2)
    parser.add_argument("--nonpreferred-count", type=int, default=2)
    parser.add_argument("--n-prefix-tokens", type=int, default=10)
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=987)
    # 默认值来自论文附录 A 的表 6。
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-steps", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--resume-preferences", action="store_true")
    parser.add_argument(
        "--feedback-only",
        action="store_true",
        help="Generate Eq. (5) preferences and exit before KTO training.",
    )
    args = parser.parse_args()
    for name in (
        "e5_batch_size", "feedback_batch_size", "e5_pool_size", "feedback_candidates",
        "preferred_count", "nonpreferred_count", "n_prefix_tokens", "max_input_length",
        "max_steps", "batch_size",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.feedback_candidates > args.e5_pool_size:
        parser.error("--feedback-candidates cannot exceed --e5-pool-size")
    if args.preferred_count + args.nonpreferred_count > args.feedback_candidates:
        parser.error("preferred and nonpreferred counts exceed feedback candidates")
    if args.max_steps % 2:
        parser.error(
            "--max-steps must be even so training cannot stop after Ld without its paired La update"
        )
    if args.save_steps % 2:
        parser.error(
            "--save-steps must be even so checkpoints are written only after a complete Ld/La pair"
        )
    return args


def score_completions(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    completions: Sequence[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """计算论文公式（5）的分数，即补全在给定提示下的对数概率之和。"""

    import torch

    result: List[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            prompt_batch = prompts[start : start + batch_size]
            completion_batch = completions[start : start + batch_size]
            prompt_ids = [tokenizer(text, add_special_tokens=True).input_ids for text in prompt_batch]
            completion_ids = [
                tokenizer(text, add_special_tokens=False).input_ids for text in completion_batch
            ]
            full_ids = [p + c for p, c in zip(prompt_ids, completion_ids)]
            trimmed = [ids[-max_length:] for ids in full_ids]
            prompt_lengths = [
                max(0, len(p) - (len(full) - len(trim)))
                for p, full, trim in zip(prompt_ids, full_ids, trimmed)
            ]
            width = max(map(len, trimmed))
            ids_tensor = torch.full(
                (len(trimmed), width), tokenizer.pad_token_id, dtype=torch.long, device=device
            )
            mask = torch.zeros((len(trimmed), width), dtype=torch.long, device=device)
            for row, ids in enumerate(trimmed):
                ids_tensor[row, : len(ids)] = torch.tensor(ids, device=device)
                mask[row, : len(ids)] = 1
            logits = model(input_ids=ids_tensor, attention_mask=mask).logits[:, :-1]
            logps = logits.log_softmax(-1).gather(2, ids_tensor[:, 1:].unsqueeze(-1)).squeeze(-1)
            for row, (prompt_length, ids) in enumerate(zip(prompt_lengths, trimmed)):
                result.append(
                    float(logps[row, max(prompt_length - 1, 0) : len(ids) - 1].sum().item())
                )
    return np.asarray(result, dtype=np.float32)


def make_feedback(
    candidates: Sequence[Dict[str, Any]],
    labels: Sequence[str],
    dataset_name: str,
    e5: Any,
    model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """使用公式（5）构造偏好与非偏好样本对，不使用代理标签。"""

    rng = random.Random(args.seed)
    records = list(candidates)
    if args.max_train_samples is not None and args.max_train_samples < len(records):
        indices = list(range(len(records)))
        rng.shuffle(indices)
        records = [records[index] for index in indices[: args.max_train_samples]]

    all_texts = [candidate_text(item) for item in candidates]
    corpus_embeddings = np.asarray(e5.encode(all_texts), dtype=np.float32)
    query_texts = [candidate_text(item).rsplit("\n", 1)[0] for item in records]
    query_embeddings = np.asarray(e5.encode(query_texts), dtype=np.float32)
    preferences: List[Dict[str, Any]] = []
    id_to_index = {str(item.get("id", index)): index for index, item in enumerate(candidates)}

    for query_index, (query, query_text, query_embedding) in enumerate(
        zip(records, query_texts, query_embeddings), start=1
    ):
        similarities = corpus_embeddings @ query_embedding
        own_index = id_to_index.get(str(query.get("id", "")))
        if own_index is not None:
            similarities[own_index] = -np.inf
        e5_pool = np.argsort(-similarities, kind="stable")[: args.e5_pool_size]
        # 作者公开实现会从 E5 子集中抽取 32 个候选进行评分。
        # 这里使用固定随机种子，以可复现的方式保留该流程。
        scored_pool = list(map(int, e5_pool))
        rng.shuffle(scored_pool)
        scored_pool = scored_pool[: args.feedback_candidates]
        prompts = [f"{all_texts[index]}\n{query_text}" for index in scored_pool]
        scores = score_completions(
            model, tokenizer, prompts, [str(query["gold"])] * len(prompts),
            args.feedback_device, args.feedback_batch_size, args.max_input_length,
        )
        ranking = np.argsort(-scores, kind="stable")
        preferred = [scored_pool[int(index)] for index in ranking[: args.preferred_count]]
        nonpreferred = [
            scored_pool[int(index)] for index in ranking[-args.nonpreferred_count :]
        ]
        preferences.append(
            {
                "query": query_text,
                "gold": str(query["gold"]),
                "preferred": [all_texts[index] for index in preferred],
                "nonpreferred": [all_texts[index] for index in nonpreferred],
                "preferred_scores": [float(scores[scored_pool.index(index)]) for index in preferred],
                "nonpreferred_scores": [
                    float(scores[scored_pool.index(index)]) for index in nonpreferred
                ],
            }
        )
        if query_index % 25 == 0:
            print(f"Generated GENICL feedback for {query_index}/{len(records)} queries", flush=True)
    return preferences


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if args.batch_size % world_size:
        raise ValueError(
            f"Global GENICL batch size {args.batch_size} must be divisible by WORLD_SIZE={world_size}"
        )
    if world_size > 1 and not args.resume_preferences:
        raise ValueError(
            "Distributed GENICL training requires pre-generated preferences; run once with "
            "--feedback-only, then relaunch all ranks with --resume-preferences"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    task = DATASET_LOADERS[args.dataset](
        cache_dir=args.cache_dir,
        banking77_dir=args.banking77_dir,
        load_demonstrations=True,
    )
    preferences_path = args.output_dir / "preferences.json"
    if args.resume_preferences and preferences_path.is_file():
        preferences = json.loads(preferences_path.read_text(encoding="utf-8"))
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        e5 = GenICLE5Encoder(
            args.e5_model, device=args.e5_device, batch_size=args.e5_batch_size,
            model_cache_dir=args.cache_dir / "models",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.float16 if args.feedback_device.startswith("cuda") else torch.float32
        feedback_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
        feedback_model.to(args.feedback_device)
        preferences = make_feedback(
            task.demonstration_pool, task.labels, task.name, e5,
            feedback_model, tokenizer, args,
        )
        preferences_path.write_text(
            json.dumps(preferences, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        del feedback_model, e5
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.feedback_only:
        print(f"GENICL preferences ready: {preferences_path}")
        return

    split_preferences = list(preferences)
    random.Random(args.seed).shuffle(split_preferences)
    split_at = max(1, int(len(split_preferences) * 0.9))
    z = latent_description(task.name, args.n_prefix_tokens)
    train_plan = build_alternating_plan(
        split_preferences[:split_at], task.labels, z, args.batch_size, args.seed, True
    )
    eval_plan = build_alternating_plan(
        split_preferences[split_at:], task.labels, z, args.batch_size, args.seed + 1, False
    )
    if not eval_plan.rows:
        raise ValueError("GENICL training requires at least two preference queries")
    schedule_metadata = {
        "algorithm_1_order": ["demonstration_loss_Ld", "answer_loss_La"],
        "independent_optimizer_steps": True,
        "gradient_accumulation_steps": 1,
        "train_query_batches": train_plan.query_batches,
        "optimizer_batches_per_epoch": len(train_plan.batches),
        "first_stages": train_plan.stages[:10],
    }
    schedule_metadata.update(
        {
            "global_record_batch_size": args.batch_size,
            "world_size": world_size,
            "per_device_record_batch_size": args.batch_size // world_size,
            "distributed_split_batches": True,
        }
    )
    if rank == 0:
        (args.output_dir / "training_schedule.json").write_text(
            json.dumps(schedule_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import KTOConfig

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_model))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(args.base_model))
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    config = KTOConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size // world_size,
        per_device_eval_batch_size=args.batch_size // world_size,
        # 必须保持为 1；若累积 Ld 与 La 的梯度，就会把算法 1 中两次独立参数更新
        # 重新合并为一次混合目标更新。
        gradient_accumulation_steps=1,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        eval_strategy="steps",
        logging_steps=min(100, args.save_steps),
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_length=args.max_input_length,
        max_prompt_length=args.max_input_length - 64,
        max_completion_length=64,
        beta=args.beta,
        desirable_weight=1.0,
        undesirable_weight=1.0,
        report_to="none",
        seed=args.seed,
        precompute_ref_log_probs=False,
        accelerator_config={"split_batches": True},
    )
    AlternatingKTOTrainer = make_alternating_kto_trainer_class()
    trainer = AlternatingKTOTrainer(
        model=model,
        ref_model=None,
        args=config,
        peft_config=peft_config,
        train_dataset=Dataset.from_list(train_plan.rows),
        eval_dataset=Dataset.from_list(eval_plan.rows),
        tokenizer=tokenizer,
        alternating_batches=train_plan.batches,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    metadata = {
        "algorithm": "genicl-kto-lora-alternating-v1",
        "paper": "Learning to Select In-Context Demonstration Preferred by Large Language Model",
        "dataset": task.name,
        "base_model": str(args.base_model.resolve()),
        "e5_model": args.e5_model,
        "e5_pool_size": args.e5_pool_size,
        "feedback_candidates": args.feedback_candidates,
        "preferred_count": args.preferred_count,
        "nonpreferred_count": args.nonpreferred_count,
        "n_prefix_tokens": args.n_prefix_tokens,
        "beta": args.beta,
        "lambda_w": 1.0,
        "lambda_l": 1.0,
        "learning_rate": args.learning_rate,
        "eta_1": args.learning_rate,
        "eta_2": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "training_queries": len(preferences),
        "optimization": "two-stage-alternating-ld-then-la",
        "alternating_two_stage": True,
        "optimizer_steps_per_query_batch": 2,
        "gradient_accumulation_steps": 1,
        "training_schedule": "training_schedule.json",
    }
    metadata.update(
        {
            "world_size": world_size,
            "per_device_batch_size": args.batch_size // world_size,
        }
    )
    if rank == 0:
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
