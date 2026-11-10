"""统一命令行流程：参数校验、加载任务、依次评测模型并汇总结果。

当前支持零样本、随机、BM25、KATE 和 CEIL ICL 评测。
"""

import argparse
import gc
import json
from pathlib import Path
from typing import List

from .config import PROJECT_ROOT, MODEL_CONFIGS, DATASET_NAMES
from .task_data import TaskData
from .data_utils import DATASET_LOADERS
from .model_client import server_model_ids, resolve_served_model_id
from .runner import evaluate_task

try:
    from ..example_selectors import (
        DEFAULT_KATE_ENCODER,
        DEFAULT_GENICL_E5_MODEL,
        CEILEncoder,
        CEILSelector,
        GenICLE5Encoder,
        GenICLScorer,
        GenICLSelector,
        KATESelector,
        TransformerMeanEncoder,
    )
except ImportError:
    from example_selectors import (
        DEFAULT_GENICL_E5_MODEL, DEFAULT_KATE_ENCODER, CEILEncoder,
        CEILSelector, GenICLE5Encoder, GenICLScorer, GenICLSelector,
        KATESelector, TransformerMeanEncoder,
    )

def parse_args() -> argparse.Namespace:
    """定义并检查实验所需的命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Zero-shot, random-ICL, BM25-ICL, KATE-ICL and CEIL-ICL evaluation "
            "with local models."
        )
    )
    parser.add_argument(
        "--method",
        choices=("zero_shot", "random", "bm25", "kate", "ceil", "genicl"),
        default="zero_shot",
        help="Evaluation method (default: zero_shot).",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=None,
        help="Number of demonstrations; ICL methods default to 4, zero_shot uses 0.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random selector seed (default: 42)."
    )
    parser.add_argument(
        "--bm25-k1", type=float, default=1.5, help="BM25 term-frequency saturation."
    )
    parser.add_argument(
        "--bm25-b", type=float, default=0.75, help="BM25 length normalization [0, 1]."
    )
    parser.add_argument(
        "--kate-encoder",
        default=DEFAULT_KATE_ENCODER,
        help="Hugging Face encoder used by KATE.",
    )
    parser.add_argument(
        "--kate-revision",
        default=None,
        help="Optional immutable Hugging Face model revision for reproducibility.",
    )
    parser.add_argument(
        "--kate-device",
        default="auto",
        help="Embedding device, e.g. auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--kate-batch-size",
        type=int,
        default=32,
        help="KATE encoder batch size (default: 32).",
    )
    parser.add_argument(
        "--kate-max-length",
        type=int,
        default=128,
        help="KATE encoder truncation length; 128 matches the original encoder.",
    )
    parser.add_argument(
        "--kate-order",
        choices=("most_similar_first", "most_similar_last"),
        default="most_similar_first",
        help="Order the retrieved top-k demonstrations in the prompt.",
    )
    parser.add_argument(
        "--kate-cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "kate_cache",
        help="Directory for fingerprinted training-pool embedding caches.",
    )
    parser.add_argument(
        "--ceil-checkpoint",
        type=Path,
        default=None,
        help="Trained CEIL checkpoint containing metadata.json and encoder/.",
    )
    parser.add_argument(
        "--ceil-device",
        default="auto",
        help="CEIL encoder device, e.g. auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--ceil-batch-size",
        type=int,
        default=32,
        help="CEIL encoder batch size (default: 32).",
    )
    parser.add_argument(
        "--ceil-candidate-pool-size",
        type=int,
        default=None,
        help="Override the checkpoint's DPP preselection pool size.",
    )
    parser.add_argument(
        "--genicl-checkpoint",
        type=Path,
        default=None,
        help="KTO/LoRA GENICL PEFT checkpoint containing metadata.json.",
    )
    parser.add_argument(
        "--genicl-e5-model",
        default=DEFAULT_GENICL_E5_MODEL,
        help="E5-base model used for GENICL search-space reduction.",
    )
    parser.add_argument("--genicl-e5-revision", default=None)
    parser.add_argument("--genicl-e5-device", default="auto")
    parser.add_argument("--genicl-e5-batch-size", type=int, default=128)
    parser.add_argument("--genicl-e5-max-length", type=int, default=256)
    parser.add_argument("--genicl-device", default="auto")
    parser.add_argument("--genicl-batch-size", type=int, default=8)
    parser.add_argument("--genicl-max-input-length", type=int, default=1024)
    parser.add_argument(
        "--genicl-candidate-pool-size",
        type=int,
        default=100,
        help="Number of E5 candidates reranked with Eq. (11) (paper inference uses 100).",
    )
    parser.add_argument("--genicl-n-prefix-tokens", type=int, default=10)
    parser.add_argument(
        "--genicl-cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "genicl_cache",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        default=list(DATASET_NAMES),
        help="Datasets to evaluate (default: all five).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_CONFIGS),
        default=["qwen", "llama"],
        help="Models to evaluate (default: qwen llama).",
    )
    parser.add_argument(
        "--qwen-base-url",
        default=MODEL_CONFIGS["qwen"]["base_url"],
        help="Qwen vLLM server root URL.",
    )
    parser.add_argument(
        "--llama-base-url",
        default=MODEL_CONFIGS["llama"]["base_url"],
        help="Llama vLLM server root URL.",
    )
    parser.add_argument(
        "--qwen-model-id",
        default=MODEL_CONFIGS["qwen"]["model_id"],
        help="Qwen model ID/path sent to vLLM.",
    )
    parser.add_argument(
        "--qwen1-5b-base-url",
        default=MODEL_CONFIGS["qwen1_5b"]["base_url"],
        help="Qwen 1.5B vLLM server root URL.",
    )
    parser.add_argument(
        "--qwen1-5b-model-id",
        default=MODEL_CONFIGS["qwen1_5b"]["model_id"],
        help="Qwen 1.5B model ID/path sent to vLLM.",
    )
    parser.add_argument(
        "--llama-model-id",
        default=MODEL_CONFIGS["llama"]["model_id"],
        help="Llama model ID/path sent to vLLM.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "hf_cache",
        help="Hugging Face dataset and model cache directory.",
    )
    parser.add_argument(
        "--banking77-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "banking77",
        help="Directory containing Banking77 test.jsonl and labels.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSONL predictions and summary.json.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N samples of each dataset (for smoke tests).",
    )
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=2, help="Retries per failed request.")
    parser.add_argument(
        "--no-structured-output",
        action="store_true",
        help="Do not send vLLM structured choice constraints.",
    )
    args = parser.parse_args()

    if args.method == "zero_shot":
        if args.num_shots not in (None, 0):
            parser.error("--method zero_shot requires --num-shots 0 (or omit it)")
        args.num_shots = 0
    else:
        if args.num_shots is None:
            # GENICL 论文第 4.1 节报告的实验协议使用 8 条演示样本。
            args.num_shots = 8 if args.method == "genicl" else 4
        if args.num_shots <= 0:
            parser.error(f"--method {args.method} requires --num-shots greater than zero")
    if args.bm25_k1 <= 0:
        parser.error("--bm25-k1 must be greater than zero")
    if not 0 <= args.bm25_b <= 1:
        parser.error("--bm25-b must be between zero and one")
    if args.kate_batch_size <= 0:
        parser.error("--kate-batch-size must be greater than zero")
    if args.kate_max_length <= 0:
        parser.error("--kate-max-length must be greater than zero")
    if args.ceil_batch_size <= 0:
        parser.error("--ceil-batch-size must be greater than zero")
    if args.ceil_candidate_pool_size is not None and args.ceil_candidate_pool_size <= 0:
        parser.error("--ceil-candidate-pool-size must be greater than zero")
    if args.method == "ceil" and args.ceil_checkpoint is None:
        parser.error("--method ceil requires --ceil-checkpoint")
    for name in (
        "genicl_e5_batch_size", "genicl_e5_max_length", "genicl_batch_size",
        "genicl_max_input_length", "genicl_candidate_pool_size", "genicl_n_prefix_tokens",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than zero")
    if args.method == "genicl" and args.genicl_checkpoint is None:
        parser.error("--method genicl requires --genicl-checkpoint")
    if args.output_dir is None:
        directory_name = (
            "zero_shot_baseline"
            if args.method == "zero_shot"
            else f"{args.method}_k{args.num_shots}"
        )
        args.output_dir = PROJECT_ROOT / "results" / directory_name
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be greater than zero")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    if args.retries < 0:
        parser.error("--retries cannot be negative")
    return args


def main() -> None:
    """加载数据与模型服务，执行全部指定实验并保存最终汇总。"""

    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    if args.method == "zero_shot":
        title = "Zero-shot baseline (no demonstrations)"
    elif args.method == "random":
        title = f"Random ICL baseline ({args.num_shots}-shot, seed={args.seed})"
    elif args.method == "bm25":
        title = (
            f"BM25 ICL baseline ({args.num_shots}-shot, "
            f"k1={args.bm25_k1}, b={args.bm25_b})"
        )
    elif args.method == "kate":
        title = (
            f"KATE ICL baseline ({args.num_shots}-shot, "
            f"encoder={args.kate_encoder}, order={args.kate_order})"
        )
    elif args.method == "ceil":
        title = f"CEIL ICL baseline ({args.num_shots}-shot, checkpoint={args.ceil_checkpoint})"
    else:
        title = (
            f"GENICL ({args.num_shots}-shot, checkpoint={args.genicl_checkpoint}, "
            f"E5={args.genicl_e5_model})"
        )
    print(title)
    print(f"Models:   {', '.join(args.models)}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Output:   {args.output_dir}")
    print("=" * 72)

    loaded_tasks: List[TaskData] = []
    load_failures = []
    # 先统一加载数据；某个数据集失败时记录原因，其余数据集仍继续运行。
    for dataset_name in args.datasets:
        print(f"\nLoading {dataset_name}...")
        try:
            task = DATASET_LOADERS[dataset_name](
                cache_dir=args.cache_dir,
                banking77_dir=args.banking77_dir,
                timeout=args.timeout,
                load_demonstrations=args.method in ("random", "bm25", "kate", "ceil", "genicl"),
            )
            if args.method in ("random", "bm25", "kate", "ceil", "genicl") and len(task.demonstration_pool) < args.num_shots:
                raise ValueError(
                    f"Only {len(task.demonstration_pool)} training examples are available, "
                    f"but --num-shots is {args.num_shots}"
                )
            if args.max_samples is not None:
                task.samples = task.samples[: args.max_samples]
            loaded_tasks.append(task)
            print(f"Loaded {len(task.samples)} evaluation samples.")
            if args.method in ("random", "bm25", "kate", "ceil", "genicl"):
                stats = task.demonstration_filter_stats
                print(
                    "Demonstration pool: "
                    f"{stats['original_pool_size']} -> {stats['final_pool_size']} "
                    f"(test overlap rows={stats['test_overlap_rows_removed']}, "
                    f"overlap label conflicts={stats['test_overlap_label_conflicts']}, "
                    f"internal duplicates={stats['internal_duplicate_rows_removed']}, "
                    f"conflict rows={stats['conflicting_rows_removed']})"
                )
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            print(f"SKIP {dataset_name}: {message}")
            load_failures.append({"dataset": dataset_name, "error": message})

    # KATE 检索与被评测 LLM 无关：只加载一次 encoder，并在模型循环前固定结果。
    if args.method == "kate" and loaded_tasks:
        print(f"\nLoading KATE encoder: {args.kate_encoder}")
        encoder = TransformerMeanEncoder(
            model_name=args.kate_encoder,
            revision=args.kate_revision,
            device=args.kate_device,
            batch_size=args.kate_batch_size,
            max_length=args.kate_max_length,
            model_cache_dir=args.cache_dir / "models",
        )
        for task in loaded_tasks:
            print(
                f"Precomputing KATE selections for {task.name}: "
                f"{len(task.demonstration_pool)} candidates, {len(task.samples)} queries"
            )
            selector = KATESelector(
                candidates=task.demonstration_pool,
                encoder=encoder,
                dataset_name=task.name,
                embedding_cache_dir=args.kate_cache_dir,
            )
            task.precomputed_selections = selector.select_many(
                [sample["input"] for sample in task.samples],
                num_shots=args.num_shots,
                order=args.kate_order,
            )

    if args.method == "ceil" and loaded_tasks:
        print(f"\nLoading CEIL checkpoint: {args.ceil_checkpoint}")
        encoder = CEILEncoder(
            checkpoint_dir=args.ceil_checkpoint,
            device=args.ceil_device,
            batch_size=args.ceil_batch_size,
        )
        for task in loaded_tasks:
            checkpoint_dataset = encoder.metadata.get("dataset")
            if checkpoint_dataset and checkpoint_dataset != task.name:
                raise ValueError(
                    f"CEIL checkpoint was trained for {checkpoint_dataset}, not {task.name}"
                )
            training_set_size = encoder.metadata.get("training_set_size")
            if training_set_size:
                print(
                    "CEIL training set size: "
                    f"{training_set_size}; inference shots: {args.num_shots}"
                )
            print(
                f"Precomputing CEIL selections for {task.name}: "
                f"{len(task.demonstration_pool)} candidates, {len(task.samples)} queries"
            )
            selector = CEILSelector(
                candidates=task.demonstration_pool,
                encoder=encoder,
                candidate_pool_size=args.ceil_candidate_pool_size,
            )
            task.precomputed_selections = selector.select_many(
                [sample["input"] for sample in task.samples], args.num_shots
            )
    # GENICL 遵循论文第 3.4 节：先用 E5-base 缩小搜索空间，再由适配后的
    # 因果语言模型按照潜变量的条件对数概率对候选排序。选出的演示样本固定后
    # 可由两个答题 LLM 复用，对应论文表 4 的迁移实验。
    if args.method == "genicl" and loaded_tasks:
        print(f"\nLoading GENICL E5 encoder: {args.genicl_e5_model}")
        e5_encoder = GenICLE5Encoder(
            model_name=args.genicl_e5_model,
            revision=args.genicl_e5_revision,
            device=args.genicl_e5_device,
            batch_size=args.genicl_e5_batch_size,
            max_length=args.genicl_e5_max_length,
            model_cache_dir=args.cache_dir / "models",
        )
        print(f"Loading GENICL checkpoint: {args.genicl_checkpoint}")
        scorer = GenICLScorer(
            checkpoint_dir=args.genicl_checkpoint,
            device=args.genicl_device,
            batch_size=args.genicl_batch_size,
            max_input_length=args.genicl_max_input_length,
            model_cache_dir=args.cache_dir / "models",
        )
        checkpoint_dataset = scorer.metadata.get("dataset")
        for task in loaded_tasks:
            if checkpoint_dataset and checkpoint_dataset not in (task.name, "multi_task"):
                raise ValueError(
                    f"GENICL checkpoint was trained for {checkpoint_dataset}, not {task.name}"
                )
            print(
                f"Precomputing GENICL selections for {task.name}: "
                f"{len(task.demonstration_pool)} candidates, {len(task.samples)} queries"
            )
            selector = GenICLSelector(
                candidates=task.demonstration_pool,
                e5_encoder=e5_encoder,
                scorer=scorer,
                dataset_name=task.name,
                candidate_pool_size=args.genicl_candidate_pool_size,
                n_prefix_tokens=args.genicl_n_prefix_tokens,
                embedding_cache_dir=args.genicl_cache_dir,
            )
            task.precomputed_selections = selector.select_many(
                [sample["input"] for sample in task.samples], args.num_shots
            )
        # 在答题模型评测前将选择结果完全物化，并释放本地 E5 和因果语言模型权重，
        # 避免它们继续与 vLLM 争用 GPU 显存。
        del selector, scorer, e5_encoder
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    summaries = []
    model_failures = []
    # 每个模型对应一个独立 vLLM 服务，按“模型 × 数据集”依次评测。
    for model_name in args.models:
        base_url = getattr(args, f"{model_name}_base_url").rstrip("/")
        configured_model_id = getattr(args, f"{model_name}_model_id")
        print(f"\nChecking {model_name} server: {base_url}")
        try:
            served_ids = server_model_ids(base_url, args.timeout)
            if not served_ids:
                raise RuntimeError("The server returned no models")
            model_id = resolve_served_model_id(configured_model_id, served_ids)
            print(f"Using model ID: {model_id}")
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            print(f"SKIP {model_name}: {message}")
            model_failures.append(
                {"model": model_name, "base_url": base_url, "error": message}
            )
            continue

        for task in loaded_tasks:
            summary = evaluate_task(
                task=task,
                model_name=model_name,
                model_id=model_id,
                base_url=base_url,
                output_dir=args.output_dir,
                timeout=args.timeout,
                retries=args.retries,
                structured_output=not args.no_structured_output,
                method=args.method,
                num_shots=args.num_shots,
                seed=args.seed if args.method == "random" else None,
                bm25_k1=args.bm25_k1,
                bm25_b=args.bm25_b,
                kate_encoder=args.kate_encoder,
                kate_order=args.kate_order,
                kate_max_length=args.kate_max_length,
                ceil_checkpoint=str(args.ceil_checkpoint) if args.ceil_checkpoint else None,
                genicl_checkpoint=(
                    str(args.genicl_checkpoint) if args.genicl_checkpoint else None
                ),
            )
            summaries.append(summary)
            print(
                f"{model_name}/{task.name}: "
                f"accuracy={summary['accuracy']:.4f} "
                f"({summary['correct']}/{summary['samples']}), "
                f"invalid={summary['invalid']}, errors={summary['request_errors']}"
            )

    final_report = {
        "method": args.method,
        "zero_shot": args.method == "zero_shot",
        "num_shots": args.num_shots,
        "seed": args.seed if args.method == "random" else None,
        "bm25_k1": args.bm25_k1 if args.method == "bm25" else None,
        "bm25_b": args.bm25_b if args.method == "bm25" else None,
        "kate_encoder": args.kate_encoder if args.method == "kate" else None,
        "kate_revision": args.kate_revision if args.method == "kate" else None,
        "kate_order": args.kate_order if args.method == "kate" else None,
        "kate_max_length": args.kate_max_length if args.method == "kate" else None,
        "ceil_checkpoint": str(args.ceil_checkpoint) if args.method == "ceil" else None,
        "ceil_candidate_pool_size": (
            args.ceil_candidate_pool_size if args.method == "ceil" else None
        ),
        "genicl_checkpoint": (
            str(args.genicl_checkpoint) if args.method == "genicl" else None
        ),
        "genicl_e5_model": args.genicl_e5_model if args.method == "genicl" else None,
        "genicl_candidate_pool_size": (
            args.genicl_candidate_pool_size if args.method == "genicl" else None
        ),
        "datasets": list(args.datasets),
        "models": list(args.models),
        "structured_output_for_label_tasks": not args.no_structured_output,
        "results": summaries,
        "dataset_load_failures": load_failures,
        "model_failures": model_failures,
    }
    # summary.json 保存整体指标和失败信息，JSONL 文件保存逐样本细节。
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(final_report, file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print(f"Finished. Summary: {summary_path}")
    print("=" * 72)

    if not loaded_tasks:
        raise SystemExit("No datasets could be loaded.")
    if not summaries:
        raise SystemExit("No model/dataset evaluation completed.")
