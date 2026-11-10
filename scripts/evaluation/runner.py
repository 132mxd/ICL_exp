"""公共评测循环：发送样本提示词、调用评分函数、保存逐条结果和指标。"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

from .task_data import TaskData
from .model_client import query_vllm
from .metrics import exact_match, parse_prediction
from .prompts import build_icl_prompt

# example_selectors 与 evaluation 同属 scripts 下；兼容脚本运行和模块运行。
try:
    from ..example_selectors import BM25Selector, select_random_examples
except ImportError:
    from example_selectors import BM25Selector, select_random_examples

def evaluate_task(
    task: TaskData,
    model_name: str,
    model_id: str,
    base_url: str,
    output_dir: Path,
    timeout: int,
    retries: int,
    structured_output: bool,
    method: str = "zero_shot",
    num_shots: int = 0,
    seed: Optional[int] = 42,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    kate_encoder: Optional[str] = None,
    kate_order: Optional[str] = None,
    kate_max_length: Optional[int] = None,
    ceil_checkpoint: Optional[str] = None,
    genicl_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """使用一个模型评测一个数据集，并逐样本记录预测及统计准确率。"""

    output_path = output_dir / f"{model_name}_{task.name}.jsonl"
    correct = 0
    invalid = 0
    errors = 0
    start_time = time.time()
    generation_max_tokens = 32
    # BM25 倒排索引对一个“模型-数据集”组合只构建一次，而不是每道题重建。
    bm25_selector = (
        BM25Selector(task.demonstration_pool, k1=bm25_k1, b=bm25_b)
        if method == "bm25"
        else None
    )

    # 每个“模型-数据集”组合单独生成 JSONL，便于中间检查和后续误差分析。
    with output_path.open("w", encoding="utf-8") as output_file:
        progress = tqdm(task.samples, desc=f"{method} {model_name}/{task.name}")
        for index, sample in enumerate(progress):
            request_start = time.time()
            error_message = None
            sample_labels = sample.get("labels", task.labels)
            use_structured_output = structured_output and task.answer_type == "label"
            demonstrations = []
            prompt = sample["prompt"]
            if method == "random":
                if seed is None:
                    raise ValueError("Random selection requires a seed")
                demonstrations = select_random_examples(
                    task.demonstration_pool, num_shots, seed, task.name, index
                )
                prompt = build_icl_prompt(task.name, prompt, demonstrations)
            elif method == "bm25":
                if bm25_selector is None:
                    raise RuntimeError("BM25 selector was not initialized")
                demonstrations = bm25_selector.select(sample["input"], num_shots)
                prompt = build_icl_prompt(task.name, prompt, demonstrations)
            elif method in ("kate", "ceil", "genicl"):
                if len(task.precomputed_selections) != len(task.samples):
                    raise RuntimeError(
                        f"{method.upper()} selections were not precomputed for every evaluation sample"
                    )
                demonstrations = task.precomputed_selections[index]
                prompt = build_icl_prompt(task.name, prompt, demonstrations)
            try:
                raw_output = query_vllm(
                    prompt=prompt,
                    labels=sample_labels,
                    base_url=base_url,
                    model_id=model_id,
                    timeout=timeout,
                    retries=retries,
                    structured_output=use_structured_output,
                    max_tokens=generation_max_tokens,
                )
                prediction = parse_prediction(raw_output, sample_labels)
            # 单条样本失败时记录错误并继续，避免长时间实验因一个请求而全部终止。
            except Exception as error:
                raw_output = ""
                prediction = None
                error_message = f"{type(error).__name__}: {error}"
                errors += 1

            is_correct = exact_match(prediction, sample["gold"])
            correct += int(is_correct)
            invalid += int(prediction is None)
            processed = index + 1
            progress.set_postfix(accuracy=f"{correct / processed:.4f}", invalid=invalid)

            record = {
                "index": index,
                "dataset": task.name,
                "model": model_name,
                "model_id": model_id,
                "method": method,
                "zero_shot": method == "zero_shot",
                "num_shots": len(demonstrations),
                "seed": seed,
                "genicl_checkpoint": genicl_checkpoint if method == "genicl" else None,
                # 保存示例 ID、输入和答案，方便复现实验并检查是否选对样本。
                "demonstrations": demonstrations,
                "input": sample["input"],
                "gold": sample["gold"],
                "prediction": prediction,
                "raw_output": raw_output,
                "max_tokens": generation_max_tokens,
                "correct": is_correct,
                "error": error_message,
                "latency_seconds": round(time.time() - request_start, 4),
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            # 每条样本立即写入磁盘，意外中断时仍可保留已经完成的结果。
            output_file.flush()

    # 返回该模型在该数据集上的汇总指标，之后统一写入 summary.json。
    total = len(task.samples)
    elapsed = time.time() - start_time
    return {
        "model": model_name,
        "model_id": model_id,
        "dataset": task.name,
        "method": method,
        "zero_shot": method == "zero_shot",
        "num_shots": num_shots,
        "seed": seed,
        "bm25_k1": bm25_k1 if method == "bm25" else None,
        "bm25_b": bm25_b if method == "bm25" else None,
        "kate_encoder": kate_encoder if method == "kate" else None,
        "kate_order": kate_order if method == "kate" else None,
        "kate_max_length": kate_max_length if method == "kate" else None,
        "ceil_checkpoint": ceil_checkpoint if method == "ceil" else None,
        "genicl_checkpoint": genicl_checkpoint if method == "genicl" else None,
        "answer_type": task.answer_type,
        "demonstration_filter_stats": task.demonstration_filter_stats,
        "metric": "exact_match_accuracy",
        "structured_output": structured_output and task.answer_type == "label",
        "max_tokens": generation_max_tokens,
        "samples": total,
        "correct": correct,
        "invalid": invalid,
        "request_errors": errors,
        "accuracy": correct / total if total else 0.0,
        "elapsed_seconds": round(elapsed, 2),
        "output": str(output_path),
    }
