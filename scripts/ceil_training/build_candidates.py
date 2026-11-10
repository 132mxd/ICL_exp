"""为每个选择器训练 query 构造小规模 BM25 候选示例池。"""

import argparse
from pathlib import Path

from .common import load_ceil_training_data, write_jsonl
from ..evaluation.config import DATASET_NAMES, PROJECT_ROOT
from ..example_selectors import BM25Selector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="sst5")
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--max-pool-size", type=int, default=2000)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "hf_cache")
    parser.add_argument(
        "--banking77-dir", type=Path, default=PROJECT_ROOT / "data" / "banking77"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.max_queries, args.max_pool_size, args.candidate_count) <= 0:
        parser.error("query, pool, and candidate counts must be positive")

    # 监督query只从validation或train内部划分；最终test/validation评测集不参与训练。
    queries, pool = load_ceil_training_data(
        args.dataset, args.cache_dir, args.banking77_dir,
        args.max_queries, args.max_pool_size, args.seed,
    )
    if args.candidate_count > len(pool):
        parser.error("--candidate-count exceeds the cleaned demonstration pool")
    # BM25 仅负责廉价召回；后续由目标 LLM 给候选示例产生真实监督分数。
    selector = BM25Selector(pool)
    rows = []
    for query in queries:
        candidates = selector.select(query["input"], args.candidate_count)
        for candidate in candidates:
            candidate.pop("selection_score", None)
        rows.append({**query, "dataset": args.dataset, "candidates": candidates})
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} query candidate pools to {args.output}")


if __name__ == "__main__":
    main()
