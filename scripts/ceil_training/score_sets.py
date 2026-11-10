"""使用冻结目标 LLM 给完整 demonstration set 打分并缓存监督数据。"""

import argparse
from pathlib import Path

from .common import (
    TeacherScorer, candidate_by_id, prompt_with_demonstrations, read_jsonl, write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    scorer = TeacherScorer(args.teacher_model, args.device)
    rows = read_jsonl(args.input)
    for row_index, row in enumerate(rows, start=1):
        lookup = candidate_by_id(row)
        scored_sets = []
        # 与单示例评分不同，这里评估整个集合插入 prompt 后的联合效用。
        sets = row["candidate_sets"]
        for start in range(0, len(sets), args.batch_size):
            batch = sets[start:start + args.batch_size]
            prompts = [prompt_with_demonstrations(row, [lookup[x] for x in ids]) for ids in batch]
            values = scorer.score_many(prompts, [row["gold"]] * len(batch))
            scored_sets.extend({"ids": ids, "score": score} for ids, score in zip(batch, values))
        row["set_scores"] = scored_sets
        row["set_score_type"] = "mean_nll_lower_is_better"
        row["teacher_prompt_format"] = "chat_template_user_assistant"
        print(f"set scoring {row_index}/{len(rows)}", flush=True)
    write_jsonl(args.output, rows)
    print(f"Wrote set-level supervision to {args.output}")


if __name__ == "__main__":
    main()
