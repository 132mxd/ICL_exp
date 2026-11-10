"""使用冻结目标 LLM 给单个 demonstration 打分并缓存监督数据。"""

import argparse
from pathlib import Path

from .common import TeacherScorer, prompt_with_demonstrations, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    scorer = TeacherScorer(args.teacher_model, args.device)
    rows = read_jsonl(args.input)
    # 每个分数都写入最终 JSONL；后续训练 EPR 时不会再次加载教师模型。
    for row_index, row in enumerate(rows, start=1):
        scores = {}
        candidates = row["candidates"]
        for start in range(0, len(candidates), args.batch_size):
            batch = candidates[start:start + args.batch_size]
            prompts = [prompt_with_demonstrations(row, [candidate]) for candidate in batch]
            values = scorer.score_many(prompts, [row["gold"]] * len(batch))
            scores.update({candidate["id"]: value for candidate, value in zip(batch, values)})
        row["singleton_scores"] = scores
        row["singleton_score_type"] = "mean_nll_lower_is_better"
        row["teacher_prompt_format"] = "chat_template_user_assistant"
        print(f"singleton scoring {row_index}/{len(rows)}", flush=True)
    write_jsonl(args.output, rows)
    print(f"Wrote singleton supervision to {args.output}")


if __name__ == "__main__":
    main()
