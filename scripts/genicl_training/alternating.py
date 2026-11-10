"""严格遵循 GENICL 论文算法 1 的两阶段批次调度。

每个查询小批次都按照固定的“演示样本（Ld）-> 答案（La）”顺序生成两个独立的
优化器批次。梯度累积固定为 1，因此 Hugging Face 训练器会对每个输出批次分别
执行一次参数更新，从而复现算法 1 的第 5—9 行。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Sequence, Tuple


@dataclass(frozen=True)
class AlternatingPlan:
    """保存展平后的 KTO 记录以及精确的优化器批次索引调度。"""

    rows: List[Dict[str, Any]]
    batches: List[List[int]]
    stages: List[str]
    query_batches: int


def build_query_stage_rows(
    preference: Dict[str, Any],
    labels: Sequence[str],
    latent: str,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """为单个查询构造公式（6）的 Ld 记录和公式（8）的 La 记录。"""

    wrong_answers = [str(label) for label in labels if str(label) != preference["gold"]]
    if not wrong_answers:
        raise ValueError("GENICL answer-level KTO requires at least two answer labels")

    demonstration_rows: List[Dict[str, Any]] = []
    for demonstration in preference["preferred"]:
        demonstration_rows.append(
            {
                "prompt": f"{demonstration}\n{preference['query']}",
                "completion": latent,
                "label": True,
            }
        )
    for demonstration in preference["nonpreferred"]:
        demonstration_rows.append(
            {
                "prompt": f"{demonstration}\n{preference['query']}",
                "completion": latent,
                "label": False,
            }
        )

    answer_prompt = f"{latent}\n{preference['query']}"
    answer_rows = [
        {"prompt": answer_prompt, "completion": preference["gold"], "label": True},
        {
            "prompt": answer_prompt,
            "completion": rng.choice(wrong_answers),
            "label": False,
        },
    ]
    return demonstration_rows, answer_rows


def build_alternating_plan(
    preferences: Sequence[Dict[str, Any]],
    labels: Sequence[str],
    latent: str,
    record_batch_size: int,
    seed: int,
    shuffle_queries: bool,
) -> AlternatingPlan:
    """构造互不混合的 ``Ld -> La`` 优化器批次。"""

    if record_batch_size <= 0:
        raise ValueError("GENICL record batch size must be positive")
    ordered = list(preferences)
    rng = random.Random(seed)
    if shuffle_queries:
        rng.shuffle(ordered)
    if not ordered:
        return AlternatingPlan([], [], [], 0)

    grouped = [build_query_stage_rows(item, labels, latent, rng) for item in ordered]
    demonstration_per_query = len(grouped[0][0])
    answer_per_query = len(grouped[0][1])
    if any(
        len(demonstration) != demonstration_per_query or len(answer) != answer_per_query
        for demonstration, answer in grouped
    ):
        raise ValueError("Every GENICL query must contribute the same number of stage records")
    if record_batch_size % demonstration_per_query or record_batch_size % answer_per_query:
        raise ValueError(
            "GENICL batch size must be divisible by both demonstration- and "
            "answer-level records per query so KL batches cannot cross stages"
        )
    max_queries = min(
        record_batch_size // demonstration_per_query,
        record_batch_size // answer_per_query,
    )

    def next_query_count(remaining: int) -> int:
        # 选择最大的查询分组，使两个阶段的记录数都能整除物理 KTO 批次大小，
        # 从而能够通过等比例重复精确补齐整个批次。
        for count in range(min(max_queries, remaining), 0, -1):
            if (
                record_batch_size % (count * demonstration_per_query) == 0
                and record_batch_size % (count * answer_per_query) == 0
            ):
                return count
        raise ValueError("Could not construct a stage-isolated GENICL KTO batch")

    def repeat_to_batch(rows_to_repeat: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        repetitions = record_batch_size // len(rows_to_repeat)
        return rows_to_repeat * repetitions

    rows: List[Dict[str, Any]] = []
    batches: List[List[int]] = []
    stages: List[str] = []
    start = 0
    query_batches = 0
    while start < len(grouped):
        query_count = next_query_count(len(grouped) - start)
        query_group = grouped[start : start + query_count]
        demonstration_batch = [row for demonstration, _ in query_group for row in demonstration]
        answer_batch = [row for _, answer in query_group for row in answer]
        demonstration_batch = repeat_to_batch(demonstration_batch)
        answer_batch = repeat_to_batch(answer_batch)

        demonstration_indices = list(range(len(rows), len(rows) + len(demonstration_batch)))
        rows.extend(demonstration_batch)
        answer_indices = list(range(len(rows), len(rows) + len(answer_batch)))
        rows.extend(answer_batch)

        # 对应算法 1 第 5—9 行：先执行一次 Ld 优化器更新，再执行一次 La 更新。
        batches.extend((demonstration_indices, answer_indices))
        stages.extend(("demonstration", "answer"))
        query_batches += 1
        start += query_count

    return AlternatingPlan(rows, batches, stages, query_batches)


class FixedBatchSampler:
    """重放不可变的批次索引计划；训练轮次重复由训练器负责。"""

    def __init__(self, batches: Sequence[Sequence[int]]) -> None:
        self.batches = [list(batch) for batch in batches]

    def __iter__(self) -> Iterator[List[int]]:
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


def make_alternating_kto_trainer_class() -> Any:
    """延迟创建 TRL 子类，使项目单元测试无需安装 TRL。"""

    from torch.utils.data import DataLoader
    from trl import KTOTrainer

    class AlternatingKTOTrainer(KTOTrainer):
        """训练加载器分别输出 Ld 和 La 更新的 KTOTrainer。"""

        def __init__(self, *args: Any, alternating_batches: Sequence[Sequence[int]], **kwargs: Any) -> None:
            self.alternating_batches = [list(batch) for batch in alternating_batches]
            super().__init__(*args, **kwargs)

        def get_train_dataloader(self) -> DataLoader:
            if self.train_dataset is None:
                raise ValueError("AlternatingKTOTrainer requires a train dataset")
            dataloader = DataLoader(
                self.train_dataset,
                batch_sampler=FixedBatchSampler(self.alternating_batches),
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=(
                    self.args.dataloader_persistent_workers
                    and self.args.dataloader_num_workers > 0
                ),
            )
            return self.accelerator.prepare(dataloader)

    return AlternatingKTOTrainer
