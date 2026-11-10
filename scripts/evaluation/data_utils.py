"""加载五个数据集，统一为包含 input、prompt、gold 的 TaskData。"""

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from datasets import load_dataset

from .config import (
    BOOLQ_LABELS, CMSQA_LABELS, PAWS_LABELS, SST5_LABELS,
)
from .task_data import TaskData
from .prompts import (
    build_banking77_prompt, build_boolq_prompt, build_cmsqa_prompt,
    build_paws_prompt, build_sst5_prompt,
)

def demonstration(example_id: str, inputs: Dict[str, Any], gold: str) -> Dict[str, Any]:
    """构造统一训练示例，并保留来源 ID 以便结果追踪和复现。"""

    return {"id": example_id, "input": inputs, "gold": gold}


def _normalize_input_value(value: Any) -> Any:
    """规范化输入以检测文本相同但大小写或空白不同的样本。"""

    if isinstance(value, str):
        normalized = unicodedata.normalize("NFKC", value).strip().casefold()
        return re.sub(r"\s+", " ", normalized)
    if isinstance(value, dict):
        return {
            str(key): _normalize_input_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_input_value(item) for item in value]
    return value


def normalized_input_key(inputs: Dict[str, Any]) -> str:
    """为完整 input 生成稳定键；不使用 gold，避免标签参与过滤。"""

    return json.dumps(
        _normalize_input_value(inputs),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sanitize_demonstration_pool(
    samples: Sequence[Dict[str, Any]],
    pool: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """删除测试集污染、标签冲突和候选池内部重复，并返回审计统计。"""

    test_labels_by_key: Dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        test_labels_by_key[normalized_input_key(sample["input"])].add(
            str(sample["gold"])
        )
    test_keys = set(test_labels_by_key)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for candidate in pool:
        grouped[normalized_input_key(candidate["input"])].append(candidate)

    cleaned: List[Dict[str, Any]] = []
    overlap_keys = 0
    overlap_rows = 0
    overlap_label_conflicts = 0
    conflict_keys = 0
    conflict_rows = 0
    duplicate_rows = 0
    for key, candidates in grouped.items():
        if key in test_keys:
            overlap_keys += 1
            overlap_rows += len(candidates)
            candidate_labels = {str(candidate["gold"]) for candidate in candidates}
            if candidate_labels != test_labels_by_key[key]:
                overlap_label_conflicts += 1
            continue

        labels = {str(candidate["gold"]) for candidate in candidates}
        if len(labels) > 1:
            conflict_keys += 1
            conflict_rows += len(candidates)
            continue

        cleaned.append(candidates[0])
        duplicate_rows += len(candidates) - 1

    stats = {
        "original_pool_size": len(pool),
        "test_overlap_unique_inputs": overlap_keys,
        "test_overlap_rows_removed": overlap_rows,
        "test_overlap_label_conflicts": overlap_label_conflicts,
        "conflicting_unique_inputs": conflict_keys,
        "conflicting_rows_removed": conflict_rows,
        "internal_duplicate_rows_removed": duplicate_rows,
        "final_pool_size": len(cleaned),
    }
    return cleaned, stats


def make_task_data(
    name: str,
    labels: Sequence[str],
    samples: List[Dict[str, Any]],
    pool: List[Dict[str, Any]],
    answer_type: str = "label",
) -> TaskData:
    """统一构造任务，并保证所有 ICL 方法共享同一份已清理候选池。"""

    cleaned_pool, filter_stats = sanitize_demonstration_pool(samples, pool)
    return TaskData(
        name,
        labels,
        samples,
        answer_type=answer_type,
        demonstration_pool=cleaned_pool,
        demonstration_filter_stats=filter_stats,
    )


def load_sst5(cache_dir: Path, load_demonstrations: bool = False, **_: Any) -> TaskData:
    """从 Hugging Face 缓存或网络加载 SST-5 官方测试集。"""

    dataset = load_dataset(
        "SetFit/sst5",
        split="test",
        cache_dir=str(cache_dir),
    )
    samples = []
    for example in dataset:
        gold = str(example.get("label_text", SST5_LABELS[int(example["label"])]))
        samples.append(
            {
                "input": {"text": example["text"]},
                "prompt": build_sst5_prompt(example["text"]),
                "gold": gold,
            }
        )
    pool = []
    if load_demonstrations:
        train_dataset = load_dataset(
            "SetFit/sst5", split="train", cache_dir=str(cache_dir)
        )
        for index, example in enumerate(train_dataset):
            gold = str(example.get("label_text", SST5_LABELS[int(example["label"])]))
            pool.append(
                demonstration(f"train_{index}", {"text": example["text"]}, gold)
            )
    return make_task_data("sst5", SST5_LABELS, samples, pool)


def load_paws(cache_dir: Path, load_demonstrations: bool = False, **_: Any) -> TaskData:
    """加载 PAWS-Wiki 有标签测试集，并使用训练集作为示例池。"""

    dataset = load_dataset(
        "google-research-datasets/paws", "labeled_final",
        split="test", cache_dir=str(cache_dir),
    )
    samples = []
    for example in dataset:
        sentence1, sentence2 = str(example["sentence1"]), str(example["sentence2"])
        gold = PAWS_LABELS[int(example["label"])]
        samples.append({
            "input": {"sentence1": sentence1, "sentence2": sentence2},
            "prompt": build_paws_prompt(sentence1, sentence2),
            "gold": gold,
        })
    pool = []
    if load_demonstrations:
        train_dataset = load_dataset(
            "google-research-datasets/paws", "labeled_final",
            split="train", cache_dir=str(cache_dir),
        )
        for index, example in enumerate(train_dataset):
            pool.append(demonstration(
                f"train_{index}",
                {"sentence1": str(example["sentence1"]), "sentence2": str(example["sentence2"])},
                PAWS_LABELS[int(example["label"])],
            ))
    return make_task_data("paws", PAWS_LABELS, samples, pool)


def read_json(path: Path) -> Any:
    """读取一个 UTF-8 编码的 JSON 文件。"""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """逐行读取 JSONL 文件，并在格式错误时报告准确行号。"""

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
    return records


def load_banking77(
    banking77_dir: Path, load_demonstrations: bool = False, **_: Any
) -> TaskData:
    """从项目 data 目录读取已准备好的 Banking77 测试集和标签表。"""

    test_path = banking77_dir / "test.jsonl"
    train_path = banking77_dir / "train.jsonl"
    labels_path = banking77_dir / "labels.json"
    if not test_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(
            f"Banking77 files not found under {banking77_dir}. "
            "Expected test.jsonl and labels.json."
        )

    label_records = read_json(labels_path)
    labels = tuple(
        item["label"] for item in sorted(label_records, key=lambda item: item["label_id"])
    )
    samples = []
    for example in read_jsonl(test_path):
        samples.append(
            {
                "input": {"text": example["text"]},
                "prompt": build_banking77_prompt(example["text"], labels),
                "gold": example["label"],
            }
        )
    pool = []
    if load_demonstrations:
        if not train_path.is_file():
            raise FileNotFoundError(f"Banking77 training file not found: {train_path}")
        for index, example in enumerate(read_jsonl(train_path)):
            pool.append(
                demonstration(
                    str(example.get("id", f"train_{index}")),
                    {"text": example["text"]},
                    example["label"],
                )
            )
    return make_task_data("banking77", labels, samples, pool)


def load_boolq(cache_dir: Path, load_demonstrations: bool = False, **_: Any) -> TaskData:
    """加载 BoolQ 验证集，并把布尔答案统一映射为 yes/no。"""

    dataset = load_dataset(
        "google/boolq",
        split="validation",
        cache_dir=str(cache_dir),
    )
    samples = []
    for example in dataset:
        gold = "yes" if bool(example["answer"]) else "no"
        samples.append(
            {
                "input": {
                    "passage": example["passage"],
                    "question": example["question"],
                },
                "prompt": build_boolq_prompt(example["passage"], example["question"]),
                "gold": gold,
            }
        )
    pool = []
    if load_demonstrations:
        train_dataset = load_dataset(
            "google/boolq", split="train", cache_dir=str(cache_dir)
        )
        for index, example in enumerate(train_dataset):
            gold = "yes" if bool(example["answer"]) else "no"
            pool.append(
                demonstration(
                    f"train_{index}",
                    {"passage": example["passage"], "question": example["question"]},
                    gold,
                )
            )
    return make_task_data("boolq", BOOLQ_LABELS, samples, pool)


def _cmsqa_choices(example: Dict[str, Any]) -> List[Dict[str, str]]:
    """把 datasets 的列式 choices 字段转换为便于提示词复用的记录列表。"""

    choices = example["choices"]
    result = [
        {"label": str(label), "text": str(text)}
        for label, text in zip(choices["label"], choices["text"])
    ]
    if tuple(choice["label"] for choice in result) != CMSQA_LABELS:
        raise ValueError("CommonsenseQA choices must be ordered and labeled A through E")
    return result


def load_cmsqa(cache_dir: Path, load_demonstrations: bool = False, **_: Any) -> TaskData:
    """加载 CommonsenseQA 验证集；官方测试答案不可本地使用。"""

    dataset = load_dataset("tau/commonsense_qa", split="validation", cache_dir=str(cache_dir))
    samples = []
    for example in dataset:
        question = str(example["question"])
        choices = _cmsqa_choices(example)
        gold = str(example["answerKey"])
        if gold not in CMSQA_LABELS:
            raise ValueError(f"Invalid CommonsenseQA answer key: {gold!r}")
        samples.append({
            "input": {"question": question, "choices": choices},
            "prompt": build_cmsqa_prompt(question, choices),
            "gold": gold,
        })
    pool = []
    if load_demonstrations:
        train_dataset = load_dataset("tau/commonsense_qa", split="train", cache_dir=str(cache_dir))
        for index, example in enumerate(train_dataset):
            choices = _cmsqa_choices(example)
            pool.append(demonstration(
                f"train_{index}",
                {"question": str(example["question"]), "choices": choices},
                str(example["answerKey"]),
            ))
    return make_task_data("cmsqa", CMSQA_LABELS, samples, pool)


# 通过名称查找对应的数据加载函数，便于用命令行自由选择数据集。
DATASET_LOADERS: Dict[str, Callable[..., TaskData]] = {
    "sst5": load_sst5,
    "paws": load_paws,
    "banking77": load_banking77,
    "boolq": load_boolq,
    "cmsqa": load_cmsqa,
}
