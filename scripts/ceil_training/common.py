"""CEIL 流水线共用的数据、序列化、编码和教师模型评分工具。"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from ..evaluation.config import BOOLQ_LABELS, CMSQA_LABELS, PAWS_LABELS, SST5_LABELS
from ..evaluation.data_utils import (
    _cmsqa_choices, demonstration, read_json, read_jsonl as read_evaluation_jsonl,
    sanitize_demonstration_pool,
)
from ..evaluation.prompts import (
    build_banking77_prompt, build_boolq_prompt, build_cmsqa_prompt,
    build_icl_prompt, build_paws_prompt, build_sst5_prompt,
)
from ..example_selectors.bm25_selector import input_to_text


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """读取 UTF-8 JSONL 中的全部非空记录。"""

    with Path(path).open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """先写临时文件再原子替换，避免中断后留下半个监督文件。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_sst5_training_data(
    cache_dir: Path,
    max_queries: int,
    max_pool_size: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """使用 validation 训练选择器，并严格保留 test 只做最终评测。"""

    validation = load_dataset("SetFit/sst5", split="validation", cache_dir=str(cache_dir))
    train = load_dataset("SetFit/sst5", split="train", cache_dir=str(cache_dir))
    queries = []
    for index, example in enumerate(validation):
        if len(queries) >= max_queries:
            break
        gold = str(example.get("label_text", SST5_LABELS[int(example["label"])]))
        text = str(example["text"])
        queries.append(
            {
                "id": f"validation_{index}",
                "input": {"text": text},
                "prompt": build_sst5_prompt(text),
                "gold": gold,
            }
        )
    pool = []
    for index, example in enumerate(train):
        if len(pool) >= max_pool_size:
            break
        gold = str(example.get("label_text", SST5_LABELS[int(example["label"])]))
        pool.append(demonstration(f"train_{index}", {"text": str(example["text"])}, gold))
    pool, _ = sanitize_demonstration_pool(queries, pool)
    return queries, pool


def _limit_and_clean(
    queries: List[Dict[str, Any]], pool: List[Dict[str, Any]],
    max_queries: int, max_pool_size: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """统一限制监督query和候选池规模，并删除query自身与重复候选。"""

    queries = queries[:max_queries]
    # 先从完整训练池删除监督query自身，再截断池；反过来会在两者都取前N条时
    # 意外得到空池。
    pool, _ = sanitize_demonstration_pool(queries, pool)
    return queries, pool[:max_pool_size]


def _balanced_sample_without_replacement(
    rows: Sequence[Dict[str, Any]], size: int, rng: random.Random,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """按标签轮询抽样，并返回选中项与未选中项。用于有序的Banking77文件。"""

    if size < 0 or size > len(rows):
        raise ValueError(f"Cannot sample {size} rows from {len(rows)} available rows")
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["gold"])].append(row)
    labels = sorted(groups)
    for label in labels:
        rng.shuffle(groups[label])

    selected: List[Dict[str, Any]] = []
    offsets = {label: 0 for label in labels}
    while len(selected) < size:
        made_progress = False
        for label in labels:
            offset = offsets[label]
            if offset < len(groups[label]):
                selected.append(groups[label][offset])
                offsets[label] = offset + 1
                made_progress = True
                if len(selected) == size:
                    break
        if not made_progress:
            raise RuntimeError("Balanced sampling exhausted rows before reaching requested size")

    remaining = [
        row for label in labels for row in groups[label][offsets[label]:]
    ]
    return selected, remaining


def load_ceil_training_data(
    dataset: str,
    cache_dir: Path,
    banking77_dir: Path,
    max_queries: int,
    max_pool_size: int,
    seed: int = 42,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """为五个任务构造不接触最终评测集的CEIL监督query和示例池。"""

    if dataset == "sst5":
        return load_sst5_training_data(cache_dir, max_queries, max_pool_size)

    queries: List[Dict[str, Any]] = []
    pool: List[Dict[str, Any]] = []
    if dataset == "paws":
        query_rows = load_dataset(
            "google-research-datasets/paws", "labeled_final",
            split="validation", cache_dir=str(cache_dir),
        )
        pool_rows = load_dataset(
            "google-research-datasets/paws", "labeled_final",
            split="train", cache_dir=str(cache_dir),
        )
        for index, example in enumerate(query_rows):
            sentence1, sentence2 = str(example["sentence1"]), str(example["sentence2"])
            queries.append({
                "id": f"validation_{index}",
                "input": {"sentence1": sentence1, "sentence2": sentence2},
                "prompt": build_paws_prompt(sentence1, sentence2),
                "gold": PAWS_LABELS[int(example["label"])],
            })
        for index, example in enumerate(pool_rows):
            pool.append(demonstration(
                f"train_{index}",
                {"sentence1": str(example["sentence1"]), "sentence2": str(example["sentence2"])},
                PAWS_LABELS[int(example["label"])],
            ))
    elif dataset == "banking77":
        labels_path = banking77_dir / "labels.json"
        train_path = banking77_dir / "train.jsonl"
        if not labels_path.is_file() or not train_path.is_file():
            raise FileNotFoundError(
                f"Banking77 training files not found under {banking77_dir}; "
                "expected labels.json and train.jsonl"
            )
        labels = tuple(
            item["label"]
            for item in sorted(read_json(labels_path), key=lambda item: item["label_id"])
        )
        all_rows = []
        for index, example in enumerate(read_evaluation_jsonl(train_path)):
            item = demonstration(
                str(example.get("id", f"train_{index}")),
                {"text": str(example["text"])}, str(example["label"]),
            )
            all_rows.append(item)
        # 本地Banking77按类别成块排列，不能使用前N条截断。先删除重复/冲突文本，
        # 再按77个标签做确定性均衡抽样，且pool只从query之外的剩余样本产生。
        all_rows, _ = sanitize_demonstration_pool([], all_rows)
        rng = random.Random(seed)
        selected_queries, remaining = _balanced_sample_without_replacement(
            all_rows, max_queries, rng,
        )
        pool, _ = _balanced_sample_without_replacement(remaining, max_pool_size, rng)
        queries = [
            {**item, "prompt": build_banking77_prompt(str(item["input"]["text"]), labels)}
            for item in selected_queries
        ]
        return queries, pool
    elif dataset == "boolq":
        train_rows = load_dataset("google/boolq", split="train", cache_dir=str(cache_dir))
        for index, example in enumerate(train_rows):
            inputs = {"passage": str(example["passage"]), "question": str(example["question"])}
            gold = BOOLQ_LABELS[0] if bool(example["answer"]) else BOOLQ_LABELS[1]
            item = demonstration(f"train_{index}", inputs, gold)
            queries.append({**item, "prompt": build_boolq_prompt(inputs["passage"], inputs["question"])})
            pool.append(item)
    elif dataset == "cmsqa":
        train_rows = load_dataset("tau/commonsense_qa", split="train", cache_dir=str(cache_dir))
        for index, example in enumerate(train_rows):
            choices = _cmsqa_choices(example)
            inputs = {"question": str(example["question"]), "choices": choices}
            gold = str(example["answerKey"])
            if gold not in CMSQA_LABELS:
                raise ValueError(f"Invalid CommonsenseQA answer key: {gold!r}")
            item = demonstration(f"train_{index}", inputs, gold)
            queries.append({**item, "prompt": build_cmsqa_prompt(inputs["question"], choices)})
            pool.append(item)
    else:
        raise ValueError(f"Unsupported CEIL training dataset: {dataset}")

    return _limit_and_clean(queries, pool, max_queries, max_pool_size)


class MeanEncoder(torch.nn.Module):
    """EPR 与 conditional DPP 共用的稠密文本编码器。"""

    def __init__(self, model_name_or_path: str, cache_dir: Path | None = None) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name_or_path,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )

    def forward(self, encoded: Dict[str, torch.Tensor]) -> torch.Tensor:
        """对有效 token 做 mean pooling，并返回 L2 归一化句向量。"""

        output = self.backbone(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).to(output.last_hidden_state.dtype)
        pooled = (output.last_hidden_state * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)


def tokenize_texts(tokenizer: Any, texts: Sequence[str], max_length: int, device: str):
    """批量分词并把全部张量移动到指定训练设备。"""

    encoded = tokenizer(
        list(texts), padding=True, truncation=True, max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


class TeacherScorer:
    """用冻结目标 LLM 对正确答案 token 的平均 NLL 评分（越低越好）。"""

    def __init__(self, model_name_or_path: str, device: str = "auto") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = (
            "cuda" if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto"
            else device
        )
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, dtype=dtype,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, prompt: str, gold: str) -> float:
        """按 Qwen chat template 只统计 assistant gold，不统计 user prompt。"""

        return self.score_many([prompt], [gold])[0]

    @torch.inference_mode()
    def score_many(self, prompts: Sequence[str], golds: Sequence[str]) -> List[float]:
        """批量计算相同定义的 gold NLL；右侧 padding 不进入均值。"""

        if len(prompts) != len(golds) or not prompts:
            raise ValueError("prompts and golds must have the same non-zero length")
        sequences, prefix_lengths, gold_lengths = [], [], []
        for prompt, gold in zip(prompts, golds):
            prefix = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True,
            )
            if hasattr(prefix, "get"):
                prefix = prefix["input_ids"]
            if prefix and isinstance(prefix[0], list):
                prefix = prefix[0]
            gold_ids = self.tokenizer.encode(gold, add_special_tokens=False)
            if not gold_ids:
                raise ValueError("Gold continuation produced no scoreable tokens")
            sequences.append(torch.tensor(prefix + gold_ids, dtype=torch.long))
            prefix_lengths.append(len(prefix)); gold_lengths.append(len(gold_ids))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.device)
        lengths = torch.tensor([len(x) for x in sequences], device=self.device)
        attention_mask = torch.arange(input_ids.shape[1], device=self.device)[None] < lengths[:, None]
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
        targets = input_ids[:, 1:]
        gathered = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        results = []
        for index, (prefix_length, gold_length) in enumerate(zip(prefix_lengths, gold_lengths)):
            start = prefix_length - 1
            results.append(float(-gathered[index, start:start + gold_length].mean().item()))
        return results


def prompt_with_demonstrations(row: Dict[str, Any], demonstrations: Sequence[Dict[str, Any]]) -> str:
    """复用统一评测模板，把指定 demonstrations 插入训练 query prompt。"""

    return build_icl_prompt(str(row["dataset"]), row["prompt"], demonstrations)


def candidate_by_id(row: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """为一条训练记录建立 demonstration ID 到完整样本的索引。"""

    return {candidate["id"]: candidate for candidate in row["candidates"]}


def dpp_logdet(
    query_embedding: torch.Tensor,
    demonstration_embeddings: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """计算 query 条件下一个 demonstration set 的 DPP log-determinant。"""

    relevance = demonstration_embeddings @ query_embedding
    quality = torch.exp(torch.clamp(relevance / temperature, min=-20.0, max=20.0))
    similarity = demonstration_embeddings @ demonstration_embeddings.T
    kernel = quality[:, None] * similarity * quality[None, :]
    kernel = (kernel + kernel.T) / 2
    kernel = kernel + torch.eye(len(kernel), device=kernel.device, dtype=kernel.dtype) * 1e-5
    return torch.logdet(kernel)


def official_dpp_logdets(
    query_embedding: torch.Tensor,
    context_embeddings: torch.Tensor,
    set_indices: torch.Tensor,
    scale_factor: float,
) -> torch.Tensor:
    """官方 CEIL conditional-DPP kernel 上各候选集合的 log|det|。"""

    relevance = (context_embeddings @ query_embedding + 1.0) / 2.0
    relevance = relevance - relevance.max().detach()
    quality = torch.exp(relevance / (2.0 * scale_factor))
    similarity = (context_embeddings @ context_embeddings.T + 1.0) / 2.0
    kernel = quality[:, None] * similarity * quality[None, :]
    submatrices = kernel[set_indices[:, :, None], set_indices[:, None, :]]
    return torch.linalg.slogdet(submatrices).logabsdet


def official_ranking_loss(scores: torch.Tensor, margin: float = 0.02) -> torch.Tensor:
    """官方全部候选集合 pairwise margin-ranking loss。输入已按教师偏好排序。"""

    maximum = scores.max(dim=-1, keepdim=True).values.detach()
    minimum = scores.min(dim=-1, keepdim=True).values.detach()
    normalized = (scores - minimum) / (maximum - minimum).clamp_min(1e-12)
    total = scores.new_zeros(())
    for distance in range(1, scores.shape[1]):
        total = total + torch.nn.functional.margin_ranking_loss(
            normalized[:, :-distance], normalized[:, distance:],
            torch.ones_like(normalized[:, :-distance]), margin=margin * distance,
        )
    return total


def split_train_validation(
    rows: Sequence[Dict[str, Any]], fraction: float, seed: int,
    stratify: bool = False,
):
    """确定性拆分；Banking77可按标签分层，其他任务保持原随机逻辑。"""

    if not 0.0 < fraction < 1.0 or len(rows) < 2:
        raise ValueError("validation fraction must be in (0, 1) and input needs >=2 rows")
    validation_size = max(1, round(len(rows) * fraction))
    if stratify:
        validation, train = _balanced_sample_without_replacement(
            rows, validation_size, random.Random(seed),
        )
        return train, validation
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(rows), generator=generator).tolist()
    validation_ids = set(order[:validation_size])
    train = [row for index, row in enumerate(rows) if index not in validation_ids]
    validation = [row for index, row in enumerate(rows) if index in validation_ids]
    return train, validation


def linear_warmup_decay(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    """与 Transformers linear scheduler 等价的 warmup + linear decay。"""

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
