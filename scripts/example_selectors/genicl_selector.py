"""GENICL 生成式偏好演示样本选择器。

本模块实现 Zhang 等人（ACL 2025）提出的推理流程：

1. 使用 E5-base 将演示池缩小为与查询相关的候选子集；
2. 使用经过 KTO/LoRA 适配的因果语言模型，按照论文公式（11）的潜变量
   条件对数概率为剩余候选打分；
3. 按分数降序返回前 K 个候选。

适配器训练与在线评测流程保持分离。检查点目录必须包含 PEFT 适配器和
``metadata.json``。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .bm25_selector import input_to_text


DEFAULT_GENICL_E5_MODEL = "intfloat/e5-base-v2"


def latent_description(dataset_name: str, n_prefix_tokens: int = 10) -> str:
    """返回官方代码使用的任务特定潜变量描述。"""

    if n_prefix_tokens <= 0:
        raise ValueError("GENICL n_prefix_tokens must be greater than zero")
    return "".join(f"<{dataset_name}-{index}>" for index in range(n_prefix_tokens))


def candidate_text(candidate: Dict[str, Any]) -> str:
    """序列化带标签的演示样本，不引入其编号或元数据。"""

    return f"{input_to_text(candidate['input'])}\n{candidate['gold']}"


class GenICLE5Encoder:
    """与 GENICL 第一阶段稠密检索一致的 E5-base 编码器。"""

    def __init__(
        self,
        model_name: str = DEFAULT_GENICL_E5_MODEL,
        revision: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 128,
        max_length: int = 256,
        model_cache_dir: Optional[Path] = None,
    ) -> None:
        if batch_size <= 0 or max_length <= 0:
            raise ValueError("GENICL E5 batch size and max length must be positive")

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.max_length = max_length
        kwargs: Dict[str, Any] = {
            "cache_dir": str(model_cache_dir) if model_cache_dir else None
        }
        if revision is not None:
            kwargs["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
        self.model = AutoModel.from_pretrained(model_name, **kwargs)
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, requested: str) -> str:
        if requested == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not self.torch.cuda.is_available():
            raise RuntimeError(f"GENICL E5 device {requested!r} requested without CUDA")
        return requested

    @property
    def cache_key(self) -> str:
        return json.dumps(
            {
                "model": self.model_name,
                "revision": self.revision,
                "prefix": "query: ",
                "pooling": "attention_mask_mean",
                "max_length": self.max_length,
                "normalized": True,
            },
            sort_keys=True,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """严格按照官方 E5 路径编码文本：添加查询前缀并执行均值池化。"""

        if not texts:
            hidden = int(getattr(self.model.config, "hidden_size", 0))
            return np.empty((0, hidden), dtype=np.float32)
        batches = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = ["query: " + text for text in texts[start : start + self.batch_size]]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                mask = encoded["attention_mask"].unsqueeze(-1).to(output.last_hidden_state.dtype)
                pooled = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=-1)
                batches.append(pooled.float().cpu().numpy())
        return np.concatenate(batches, axis=0)


class GenICLScorer:
    """使用适配后语言模型生成潜变量 z 的概率为候选与查询对打分。"""

    def __init__(
        self,
        checkpoint_dir: Path,
        device: str = "auto",
        batch_size: int = 8,
        max_input_length: int = 1024,
        model_cache_dir: Optional[Path] = None,
    ) -> None:
        if batch_size <= 0 or max_input_length <= 0:
            raise ValueError("GENICL scorer batch size and max length must be positive")
        checkpoint_dir = Path(checkpoint_dir)
        metadata_path = checkpoint_dir / "metadata.json"
        if not metadata_path.is_file() or not (checkpoint_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"Invalid GENICL checkpoint {checkpoint_dir}; expected metadata.json "
                "and adapter_config.json"
            )
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("algorithm") != "genicl-kto-lora-alternating-v1":
            raise ValueError(
                "Checkpoint is not a paper-faithful alternating GENICL checkpoint; "
                "expected algorithm=genicl-kto-lora-alternating-v1"
            )

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.device = self._resolve_device(device)
        base_model = self.metadata["base_model"]
        cache_dir = str(model_cache_dir) if model_cache_dir else None
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=cache_dir)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base_model, cache_dir=cache_dir, torch_dtype=dtype)
        self.model = PeftModel.from_pretrained(model, checkpoint_dir)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, requested: str) -> str:
        if requested == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not self.torch.cuda.is_available():
            raise RuntimeError(f"GENICL scorer device {requested!r} requested without CUDA")
        return requested

    def score(self, prompts: Sequence[str], completions: Sequence[str]) -> np.ndarray:
        """按照官方 KTO 代码返回补全部分各词元的对数概率之和。"""

        if len(prompts) != len(completions):
            raise ValueError("GENICL prompts and completions must have equal length")
        scores: List[float] = []
        with self.torch.inference_mode():
            for start in range(0, len(prompts), self.batch_size):
                batch_prompts = list(prompts[start : start + self.batch_size])
                batch_completions = list(completions[start : start + self.batch_size])
                prompt_ids = [self.tokenizer(text, add_special_tokens=True).input_ids for text in batch_prompts]
                completion_ids = [
                    self.tokenizer(text, add_special_tokens=False).input_ids
                    for text in batch_completions
                ]
                # TRL 仅计算补全部分的 KTO 整理器会分别对提示和补全进行分词，
                # 然后再将二者拼接。这里复现相同边界，而不是对拼接后的文本重新分词。
                full_ids = [
                    prompt + completion
                    for prompt, completion in zip(prompt_ids, completion_ids)
                ]
                # 候选过长时从左侧截断，确保保留查询和潜变量补全部分。
                trimmed = [ids[-self.max_input_length :] for ids in full_ids]
                prompt_lengths = [
                    max(0, len(ids) - (len(full) - len(trim)))
                    for ids, full, trim in zip(prompt_ids, full_ids, trimmed)
                ]
                max_len = max(len(ids) for ids in trimmed)
                input_ids = []
                attention_mask = []
                for ids in trimmed:
                    pad = max_len - len(ids)
                    input_ids.append(ids + [self.tokenizer.pad_token_id] * pad)
                    attention_mask.append([1] * len(ids) + [0] * pad)
                input_tensor = self.torch.tensor(input_ids, device=self.device)
                mask_tensor = self.torch.tensor(attention_mask, device=self.device)
                logits = self.model(input_ids=input_tensor, attention_mask=mask_tensor).logits
                token_logps = logits[:, :-1].log_softmax(-1)
                target_ids = input_tensor[:, 1:]
                gathered = token_logps.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
                for row, (prompt_length, ids) in enumerate(zip(prompt_lengths, trimmed)):
                    first_target = max(prompt_length - 1, 0)
                    last_target = len(ids) - 1
                    scores.append(float(gathered[row, first_target:last_target].sum().item()))
        return np.asarray(scores, dtype=np.float32)


class GenICLSelector:
    """严格的两阶段 GENICL 推理选择器：先用 E5 缩池，再按公式（11）排序。"""

    def __init__(
        self,
        candidates: Sequence[Dict[str, Any]],
        e5_encoder: Any,
        scorer: Any,
        dataset_name: str,
        candidate_pool_size: int = 100,
        n_prefix_tokens: int = 10,
        embedding_cache_dir: Optional[Path] = None,
    ) -> None:
        if not candidates:
            raise ValueError("GENICL requires a non-empty demonstration pool")
        if candidate_pool_size <= 0:
            raise ValueError("GENICL candidate pool size must be greater than zero")
        self.candidates = list(candidates)
        self.e5_encoder = e5_encoder
        self.scorer = scorer
        self.dataset_name = dataset_name
        self.candidate_pool_size = candidate_pool_size
        self.latent = latent_description(dataset_name, n_prefix_tokens)
        self.corpus_texts = [candidate_text(item) for item in self.candidates]
        self.corpus_embeddings = self._load_or_encode(embedding_cache_dir)

    def _load_or_encode(self, cache_dir: Optional[Path]) -> np.ndarray:
        digest = hashlib.sha256(str(self.e5_encoder.cache_key).encode())
        for candidate, text in zip(self.candidates, self.corpus_texts):
            digest.update(b"\0" + str(candidate.get("id", "")).encode() + b"\0" + text.encode())
        fingerprint = digest.hexdigest()
        cache_path = None
        if cache_dir is not None:
            cache_path = Path(cache_dir) / f"{self.dataset_name}_{fingerprint[:20]}.npz"
            if cache_path.is_file():
                with np.load(cache_path, allow_pickle=False) as data:
                    if str(data["fingerprint"].item()) == fingerprint:
                        return data["embeddings"].astype(np.float32, copy=False)
        embeddings = np.asarray(self.e5_encoder.encode(self.corpus_texts), dtype=np.float32)
        if embeddings.ndim != 2 or len(embeddings) != len(self.candidates):
            raise ValueError(f"GENICL E5 returned invalid shape {embeddings.shape}")
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, embeddings=embeddings, fingerprint=np.asarray(fingerprint))
            temporary.replace(cache_path)
        return embeddings

    def select_many(
        self, query_inputs: Sequence[Dict[str, Any]], num_shots: int
    ) -> List[List[Dict[str, Any]]]:
        if num_shots <= 0:
            raise ValueError("num_shots must be greater than zero for GENICL")
        if num_shots > len(self.candidates):
            raise ValueError("GENICL cannot select more demonstrations than candidates")
        query_texts = [input_to_text(item) for item in query_inputs]
        query_embeddings = np.asarray(self.e5_encoder.encode(query_texts), dtype=np.float32)
        query_embeddings /= np.maximum(np.linalg.norm(query_embeddings, axis=1, keepdims=True), 1e-12)
        pool_size = min(max(num_shots, self.candidate_pool_size), len(self.candidates))
        selections = []
        for query_text, query_embedding in zip(query_texts, query_embeddings):
            e5_scores = self.corpus_embeddings @ query_embedding
            pool = np.argsort(-e5_scores, kind="stable")[:pool_size]
            prompts = [f"{self.corpus_texts[index]}\n{query_text}" for index in pool]
            genicl_scores = self.scorer.score(prompts, [self.latent] * len(prompts))
            ranked_local = np.argsort(-genicl_scores, kind="stable")[:num_shots]
            selected = []
            for position, local_index in enumerate(ranked_local, start=1):
                candidate_index = int(pool[int(local_index)])
                item = dict(self.candidates[candidate_index])
                item["selection_score"] = round(float(genicl_scores[int(local_index)]), 8)
                item["e5_score"] = round(float(e5_scores[candidate_index]), 8)
                item["e5_rank"] = int(np.flatnonzero(pool == candidate_index)[0] + 1)
                item["prompt_position"] = position
                selected.append(item)
            selections.append(selected)
        return selections
