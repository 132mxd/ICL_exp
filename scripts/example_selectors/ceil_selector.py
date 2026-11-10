"""CEIL 条件 DPP 示例选择器，供统一 ICL 评测入口调用。"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .bm25_selector import input_to_text


class CEILEncoder:
    """加载 CEIL 训练得到的 query/demonstration 共享编码器。"""

    def __init__(self, checkpoint_dir: Path, device: str = "auto", batch_size: int = 32) -> None:
        if batch_size <= 0:
            raise ValueError("CEIL batch size must be greater than zero")
        checkpoint_dir = Path(checkpoint_dir)
        metadata_path = checkpoint_dir / "metadata.json"
        encoder_dir = checkpoint_dir / "encoder"
        if not metadata_path.is_file() or not encoder_dir.is_dir():
            raise FileNotFoundError(
                f"Invalid CEIL checkpoint {checkpoint_dir}; expected metadata.json and encoder/"
            )

        # 延迟导入并加载模型，避免其他选择方法无故占用 GPU 和 Transformer 依赖。
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.checkpoint_dir = checkpoint_dir
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        # 兼容旧 checkpoint：早期版本把训练候选集合大小命名为 num_shots，
        # 但该值不应约束推理时的 demonstration 数量。
        if "training_set_size" not in self.metadata and "num_shots" in self.metadata:
            self.metadata["training_set_size"] = int(self.metadata["num_shots"])
        self.batch_size = batch_size
        self.max_length = int(self.metadata.get("max_length", 128))
        self.temperature = float(self.metadata.get("temperature", 0.1))
        self.scale_factor = float(self.metadata.get("scale_factor", self.temperature))
        self.official = self.metadata.get("implementation") == "official-ceil-qwen-adapter-v1"
        self.candidate_pool_size = int(self.metadata.get("candidate_pool_size", 50))
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_dir)
        self.model = AutoModel.from_pretrained(encoder_dir)
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, requested: str) -> str:
        if requested == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not self.torch.cuda.is_available():
            raise RuntimeError(f"CEIL device {requested!r} requested, but CUDA is unavailable")
        return requested

    @property
    def cache_key(self) -> str:
        return json.dumps(
            {
                "checkpoint": str(self.checkpoint_dir.resolve()),
                "max_length": self.max_length,
                "temperature": self.temperature,
                "scale_factor": self.scale_factor,
                "implementation": self.metadata.get("implementation", "legacy-simplified-v1"),
            },
            sort_keys=True,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """分批编码文本，执行带 attention mask 的 mean pooling 和 L2 归一化。"""

        if not texts:
            hidden_size = int(getattr(self.model.config, "hidden_size", 0))
            return np.empty((0, hidden_size), dtype=np.float32)
        batches = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                encoded = self.tokenizer(
                    list(texts[start : start + self.batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                mask = encoded["attention_mask"].unsqueeze(-1).to(output.last_hidden_state.dtype)
                pooled = (output.last_hidden_state * mask).sum(dim=1)
                pooled = pooled / mask.sum(dim=1).clamp(min=1e-9)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                batches.append(pooled.float().cpu().numpy())
        return np.concatenate(batches, axis=0)


def greedy_dpp_map(kernel: np.ndarray, size: int) -> List[int]:
    """通过逐步比较精确 log-determinant，贪心求解固定大小的 DPP MAP。"""

    if size <= 0:
        raise ValueError("DPP selection size must be greater than zero")
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"DPP kernel must be square, got {kernel.shape}")
    if size > len(kernel):
        raise ValueError(f"Cannot select {size} items from a kernel of size {len(kernel)}")

    selected: List[int] = []
    remaining = list(range(len(kernel)))
    for _ in range(size):
        best_index = None
        best_value = -np.inf
        for candidate in remaining:
            indices = selected + [candidate]
            sign, logdet = np.linalg.slogdet(kernel[np.ix_(indices, indices)])
            value = float(logdet) if sign > 0 else -np.inf
            if value > best_value:
                best_value = value
                best_index = candidate
        # 完全重复的向量可能导致子矩阵退化；此时按对角线质量确定性回退。
        if best_index is None:
            best_index = max(remaining, key=lambda index: (kernel[index, index], -index))
        selected.append(best_index)
        remaining.remove(best_index)
    return selected


class CEILSelector:
    """使用条件 DPP 选择与 query 相关且彼此不冗余的演示示例。"""

    def __init__(
        self,
        candidates: Sequence[Dict[str, Any]],
        encoder: Any,
        candidate_pool_size: Optional[int] = None,
    ) -> None:
        if not candidates:
            raise ValueError("CEIL requires a non-empty demonstration pool")
        self.candidates = list(candidates)
        self.encoder = encoder
        configured_pool = int(getattr(encoder, "candidate_pool_size", 50))
        self.candidate_pool_size = candidate_pool_size or configured_pool
        if self.candidate_pool_size <= 0:
            raise ValueError("CEIL candidate pool size must be greater than zero")
        self.temperature = max(float(getattr(encoder, "temperature", 0.1)), 1e-6)
        self.scale_factor = max(float(getattr(encoder, "scale_factor", self.temperature)), 1e-6)
        self.official = bool(getattr(encoder, "official", False))
        # demonstration pool 在所有测试 query 之间共享，因此只编码一次。
        texts = [input_to_text(candidate["input"]) for candidate in self.candidates]
        self.corpus_embeddings = self._normalize(encoder.encode(texts))

    @staticmethod
    def _normalize(embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"CEIL encoder returned invalid shape: {embeddings.shape}")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)

    def select_many(
        self, query_inputs: Sequence[Dict[str, Any]], num_shots: int
    ) -> List[List[Dict[str, Any]]]:
        """为一批测试 query 预计算固定数量的 CEIL 演示示例。"""

        if num_shots <= 0:
            raise ValueError("num_shots must be greater than zero for CEIL selection")
        if num_shots > len(self.candidates):
            raise ValueError(
                f"Requested {num_shots} demonstrations, but only {len(self.candidates)} candidates are available"
            )
        query_texts = [input_to_text(inputs) for inputs in query_inputs]
        query_embeddings = self._normalize(self.encoder.encode(query_texts))
        selections = []
        pool_size = min(max(num_shots, self.candidate_pool_size), len(self.candidates))
        for query_embedding in query_embeddings:
            # 先按 query–candidate 相关性缩小候选池，避免在整个训练集上构造 DPP。
            relevance = self.corpus_embeddings @ query_embedding
            ranked = np.argsort(-relevance, kind="stable")[:pool_size]
            pool_embeddings = self.corpus_embeddings[ranked]
            # L = diag(q) S diag(q)：q 表示 query 条件质量，S 是候选向量 Gram 矩阵。
            if self.official:
                official_relevance = (relevance[ranked] + 1.0) / 2.0
                official_relevance -= official_relevance.max()
                quality = np.exp(official_relevance / (2.0 * self.scale_factor))
                similarity = (pool_embeddings @ pool_embeddings.T + 1.0) / 2.0
            else:
                # 仅为已存在的旧 checkpoint 保留，新的训练一律走官方分支。
                quality = np.exp(np.clip(relevance[ranked] / self.temperature, -20.0, 20.0))
                similarity = pool_embeddings @ pool_embeddings.T
            kernel = quality[:, None] * similarity * quality[None, :]
            kernel = (kernel + kernel.T) / 2.0
            kernel += np.eye(pool_size, dtype=np.float32) * 1e-6
            chosen_local = greedy_dpp_map(kernel, num_shots)
            # 官方 dense_retriever 对 top-k 候选下标排序；候选池本身按相关性降序。
            if self.official:
                chosen_local = sorted(chosen_local)
            selected = []
            for prompt_position, local_index in enumerate(chosen_local, start=1):
                candidate_index = int(ranked[local_index])
                candidate = dict(self.candidates[candidate_index])
                # 保存选择分数、候选池排名和 prompt 位置，便于结果审计和复现。
                candidate["selection_score"] = round(float(relevance[candidate_index]), 8)
                candidate["selection_rank"] = int(np.flatnonzero(ranked == candidate_index)[0] + 1)
                candidate["prompt_position"] = prompt_position
                selected.append(candidate)
            selections.append(selected)
        return selections
