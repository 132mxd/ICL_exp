"""KATE：使用稠密句向量语义相似度选择 ICL 演示示例。"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .bm25_selector import input_to_text


DEFAULT_KATE_ENCODER = (
    "sentence-transformers/roberta-large-nli-stsb-mean-tokens"
)


class TransformerMeanEncoder:
    """用 Hugging Face Transformer 和 mean pooling 复现原始 KATE encoder。"""

    def __init__(
        self,
        model_name: str = DEFAULT_KATE_ENCODER,
        revision: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 128,
        model_cache_dir: Optional[Path] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("KATE batch size must be greater than zero")
        if max_length <= 0:
            raise ValueError("KATE max length must be greater than zero")

        # 延迟导入，zero-shot、random 和 BM25 不需要加载 Transformer 依赖。
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.max_length = max_length
        cache_dir = str(model_cache_dir) if model_cache_dir is not None else None
        load_kwargs = {"cache_dir": cache_dir}
        if revision is not None:
            load_kwargs["revision"] = revision

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
        self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
        self.device = self._resolve_device(device)
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, requested: str) -> str:
        if requested == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not self.torch.cuda.is_available():
            raise RuntimeError(f"KATE device {requested!r} requested, but CUDA is unavailable")
        return requested

    @property
    def cache_key(self) -> str:
        """返回会影响 embedding 的完整配置，用于生成安全缓存键。"""

        return json.dumps(
            {
                "model_name": self.model_name,
                "revision": self.revision,
                "pooling": "attention_mask_mean",
                "max_length": self.max_length,
                "normalized": True,
            },
            sort_keys=True,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """批量生成 L2-normalized float32 句向量。"""

        if not texts:
            hidden_size = int(getattr(self.model.config, "hidden_size", 0))
            return np.empty((0, hidden_size), dtype=np.float32)

        batches = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                output = self.model(**encoded)
                token_embeddings = output.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1).to(
                    token_embeddings.dtype
                )
                pooled = (token_embeddings * attention_mask).sum(dim=1)
                pooled = pooled / attention_mask.sum(dim=1).clamp(min=1e-9)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                batches.append(pooled.cpu().to(self.torch.float32).numpy())
        return np.concatenate(batches, axis=0)


class KATESelector:
    """缓存训练池 embedding，并按 cosine similarity 返回 top-k 示例。"""

    def __init__(
        self,
        candidates: Sequence[Dict[str, Any]],
        encoder: Any,
        dataset_name: str,
        embedding_cache_dir: Optional[Path] = None,
    ) -> None:
        if not candidates:
            raise ValueError("KATE requires a non-empty demonstration pool")
        self.candidates = list(candidates)
        self.encoder = encoder
        self.dataset_name = dataset_name
        self.corpus_texts = [input_to_text(item["input"]) for item in self.candidates]
        self.fingerprint = self._fingerprint()
        self.cache_path = self._cache_path(embedding_cache_dir)
        self.corpus_embeddings = self._load_or_encode_corpus()

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(str(self.encoder.cache_key).encode("utf-8"))
        for index, (candidate, text) in enumerate(zip(self.candidates, self.corpus_texts)):
            digest.update(b"\0")
            digest.update(str(candidate.get("id", index)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(text.encode("utf-8"))
        return digest.hexdigest()

    def _cache_path(self, cache_dir: Optional[Path]) -> Optional[Path]:
        if cache_dir is None:
            return None
        safe_dataset = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in self.dataset_name
        )
        return Path(cache_dir) / f"{safe_dataset}_{self.fingerprint[:20]}.npz"

    def _load_or_encode_corpus(self) -> np.ndarray:
        if self.cache_path is not None and self.cache_path.is_file():
            with np.load(self.cache_path, allow_pickle=False) as cache:
                embeddings = cache["embeddings"].astype(np.float32, copy=False)
                fingerprint = str(cache["fingerprint"].item())
            if fingerprint == self.fingerprint and len(embeddings) == len(self.candidates):
                return embeddings

        embeddings = np.asarray(self.encoder.encode(self.corpus_texts), dtype=np.float32)
        if embeddings.ndim != 2 or len(embeddings) != len(self.candidates):
            raise ValueError(
                "KATE encoder returned invalid corpus embedding shape: "
                f"{embeddings.shape}"
            )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)

        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary_path,
                embeddings=embeddings,
                fingerprint=np.asarray(self.fingerprint),
            )
            temporary_path.replace(self.cache_path)
        return embeddings

    def select_many(
        self,
        query_inputs: Sequence[Dict[str, Any]],
        num_shots: int,
        order: str = "most_similar_first",
        query_batch_size: int = 256,
    ) -> List[List[Dict[str, Any]]]:
        """为全部测试输入预计算示例，使不同 LLM 使用完全相同的检索结果。"""

        if num_shots <= 0:
            raise ValueError("num_shots must be greater than zero for KATE selection")
        if num_shots > len(self.candidates):
            raise ValueError(
                f"Requested {num_shots} demonstrations, but only "
                f"{len(self.candidates)} candidates are available"
            )
        if order not in ("most_similar_first", "most_similar_last"):
            raise ValueError(f"Unsupported KATE order: {order}")
        if query_batch_size <= 0:
            raise ValueError("KATE query batch size must be greater than zero")

        query_texts = [input_to_text(inputs) for inputs in query_inputs]
        query_embeddings = np.asarray(self.encoder.encode(query_texts), dtype=np.float32)
        if query_embeddings.ndim != 2 or len(query_embeddings) != len(query_inputs):
            raise ValueError(
                "KATE encoder returned invalid query embedding shape: "
                f"{query_embeddings.shape}"
            )
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        query_embeddings = query_embeddings / np.maximum(norms, 1e-12)

        selections: List[List[Dict[str, Any]]] = []
        for start in range(0, len(query_embeddings), query_batch_size):
            similarities = query_embeddings[start : start + query_batch_size] @ (
                self.corpus_embeddings.T
            )
            # stable sort 让完全相同的分数按训练池原始顺序确定性打破平局。
            ranked_rows = np.argsort(-similarities, axis=1, kind="stable")[:, :num_shots]
            for row_offset, ranked_indices in enumerate(ranked_rows):
                if order == "most_similar_last":
                    ranked_indices = ranked_indices[::-1]
                row_scores = similarities[row_offset]
                selected = []
                for prompt_position, candidate_index in enumerate(ranked_indices, start=1):
                    candidate = dict(self.candidates[int(candidate_index)])
                    candidate["selection_score"] = round(
                        float(row_scores[int(candidate_index)]), 8
                    )
                    candidate["selection_rank"] = int(
                        np.flatnonzero(ranked_rows[row_offset] == candidate_index)[0] + 1
                    )
                    candidate["prompt_position"] = prompt_position
                    selected.append(candidate)
                selections.append(selected)
        return selections
