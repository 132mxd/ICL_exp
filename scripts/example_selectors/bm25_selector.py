"""基于 BM25 词项相关性的 ICL 示例选择器，不依赖第三方检索库。"""

import heapq
import math
import re
from collections import Counter, defaultdict
from typing import Any, DefaultDict, Dict, List, Sequence, Tuple


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", flags=re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    """使用适合当前英文数据集的轻量分词：小写化并保留单词、数字和缩写。"""

    return TOKEN_PATTERN.findall(text.casefold())


def input_to_text(inputs: Dict[str, Any]) -> str:
    """把五种数据集的统一 input 字段转换成 BM25 可检索文本。"""

    def flatten(value: Any) -> List[str]:
        if isinstance(value, dict):
            return [text for item in value.values() for text in flatten(item)]
        if isinstance(value, (list, tuple)):
            return [text for item in value for text in flatten(item)]
        return [str(value)] if isinstance(value, (str, int, float)) else []

    # 先放任务的核心输入，再收集PAWS句对和CMSQA选项等嵌套文本。
    ordered_keys = ("question", "sentence1", "sentence2", "text", "passage", "choices")
    values = [text for key in ordered_keys if key in inputs for text in flatten(inputs[key])]
    values.extend(text for key, value in inputs.items() if key not in ordered_keys for text in flatten(value))
    return " ".join(values)


class BM25Selector:
    """为训练示例建立一次倒排索引，并为每条测试输入选择 top-k 示例。"""

    def __init__(
        self,
        candidates: Sequence[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not candidates:
            raise ValueError("BM25 requires a non-empty demonstration pool")
        if k1 <= 0:
            raise ValueError("BM25 k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("BM25 b must be between zero and one")

        self.candidates = list(candidates)
        self.k1 = k1
        self.b = b
        self.document_lengths: List[int] = []
        self.postings: DefaultDict[str, List[Tuple[int, int]]] = defaultdict(list)
        document_frequencies: Counter[str] = Counter()

        # 倒排表只索引 input，不索引 gold，防止利用答案标签进行示例检索。
        for document_index, candidate in enumerate(self.candidates):
            term_frequencies = Counter(tokenize(input_to_text(candidate["input"])))
            self.document_lengths.append(sum(term_frequencies.values()))
            document_frequencies.update(term_frequencies.keys())
            for term, frequency in term_frequencies.items():
                self.postings[term].append((document_index, frequency))

        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
        )
        document_count = len(self.candidates)
        # Robertson/Sparck Jones IDF 的正值变体，常用于 BM25Okapi 实现。
        self.idf = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def select(self, query_input: Dict[str, Any], num_shots: int) -> List[Dict[str, Any]]:
        """按 BM25 分数降序返回 top-k；同分时按训练集原始顺序稳定排序。"""

        if num_shots <= 0:
            raise ValueError("num_shots must be greater than zero for BM25 selection")
        if num_shots > len(self.candidates):
            raise ValueError(
                f"Requested {num_shots} demonstrations, but only "
                f"{len(self.candidates)} candidates are available"
            )

        query_frequencies = Counter(tokenize(input_to_text(query_input)))
        scores: Dict[int, float] = {}
        average_length = self.average_document_length or 1.0
        for term, query_frequency in query_frequencies.items():
            if term not in self.idf:
                continue
            for document_index, term_frequency in self.postings[term]:
                length_ratio = self.document_lengths[document_index] / average_length
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * length_ratio
                )
                contribution = (
                    query_frequency
                    * self.idf[term]
                    * term_frequency
                    * (self.k1 + 1)
                    / denominator
                )
                scores[document_index] = scores.get(document_index, 0.0) + contribution

        # 只在有非零分数的文档中维护 top-k，避免每次查询完整排序训练集。
        ranked_indices = heapq.nsmallest(
            num_shots,
            scores,
            key=lambda index: (-scores[index], index),
        )
        if len(ranked_indices) < num_shots:
            selected = set(ranked_indices)
            for document_index in range(len(self.candidates)):
                if document_index not in selected:
                    ranked_indices.append(document_index)
                    selected.add(document_index)
                    if len(ranked_indices) == num_shots:
                        break

        results = []
        for document_index in ranked_indices:
            candidate = dict(self.candidates[document_index])
            candidate["selection_score"] = round(scores.get(document_index, 0.0), 8)
            results.append(candidate)
        return results
