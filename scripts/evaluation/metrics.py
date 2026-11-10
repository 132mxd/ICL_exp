"""答案清理与解析：分类标签匹配和精确匹配评分。"""

import re
from typing import Optional, Sequence


def normalize_output(text: str) -> str:
    """清理模型输出两侧的包装字符、字段名前缀和常见标点。"""

    text = text.strip().strip("`\"'")
    text = re.sub(r"^(?:label|answer|intent(?: label)?)\s*:\s*", "", text, flags=re.I)
    return text.strip().rstrip(".!,;:").strip()


def parse_prediction(text: str, labels: Sequence[str]) -> Optional[str]:
    """将模型文本解析为唯一标准标签；含多个候选标签时判为无效输出。"""

    cleaned = normalize_output(text)
    for label in labels:
        if cleaned.casefold() == label.casefold():
            return label

    matches = []
    for label in labels:
        pattern = rf"(?<![\w]){re.escape(label)}(?![\w])"
        matches.extend(
            (match.start(), match.end(), label)
            for match in re.finditer(pattern, cleaned, flags=re.I)
        )

    # 当标签文本重叠时保留更长的标签，例如 SST-5 中的 ``very negative``
    # 包含 ``negative``，此时不能把一次输出误判为两个标签。
    maximal_matches = [
        match
        for match in matches
        if not any(
            other[0] <= match[0]
            and match[1] <= other[1]
            and (other[1] - other[0]) > (match[1] - match[0])
            for other in matches
        )
    ]

    unique_matches = list(dict.fromkeys(match[2] for match in maximal_matches))
    return unique_matches[0] if len(unique_matches) == 1 else None


def exact_match(prediction: Optional[str], gold: str) -> bool:
    """仅有效且等于标准答案的预测计为正确；请求失败或无效输出计为错误。"""

    return prediction is not None and prediction == gold
