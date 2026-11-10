"""可复现的随机 ICL 示例选择方法。"""

import hashlib
import random
from typing import Any, Dict, List, Sequence


def select_random_examples(
    candidates: Sequence[Dict[str, Any]],
    num_shots: int,
    seed: int,
    dataset_name: str,
    sample_index: int,
) -> List[Dict[str, Any]]:
    """从训练池无放回抽取示例，同一配置下每次运行结果完全一致。"""

    if num_shots <= 0:
        raise ValueError("num_shots must be greater than zero for random selection")
    if num_shots > len(candidates):
        raise ValueError(
            f"Requested {num_shots} demonstrations, but {dataset_name} only has "
            f"{len(candidates)} candidates"
        )

    # 为每个测试样本派生独立随机种子，避免模型顺序或部分重跑改变所选示例。
    seed_text = f"{seed}\0{dataset_name}\0{sample_index}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:8], "big")
    generator = random.Random(derived_seed)
    return generator.sample(list(candidates), num_shots)
