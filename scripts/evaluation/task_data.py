"""定义各数据集共用的任务结构，供数据加载器和评测循环交换数据。"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

@dataclass
class TaskData:
    """统一保存评测样本及可供 ICL 选择的训练示例池。"""

    name: str
    labels: Sequence[str]
    samples: List[Dict[str, Any]]
    answer_type: str = "label"
    # zero-shot 不加载该字段；random 等 ICL 方法只从这里选择演示样本。
    demonstration_pool: List[Dict[str, Any]] = field(default_factory=list)
    # 稠密/学习型选择器统一预计算，确保不同 LLM 使用完全相同的示例和顺序。
    precomputed_selections: List[List[Dict[str, Any]]] = field(default_factory=list)
    # 记录候选池去重与测试集污染过滤，写入实验汇总以便审计和复现。
    demonstration_filter_stats: Dict[str, int] = field(default_factory=dict)
