#!/usr/bin/env python3
"""统一评测入口：例如 python scripts/evaluate.py --models qwen --datasets sst5。

具体实现位于 evaluation 包；支持 zero-shot、随机、BM25、KATE 和 CEIL 评测。
"""

# 同时支持直接运行脚本和 python -m scripts.evaluate 两种启动方式。
if __package__:
    from .evaluation.cli import main
else:
    from evaluation.cli import main


if __name__ == "__main__":
    main()
