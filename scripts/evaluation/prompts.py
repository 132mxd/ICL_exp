"""集中构造 zero-shot 提示词、ICL 演示示例及最终 few-shot 提示词。"""

from typing import Any, Dict, Sequence

from .config import BOOLQ_LABELS, CMSQA_LABELS, PAWS_LABELS, SST5_LABELS

def labels_block(labels: Sequence[str]) -> str:
    """将候选标签转换成适合插入提示词的逐行文本。"""

    return "\n".join(labels)


# 以下函数为各数据集构造 zero-shot 提示词。提示词只描述任务和输出格式，
# 不加入任何带答案的演示样本，因此不会引入 few-shot/ICL 信息。
def build_sst5_prompt(text: str) -> str:
    return f"""Classify the sentiment of the following movie-review sentence.

You must output exactly one label from this list:
{labels_block(SST5_LABELS)}

Do not provide an explanation.

Sentence:
{text}

Label:"""


def build_banking77_prompt(text: str, labels: Sequence[str]) -> str:
    return f"""Classify the intent of the following banking customer request.

You must output exactly one intent label from this list:
{labels_block(labels)}

Do not provide an explanation and do not change the label spelling.

Customer request:
{text}

Intent label:"""


def build_paws_prompt(sentence1: str, sentence2: str) -> str:
    return f"""Determine whether the following two sentences have the same meaning.

You must output exactly one label from this list:
{labels_block(PAWS_LABELS)}

Use "paraphrase" when they have the same meaning and "not paraphrase" otherwise.
Do not provide an explanation.

Sentence 1:
{sentence1}

Sentence 2:
{sentence2}

Label:"""


def build_boolq_prompt(passage: str, question: str) -> str:
    return f"""Read the passage and answer the question with yes or no.

Passage:
{passage}

Question:
{question}

You must output exactly one label:
yes
no

Do not provide an explanation.

Answer:"""


def format_cmsqa_choices(choices: Sequence[Dict[str, str]]) -> str:
    """把 CommonsenseQA 的五个选项格式化为稳定的 A-E 列表。"""

    return "\n".join(f"{choice['label']}. {choice['text']}" for choice in choices)


def build_cmsqa_prompt(question: str, choices: Sequence[Dict[str, str]]) -> str:
    return f"""Choose the best answer to the following commonsense question.

You must output exactly one option label: {', '.join(CMSQA_LABELS)}.
Do not output the answer text and do not provide an explanation.

Question:
{question}

Options:
{format_cmsqa_choices(choices)}

Answer:"""


def format_demonstration(dataset_name: str, example: Dict[str, Any], number: int) -> str:
    """把一条训练样本转换为对应任务的带答案演示示例。"""

    inputs = example["input"]
    gold = example["gold"]
    if dataset_name == "sst5":
        body = f"Sentence:\n{inputs['text']}\nLabel: {gold}"
    elif dataset_name == "paws":
        body = (
            f"Sentence 1:\n{inputs['sentence1']}\n\n"
            f"Sentence 2:\n{inputs['sentence2']}\nLabel: {gold}"
        )
    elif dataset_name == "banking77":
        body = f"Customer request:\n{inputs['text']}\nIntent label: {gold}"
    elif dataset_name == "boolq":
        body = (
            f"Passage:\n{inputs['passage']}\n\n"
            f"Question:\n{inputs['question']}\nAnswer: {gold}"
        )
    elif dataset_name == "cmsqa":
        body = (
            f"Question:\n{inputs['question']}\n\nOptions:\n"
            f"{format_cmsqa_choices(inputs['choices'])}\nAnswer: {gold}"
        )
    else:
        raise ValueError(f"Unsupported dataset for demonstrations: {dataset_name}")
    return f"Example {number}:\n{body}"


def build_icl_prompt(
    dataset_name: str,
    zero_shot_prompt: str,
    demonstrations: Sequence[Dict[str, Any]],
) -> str:
    """在原任务说明与待预测问题之间加入带答案的演示样本。"""

    if not demonstrations:
        return zero_shot_prompt

    # 各 zero-shot 模板最后一个空行之后是待预测问题；将示例插入该位置，
    # 可以复用原任务说明，并保持最后一段始终是模型真正需要回答的问题。
    marker_by_dataset = {
        "sst5": "\nSentence:\n",
        "paws": "\nSentence 1:\n",
        "banking77": "\nCustomer request:\n",
        "boolq": "\nPassage:\n",
        "cmsqa": "\nQuestion:\n",
    }
    marker = marker_by_dataset[dataset_name]
    split_at = zero_shot_prompt.rfind(marker)
    if split_at < 0:
        raise ValueError(f"Could not locate query section for {dataset_name}")

    examples = "\n\n".join(
        format_demonstration(dataset_name, example, index)
        for index, example in enumerate(demonstrations, start=1)
    )
    return (
        zero_shot_prompt[:split_at]
        + "\n\nHere are some labeled examples:\n\n"
        + examples
        + "\n\nNow answer the following item."
        + zero_shot_prompt[split_at:]
    )
