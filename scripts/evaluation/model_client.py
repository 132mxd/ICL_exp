"""公共 vLLM 客户端：查询模型 ID、发送请求并在失败时重试。"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

import requests


# 本地 vLLM 流量必须留在服务器回环接口，不能继承 HTTP_PROXY/ALL_PROXY。
# 否则 127.0.0.1 请求可能被送进 SSH RemoteForward，既显著变慢，也会占用
# 用户本地代理和服务器外部网络。远程模型地址仍使用 requests 的默认代理行为。
_local_session = requests.Session()
_local_session.trust_env = False


def _http_client(base_url: str) -> Any:
    """本地地址使用无代理 Session，远程地址保留环境代理设置。"""

    hostname = urlparse(base_url).hostname
    if hostname in {"127.0.0.1", "localhost", "::1"}:
        return _local_session
    return requests

def server_model_ids(base_url: str, timeout: int) -> List[str]:
    """查询 vLLM 服务当前实际提供的模型 ID。"""

    response = _http_client(base_url).get(
        f"{base_url.rstrip('/')}/v1/models", timeout=min(timeout, 10)
    )
    response.raise_for_status()
    return [item["id"] for item in response.json().get("data", [])]


def resolve_served_model_id(configured_id: str, served_ids: Sequence[str]) -> str:
    """将脚本配置的本地模型路径与 vLLM 返回的模型 ID 对齐。"""

    if configured_id in served_ids:
        return configured_id

    configured_name = Path(configured_id).name.casefold()
    name_matches = [
        model_id for model_id in served_ids if Path(model_id).name.casefold() == configured_name
    ]
    if len(name_matches) == 1:
        return name_matches[0]
    if len(served_ids) == 1:
        print(
            f"WARNING: configured model ID {configured_id!r} differs from the "
            f"server ID {served_ids[0]!r}; using the server ID."
        )
        return served_ids[0]
    raise RuntimeError(
        f"Could not match configured model {configured_id!r} to server models: "
        f"{list(served_ids)!r}"
    )


def query_vllm(
    prompt: str,
    labels: Sequence[str],
    base_url: str,
    model_id: str,
    timeout: int,
    retries: int,
    structured_output: bool,
    max_tokens: int = 32,
) -> str:
    """向 vLLM 的 Chat Completions 接口发送一次聊天推理请求。"""

    # temperature=0 用于减少采样随机性，使重复实验更稳定。
    # 当前五个任务均输出固定短标签，32 token 足以容纳响应和少量模型包装文本。
    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    # 分类任务使用 vLLM 的 choice 约束输出必须来自标签集合。
    if structured_output and labels:
        payload["structured_outputs"] = {"choice": list(labels)}

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    last_error: Optional[Exception] = None
    # 网络请求失败时进行指数退避重试，避免一次瞬时错误中断整组实验。
    for attempt in range(retries + 1):
        try:
            response = _http_client(base_url).post(
                url, json=payload, timeout=timeout
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError, TypeError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"vLLM request failed after {retries + 1} attempt(s): {last_error}")
