"""最小可运行的大模型 Function-Calling 示例（DeepSeek / OpenAI 兼容接口）。

仅依赖标准库。运行前请配置环境变量：
    Windows cmd:  set LLM_API_KEY=sk-xxx
    Linux/macOS:  export LLM_API_KEY=sk-xxx
可选：
    LLM_BASE_URL  默认 https://api.deepseek.com
    LLM_MODEL     默认 deepseek-chat

注意：本文件名 `import json,os,urllib.py` 不是合法的模块名，无法被 import。
建议重命名为 `llm_tool_demo.py`，直接 `python llm_tool_demo.py` 运行不受影响。
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_TOKENS = 512
DEFAULT_READ_LIMIT = 1000

# 配置在模块级统一收敛，URL 去掉尾部斜杠，避免拼出 "//chat/completions"
API_BASE_URL = os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
MODEL = os.environ.get("LLM_MODEL", DEFAULT_MODEL)


class LLMError(RuntimeError):
    """调用大模型接口失败（缺少密钥、网络错误、响应格式异常）时抛出。"""


def _api_key() -> str:
    """延迟读取密钥：避免 import 阶段缺少环境变量就 KeyError 崩溃。"""
    key = os.environ.get("LLM_API_KEY", "").strip()
    if not key:
        raise LLMError(
            "环境变量 LLM_API_KEY 未设置（Windows: set LLM_API_KEY=sk-xxx）"
        )
    return key


def chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """发送一次 /chat/completions 请求。

    返回 assistant 的 message 对象（dict），可能包含 `tool_calls`。
    这样调用方既能拿 content，也能拿 tool_calls，契约始终一致。
    """
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,  # 必须是复数 messages，不是 message
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    request = urllib.request.Request(
        # 注意花括号：f"{API_BASE_URL}/chat/completions"，否则会被当成除法表达式
        f"{API_BASE_URL}/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_api_key()}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # 4xx/5xx 时读响应体，否则拿不到服务端的具体报错
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise LLMError(f"HTTP {exc.code} {exc.reason}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMError(f"网络请求失败: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMError(f"请求超时（>{timeout}s）") from exc
    except json.JSONDecodeError as exc:
        raise LLMError("服务端返回的不是合法 JSON") from exc

    if "error" in payload:
        raise LLMError(f"接口返回错误: {payload['error']}")

    try:
        return payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"响应格式异常，缺少 choices[0].message: {payload}") from exc


# --------------------------------------------------------------------------- #
# 工具定义：错误提示里提到的 list_dir，必须真的注册给模型，否则模型无从调用
# --------------------------------------------------------------------------- #
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "列出目录下的文件名，读取文件前可先确认路径",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "要列出的目录路径，默认当前目录",
                        "default": ".",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取 UTF-8 文本文件的内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "要读取的文件路径",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "最多读取的字符数，<=0 表示读取全文",
                        "default": DEFAULT_READ_LIMIT,
                    },
                },
                "required": ["file_path"],
            },
        },
    },
]


def _tool_error(error: str, target: str, suggestion: str) -> str:
    """工具失败的统一返回结构（始终是 JSON 字符串）。"""
    return json.dumps(
        {"ok": False, "error": error, "target": target, "suggestion": suggestion},
        ensure_ascii=False,
    )


def _tool_ok(**payload: Any) -> str:
    """工具成功的统一返回结构（始终是 JSON 字符串）。"""
    return json.dumps({"ok": True, **payload}, ensure_ascii=False)


def read_file(file_path: str, limit: int = DEFAULT_READ_LIMIT) -> str:
    """读取文本文件，成功与失败都返回统一 JSON 字符串，不向调用方抛异常。"""
    if not isinstance(file_path, str) or not file_path.strip():
        return _tool_error(
            "Invalid file_path", repr(file_path), "请提供非空的文件路径字符串"
        )

    # limit <= 0 或类型异常时视为“读全文”，避免对 None/负数调用 read()
    max_chars = limit if isinstance(limit, int) and limit > 0 else None

    try:
        # 显式指定 UTF-8：Windows 默认 cp936，读 UTF-8 中文文件会 UnicodeDecodeError
        with open(file_path, "r", encoding="utf-8") as handle:
            # 多读 1 个字符用于精确判断是否被截断
            content = handle.read() if max_chars is None else handle.read(max_chars + 1)
    except FileNotFoundError:
        return _tool_error(
            "File not found",
            file_path,
            "先用 list_dir 列出文件列表，再用 read_file 读取文件",
        )
    except IsADirectoryError:
        return _tool_error(
            "Path is a directory", file_path, "这是目录，请改用 list_dir"
        )
    except PermissionError:
        return _tool_error(
            "Permission denied", file_path, "检查文件是否被占用或权限不足"
        )
    except UnicodeDecodeError:
        return _tool_error(
            "Not a UTF-8 text file", file_path, "疑似二进制文件，请改用其他工具处理"
        )
    except OSError as exc:
        return _tool_error(
            f"OSError: {exc.strerror or exc}", file_path, "请稍后重试或更换路径"
        )

    truncated = max_chars is not None and len(content) > max_chars
    if truncated:
        return _tool_ok(
            path=file_path,
            content=content[:max_chars],
            truncated=True,
            hint=f"内容已截断（limit={max_chars}），可增大 limit 后重读",
        )
    return _tool_ok(path=file_path, content=content, truncated=False)


def list_dir(directory: str = ".") -> str:
    """列出目录条目，返回 JSON 字符串。"""
    try:
        entries = sorted(os.listdir(directory))
    except FileNotFoundError:
        return _tool_error("Directory not found", directory, "确认目录路径是否正确")
    except NotADirectoryError:
        return _tool_error(
            "Not a directory", directory, "该路径是文件，读取请用 read_file"
        )
    except PermissionError:
        return _tool_error("Permission denied", directory, "检查目录访问权限")
    except OSError as exc:
        return _tool_error(
            f"OSError: {exc.strerror or exc}", directory, "请稍后重试或更换路径"
        )
    return _tool_ok(directory=directory, count=len(entries), entries=entries)


# 工具注册表：新增工具只改这里，主循环无需改动
TOOL_HANDLERS: dict[str, Callable[..., str]] = {
    "list_dir": list_dir,
    "read_file": read_file,
}


def dispatch_tool_call(call: dict[str, Any]) -> str:
    """执行单个 tool_call；任何异常都转成 JSON 错误串回灌模型，保证循环不中断。"""
    function = call.get("function") or {}
    name = function.get("name", "")
    handler = TOOL_HANDLERS.get(name)
    if handler is None:
        return _tool_error(
            "Unknown tool", name, f"可用工具: {sorted(TOOL_HANDLERS)}"
        )

    raw_args = function.get("arguments") or "{}"
    try:
        arguments = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _tool_error(
            "Invalid arguments", name, f"arguments 不是合法 JSON 对象: {raw_args!r}"
        )
    if not isinstance(arguments, dict):
        return _tool_error(
            "Invalid arguments", name, "arguments 必须是 JSON 对象（键值对）"
        )

    try:
        return handler(**arguments)
    except TypeError as exc:
        # 参数名/类型与函数签名不匹配，直接暴露给模型让它重试
        return _tool_error(f"Bad arguments: {exc}", name, "请按工具 schema 传参")


def run_agent(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    max_turns: int = 5,
) -> str:
    """主循环：请求模型 -> 执行 tool_calls -> 回灌结果 -> 直到得到最终文本。"""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for turn in range(1, max_turns + 1):
        message = chat(messages, tools=TOOLS)
        messages.append(message)

        tool_calls = message.get("tool_calls") or []  # 拼写是 tool_calls
        if not tool_calls:
            return message.get("content") or ""

        logger.info("第 %d 轮：模型请求 %d 个工具调用", turn, len(tool_calls))
        for call in tool_calls:
            tool_name = (call.get("function") or {}).get("name", "?")
            result = dispatch_tool_call(call)
            logger.info("工具 %s 执行完成，返回 %d 字符", tool_name, len(result))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": result,
                }
            )

    raise LLMError(f"超过 {max_turns} 轮仍未得到最终回答")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        answer = run_agent(
            "请帮我列出当前目录下的文件列表，然后用 read_file 读取第一个 .py 文件的前 200 个字符"
        )
    except LLMError as exc:
        logger.error("运行失败：%s", exc)
        return 1
    print("模型最终回答：")
    print(answer)
    return 0


if __name__ == "__main__":  # 双下划线，且必须位于模块级
    sys.exit(main())
