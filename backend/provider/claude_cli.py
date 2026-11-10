"""Claude subscription access through the official ``claude -p`` CLI.

Evonic never reads, copies or refreshes Claude OAuth tokens. The user logs in
with ``claude auth login`` and Evonic only spawns the CLI in print mode, so the
Claude Code client itself talks to Anthropic. The CLI runs here as a plain LLM:
its built-in tools, settings, MCP servers and session persistence are disabled,
and Evonic's own tools are emulated through ``--json-schema`` so the agent loop,
workplaces and safety checks in Evonic stay in charge of every action.

The module is an adapter: Evonic's rules apply on the way out (OpenAI-shaped
messages, honest ``tool_calls``/``finish_reason``), Claude's rules apply on the
way in. The history goes to the CLI as one content block per message so the
prompt cache keeps matching the unchanged prefix between calls; a single
growing text block would be re-written from scratch on every step.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

DEFAULT_TIMEOUT = 300
STATUS_TTL = 30
# Above this size the system prompt goes through stdin instead of argv
# (single argv strings are capped at ~128 KB on Linux).
MAX_ARGV_SYSTEM_PROMPT = 100_000

KNOWN_MODELS = [
    {"id": "claude-fable-5-1", "name": "Claude Fable 5.1"},
    {"id": "claude-opus-5-5", "name": "Claude Opus 5.5"},
    {"id": "claude-sonnet-5-5", "name": "Claude Sonnet 5.5"},
    {"id": "claude-haiku-5-5", "name": "Claude Haiku 5.5"},
]

_logger = logging.getLogger(__name__)

_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "done": {"type": "boolean"},
        "tool_calls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["name", "arguments"],
            },
        },
    },
    "required": ["text", "done"],
}

# The structured-output channel itself is a synthetic tool inside the CLI.
_STRUCTURED_OUTPUT_TOOL = "StructuredOutput"

# Line appended by ``build_system_prompt`` when ``inject_datetime`` is on
# (``context.DATETIME_LINE_PREFIX``). It changes every turn, so it is moved
# out of the system prompt, which must stay byte-stable to be cached.
_DATETIME_LINE = re.compile(r"\n*^Current date/time: [^\n]*", re.MULTILINE)

_NUDGE = (
    "[evonic]\nYour last reply set done=false but contained no tool_calls, so nothing "
    "was executed and your turn would end here. Reply again now with the tool_calls "
    "for your next step. Set done=true only if the task is finished or you need "
    "input from the user. Do not repeat the plan.")

# Same child-env policy as pi-claude-bridge: no claude.ai cloud MCP discovery,
# and Evonic owns context management so the CLI must not auto-compact.
CHILD_ENV = {
    "ENABLE_CLAUDEAI_MCP_SERVERS": "0",
    "DISABLE_AUTO_COMPACT": "1",
}

# Thinking-budget (tokens) upper bounds -> Claude Code ``--effort`` level.
_EFFORT_STEPS = ((2_048, "low"), (8_192, "medium"), (24_576, "high"), (65_536, "xhigh"))
_VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")

_status_lock = threading.Lock()
_status_cache: Dict[str, Any] = {"at": 0.0, "value": None}
_workdir: Optional[str] = None


def claude_binary() -> Optional[str]:
    return shutil.which("claude")


def _neutral_cwd() -> str:
    """Empty directory so no project CLAUDE.md or settings leak into prompts."""
    global _workdir
    if not _workdir or not os.path.isdir(_workdir):
        _workdir = tempfile.mkdtemp(prefix="evonic-claude-cli-")
    return _workdir


def get_status(force: bool = False) -> Dict[str, Any]:
    """Return ``{installed, logged_in, version, auth_method, error}`` (cached)."""
    with _status_lock:
        now = time.time()
        cached = _status_cache["value"]
        if cached is not None and not force and now - _status_cache["at"] < STATUS_TTL:
            return cached

    binary = claude_binary()
    status: Dict[str, Any] = {
        "installed": bool(binary), "logged_in": False, "version": "",
        "auth_method": "", "error": "",
    }
    if not binary:
        status["error"] = "Claude Code CLI not found. Install it, then run `claude auth login`."
    else:
        try:
            version = subprocess.run(
                [binary, "--version"], capture_output=True, text=True, timeout=10,
                stdin=subprocess.DEVNULL, cwd=_neutral_cwd())
            status["version"] = version.stdout.strip().split(" ")[0]
            auth = subprocess.run(
                [binary, "auth", "status"], capture_output=True, text=True, timeout=15,
                stdin=subprocess.DEVNULL, cwd=_neutral_cwd())
            info = json.loads(auth.stdout or "{}")
            status["logged_in"] = bool(info.get("loggedIn"))
            status["auth_method"] = info.get("authMethod") or ""
            if not status["logged_in"]:
                status["error"] = "Not logged in. Run `claude auth login` in a terminal."
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            status["error"] = f"Could not query Claude Code CLI: {exc}"

    with _status_lock:
        _status_cache.update(at=time.time(), value=status)
    return status


def resolve_effort(thinking: bool, budget: int = 0) -> str:
    """Map Evonic's thinking flag/budget onto a ``--effort`` level.

    Thinking off -> ``low``. Thinking on without a budget -> ``medium``; with a
    budget the closest level is chosen. An explicit level name is passed through.
    """
    if isinstance(budget, str) and budget.lower() in _VALID_EFFORTS:
        return budget.lower()
    if not thinking:
        return "low"
    try:
        budget = int(budget or 0)
    except (TypeError, ValueError):
        budget = 0
    if budget <= 0:
        return "medium"
    for limit, level in _EFFORT_STEPS:
        if budget <= limit:
            return level
    return "max"


def _content_text(content: Any, images: Optional[List[Dict[str, Any]]] = None) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif part.get("type") in ("image_url", "image"):
                    block = _image_block(part)
                    if block is not None and images is not None:
                        images.append(block)
                        parts.append(f"[image {len(images)} attached]")
                    else:
                        parts.append("[image omitted: could not be read as an inline image]")
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    return json.dumps(content, ensure_ascii=False)


_DATA_URL = re.compile(r"^data:(image/[\w.+-]+);base64,(.+)$", re.DOTALL)


def _image_block(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """OpenAI ``image_url`` (data URL) / Anthropic ``image`` part -> Claude image block."""
    if part.get("type") == "image" and isinstance(part.get("source"), dict):
        return {"type": "image", "source": part["source"]}
    url = part.get("image_url")
    url = url.get("url", "") if isinstance(url, dict) else (url or "")
    match = _DATA_URL.match(url)
    if not match:
        return None
    return {"type": "image", "source": {
        "type": "base64", "media_type": match.group(1), "data": match.group(2)}}


def _text_block(text: str) -> Dict[str, Any]:
    return {"type": "text", "text": text}


def _split_system(messages: List[Dict[str, Any]]):
    """Split messages into ``(system_prompt, context_parts, history)``.

    Only the first system message becomes the CLI system prompt. The other
    leading system messages (agent state, skill context) are rewritten by the
    agent loop between iterations; at the front of the prompt each rewrite
    would invalidate the cache for the whole history, so they travel as
    context next to the latest user message instead. System messages inside
    the history stay where they are.
    """
    system_prompt = ""
    context: List[str] = []
    index = 0
    while index < len(messages) and messages[index].get("role") == "system":
        text = _content_text(messages[index].get("content"))
        if index == 0:
            system_prompt = text
        elif text:
            context.append(text)
        index += 1

    stamps = [m.group(0).strip() for m in _DATETIME_LINE.finditer(system_prompt)]
    if stamps:
        system_prompt = _DATETIME_LINE.sub("", system_prompt)
        context = stamps + context
    return system_prompt, context, messages[index:]


def _history_blocks(history: List[Dict[str, Any]], context: List[str]) -> List[Dict[str, Any]]:
    """One content block per message; consecutive tool results share one block.

    Blocks for older messages must come out byte-identical on every call; the
    cache only hits on an unchanged prefix of whole blocks. Volatile context is
    placed right after the latest user message: it then only moves when a new
    user message arrives, not on every tool step.
    """
    anchor = max((i for i, m in enumerate(history) if m.get("role") == "user"), default=None)
    context_block = _text_block("[context]\n" + "\n\n".join(context)) if context else None
    blocks: List[Dict[str, Any]] = [context_block] if context_block and anchor is None else []
    images: List[Dict[str, Any]] = []
    tool_results: List[str] = []

    def flush_tool_results():
        if tool_results:
            blocks.append(_text_block("\n\n".join(tool_results)))
            tool_results.clear()

    for index, message in enumerate(history):
        role = message.get("role")
        if role == "tool":
            text = _content_text(message.get("content"))
            tool_results.append(f"[tool_result id={message.get('tool_call_id', '')}]\n{text}")
            continue
        flush_tool_results()
        if role == "user":
            first_image = len(images)
            text = _content_text(message.get("content"), images)
            blocks.extend(images[first_image:])
            blocks.append(_text_block(f"[user]\n{text}"))
        elif role == "assistant":
            text = _content_text(message.get("content"))
            lines = [f"[assistant]\n{text}"] if text else []
            for call in message.get("tool_calls") or []:
                fn = call.get("function", {})
                lines.append(
                    f"[assistant tool_call id={call.get('id', '')} name={fn.get('name', '')}]\n"
                    f"{fn.get('arguments', '{}')}")
            if lines:
                blocks.append(_text_block("\n".join(lines)))
        elif role == "system":
            blocks.append(_text_block(f"[system]\n{_content_text(message.get('content'))}"))
        if index == anchor and context_block:
            blocks.append(context_block)
    flush_tool_results()
    return blocks or [_text_block("[user]\n(empty)")]


def _tool_instructions(tools: List[Dict[str, Any]], tool_choice: Optional[str]) -> str:
    specs = []
    for tool in tools:
        fn = tool.get("function", {})
        specs.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {}),
        })
    rule = (f"You MUST call the tool `{tool_choice}` in this reply."
            if tool_choice else
            "Call tools only when they are needed; otherwise leave tool_calls empty.")
    return (
        "\n\n# Tools\n"
        "The tools listed below are NOT native function-calling tools in this session; "
        "never invoke them as native tool calls (that fails with 'No such tool "
        "available'). Instead you call them by writing them into your final structured "
        "reply, a JSON object with `text` (what you say to the user, may be empty when "
        "calling tools), `tool_calls` (list of {name, arguments}) and `done`. Evonic "
        "executes every entry in `tool_calls` and the conversation continues with "
        "[tool_result] entries. `arguments` must follow the tool's JSON schema. You have "
        "no other tools.\n"
        "A reply without tool_calls ends your turn. If you are about to do something, "
        "put its tool_calls in this same reply instead of describing the plan. Set "
        "`done` to true only when the task is finished or you need input from the "
        "user; when `done` is false, tool_calls must not be empty. "
        f"{rule}\n\n{json.dumps(specs, ensure_ascii=False)}")


def _parse_events(stdout: str):
    """Return ``(result_event, native_tool_attempts)`` from stream-json output.

    Native tool attempts are tool_use blocks the model sent inside the CLI's
    own loop (they fail there with 'No such tool available'). Evonic never sees
    them otherwise, so they are surfaced for logging.
    """
    result = None
    attempts: List[str] = []
    for line in (stdout or "").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "result":
            result = event
        elif event.get("type") == "assistant":
            for block in (event.get("message") or {}).get("content") or []:
                if (isinstance(block, dict) and block.get("type") == "tool_use"
                        and block.get("name") != _STRUCTURED_OUTPUT_TOOL):
                    attempts.append(str(block.get("name") or ""))
    if result is None:
        raise json.JSONDecodeError("no result event", stdout or "", 0)
    return result, attempts


def _classify_error(text: str) -> str:
    low = (text or "").lower()
    if any(s in low for s in ("not logged in", "login", "authenticat", "401", "oauth token")):
        return "auth_error"
    if any(s in low for s in ("rate limit", "usage limit", "429", "limit reached")):
        return "rate_limit_error"
    if any(s in low for s in ("prompt is too long", "context")):
        return "context_length_error"
    if any(s in low for s in ("overloaded", "529", "500", "502", "503")):
        return "provider_error"
    return "llm_error"


def _fail(error_type: str, detail: str, duration_ms: int) -> Dict[str, Any]:
    return {"success": False, "error_type": error_type, "error": detail,
            "duration_ms": duration_ms}


def _run_cli(cmd: List[str], blocks: List[Dict[str, Any]], timeout: int):
    """Run one CLI call. Returns ``(result_event, native_attempts, None)`` or ``(None, None, failure)``."""
    start = time.time()
    payload = json.dumps({"type": "user", "message": {"role": "user", "content": blocks}},
                         ensure_ascii=False) + "\n"
    try:
        proc = subprocess.run(
            cmd, input=payload, capture_output=True, text=True,
            timeout=timeout, cwd=_neutral_cwd(), env={**os.environ, **CHILD_ENV})
    except subprocess.TimeoutExpired:
        return None, None, _fail("timeout_error", "claude -p timed out",
                                 int((time.time() - start) * 1000))
    except OSError as exc:
        return None, None, _fail("connection_error", str(exc), int((time.time() - start) * 1000))

    duration_ms = int((time.time() - start) * 1000)
    try:
        data, attempts = _parse_events(proc.stdout)
    except (json.JSONDecodeError, TypeError):
        detail = (proc.stderr or proc.stdout or "empty output").strip()[:500]
        return None, None, _fail(_classify_error(detail), detail, duration_ms)

    if data.get("is_error") or proc.returncode != 0:
        detail = str(data.get("result") or proc.stderr or "claude -p failed")[:500]
        error_type = "rate_limit_error" if data.get("api_error_status") == 429 else _classify_error(detail)
        return None, None, _fail(error_type, detail, duration_ms)
    data["_duration_ms"] = duration_ms
    return data, attempts, None


def _structured_reply(data: Dict[str, Any]) -> Dict[str, Any]:
    structured = data.get("structured_output")
    if isinstance(structured, dict):
        return structured
    text = data.get("result") or ""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    return parsed if isinstance(parsed, dict) else {"text": text}


def _tool_calls(structured: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every named call is handed to Evonic, known or not.

    Unknown names are not dropped here: Evonic's quality monitor answers them
    with a correction listing the real tools, so the model can recover instead
    of its turn silently ending.
    """
    calls = []
    for call in structured.get("tool_calls") or []:
        if not isinstance(call, dict) or not call.get("name"):
            continue
        arguments = call.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments or {}, ensure_ascii=False)
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": str(call["name"]), "arguments": arguments},
        })
    return calls


def send_request(
    model: str,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    thinking: bool = False,
    thinking_budget: int = 0,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one completion. Returns ``{success, response}`` in OpenAI chat shape."""
    binary = claude_binary()
    if not binary:
        return _fail("connection_error", get_status()["error"], 0)

    system_prompt, context, history = _split_system(messages)
    system_prompt = system_prompt or "You are a helpful assistant."
    if tools:
        system_prompt += _tool_instructions(tools, tool_choice)
    blocks = _history_blocks(history, context)

    cmd = [
        binary, "-p", "--model", model,
        "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
        "--tools", "", "--no-session-persistence", "--disable-slash-commands",
        "--setting-sources", "", "--strict-mcp-config",
    ]
    cmd += ["--effort", resolve_effort(thinking, thinking_budget)]
    if tools:
        cmd += ["--json-schema", json.dumps(_OUTPUT_SCHEMA)]
    if len(system_prompt.encode("utf-8")) <= MAX_ARGV_SYSTEM_PROMPT:
        cmd += ["--system-prompt", system_prompt]
    else:
        cmd += ["--system-prompt", "Follow the instructions in the <system> block of the input."]
        blocks = [_text_block(f"<system>\n{system_prompt}\n</system>")] + blocks

    timeout = timeout or DEFAULT_TIMEOUT
    data, attempts, failure = _run_cli(cmd, blocks, timeout)
    if failure:
        return failure
    runs = [data]
    native_attempts = list(attempts)

    text = data.get("result") or ""
    tool_calls: List[Dict[str, Any]] = []
    if tools:
        structured = _structured_reply(data)
        text = structured.get("text") or ""
        tool_calls = _tool_calls(structured)
        if not tool_calls and structured.get("done") is False:
            # The model announced work without emitting it. Ask once more on the
            # same (cached) prefix instead of letting the turn end on a plan.
            draft = _text_block(f"[assistant draft, not executed]\n{text}")
            retry, attempts, failure = _run_cli(cmd, blocks + [draft, _text_block(_NUDGE)], timeout)
            if not failure:
                runs.append(retry)
                native_attempts += attempts
                structured = _structured_reply(retry)
                text = structured.get("text") or text
                tool_calls = _tool_calls(structured)
            _logger.warning("claude_cli: done=false without tool_calls; retried (%s tool_calls after retry)",
                            len(tool_calls))
    if native_attempts:
        _logger.warning("claude_cli: model tried native tool calls inside the CLI: %s",
                        ", ".join(native_attempts))

    usage: Dict[str, int] = {}
    for run in runs:
        for key, value in (run.get("usage") or {}).items():
            if isinstance(value, int):
                usage[key] = usage.get(key, 0) + value
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_write = usage.get("cache_creation_input_tokens", 0)
    prompt_tokens = usage.get("input_tokens", 0) + cache_read + cache_write
    completion_tokens = usage.get("output_tokens", 0)
    message: Dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "success": True,
        "response": {
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "prompt_tokens_details": {"cached_tokens": cache_read,
                                          "cache_write_tokens": cache_write},
            },
        },
        "duration_ms": sum(run.get("_duration_ms", 0) for run in runs),
        "diagnostics": {
            "cli_calls": len(runs),
            "num_turns": sum(run.get("num_turns") or 0 for run in runs),
            "native_tool_attempts": native_attempts,
        },
    }
