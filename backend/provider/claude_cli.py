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

import hashlib
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
from collections import OrderedDict
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
    would invalidate the cache for the whole history, so they become context
    parts, keyed by their first line. System messages inside the history stay
    where they are.
    """
    system_prompt = ""
    parts: List[tuple] = []
    index = 0
    while index < len(messages) and messages[index].get("role") == "system":
        text = _content_text(messages[index].get("content"))
        if index == 0:
            system_prompt = text
        elif text:
            parts.append((text.strip().splitlines()[0], text))
        index += 1

    stamps = [m.group(0).strip() for m in _DATETIME_LINE.finditer(system_prompt)]
    if stamps:
        system_prompt = _DATETIME_LINE.sub("", system_prompt)
        parts = [("datetime", "\n".join(stamps))] + parts
    return system_prompt, _unique_keys(parts), messages[index:]


def _unique_keys(parts: List[tuple]) -> List[tuple]:
    seen: Dict[str, int] = {}
    unique = []
    for key, text in parts:
        seen[key] = seen.get(key, 0) + 1
        unique.append((key if seen[key] == 1 else f"{key} #{seen[key]}", text))
    return unique


def _history_blocks(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One content block per message; consecutive tool results share one block.

    Blocks for older messages must come out byte-identical on every call; the
    cache only hits on an unchanged prefix of whole blocks.
    """
    blocks: List[Dict[str, Any]] = []
    images: List[Dict[str, Any]] = []
    tool_results: List[str] = []

    def flush_tool_results():
        if tool_results:
            blocks.append(_text_block("\n\n".join(tool_results)))
            tool_results.clear()

    for message in history:
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
    flush_tool_results()
    return blocks or [_text_block("[user]\n(empty)")]


# Share of the frozen system prompt that instruction overrides may reach before
# the frozen copy is refreshed (one full cache miss) instead of growing further.
_MAX_OVERRIDE_SHARE = 0.3
_INSTRUCTIONS = "instructions: "
_SECTION_HEADING = re.compile(r"^#{1,2} \S")


def _sections(prompt: str) -> List[tuple]:
    """Split a prompt into ``(heading, text)`` at ``#``/``##`` headings outside code fences."""
    sections: List[tuple] = []
    heading, lines, fenced = "(preamble)", [], False
    for line in prompt.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if not fenced and _SECTION_HEADING.match(line):
            sections.append((heading, "\n".join(lines)))
            heading, lines = line.strip(), []
        lines.append(line)
    sections.append((heading, "\n".join(lines)))
    return _unique_keys([(f"{_INSTRUCTIONS}{h}", t.strip("\n")) for h, t in sections])


def _instruction_overrides(frozen: str, prompt: str):
    """Sections of ``prompt`` that differ from the frozen system prompt.

    Returns ``(parts, defaults, size)``: the changed or new sections as context
    parts, the frozen text of every section (what an override falls back to),
    and the total size of the overrides in characters.
    """
    old = dict(_sections(frozen))
    new = _sections(prompt)
    parts = []
    for key, text in new:
        if old.get(key) != text:
            parts.append((key, text if key != f"{_INSTRUCTIONS}(preamble)"
                          else "[updated opening of your instructions]\n" + text))
    names = {key for key, _ in new}
    for key in old:
        if key not in names:
            parts.append((key, f"{key[len(_INSTRUCTIONS):]}: (removed from your instructions)"))
    return parts, old, sum(len(text) for _, text in parts)


# Receipt the agent loop puts in place of a skill's SYSTEM.md after its first
# iteration (backend/agent_runtime/llm_loop.py). The full text already sent
# stays valid, so the receipt is not an update.
_SKILL_RECEIPT = re.compile(r'\[Skill ".*" is loaded\. Refer to earlier system message')

_CONTEXT_HEADER = "[context]"
_CONTEXT_UPDATE_HEADER = ("[context update] Each section below replaces the earlier "
                          "value of the section with the same heading.")


def _place_context(base: List[Dict[str, Any]], parts: List[tuple],
                   log: List[Dict[str, Any]], defaults: Optional[Dict[str, str]] = None):
    """Insert context blocks into the history so they never move.

    ``log`` holds the context blocks sent before, each anchored to the chain
    hash of the history block it followed. Entries whose anchor is still in the
    history are re-sent in place, byte for byte. Only sections whose value
    changed since then go out as a new entry after the last history block, so
    a datetime or agent-state change no longer invalidates the cached history.
    When the history was rewritten (summary) the anchors are gone and the
    whole context is sent again from scratch. ``defaults`` gives the value a
    section falls back to when it is no longer overridden (the frozen system
    prompt's own text). Returns ``(blocks, log)``.
    """
    hashes = _block_hashes(base)
    present = set(hashes)
    kept = [entry for entry in log if entry["anchor"] == "" or entry["anchor"] in present]

    sent: Dict[str, Optional[str]] = {}
    for entry in kept:
        sent.update(entry["values"])
    current = dict(parts)
    for key, text in (defaults or {}).items():
        if key in sent and key not in current:
            current[key] = text
    changed: Dict[str, Optional[str]] = {}
    for key, text in current.items():
        if sent.get(key) == text:
            continue
        if sent.get(key) is not None and _SKILL_RECEIPT.search(text):
            continue
        changed[key] = text
    for key, text in sent.items():
        if text is not None and key not in current:
            changed[key] = None

    if changed:
        lines = [text if text is not None else f"{key}: (no longer active)"
                 for key, text in changed.items()]
        replaces = sent or any(key.startswith(_INSTRUCTIONS) for key in changed)
        header = _CONTEXT_UPDATE_HEADER if replaces else _CONTEXT_HEADER
        kept = kept + [{"anchor": hashes[-1] if hashes else "",
                        "values": changed,
                        "text": header + "\n" + "\n\n".join(lines)}]

    by_anchor: Dict[str, List[Dict[str, Any]]] = {}
    for entry in kept:
        by_anchor.setdefault(entry["anchor"], []).append(_text_block(entry["text"]))
    blocks = list(by_anchor.get("", []))
    for block, chain in zip(base, hashes):
        blocks.append(block)
        blocks.extend(by_anchor.get(chain, []))
    return blocks, kept


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


# ── Cache diagnostics ────────────────────────────────────────────────────────
# Per-conversation memory of what the previous call sent, so a cache miss can
# be explained (system prompt changed, history rewritten, context moved). It is
# best-effort: a restart only loses the comparison for the next call.

_CONVERSATION_LIMIT = 256
_conversations_lock = threading.Lock()
_conversations: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


def _digest(value: Any) -> str:
    raw = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _block_hashes(blocks: List[Dict[str, Any]]) -> List[str]:
    """Chain hashes: each one identifies a block *and* everything before it."""
    hashes, previous = [], ""
    for block in blocks:
        previous = _digest(previous + _digest(block))
        hashes.append(previous)
    return hashes


def _conversation_key(model: str, system_prompt: str, blocks: List[Dict[str, Any]]) -> str:
    from backend.llm_usage_events import current_usage_context

    ctx = current_usage_context()
    if ctx.get("session_id") or ctx.get("agent_id"):
        return "|".join(str(ctx.get(k) or "") for k in ("source", "agent_id", "session_id")) + "|" + model
    return "anon|" + _digest([model, system_prompt[:2000], blocks[0] if blocks else None])


def _conversation(key: str) -> Optional[Dict[str, Any]]:
    with _conversations_lock:
        state = _conversations.get(key)
        if state is not None:
            _conversations.move_to_end(key)
        return state


def _remember(key: str, state: Dict[str, Any]) -> None:
    with _conversations_lock:
        _conversations[key] = state
        _conversations.move_to_end(key)
        while len(_conversations) > _CONVERSATION_LIMIT:
            _conversations.popitem(last=False)


def _context_index(blocks: List[Dict[str, Any]]) -> Optional[int]:
    return next((i for i, b in enumerate(blocks)
                 if str(b.get("text", "")).startswith("[context")), None)


def _diagnose(previous: Optional[Dict[str, Any]], system_hash: str,
              hashes: List[str]) -> Dict[str, Any]:
    """Explain, before the call, which part of the prompt can no longer hit the cache."""
    if previous is None:
        return {"reasons": ["first_call"], "reused_blocks": 0, "previous_blocks": 0}
    old = previous["hashes"]
    reused = 0
    while reused < min(len(old), len(hashes)) and old[reused] == hashes[reused]:
        reused += 1
    reasons = []
    if previous["system_hash"] != system_hash:
        reasons.append("system_prompt_changed")
    if reused < len(old):
        if previous.get("context_index") == reused:
            reasons.append("context_moved")
        else:
            reasons.append(f"history_rewritten_at_block_{reused}")
    return {"reasons": reasons, "reused_blocks": reused, "previous_blocks": len(old)}


def _report_cache(key: str, diagnosis: Dict[str, Any], cache_read: int, cache_write: int) -> None:
    diagnosis.update(cache_read=cache_read, cache_write=cache_write)
    line = ("claude_cli cache %s: read=%d write=%d reused_blocks=%d/%d reasons=%s"
            % (key, cache_read, cache_write, diagnosis["reused_blocks"],
               diagnosis["previous_blocks"], ",".join(diagnosis["reasons"]) or "-"))
    # A follow-up call that writes more than it reads lost a large part of its
    # prefix; that is the case worth seeing in the log.
    if diagnosis["reasons"] != ["first_call"] and cache_write > cache_read:
        _logger.warning(line)
    else:
        _logger.info(line)


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

    system_prompt, parts, history = _split_system(messages)
    system_prompt = system_prompt or "You are a helpful assistant."
    if tools:
        system_prompt += _tool_instructions(tools, tool_choice)
    base = _history_blocks(history)
    key = _conversation_key(model, system_prompt, base)
    previous = _conversation(key) or {}

    # The system prompt sent to the CLI stays frozen for the conversation:
    # Evonic rebuilds it when KB files, memory keys or the agent roster change,
    # and any byte changed there invalidates the cache for the entire history.
    # Changed sections travel as context updates instead.
    frozen = previous.get("frozen_system")
    context_log = previous.get("context_log", [])
    overrides: List[tuple] = []
    defaults: Dict[str, str] = {}
    refreshed = False
    if frozen is not None and frozen != system_prompt:
        overrides, defaults, size = _instruction_overrides(frozen, system_prompt)
        if size > _MAX_OVERRIDE_SHARE * len(frozen):
            frozen, overrides, defaults, context_log, refreshed = system_prompt, [], {}, [], True
    elif frozen is not None:
        defaults = dict(_sections(frozen))
    frozen = system_prompt if frozen is None else frozen
    blocks, context_log = _place_context(base, parts + overrides, context_log, defaults)
    system_prompt = frozen

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

    system_hash = _digest(system_prompt)
    hashes = _block_hashes(blocks)
    diagnosis = _diagnose(previous or None, system_hash, hashes)
    diagnosis["instruction_overrides"] = len(overrides)
    if refreshed:
        diagnosis["reasons"].append("frozen_system_refreshed")

    timeout = timeout or DEFAULT_TIMEOUT
    data, attempts, failure = _run_cli(cmd, blocks, timeout)
    if failure:
        return failure
    _remember(key, {"system_hash": system_hash, "hashes": hashes,
                    "context_index": _context_index(blocks), "context_log": context_log,
                    "frozen_system": frozen})
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
        for field, value in (run.get("usage") or {}).items():
            if isinstance(value, int):
                usage[field] = usage.get(field, 0) + value
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_write = usage.get("cache_creation_input_tokens", 0)
    prompt_tokens = usage.get("input_tokens", 0) + cache_read + cache_write
    completion_tokens = usage.get("output_tokens", 0)
    first_usage = runs[0].get("usage") or {}
    _report_cache(key, diagnosis, first_usage.get("cache_read_input_tokens", 0) or 0,
                  first_usage.get("cache_creation_input_tokens", 0) or 0)
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
            "cache": diagnosis,
        },
    }
