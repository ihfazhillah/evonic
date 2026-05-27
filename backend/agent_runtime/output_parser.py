"""
output_parser.py — Detect and report malformed tool calls in assistant text.

Adopted from little-coder's output-parser extension. When a model (especially
small ones) isn't trained for native function calling, it sometimes embeds
tool calls as text in the response body instead of using the tool_calls field.

This module detects three common malformed patterns:

1. Fenced ```tool blocks — ```tool\n{...}\n```
2. <tool_call> XML tags — <tool_call>{"name": "...", ...}</tool_call>
3. Bare JSON objects that look like tool calls

Two recovery strategies:

- build_nudge_message — instruct the model to re-issue using native tool calls
  (works for capable models that briefly drift off-format)
- promote_to_tool_calls — convert the extracted JSON directly to OpenAI
  tool_calls format (last-resort for small/weak models that cannot recover
  from a nudge). Use is_small_model() to decide.
"""

from __future__ import annotations

import json
import logging
import re
import uuid

_logger = logging.getLogger(__name__)

# --- Pattern definitions ---

# Matches fenced ```tool ... ``` blocks
_TOOL_FENCE_RE = re.compile(
    r"```tool\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

# Matches <tool_call>...</tool_call> XML tags (Qwen-style)
_TOOL_CALL_XML_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)

# Sentinel prefix used at the start of every output-parser nudge message.
# llm_loop.py uses it to count prior nudges and decide when to escalate
# to auto-promotion.
NUDGE_PREFIX = "[SYSTEM] Your response contains tool calls embedded as text"


def _scan_balanced_json(text: str, start: int) -> int | None:
    """Return the index AFTER the closing '}' of the JSON object that starts at
    text[start] == '{'. Tracks string state so braces inside string values
    don't affect depth. Returns None if no balanced object is found.
    """
    if start >= len(text) or text[start] != '{':
        return None
    depth = 0
    i = start
    in_string = False
    escape = False
    while i < len(text):
        c = text[i]
        if escape:
            escape = False
        elif in_string:
            if c == '\\':
                escape = True
            elif c == '"':
                in_string = False
        else:
            if c == '"':
                in_string = True
            elif c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return None


def detect_fenced_tool_blocks(content: str) -> list[dict]:
    """Detect ```tool ... ``` fenced blocks in text.

    Returns a list of dicts with keys:
        format:   "fenced_code"
        raw:      The raw matched text (```tool\n...\n```)
        content:  The inner content (without fence markers)

    Each inner content is parsed as JSON if possible.
    """
    results = []
    for match in _TOOL_FENCE_RE.finditer(content):
        inner = match.group(1).strip()
        parsed = None
        try:
            parsed = json.loads(inner)
        except (json.JSONDecodeError, TypeError):
            pass
        results.append({
            "format": "fenced_code",
            "raw": match.group(0),
            "content": inner,
            "parsed": parsed,
        })
    return results


def detect_xml_tool_calls(content: str) -> list[dict]:
    """Detect <tool_call>...</tool_call> XML tags in text.

    Returns a list of dicts with keys:
        format:   "xml_tag"
        raw:      The raw matched text (<tool_call>...</tool_call>)
        content:  The inner tag content
        parsed:   Parsed JSON if the content is valid JSON, else None.
    """
    results = []
    for match in _TOOL_CALL_XML_RE.finditer(content):
        inner = match.group(1).strip()
        parsed = None
        try:
            parsed = json.loads(inner)
        except (json.JSONDecodeError, TypeError):
            pass
        results.append({
            "format": "xml_tag",
            "raw": match.group(0),
            "content": inner,
            "parsed": parsed,
        })
    return results


def detect_bare_json_tool_calls(content: str) -> list[dict]:
    """Detect bare JSON objects that look like tool calls.

    Uses a balanced-brace scanner instead of a regex so it correctly handles
    empty ``arguments: {}`` and nested objects inside ``arguments``. Only
    returns entries that parse as JSON dicts containing both ``name`` and
    ``arguments`` keys — keeps false positives off.
    """
    if not content:
        return []

    # Strip fenced/XML content first so we don't double-detect.
    clean = _TOOL_FENCE_RE.sub("", content)
    clean = _TOOL_CALL_XML_RE.sub("", clean)

    results = []
    i = 0
    while i < len(clean):
        if clean[i] != '{':
            i += 1
            continue
        end = _scan_balanced_json(clean, i)
        if end is None:
            i += 1
            continue
        raw = clean[i:end]
        # Cheap pre-filter before attempting json.loads.
        if '"name"' not in raw or '"arguments"' not in raw:
            i += 1
            continue
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            i += 1
            continue
        if (
            isinstance(parsed, dict)
            and 'name' in parsed
            and 'arguments' in parsed
        ):
            results.append({
                "format": "bare_json",
                "raw": raw,
                "content": raw,
                "parsed": parsed,
            })
            i = end
        else:
            i += 1
    return results


def detect_all(content: str) -> list[dict]:
    """Run all three detectors and return combined results.

    Order: fenced blocks first, then XML tags, then bare JSON.
    Bare JSON detector strips fenced/XML content first to avoid duplicates.

    Returns a list of dicts, each with at least:
        format, raw, content, parsed
    """
    results = []
    results.extend(detect_fenced_tool_blocks(content))
    results.extend(detect_xml_tool_calls(content))
    results.extend(detect_bare_json_tool_calls(content))
    return results


def has_malformed_calls(content: str) -> bool:
    """Quick check: does this text contain any malformed tool call patterns?"""
    if not content:
        return False
    if _TOOL_FENCE_RE.search(content):
        return True
    if _TOOL_CALL_XML_RE.search(content):
        return True
    # Bare-JSON case requires a balanced-brace scan (see detect_bare_json_tool_calls).
    return bool(detect_bare_json_tool_calls(content))


# ── Small-model detection ────────────────────────────────────────────────

# Param-count tokens preceded by a non-digit separator. Examples that match:
#   "1.5b", "1_5b", "1-5b", "7b", "0.5b", "0.6b"
# Examples that don't match:
#   "qwen2.5"  (no trailing 'b')
#   "v1b"      (preceded by letter, intentional — names like "v1b1" aren't sizes)
_PARAM_SIZE_RE = re.compile(
    r'(?:^|[\s\-_/])(\d+(?:[._\-]\d+)?)\s*b(?=[\s\-_./]|$)',
    re.IGNORECASE,
)

# Markers that always indicate a tiny model regardless of param count.
_SMALL_TOKENS = ('tiny', 'nano', 'smollm', 'smol-lm', 'smol_lm', 'smol-llm')

# Models <= this many billion parameters are treated as "small" — too weak
# to reliably recover from a nudge, so we promote bare JSON to tool_calls.
_SMALL_MODEL_THRESHOLD_B = 3.0


def is_small_model(model_name: str | None) -> bool:
    """Heuristic: is this model small/weak enough to need auto-promotion?

    Returns True for explicit tiny markers (``tiny``, ``nano``, smol family)
    or when the name advertises <= 3B parameters. Used as a signal to skip
    the "please re-issue using native tool calls" nudge — small models
    cannot recover and just loop.
    """
    if not model_name:
        return False
    s = model_name.lower()
    if any(tok in s for tok in _SMALL_TOKENS):
        return True
    for m in _PARAM_SIZE_RE.finditer(s):
        size_str = m.group(1).replace('_', '.').replace('-', '.')
        try:
            size = float(size_str)
        except ValueError:
            continue
        if size <= _SMALL_MODEL_THRESHOLD_B:
            return True
    return False


# ── Promotion: bare JSON → OpenAI tool_calls ────────────────────────────

def promote_to_tool_calls(extracted: list[dict]) -> list[dict]:
    """Convert detector results to OpenAI tool_calls. Dedupes by (name, args).

    Skips entries whose ``parsed`` is not a dict with both ``name`` and
    ``arguments``. Caller is responsible for stripping the raw spans from
    the visible content (use :func:`strip_extracted_calls`).
    """
    if not extracted:
        return []
    seen: set[tuple[str, str]] = set()
    result: list[dict] = []
    for entry in extracted:
        parsed = entry.get('parsed')
        if not isinstance(parsed, dict):
            continue
        name = parsed.get('name') or parsed.get('function')
        if not isinstance(name, str) or not name.strip():
            continue
        args = parsed.get('arguments')
        if args is None:
            args = parsed.get('parameters', {})
        if not isinstance(args, dict):
            continue
        try:
            args_json = json.dumps(args)
            dedup_key = (name, json.dumps(args, sort_keys=True))
        except (TypeError, ValueError):
            continue
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        result.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": name, "arguments": args_json},
        })
    return result


# Matches a code fence pair whose body is empty or whitespace only — left
# behind when bare JSON inside ```json ... ``` is stripped.
_EMPTY_FENCE_RE = re.compile(
    r"```[A-Za-z0-9_+\-]*\s*\n?\s*```",
    re.DOTALL,
)


def strip_extracted_calls(text: str, extracted: list[dict]) -> str:
    """Remove the raw substrings of extracted calls from text.

    Used after promote_to_tool_calls so the bare JSON doesn't leak into the
    visible assistant message. Also cleans up empty code fences left behind
    when the model wrapped its tool call in ```json ... ```.
    """
    if not text or not extracted:
        return text or ""
    result = text
    for call in extracted:
        raw = call.get('raw')
        if raw:
            result = result.replace(raw, '', 1)
    # Strip code fences that are now empty after the JSON body was removed.
    result = _EMPTY_FENCE_RE.sub('', result)
    return result.strip()


def count_prior_nudges(messages: list[dict]) -> int:
    """Count how many output-parser nudges are already in the message list.

    Used by the llm_loop escalation gate: after N nudges with no recovery,
    fall through to auto-promotion instead of looping forever.
    """
    n = 0
    for m in messages:
        if m.get('role') != 'user':
            continue
        content = m.get('content')
        if isinstance(content, str) and content.startswith(NUDGE_PREFIX):
            n += 1
    return n


def build_nudge_message(extracted_calls: list[dict]) -> str:
    """Build a correction nudge message from extracted malformed tool calls.

    The nudge tells the model to use native tool calling and includes
    the extracted calls so the model can re-issue them.

    Args:
        extracted_calls: List of detection result dicts from detect_* functions.

    Returns:
        A user-role message string ready for injection into the conversation.
    """
    if not extracted_calls:
        return ""

    formats_seen = set(c["format"] for c in extracted_calls)
    format_descriptions = {
        "fenced_code": "```tool code blocks",
        "xml_tag": "<tool_call> XML tags",
        "bare_json": "bare JSON objects",
    }
    format_list = ", ".join(
        format_descriptions.get(f, f) for f in sorted(formats_seen)
    )

    # Build a summary of extracted calls
    call_summaries = []
    for i, call in enumerate(extracted_calls[:5]):  # cap at 5 to avoid message bloat
        if call.get("parsed") and isinstance(call["parsed"], dict):
            name = call["parsed"].get("name", call["parsed"].get("function", "?"))
            args = call["parsed"].get("arguments", call["parsed"].get("parameters", {}))
            args_preview = json.dumps(args)[:120]
            call_summaries.append(f"  {i+1}. {name}({args_preview})")
        else:
            preview = call.get("content", "")[:80]
            call_summaries.append(f"  {i+1}. [unparseable] {preview}...")

    if len(extracted_calls) > 5:
        call_summaries.append(f"  ... and {len(extracted_calls) - 5} more")

    summaries_block = "\n".join(call_summaries)

    return (
        f"{NUDGE_PREFIX} "
        f"({format_list}) instead of using native function calling. "
        "You must use the native tool_calls mechanism to invoke tools.\n\n"
        "Detected calls:\n"
        f"{summaries_block}\n\n"
        "Please re-issue these tool calls using the proper function calling "
        "format. Do NOT embed tool calls in text blocks, code fences, or XML tags."
    )
