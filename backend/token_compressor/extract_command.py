"""
Command detection helper for RTK token compressor.

Maps tool invocations to filter-friendly command strings
so the compressor registry can pick the right TOML pipeline.
"""

from __future__ import annotations

import re
from typing import Any


_META_PREFIXES = ("cd ", "export ", "set ", "trap ", "exec ", "echo ")


def extract_command(tool_name: str, params: dict[str, Any]) -> str:
    """Derive a command string for compressor filter matching.

    Args:
        tool_name: The tool being invoked (e.g. "bash", "runpy").
        params:    Tool arguments dict.

    Returns:
        A string like "git status", "pytest", "read_file foo.py", or tool_name.
    """
    if not isinstance(params, dict):
        return tool_name

    if tool_name == "bash":
        script = params.get("script", "")
        if not isinstance(script, str):
            return "bash"
        return _bash_cmd(script)

    if tool_name == "runpy":
        code = params.get("code", "")
        if not isinstance(code, str):
            return "python"
        return _py_cmd(code)

    if tool_name in ("read_file", "write_file"):
        fp = params.get("file_path", "")
        return f"{tool_name} {fp}" if fp else tool_name

    return tool_name


_SPLIT_RE = re.compile(r"\n|;|&&|\|\||\|")
_META_WORDS = {"cd", "export", "set", "source", ".", "trap", "exec", "echo", "pushd", "popd", "unset", "true", "false"}


def _bash_candidates(script: str) -> list[str]:
    """Split a script into substantive command segments (task #1, step 2)."""
    out: list[str] = []
    for seg in _SPLIT_RE.split(script or ""):
        seg = seg.strip().lstrip("({").strip()
        if not seg or seg.startswith("#"):
            continue
        seg = re.sub(r"^(?:\w+=\S*\s*)+", "", seg).strip()
        if not seg:
            continue
        tokens = seg.split()
        if tokens[0] in _META_WORDS:
            continue
        out.append(" ".join(tokens[:2]) if len(tokens) >= 2 else tokens[0])
    return out


def _bash_cmd(script: str, matcher=None) -> str:
    """Return the first candidate segment that matches a filter, else the last candidate."""
    cands = _bash_candidates(script)
    if not cands:
        return "bash"
    if matcher is None:
        try:
            from .compressor_registry import get_registry
            matcher = get_registry().lookup
        except Exception:
            matcher = None
    if matcher is not None:
        for c in cands:
            try:
                if matcher(c) is not None:
                    return c
            except Exception:
                break
    return cands[-1]


def _py_cmd(code: str) -> str:
    """Heuristically detect the Python command from code content."""
    if not code:
        return "python"

    for line in code.strip().split("\n")[:10]:
        s = line.strip()
        if re.match(r"^(?:import\s+pytest|from\s+pytest\s)", s):
            return "pytest"
        if re.match(r"^(?:import\s+unittest|from\s+unittest\s)", s):
            return "python -m unittest"
        if "subprocess.run" in s or "subprocess.call" in s:
            return "python subprocess"
        if "os.system" in s:
            return "python"

    return "python"
