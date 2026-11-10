"""Tool catalogue for the Claude CLI adapter, served over stdio MCP.

``claude_cli`` starts this script through ``--mcp-config`` so Evonic's tools
reach the model as real (native) tools. It only *lists* them: the adapter
stops the CLI as soon as the model's message ends with ``tool_use`` and
Evonic executes the calls itself, with its own safety checks. The CLI also
denies these tools on its own (no ``--allowedTools``), so ``tools/call`` is
not expected; if it arrives anyway it is refused here and nothing runs.

Usage: python claude_mcp_stub.py <tools.json>

Standard library only, so the interpreter can start it in isolated mode.
"""

import json
import sys

_REFUSAL = ("This tool is executed by Evonic, not inside this session. "
            "Reply with the tool call; do not retry it here.")


def _reply(request_id, result=None, error=None):
    message = {"jsonrpc": "2.0", "id": request_id}
    if error is not None:
        message["error"] = error
    else:
        message["result"] = result if result is not None else {}
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main(path: str) -> None:
    with open(path, encoding="utf-8") as handle:
        tools = json.load(handle)
    for line in sys.stdin:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(request, dict) or request.get("id") is None:
            continue  # notifications need no answer
        method = request.get("method")
        if method == "initialize":
            params = request.get("params") or {}
            _reply(request["id"], {
                "protocolVersion": params.get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "evonic", "version": "1"},
            })
        elif method == "tools/list":
            _reply(request["id"], {"tools": tools})
        elif method == "tools/call":
            _reply(request["id"], {"content": [{"type": "text", "text": _REFUSAL}], "isError": True})
        elif method == "ping":
            _reply(request["id"], {})
        else:
            _reply(request["id"], error={"code": -32601, "message": f"method not found: {method}"})


if __name__ == "__main__":
    main(sys.argv[1])
