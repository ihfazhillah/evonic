"""Claude subscription access through the official `claude -p` CLI."""

import json
import subprocess
from unittest.mock import patch

import pytest

from backend.llm_usage_events import usage_context
from backend.provider import claude_cli


@pytest.fixture(autouse=True)
def _fresh_conversations():
    claude_cli._conversations.clear()
    yield
    claude_cli._conversations.clear()

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]


def _proc(result, returncode=0, stderr="", assistant_tools=()):
    """Fake stream-json stdout: init event, optional assistant tool_use, result event."""
    lines = [json.dumps({"type": "system", "subtype": "init", "tools": []})]
    if assistant_tools:
        lines.append(json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": name, "input": {}} for name in assistant_tools]}}))
    lines.append(json.dumps({"type": "result", **result}))
    return subprocess.CompletedProcess([], returncode, stdout="\n".join(lines) + "\n", stderr=stderr)


def _run(results, messages=None, **kwargs):
    messages = messages or [{"role": "system", "content": "Be brief."},
                            {"role": "user", "content": "hi"}]
    if isinstance(results, dict):
        results = [results]
    procs = [r if isinstance(r, subprocess.CompletedProcess) else _proc(r) for r in results]
    with patch.object(claude_cli, "claude_binary", return_value="/usr/bin/claude"), \
         patch("backend.provider.claude_cli.subprocess.run", side_effect=procs) as run:
        result = claude_cli.send_request("claude-sonnet-5-5", messages, **kwargs)
    return result, run


def _blocks(run, call=-1):
    return json.loads(run.call_args_list[call].kwargs["input"])["message"]["content"]


def test_plain_completion_disables_builtin_tools_and_maps_usage():
    result, run = _run({"result": "Hello!", "usage": {
        "input_tokens": 2, "cache_creation_input_tokens": 600,
        "cache_read_input_tokens": 10, "output_tokens": 5}})
    cmd = run.call_args.args[0]
    assert result["success"]
    assert result["response"]["choices"][0]["message"]["content"] == "Hello!"
    assert result["response"]["usage"] == {
        "prompt_tokens": 612, "completion_tokens": 5, "total_tokens": 617,
        "prompt_tokens_details": {"cached_tokens": 10, "cache_write_tokens": 600}}
    assert cmd[cmd.index("--tools") + 1] == ""
    for flag in ("--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands"):
        assert flag in cmd
    assert cmd[cmd.index("--input-format") + 1] == "stream-json"
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"
    assert cmd[cmd.index("--system-prompt") + 1] == "Be brief."
    assert "--json-schema" not in cmd
    assert _blocks(run) == [{"type": "text", "text": "[user]\nhi"}]


def test_tool_calls_are_passed_through_including_unknown_names():
    result, run = _run({"result": "", "structured_output": {
        "text": "Checking.", "done": False, "tool_calls": [
            {"name": "get_weather", "arguments": {"city": "Tokyo"}},
            {"name": "rm_rf", "arguments": {}}]},
        "usage": {}}, tools=TOOLS)
    message = result["response"]["choices"][0]["message"]
    assert "--json-schema" in run.call_args.args[0]
    # Unknown names reach Evonic, whose quality monitor answers with a correction.
    assert [c["function"]["name"] for c in message["tool_calls"]] == ["get_weather", "rm_rf"]
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"city": "Tokyo"}
    assert result["response"]["choices"][0]["finish_reason"] == "tool_calls"
    assert run.call_count == 1


def test_history_is_one_block_per_message_and_tool_results_are_grouped():
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}},
            {"id": "c2", "type": "function",
             "function": {"name": "get_weather", "arguments": "{\"city\":\"Oslo\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        {"role": "tool", "tool_call_id": "c2", "content": "rain"},
    ]
    _, run = _run({"result": "ok", "usage": {}}, messages=messages)
    texts = [b["text"] for b in _blocks(run)]
    assert texts[0] == "[user]\nweather?"
    assert "tool_call id=c1 name=get_weather" in texts[1]
    assert texts[2] == "[tool_result id=c1]\nsunny\n\n[tool_result id=c2]\nrain"
    assert len(texts) == 3


def test_older_blocks_stay_identical_when_history_grows():
    """The cache only hits on an unchanged prefix of whole blocks."""
    base = [{"role": "system", "content": "S"}, {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
             "function": {"name": "get_weather", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "done"}]
    grown = base + [{"role": "assistant", "content": "", "tool_calls": [{"id": "c2",
                     "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "c2", "content": "done again"}]
    _, first = _run({"result": "ok", "usage": {}}, messages=base)
    _, second = _run({"result": "ok", "usage": {}}, messages=grown)
    before, after = _blocks(first), _blocks(second)
    assert after[:len(before)] == before
    assert first.call_args.args[0] == second.call_args.args[0]


def test_datetime_and_volatile_system_messages_move_next_to_the_latest_user_message():
    from backend.agent_runtime.context import DATETIME_LINE_PREFIX

    messages = [
        {"role": "system", "content": f"Rules.\n\n{DATETIME_LINE_PREFIX}Monday, 2026-11-10, "
                                      "09:15:02 (WIB/UTC+7)\n\nYour owner's name is: **ihf**"},
        {"role": "system", "content": "## Agent State\nfocus: task 21"},
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "now"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
         "function": {"name": "get_weather", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
    ]
    _, run = _run({"result": "ok", "usage": {}}, messages=messages)
    system = run.call_args.args[0][run.call_args.args[0].index("--system-prompt") + 1]
    assert system == "Rules.\n\nYour owner's name is: **ihf**"
    texts = [b["text"] for b in _blocks(run)]
    assert texts[:3] == ["[user]\nearlier", "[assistant]\nok", "[user]\nnow"]
    assert texts[3].startswith("[context]\nCurrent date/time: Monday, 2026-11-10, 09:15:02")
    assert "## Agent State\nfocus: task 21" in texts[3]
    assert texts[4].startswith("[assistant tool_call id=c1")


def test_done_false_without_tool_calls_is_retried_once():
    plan = {"result": "", "structured_output": {"text": "Next I will check the weather.",
            "done": False, "tool_calls": []},
            "usage": {"input_tokens": 1, "cache_read_input_tokens": 100, "output_tokens": 3}}
    act = {"result": "", "structured_output": {"text": "", "done": False, "tool_calls": [
           {"name": "get_weather", "arguments": {"city": "Tokyo"}}]},
           "usage": {"input_tokens": 1, "cache_read_input_tokens": 120, "output_tokens": 4}}
    result, run = _run([plan, act], tools=TOOLS)
    assert run.call_count == 2
    first, retry = _blocks(run, 0), _blocks(run, 1)
    assert retry[:len(first)] == first  # same cached prefix
    assert "not executed" in retry[-2]["text"] and "done=false" in retry[-1]["text"]
    choice = result["response"]["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert result["response"]["usage"]["prompt_tokens_details"]["cached_tokens"] == 220
    assert result["diagnostics"]["cli_calls"] == 2


def test_done_true_text_reply_ends_the_turn_without_retry():
    result, run = _run({"result": "", "structured_output": {"text": "All finished.", "done": True,
                        "tool_calls": []}, "usage": {}}, tools=TOOLS)
    assert run.call_count == 1
    assert result["response"]["choices"][0]["finish_reason"] == "stop"
    assert result["response"]["choices"][0]["message"]["content"] == "All finished."


def test_native_tool_attempts_are_reported():
    proc = _proc({"result": "", "structured_output": {"text": "", "done": True, "tool_calls": []},
                  "usage": {}}, assistant_tools=("bash", "StructuredOutput"))
    result, _ = _run([proc], tools=TOOLS)
    assert result["diagnostics"]["native_tool_attempts"] == ["bash"]


def test_errors_are_classified():
    for detail, expected in (("Not logged in · Please run /login", "auth_error"),
                             ("Claude usage limit reached", "rate_limit_error")):
        result, _ = _run([_proc({"is_error": True, "result": detail}, returncode=1)],
                         messages=[{"role": "user", "content": "x"}])
        assert not result["success"] and result["error_type"] == expected


def test_missing_binary_and_timeout():
    with patch.object(claude_cli, "claude_binary", return_value=None):
        assert claude_cli.send_request("m", [{"role": "user", "content": "x"}])["error_type"] == "connection_error"
    with patch.object(claude_cli, "claude_binary", return_value="/usr/bin/claude"), \
         patch("backend.provider.claude_cli.subprocess.run",
               side_effect=subprocess.TimeoutExpired("claude", 1)):
        assert claude_cli.send_request("m", [{"role": "user", "content": "x"}])["error_type"] == "timeout_error"


def test_llm_client_routes_claude_cli_format():
    from backend.llm_client import LLMClient

    client = LLMClient(model_config={"model_name": "claude-sonnet-5-5", "api_format": "claude_cli",
                                     "provider": "claude-code", "base_url": ""})
    fake = {"success": True, "duration_ms": 5, "response": {
        "choices": [{"message": {"role": "assistant", "content": "yo"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4,
                  "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 2}}}}
    # Resolving the model against its seeded "claude-code" provider must not
    # coerce the direct CLI format into an HTTP adapter (which would send the
    # request to an empty OpenAI URL).
    assert client.api_format == "claude_cli"
    with patch("backend.provider.claude_cli.send_request", return_value=fake) as send, \
         patch("backend.llm_usage_events.record_llm_usage") as record:
        result = client.chat_completion([{"role": "user", "content": "hi"}])
    assert send.called
    assert result["success"] and client.extract_content(result) == "yo"
    assert record.call_args.kwargs["cache_write_tokens"] == 2


def test_effort_follows_thinking_budget():
    assert claude_cli.resolve_effort(False, 50_000) == "low"
    assert claude_cli.resolve_effort(True, 0) == "medium"
    assert claude_cli.resolve_effort(True, 4_000) == "medium"
    assert claude_cli.resolve_effort(True, 16_000) == "high"
    assert claude_cli.resolve_effort(True, 100_000) == "max"
    _, run = _run({"result": "ok", "usage": {}}, thinking=True, thinking_budget=16_000)
    cmd = run.call_args.args[0]
    assert cmd[cmd.index("--effort") + 1] == "high"
    assert run.call_args.kwargs["env"]["DISABLE_AUTO_COMPACT"] == "1"


def test_tool_instructions_forbid_native_calls_and_plan_only_replies():
    text = claude_cli._tool_instructions(
        [{"type": "function", "function": {"name": "bash", "description": "", "parameters": {}}}], None)
    assert "never invoke them as native tool calls" in text
    assert "A reply without tool_calls ends your turn" in text
    assert "done" in claude_cli._OUTPUT_SCHEMA["required"]


def test_images_are_sent_in_place_as_blocks():
    url = "data:image/png;base64,QUJD"
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": url}}]}]
    result, run = _run({"result": "a dashboard", "usage": {}}, messages=messages)
    assert result["response"]["choices"][0]["message"]["content"] == "a dashboard"
    blocks = _blocks(run)
    assert blocks[0]["source"] == {"type": "base64", "media_type": "image/png", "data": "QUJD"}
    assert "[image 1 attached]" in blocks[1]["text"]


def _turn(messages, result=None):
    result = result or {"result": "ok", "usage": {"cache_read_input_tokens": 50,
                                                   "cache_creation_input_tokens": 5}}
    with usage_context("agent_turn", "hurayrah", "Hurayrah", "s1"):
        out, _ = _run(result, messages=messages)
    return out["diagnostics"]["cache"]


def test_cache_diagnosis_explains_each_kind_of_miss():
    from backend.agent_runtime.context import DATETIME_LINE_PREFIX

    def system(clock, kb="a.md"):
        return {"role": "system", "content": f"Rules. KB: {kb}\n\n{DATETIME_LINE_PREFIX}{clock}"}

    step = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
             "function": {"name": "get_weather", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]

    assert _turn([system("09:00"), step[0]])["reasons"] == ["first_call"]
    grown = _turn([system("09:00")] + step)
    assert grown["reasons"] == [] and grown["reused_blocks"] == 2
    next_turn = _turn([system("09:05")] + step + [{"role": "user", "content": "more"}])
    assert next_turn["reasons"] == ["context_moved"]
    kb_changed = _turn([system("09:05", kb="a.md, b.md")] + step + [{"role": "user", "content": "more"}])
    assert kb_changed["reasons"] == ["system_prompt_changed"]
    rewritten = _turn([system("09:05", kb="a.md, b.md"), {"role": "user", "content": "[summary] go"}])
    assert rewritten["reasons"] == ["history_rewritten_at_block_0"]


def test_conversations_are_keyed_by_usage_context():
    msgs = [{"role": "user", "content": "hi"}]
    with usage_context("agent_turn", "a1", "A", "s1"):
        _run({"result": "ok", "usage": {}}, messages=msgs)
    with usage_context("summarizer", "a1", "A", "s1"):
        out, _ = _run({"result": "ok", "usage": {}}, messages=msgs)
    assert out["diagnostics"]["cache"]["reasons"] == ["first_call"]
    assert set(claude_cli._conversations) == {"agent_turn|a1|s1|claude-sonnet-5-5",
                                              "summarizer|a1|s1|claude-sonnet-5-5"}
