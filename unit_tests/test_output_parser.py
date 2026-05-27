"""
Unit tests for backend/agent_runtime/output_parser.py — covers the cases
that the old regex couldn't handle (empty arguments, nested arguments)
plus the new promotion path used for small models.
"""

import json
import pytest

from backend.agent_runtime.output_parser import (
    NUDGE_PREFIX,
    build_nudge_message,
    count_prior_nudges,
    detect_all,
    detect_bare_json_tool_calls,
    detect_fenced_tool_blocks,
    detect_xml_tool_calls,
    has_malformed_calls,
    is_small_model,
    promote_to_tool_calls,
    strip_extracted_calls,
)


# ─── detect_bare_json_tool_calls ────────────────────────────────────────

def test_bare_json_empty_arguments():
    """Old regex `[^}]+` rejected empty `{}` — balanced scanner must accept it."""
    text = 'sure\n{"name": "koans_get_challenge", "arguments": {}}'
    results = detect_bare_json_tool_calls(text)
    assert len(results) == 1
    assert results[0]['parsed'] == {'name': 'koans_get_challenge', 'arguments': {}}


def test_bare_json_nested_arguments():
    """Old regex captured up to the first `}` and produced invalid JSON."""
    text = '{"name": "hello_world", "arguments": {"name": "John Doe"}}'
    results = detect_bare_json_tool_calls(text)
    assert len(results) == 1
    assert results[0]['parsed']['arguments'] == {'name': 'John Doe'}


def test_bare_json_multi_arg_flat():
    text = (
        '{"name": "koans_register", "arguments": '
        '{"agent_name": "AI Agent", "model": "Qwen", "force": false}}'
    )
    results = detect_bare_json_tool_calls(text)
    assert len(results) == 1
    assert results[0]['parsed']['arguments']['force'] is False


def test_bare_json_double_brace_wrapper():
    """Some models emit `{{...}}` (jinja artefacts). Skip the bad outer brace,
    pick up the inner valid JSON."""
    text = '{{"name": "x", "arguments": {}}}'
    results = detect_bare_json_tool_calls(text)
    assert len(results) == 1
    assert results[0]['parsed'] == {'name': 'x', 'arguments': {}}


def test_bare_json_ignores_unrelated_objects():
    text = 'note: the row is {"foo": "bar", "baz": 42}, fyi.'
    assert detect_bare_json_tool_calls(text) == []


def test_bare_json_skips_content_inside_xml_tags():
    """XML <tool_call> blocks are handled by detect_xml_tool_calls; the bare
    detector must not double-count them."""
    text = '<tool_call>{"name":"a","arguments":{}}</tool_call>'
    assert detect_bare_json_tool_calls(text) == []
    assert len(detect_xml_tool_calls(text)) == 1


def test_bare_json_string_with_brace_does_not_break_scan():
    """A `}` inside a string value must not close the JSON early."""
    text = '{"name": "x", "arguments": {"sql": "SELECT \\"a}\\" FROM t"}}'
    results = detect_bare_json_tool_calls(text)
    assert len(results) == 1
    assert results[0]['parsed']['arguments']['sql'] == 'SELECT "a}" FROM t'


# ─── has_malformed_calls ─────────────────────────────────────────────────

def test_has_malformed_calls_positive_cases():
    assert has_malformed_calls('{"name": "x", "arguments": {}}') is True
    assert has_malformed_calls('<tool_call>foo</tool_call>') is True
    assert has_malformed_calls('```tool\n{"name":"x"}\n```') is True


def test_has_malformed_calls_negative_cases():
    assert has_malformed_calls('') is False
    assert has_malformed_calls('plain text') is False
    assert has_malformed_calls('here is config: {"foo": 1}') is False


# ─── promote_to_tool_calls ───────────────────────────────────────────────

def test_promote_basic_shape():
    extracted = detect_bare_json_tool_calls(
        '{"name": "ping", "arguments": {"host": "localhost"}}'
    )
    promoted = promote_to_tool_calls(extracted)
    assert len(promoted) == 1
    tc = promoted[0]
    assert tc['type'] == 'function'
    assert tc['function']['name'] == 'ping'
    assert json.loads(tc['function']['arguments']) == {'host': 'localhost'}
    assert tc['id'].startswith('call_')


def test_promote_dedupes_identical_calls():
    """Same name + same args should only execute once."""
    extracted = [
        {'parsed': {'name': 'foo', 'arguments': {'x': 1}}, 'raw': 'a'},
        {'parsed': {'name': 'foo', 'arguments': {'x': 1}}, 'raw': 'b'},
        {'parsed': {'name': 'foo', 'arguments': {'x': 2}}, 'raw': 'c'},
    ]
    promoted = promote_to_tool_calls(extracted)
    assert len(promoted) == 2


def test_promote_skips_entries_without_valid_parse():
    extracted = [
        {'parsed': None, 'raw': '...'},
        {'parsed': {'name': '', 'arguments': {}}, 'raw': '...'},
        {'parsed': {'name': 'ok', 'arguments': 'not-a-dict'}, 'raw': '...'},
        {'parsed': {'name': 'good', 'arguments': {}}, 'raw': '...'},
    ]
    promoted = promote_to_tool_calls(extracted)
    assert [t['function']['name'] for t in promoted] == ['good']


def test_promote_accepts_parameters_alias():
    """Some models use `parameters` instead of `arguments`."""
    extracted = [{'parsed': {'name': 'foo', 'parameters': {'x': 1}}, 'raw': '...'}]
    promoted = promote_to_tool_calls(extracted)
    assert len(promoted) == 1
    assert json.loads(promoted[0]['function']['arguments']) == {'x': 1}


# ─── is_small_model ──────────────────────────────────────────────────────

@pytest.mark.parametrize('name', [
    'Qwen2.5-Coder 1.5B Instruct',
    'qwen2.5-coder-1.5b-instruct',
    'llama-3.2-1b-instruct',
    'qwen3-1.7b',
    'qwen2.5-coder-1_5b',
    'SmolLM2-360M',
    'TinyLlama-1.1B-Chat',
    'qwen2.5-3b-instruct',
])
def test_is_small_model_positive(name):
    assert is_small_model(name) is True


@pytest.mark.parametrize('name', [
    'qwen2.5-7b-instruct',
    'claude-opus-4-7',
    'gpt-4o',
    'phi-3-mini-3.8b-instruct',  # 3.8B > 3B threshold
    None,
    '',
])
def test_is_small_model_negative(name):
    assert is_small_model(name) is False


# ─── strip_extracted_calls ───────────────────────────────────────────────

def test_strip_removes_extracted_spans():
    raw = '{"name": "x", "arguments": {}}'
    text = f'prefix {raw} suffix'
    extracted = detect_bare_json_tool_calls(text)
    cleaned = strip_extracted_calls(text, extracted)
    assert cleaned == 'prefix  suffix'.strip()
    assert raw not in cleaned


def test_strip_no_op_when_no_extractions():
    assert strip_extracted_calls('hello', []) == 'hello'
    assert strip_extracted_calls('', []) == ''


def test_strip_removes_empty_json_code_fence():
    """Qwen often wraps bare JSON in ```json ... ```. After we remove the
    JSON body, the empty fence wrapper must be cleaned up too."""
    text = '```json\n{"name": "hello_world", "arguments": {"name": "Mudir"}}\n```'
    extracted = detect_bare_json_tool_calls(text)
    assert len(extracted) == 1
    cleaned = strip_extracted_calls(text, extracted)
    assert cleaned == ''


def test_strip_preserves_non_empty_text_around_call():
    text = 'thinking...\n```json\n{"name": "x", "arguments": {}}\n```\nthen done.'
    extracted = detect_bare_json_tool_calls(text)
    cleaned = strip_extracted_calls(text, extracted)
    assert 'thinking...' in cleaned
    assert 'then done.' in cleaned
    assert '```' not in cleaned
    assert '{"name"' not in cleaned


# ─── count_prior_nudges ──────────────────────────────────────────────────

def test_count_prior_nudges_matches_only_nudge_prefix():
    msgs = [
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': NUDGE_PREFIX + ' (a)'},  # assistant ≠ nudge
        {'role': 'user', 'content': NUDGE_PREFIX + ' (b)'},
        {'role': 'user', 'content': NUDGE_PREFIX + ' (c)'},
        {'role': 'user', 'content': 'normal user msg'},
    ]
    assert count_prior_nudges(msgs) == 2


def test_build_nudge_message_starts_with_prefix():
    """count_prior_nudges relies on this — keep them in sync."""
    msg = build_nudge_message([
        {'format': 'bare_json', 'raw': '{"name":"x","arguments":{}}',
         'content': '{"name":"x","arguments":{}}',
         'parsed': {'name': 'x', 'arguments': {}}}
    ])
    assert msg.startswith(NUDGE_PREFIX)
