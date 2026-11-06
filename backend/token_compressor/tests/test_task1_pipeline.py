"""Task #1 regression tests: RTK compression pipeline fixes."""
import json

from backend.token_compressor.compressor_registry import get_registry
from backend.token_compressor.extract_command import extract_command
from backend.token_compressor.filter_pipeline import fallback_compress


def _compress_dict(cmd, result):
    reg = get_registry()
    out = dict(result)
    for k in ("stdout", "stderr", "result", "content"):
        if isinstance(out.get(k), str) and out[k]:
            out[k] = reg.compress(cmd, result.get("exit_code", 0), out[k])
    return json.dumps(out)


def test_dict_result_stays_valid_json_same_keys():
    res = {"stdout": "\n".join(f"line {i}" for i in range(1000)), "stderr": "", "exit_code": 0, "execution_time": 0.1}
    s = _compress_dict("some_tool", res)
    d = json.loads(s)
    assert set(d) == set(res)
    assert len(d["stdout"]) < len(res["stdout"])


def test_cd_grep_resolves_to_grep():
    assert extract_command("bash", {"script": "cd /x && grep -rn foo ."}).startswith("grep")


def test_env_assign_and_pipe_resolves_git_status():
    assert extract_command("bash", {"script": "export A=1\nFOO=1 git status | head"}) == "git status"


def test_only_meta_resolves_bash():
    assert extract_command("bash", {"script": "cd x"}) == "bash"


def test_nonzero_exit_keeps_error_lines():
    lines = [f"noise {i}" for i in range(400)]
    lines[50:50] = ["Traceback (most recent call last):", '  File "a.py", line 1, in <module>', "ValueError: boom"]
    lines.append("FAILED tests/test_x.py::test_y")
    raw = "\n".join(lines)
    out = get_registry().compress("unknown_cmd", 1, raw)
    for must in ("Traceback (most recent call last):", "ValueError: boom", "FAILED tests/test_x.py::test_y"):
        assert must in out
    assert len(out) < len(raw)


def test_fallback_keeps_tail_and_marker():
    raw = "\n".join(f"row {i}" for i in range(500))
    out = fallback_compress(raw)
    tail = raw.split("\n")[-10:]
    assert all(t in out.split("\n") for t in tail)
    assert "lines omitted" in out


def test_fallback_strips_ansi_and_dupes():
    out = fallback_compress("\x1b[31mred\x1b[0m\nsame\nsame\nsame\nend")
    assert "\x1b" not in out and out.count("same") == 1 and out.endswith("end")


def test_small_output_unchanged_by_fallback():
    assert fallback_compress("short\noutput") == "short\noutput"
