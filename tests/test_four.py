"""Tests for four-function algebra."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from four.core import Err, Ok, Result, run, save_trajectory
from four.parse import regex_parse, toolcall_parse
from four.env import local_env
from four.core import retry_invoke, AbortError


# ── Helpers ─────────────────────────────────────────────────────────────────


def _run_and_check(
    G,
    tmpdir: str,
    max_steps: int = 5,
    system: str = "test",
    prompt: str = "Say hello",
    max_format_errors: int = 3,
    check=None,
    expected_outcome=None,
):
    """Run the loop and return parsed trajectory inside temp dir context."""
    path = run(
        G=G,
        V1=regex_parse(),
        V2=local_env(),
        emit=save_trajectory(tmpdir),
        system=system,
        prompt=prompt,
        max_steps=max_steps,
        max_format_errors=max_format_errors,
    )

    data = json.loads(path.read_text())

    if expected_outcome:
        assert data["outcome"] == expected_outcome

    if check:
        check(data)

    return data


# ── Ok / Err types ──────────────────────────────────────────────────────────


class TestOkErr:
    def test_ok_wraps_value(self):
        ok = Ok("hello")
        assert ok.value == "hello"

    def test_err_wraps_error(self):
        err = Err("boom")
        assert err.error == "boom"

    def test_isinstance_checks(self):
        assert isinstance(Ok("x"), Ok)
        assert isinstance(Err("y"), Err)
        assert not isinstance(Ok("x"), Err)
        assert not isinstance(Err("y"), Ok)


# ── regex_parse ─────────────────────────────────────────────────────────────


class TestRegexParse:
    def test_single_command(self):
        p = regex_parse()
        raw = "```mswea_bash_command\necho hello\n```"
        result = p(raw)
        assert isinstance(result, Ok)
        assert result.value == ["echo hello"]

    def test_empty_response(self):
        """No code block = plain text = task complete."""
        p = regex_parse()
        result = p("no commands here")
        assert isinstance(result, Err)
        assert result.error == "exit:task_complete"

    def test_multiple_commands(self):
        """Multiple blocks → returns all commands as a list."""
        p = regex_parse()
        raw = (
            "```mswea_bash_command\necho one\n```\n"
            "```mswea_bash_command\necho two\n```"
        )
        result = p(raw)
        assert isinstance(result, Ok)
        assert result.value == ["echo one", "echo two"]

    def test_custom_pattern(self):
        p = regex_parse(pattern=r"```bash\n(.*?)\n```")
        raw = "```bash\necho hi\n```"
        result = p(raw)
        assert isinstance(result, Ok)
        assert result.value == ["echo hi"]

    def test_accepts_bash_blocks(self):
        """Model uses ```bash``` instead of ```mswea_bash_command``` — still works."""
        p = regex_parse()
        raw = "```bash\necho hello\n```"
        result = p(raw)
        assert isinstance(result, Ok)
        assert result.value == ["echo hello"]

    def test_accepts_sh_blocks(self):
        p = regex_parse()
        raw = "```sh\necho hi\n```"
        result = p(raw)
        assert isinstance(result, Ok)
        assert result.value == ["echo hi"]

    def test_thought_with_bash_block(self):
        """Model includes THOUGHT before the code block."""
        p = regex_parse()
        raw = "THOUGHT: I will run this command\n\n```bash\necho hello\n```"
        result = p(raw)
        assert isinstance(result, Ok)
        assert result.value == ["echo hello"]

    def test_multiline_command(self):
        p = regex_parse()
        raw = "```mswea_bash_command\ncat <<'EOF'\nline1\nline2\nEOF\n```"
        result = p(raw)
        assert isinstance(result, Ok)
        assert "line1" in result.value[0]
        assert "line2" in result.value[0]


# ── toolcall_parse ──────────────────────────────────────────────────────────


class TestToolcallParse:
    def test_single_bash_call(self):
        p = toolcall_parse()
        raw = json.dumps([{
            "tool_call_id": "1",
            "name": "bash",
            "arguments": json.dumps({"command": "echo hi"}),
        }])
        result = p(raw)
        assert isinstance(result, Ok)
        assert result.value == [{"command": "echo hi", "tool_call_id": "1"}]

    def test_multiple_bash_calls(self):
        """Multiple bash tool calls → returns all commands."""
        p = toolcall_parse()
        raw = json.dumps([{
            "tool_call_id": "1",
            "name": "bash",
            "arguments": json.dumps({"command": "echo one"}),
        }, {
            "tool_call_id": "2",
            "name": "bash",
            "arguments": json.dumps({"command": "echo two"}),
        }])
        result = p(raw)
        assert isinstance(result, Ok)
        assert result.value == [
            {"command": "echo one", "tool_call_id": "1"},
            {"command": "echo two", "tool_call_id": "2"},
        ]

    def test_no_bash_tool(self):
        p = toolcall_parse()
        raw = json.dumps([{
            "tool_call_id": "1",
            "name": "python",
            "arguments": json.dumps({"code": "print(1)"}),
        }])
        result = p(raw)
        assert isinstance(result, Err)
        assert "No bash" in result.error

    def test_invalid_json(self):
        """Non-JSON = plain text = task complete."""
        p = toolcall_parse()
        result = p("not json at all")
        assert isinstance(result, Err)
        assert result.error == "exit:task_complete"

    def test_empty_array(self):
        p = toolcall_parse()
        result = p("[]")
        assert isinstance(result, Err)
        assert "No tool calls" in result.error

    def test_invalid_arguments_json(self):
        p = toolcall_parse()
        raw = json.dumps([{
            "tool_call_id": "1",
            "name": "bash",
            "arguments": "not valid json",
        }])
        result = p(raw)
        assert isinstance(result, Err)
        assert "Invalid bash" in result.error


# ── local_env ───────────────────────────────────────────────────────────────


class TestLocalEnv:
    def test_simple_command(self):
        v = local_env()
        result = v("echo hello")
        assert isinstance(result, Ok)
        assert "hello" in result.value["content"]

    def test_return_code_zero(self):
        v = local_env()
        result = v("true")
        assert isinstance(result, Ok)
        assert "<returncode>0</returncode>" in result.value["content"]

    def test_nonzero_return_code(self):
        v = local_env()
        result = v("false")
        assert isinstance(result, Ok)
        assert "<returncode>1</returncode>" in result.value["content"]

    def test_exit_signal(self):
        v = local_env()
        result = v("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
        assert isinstance(result, Err)
        assert result.error == "exit:task_complete"

    def test_custom_exit_signal(self):
        v = local_env(exit_signal="DONE")
        result = v("echo DONE")
        assert isinstance(result, Err)
        assert result.error == "exit:task_complete"

    def test_timeout(self):
        v = local_env(timeout=1)
        result = v("sleep 5")
        assert isinstance(result, Err)
        assert result.error == "timeout"

    def test_output_truncation(self):
        v = local_env(max_output=100)
        long_cmd = "python3 -c \"print('x' * 200)\""
        result = v(long_cmd)
        assert isinstance(result, Ok)
        assert "chars elided" in result.value["content"]

    def test_stderr_capture(self):
        v = local_env()
        result = v("python3 -c \"import sys; print('err', file=sys.stderr); exit(1)\"")
        assert isinstance(result, Ok)
        assert "err" in result.value["content"]


# ── save_trajectory (emit) ─────────────────────────────────────────────────


class TestSaveTrajectory:
    def test_saves_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            emit = save_trajectory(tmpdir)
            messages = [{"role": "user", "content": "hi"}]
            path = emit(messages, "ok")

            assert path.exists()
            data = json.loads(path.read_text())
            assert data["outcome"] == "ok"
            assert data["messages"] == messages

    def test_incremental_naming(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            emit = save_trajectory(tmpdir)
            emit([{"role": "user"}], "a")
            emit([{"role": "assistant"}], "b")

            files = sorted(Path(tmpdir).glob("*.json"))
            assert len(files) == 2
            assert "0000" in files[0].name
            assert "0001" in files[1].name


# ── retry_invoke ────────────────────────────────────────────────────────────


class TestRetryInvoke:
    def test_success_no_retry(self):
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            return Ok("success")

        wrapped = retry_invoke(mock_invoke, max_attempts=3)
        result = wrapped([])
        assert isinstance(result, Ok)
        assert call_count == 1

    def test_retries_on_transient_error(self):
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return Err("RateLimitError: too many requests")
            return Ok("success")

        wrapped = retry_invoke(mock_invoke, max_attempts=5)
        result = wrapped([])
        assert isinstance(result, Ok)
        assert call_count == 3

    def test_aborts_on_auth_error(self):
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            return Err("InvalidAPIKeyError: bad key")

        wrapped = retry_invoke(mock_invoke, max_attempts=5)
        result = wrapped([])
        assert isinstance(result, Err)
        assert "abort" in result.error
        assert call_count == 1

    def test_exhausts_retries(self):
        call_count = 0

        def mock_invoke(messages):
            nonlocal call_count
            call_count += 1
            return Err("RateLimitError: too many requests")

        wrapped = retry_invoke(mock_invoke, max_attempts=3)
        result = wrapped([])
        assert isinstance(result, Err)
        assert "retry_exhausted" in result.error


# ── The loop (run) ─────────────────────────────────────────────────────────


class TestRun:
    def test_successful_single_step(self):
        """G returns valid command → V1 parses → V2 executes → done."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _run_and_check(
                G=lambda msgs: Ok("```mswea_bash_command\necho hello\n```"),
                tmpdir=tmpdir,
            )
        tools = [m for m in data["messages"] if m.get("role") == "tool"]
        assert len(tools) >= 1
        assert "hello" in tools[-1]["content"]

    def test_multiple_actions(self):
        """V1 returns multiple commands → all are executed in one step."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _run_and_check(
                G=lambda msgs: Ok(
                    "```bash\necho first\n```\n```bash\necho second\n```"
                ),
                tmpdir=tmpdir,
                max_steps=1,
            )
        tools = [m for m in data["messages"] if m.get("role") == "tool"]
        assert len(tools) == 2
        assert "first" in tools[0]["content"]
        assert "second" in tools[1]["content"]

    def test_plain_text_stops(self):
        """V1 returns exit:task_complete (plain text) → loop stops immediately."""
        call_count = 0

        def mock_G(messages):
            nonlocal call_count
            call_count += 1
            return Ok("this is a final answer")

        with tempfile.TemporaryDirectory() as tmpdir:
            data = _run_and_check(
                G=mock_G,
                tmpdir=tmpdir,
                max_steps=3,
                expected_outcome="exit:task_complete",
            )

        assert call_count == 1

    def test_repeated_format_error_stops(self):
        """Consecutive non-exit format errors beyond max → abort."""
        call_count = 0

        def failing_parse(raw: str):
            from four.core import Err
            return Err("malformed response")

        def mock_G(messages):
            nonlocal call_count
            call_count += 1
            return Ok("anything")

        path = run(
            G=mock_G,
            V1=failing_parse,
            V2=local_env(),
            emit=save_trajectory(tempfile.mkdtemp()),
            system="test",
            prompt="Say hello",
            max_steps=10,
            max_format_errors=3,
        )
        data = json.loads(path.read_text())
        assert data["outcome"] == "repeated_format_error: malformed response"
        # 2 format errors appended, 3rd triggers abort (not appended)
        format_msgs = [m for m in data["messages"] if "Format error" in m.get("content", "")]
        assert len(format_msgs) == 2
        assert call_count == 3

    def test_model_error_stops(self):
        """G returns Err → loop stops immediately."""
        call_count = 0

        def mock_G(messages):
            nonlocal call_count
            call_count += 1
            return Err("API down")

        with tempfile.TemporaryDirectory() as tmpdir:
            _run_and_check(
                G=mock_G,
                tmpdir=tmpdir,
                expected_outcome="model_error: API down",
            )

        assert call_count == 1

    def test_exit_signal_stops(self):
        """V2 returns exit → loop stops immediately."""
        call_count = 0

        def mock_G(messages):
            nonlocal call_count
            call_count += 1
            return Ok("```mswea_bash_command\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```")

        with tempfile.TemporaryDirectory() as tmpdir:
            _run_and_check(
                G=mock_G,
                tmpdir=tmpdir,
                expected_outcome="exit:task_complete",
            )

        assert call_count == 1

    def test_max_steps_reached(self):
        """Exceeds max_steps → stops with timeout outcome."""
        call_count = 0

        def mock_G(messages):
            nonlocal call_count
            call_count += 1
            return Ok("```mswea_bash_command\necho step\n```")

        with tempfile.TemporaryDirectory() as tmpdir:
            _run_and_check(
                G=mock_G,
                tmpdir=tmpdir,
                max_steps=3,
                expected_outcome="max_steps_reached",
            )

        assert call_count == 3

    def test_messages_accumulate(self):
        """Each step appends to messages — history grows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = _run_and_check(
                G=lambda msgs: Ok("```mswea_bash_command\necho step\n```"),
                tmpdir=tmpdir,
                max_steps=2,
            )

        # system + user prompt + 2 assistant messages + 2 tool observations
        assert len(data["messages"]) == 6


# ── Integration ─────────────────────────────────────────────────────────────


class TestIntegration:
    def test_full_pipeline_no_llm(self):
        """Simulate a complete agent loop with mock G."""
        responses = [
            "```mswea_bash_command\necho first\n```",
            "```mswea_bash_command\necho second\n```",
        ]
        idx = 0

        def mock_G(messages):
            nonlocal idx
            resp = responses[idx] if idx < len(responses) else None
            idx += 1
            return Ok(resp) if resp else Err("stopped")

        with tempfile.TemporaryDirectory() as tmpdir:
            data = _run_and_check(
                G=mock_G,
                tmpdir=tmpdir,
                max_steps=5,
            )

        tools = [m for m in data["messages"] if m.get("role") == "tool"]
        assert len(tools) == 2
        assert "first" in tools[0]["content"]
        assert "second" in tools[1]["content"]

    def test_format_error_then_recovery(self):
        """V1 returns non-exit error, then model recovers on next step."""
        responses = [
            "```bash\necho",  # malformed — no closing block, triggers exit:task_complete
            "```bash\necho recovered\n```",
            "```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
        ]
        idx = 0

        def mock_G(messages):
            nonlocal idx
            resp = responses[idx] if idx < len(responses) else None
            idx += 1
            return Ok(resp) if resp else Err("stopped")

        with tempfile.TemporaryDirectory() as tmpdir:
            data = _run_and_check(
                G=mock_G,
                tmpdir=tmpdir,
                max_steps=5,
                max_format_errors=3,
                expected_outcome="exit:task_complete",
            )

        # First response triggers immediate exit (no format error retry)
        # system + user + assistant = 3
        assert len(data["messages"]) == 3
