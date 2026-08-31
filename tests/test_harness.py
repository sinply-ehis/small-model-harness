"""Tests for small-model-harness standalone package."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from small_model_harness import (
    ExecutionRecord,
    ExecutionStatus,
    FileInput,
    HarnessState,
    compact_text,
    compact_tool_response,
    create_harness_session,
    detect_failure_loop,
    estimate_messages_tokens,
    estimate_tokens,
    format_audit_trail,
    generate_session_summary,
    process_file_input,
    process_multiple_files,
    read_text_file,
    validate_file_input,
    validate_result_quality,
)


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------


class TestSessionCreation:
    def test_default_session(self):
        session = create_harness_session()
        assert session.session_id
        assert session.budget_remaining == 100
        assert session.total_tool_calls == 0
        assert session.success_rate == 1.0

    def test_custom_session_id(self):
        session = create_harness_session(session_id="test-123")
        assert session.session_id == "test-123"

    def test_budget_from_n_ctx(self):
        session = create_harness_session(n_ctx=4096)
        assert session.budget_remaining == 8  # 4096 // 500 = 8

    def test_budget_floor(self):
        session = create_harness_session(n_ctx=100)
        assert session.budget_remaining == 5  # max(5, 100//500)

    def test_budget_ceiling(self):
        session = create_harness_session(n_ctx=100000)
        assert session.budget_remaining == 100  # min(100, ...)


# ---------------------------------------------------------------------------
# Tool stats and steering
# ---------------------------------------------------------------------------


class TestToolStats:
    def _record(self, session: HarnessState, tool: str, status: ExecutionStatus):
        session.record_execution(
            ExecutionRecord(
                timestamp="2026-01-01T00:00:00Z",
                tool_name=tool,
                arguments={},
                status=status,
            )
        )

    def test_success_counts(self):
        s = create_harness_session()
        self._record(s, "search", ExecutionStatus.COMPLETED)
        self._record(s, "search", ExecutionStatus.COMPLETED)
        assert s.tool_success_counts["search"] == 2
        assert s.successful_calls == 2
        assert s.success_rate == 1.0

    def test_failure_counts(self):
        s = create_harness_session()
        self._record(s, "search", ExecutionStatus.FAILED)
        self._record(s, "search", ExecutionStatus.FAILED)
        assert s.tool_failure_counts["search"] == 2
        assert s.failed_calls == 2
        assert s.success_rate == 0.0

    def test_steering_hints_after_failures(self):
        s = create_harness_session()
        self._record(s, "bad_tool", ExecutionStatus.FAILED)
        self._record(s, "bad_tool", ExecutionStatus.FAILED)
        assert "bad_tool" in s.avoided_tools
        assert any("Do NOT use bad_tool" in h for h in s.steering_hints)

    def test_steering_prompt_format(self):
        s = create_harness_session()
        self._record(s, "bad_tool", ExecutionStatus.FAILED)
        self._record(s, "bad_tool", ExecutionStatus.FAILED)
        prompt = s.get_steering_prompt()
        assert "# Session Guidance" in prompt
        assert "Do NOT use bad_tool" in prompt

    def test_no_steering_when_clean(self):
        s = create_harness_session()
        assert s.get_steering_prompt() == ""
        assert s.get_avoided_tools_prompt() == ""

    def test_budget_decrements(self):
        s = create_harness_session(budget=5)
        self._record(s, "a", ExecutionStatus.COMPLETED)
        self._record(s, "b", ExecutionStatus.COMPLETED)
        assert s.budget_remaining == 3
        assert not s.is_budget_exhausted
        self._record(s, "c", ExecutionStatus.COMPLETED)
        self._record(s, "d", ExecutionStatus.COMPLETED)
        self._record(s, "e", ExecutionStatus.COMPLETED)
        assert s.is_budget_exhausted


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------


class TestLoopDetection:
    def test_no_loop_short_list(self):
        assert detect_failure_loop(["a", "b"]) is None

    def test_loop_detected(self):
        assert detect_failure_loop(["a", "a", "a"]) == "a"

    def test_no_loop_different_tools(self):
        assert detect_failure_loop(["a", "b", "c"]) is None

    def test_loop_custom_window(self):
        assert detect_failure_loop(["a", "a", "a", "a"], window=4) == "a"
        assert detect_failure_loop(["a", "a", "a"], window=4) is None


# ---------------------------------------------------------------------------
# Context pressure
# ---------------------------------------------------------------------------


class TestContextPressure:
    def test_pressure_zero(self):
        s = create_harness_session()
        assert s.context_pressure == 0.0
        assert not s.is_context_warned
        assert not s.is_context_critical

    def test_pressure_warn(self):
        s = create_harness_session()
        s.update_context_pressure(3000, 4000)  # 0.75
        assert s.is_context_warned
        assert not s.is_context_critical
        assert any("Context is getting full" in h for h in s.steering_hints)

    def test_pressure_critical(self):
        s = create_harness_session()
        s.update_context_pressure(3700, 4000)  # 0.925
        assert s.is_context_critical
        assert any("Context is nearly full" in h for h in s.steering_hints)


# ---------------------------------------------------------------------------
# Response compaction
# ---------------------------------------------------------------------------


class TestResponseCompaction:
    def test_short_response_not_compacted(self):
        result, was_compacted = compact_tool_response("search", "hello", max_tokens=100)
        assert result == "hello"
        assert not was_compacted

    def test_empty_response(self):
        result, was_compacted = compact_tool_response("search", "", max_tokens=100)
        assert result == ""
        assert not was_compacted

    def test_error_not_compacted(self):
        err = "Error: tool failed"
        result, was_compacted = compact_tool_response("search", err, max_tokens=1)
        assert result == err
        assert not was_compacted

    def test_json_audio_compaction(self):
        data = {"audio_file": "/tmp/out.wav", "duration": 5.0, "sample_rate": 44100, "extra": "verbose"}
        response = json.dumps(data)
        result, was_compacted = compact_tool_response("speak", response, max_tokens=20)
        assert was_compacted
        compacted = json.loads(result)
        assert "audio_file" in compacted
        assert "extra" not in compacted

    def test_json_list_compaction(self):
        data = {"profiles": [{"id": str(i), "name": f"v{i}"} for i in range(20)]}
        response = json.dumps(data)
        result, was_compacted = compact_tool_response("list_voices", response, max_tokens=30)
        assert was_compacted
        compacted = json.loads(result)
        assert len(compacted["profiles"]) == 10
        assert compacted["profiles_count"] == 20

    def test_generic_truncation(self):
        long = "x" * 5000
        result, was_compacted = compact_tool_response("misc", long, max_tokens=100)
        assert was_compacted
        assert "[... truncated ...]" in result


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    def test_estimate_tokens(self):
        assert estimate_tokens("") == 1
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("a" * 100) == 25

    def test_estimate_messages_tokens(self):
        msgs = [{"content": "hello world"}, {"content": "test"}]
        total = estimate_messages_tokens(msgs)
        assert total > 0


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------


class TestFileHandling:
    def test_validate_txt(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = validate_file_input(f)
        assert result.is_supported
        assert result.is_text_based
        assert not result.needs_transcription
        assert result.error is None

    def test_validate_unsupported(self, tmp_path):
        f = tmp_path / "test.xyz"
        f.write_text("hello")
        result = validate_file_input(f)
        assert not result.is_supported
        assert result.error is not None

    def test_read_text_file(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Hello\n\nWorld")
        content = read_text_file(f)
        assert "# Hello" in content

    def test_compact_text_short(self):
        content = "short text"
        result, was_truncated = compact_text(content, max_tokens=1000)
        assert result == content
        assert not was_truncated

    def test_compact_text_long(self):
        content = "word " * 5000
        result, was_truncated = compact_text(content, max_tokens=10)
        assert was_truncated
        assert "[... content truncated ...]" in result

    def test_process_file_input(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = process_file_input(f)
        assert result.content == "hello world"
        assert result.compacted

    def test_process_multiple_files(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("aaa")
        f2.write_text("bbb")
        results = process_multiple_files([f1, f2])
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Result quality validation
# ---------------------------------------------------------------------------


class TestResultQuality:
    def test_valid_result(self):
        is_valid, issues = validate_result_quality("search", "Found 10 results")
        assert is_valid
        assert issues == []

    def test_empty_result(self):
        is_valid, issues = validate_result_quality("search", "")
        assert not is_valid
        assert "Empty result" in issues[0]

    def test_error_result(self):
        is_valid, issues = validate_result_quality("search", "Error timeout")
        assert not is_valid

    def test_json_expected(self):
        is_valid, issues = validate_result_quality("search", "not json", expected_type="json")
        assert not is_valid
        assert "JSON" in issues[0]

    def test_json_valid(self):
        is_valid, issues = validate_result_quality("search", '{"key": "value"}', expected_type="json")
        assert is_valid


# ---------------------------------------------------------------------------
# Audit trail and summary
# ---------------------------------------------------------------------------


class TestAuditTrail:
    def test_format_audit_trail(self):
        s = create_harness_session(session_id="abc")
        s.record_execution(
            ExecutionRecord(
                timestamp="2026-01-01T00:00:00Z",
                tool_name="search",
                arguments={"q": "test"},
                status=ExecutionStatus.COMPLETED,
                duration_ms=100.0,
            )
        )
        trail = format_audit_trail(s)
        assert "Session: abc" in trail
        assert "search" in trail
        assert "100ms" in trail

    def test_session_summary(self):
        s = create_harness_session(session_id="xyz")
        summary = generate_session_summary(s)
        assert "Session xyz" in summary
        assert "0 calls" in summary
        assert "100% success" in summary


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict(self):
        s = create_harness_session(session_id="ser")
        d = s.to_dict()
        assert d["session_id"] == "ser"
        assert "execution_history" in d
        assert "file_inputs" in d
        assert "warnings" in d
