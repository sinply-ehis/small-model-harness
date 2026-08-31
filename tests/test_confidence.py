"""Tests for confidence scoring module."""
import pytest

from small_model_harness import HarnessState, create_harness_session
from small_model_harness.confidence import (
    score_tool_call,
    should_escalate_to_cloud,
    format_confidence_summary,
)


class TestScoreToolCall:
    def setup_method(self):
        self.schema = {
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        }

    def test_valid_call_high_confidence(self):
        score = score_tool_call(
            "search_web",
            {"query": "hello", "limit": 5},
            self.schema,
        )
        assert score.overall >= 0.7
        assert score.should_escalate is False

    def test_no_schema_moderate_confidence(self):
        score = score_tool_call("search_web", {"query": "hello"}, None)
        assert score.overall >= 0.7
        assert score.should_escalate is False

    def test_wrong_type_reduces_score(self):
        score = score_tool_call(
            "search_web",
            {"query": 123, "limit": "not a number"},
            self.schema,
        )
        assert score.schema_match < 1.0
        assert score.overall < 0.85  # schema mismatch penalizes

    def test_missing_required_reduces_completeness(self):
        score = score_tool_call(
            "search_web",
            {},  # missing required "query"
            self.schema,
        )
        assert score.completeness < 1.0

    def test_unknown_key_reduces_score(self):
        score = score_tool_call(
            "search_web",
            {"query": "hello", "unknown_param": "val"},
            self.schema,
        )
        assert score.schema_match < 1.0

    def test_history_success_increases_score(self):
        harness = create_harness_session()
        harness.tool_success_counts["search_web"] = 10
        harness.tool_failure_counts["search_web"] = 1

        score = score_tool_call(
            "search_web",
            {"query": "hello"},
            self.schema,
            harness=harness,
        )
        assert score.history_score > 0.8

    def test_history_failures_decrease_score(self):
        harness = create_harness_session()
        harness.tool_success_counts["search_web"] = 1
        harness.tool_failure_counts["search_web"] = 10

        score = score_tool_call(
            "search_web",
            {"query": "hello"},
            self.schema,
            harness=harness,
        )
        assert score.history_score < 0.5

    def test_avoided_tool_very_low_history(self):
        harness = create_harness_session()
        harness.avoided_tools.add("search_web")

        score = score_tool_call(
            "search_web",
            {"query": "hello"},
            self.schema,
            harness=harness,
        )
        assert score.history_score <= 0.1

    def test_repairs_reduce_score(self):
        score = score_tool_call(
            "search_web",
            {"query": "hello"},
            self.schema,
            repairs_applied=["Fixed JSON", "Coerced type", "Renamed key"],
        )
        assert score.repair_cost < 1.0
        assert score.overall < 0.9

    def test_no_repairs_full_repair_score(self):
        score = score_tool_call(
            "search_web",
            {"query": "hello"},
            self.schema,
            repairs_applied=[],
        )
        assert score.repair_cost == 1.0

    def test_perfect_score(self):
        harness = create_harness_session()
        harness.tool_success_counts["search_web"] = 20
        harness.tool_failure_counts["search_web"] = 0

        score = score_tool_call(
            "search_web",
            {"query": "hello"},
            self.schema,
            harness=harness,
            repairs_applied=[],
        )
        assert score.overall >= 0.9
        assert score.should_escalate is False

    def test_very_bad_score_triggers_escalation(self):
        harness = create_harness_session()
        harness.tool_success_counts["bad_tool"] = 0
        harness.tool_failure_counts["bad_tool"] = 10
        harness.avoided_tools.add("bad_tool")

        score = score_tool_call(
            "bad_tool",
            {"wrong_arg": 123},
            self.schema,
            harness=harness,
            repairs_applied=["Fixed JSON", "Coerced type", "Renamed key", "Injected defaults"],
        )
        assert score.overall < 0.5
        assert score.should_escalate is True


class TestShouldEscalate:
    def test_high_confidence_no_escalation(self):
        from small_model_harness.confidence import ConfidenceScore
        score = ConfidenceScore(
            overall=0.85,
            schema_match=0.9,
            history_score=0.8,
            completeness=1.0,
            repair_cost=1.0,
            should_escalate=False,
            reason="All checks passed",
        )
        assert should_escalate_to_cloud(score) is False

    def test_low_confidence_escalation(self):
        from small_model_harness.confidence import ConfidenceScore
        score = ConfidenceScore(
            overall=0.3,
            schema_match=0.5,
            history_score=0.1,
            completeness=0.5,
            repair_cost=0.3,
            should_escalate=True,
            reason="Multiple failures",
        )
        assert should_escalate_to_cloud(score) is True


class TestFormatSummary:
    def test_format_output(self):
        score = score_tool_call("search_web", {"query": "hello"}, {
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        })
        summary = format_confidence_summary(score)
        assert "Confidence:" in summary
        assert "Schema:" in summary
        assert "History:" in summary
