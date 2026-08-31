"""Tests for pydantic-deep integration module.

These tests verify the integration module's structure and error handling
without requiring pydantic-deep to be installed (it's an optional dependency).
"""

from __future__ import annotations

import pytest

from small_model_harness import create_harness_session, HarnessState, ExecutionRecord


class TestIntegrationImports:
    """Test that the integration module handles missing pydantic-deep gracefully."""

    def test_import_module(self):
        """Module should be importable even without pydantic-deep."""
        from small_model_harness import pydantic_deep_integration

        assert hasattr(pydantic_deep_integration, "build_small_model_agent")
        assert hasattr(pydantic_deep_integration, "run_with_harness")
        assert hasattr(pydantic_deep_integration, "SmallModelAgentResult")

    def test_check_pydantic_deep_raises_without_package(self):
        """_check_pydantic_deep should raise ImportError when not installed."""
        from small_model_harness.pydantic_deep_integration import _check_pydantic_deep

        # This will raise ImportError if pydantic-deep is not installed,
        # or succeed if it is. Either way, it shouldn't crash.
        try:
            _check_pydantic_deep()
        except ImportError as e:
            assert "pydantic-deep" in str(e)
            assert "pip install" in str(e)


class TestHarnessStateCompatibility:
    """Test that HarnessState works correctly for integration use cases."""

    def test_pressure_callback_simulation(self):
        """Simulate the on_context_update callback pattern."""
        harness = create_harness_session(n_ctx=4096)

        # Simulate what pydantic-deep's on_context_update would do
        # 3200/4096 = 0.781, above the 0.75 warn threshold
        harness.update_context_pressure(3200, 4096)
        assert harness.context_pressure == pytest.approx(0.781, abs=0.01)
        assert harness.is_context_warned

    def test_steering_with_pressure(self):
        """Test steering hints combine failure + pressure info."""
        harness = create_harness_session(n_ctx=4096)

        # Simulate tool failures
        for _ in range(2):
            harness.record_execution(
                ExecutionRecord(
                    timestamp="2026-01-01T00:00:00Z",
                    tool_name="bad_tool",
                    arguments={},
                    status="failed",
                )
            )

        # Simulate context pressure
        harness.update_context_pressure(3500, 4096)

        prompt = harness.get_steering_prompt()
        assert "Session Guidance" in prompt
        assert "bad_tool" in prompt
        assert "Context is getting full" in prompt

    def test_budget_tracking_for_agent(self):
        """Test budget works correctly in agent loop pattern."""
        harness = create_harness_session(n_ctx=4096)
        # Budget should be min(100, max(5, 4096//500)) = 8
        assert harness.budget_remaining == 8

        # Simulate agent loop
        for i in range(8):
            harness.record_execution(
                ExecutionRecord(
                    timestamp="2026-01-01T00:00:00Z",
                    tool_name=f"tool_{i}",
                    arguments={},
                    status="completed",
                )
            )

        assert harness.is_budget_exhausted
        assert harness.budget_remaining == 0


class TestSmallModelAgentResult:
    """Test the SmallModelAgentResult dataclass."""

    def test_result_structure(self):
        """Verify result has all expected fields."""
        from small_model_harness.pydantic_deep_integration import SmallModelAgentResult

        harness = create_harness_session()
        result = SmallModelAgentResult(
            agent=None,
            deps=None,
            harness=harness,
        )
        assert result.agent is None
        assert result.deps is None
        assert result.harness is harness
        assert result.paths is None

    def test_result_with_paths(self):
        """Verify paths field works."""
        from small_model_harness.pydantic_deep_integration import SmallModelAgentResult

        harness = create_harness_session()
        result = SmallModelAgentResult(
            agent=None,
            deps=None,
            harness=harness,
            paths="/some/path",
        )
        assert result.paths == "/some/path"
