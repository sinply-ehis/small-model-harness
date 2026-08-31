"""End-to-end integration test — exercises the full small-model harness pipeline."""
import json

from small_model_harness import (
    HarnessState, ExecutionRecord, FileInput,
    create_harness_session, compact_tool_response,
    detect_failure_loop, estimate_tokens, estimate_messages_tokens,
    tool_schema, pydantic_tool_schema,
    repair_json, coerce_types, rename_keys, inject_defaults, repair_tool_call,
    ConfidenceScore, score_tool_call_confidence, should_escalate_to_cloud,
    format_confidence_summary, format_audit_trail, generate_session_summary,
    ToolScore, rank_tools, get_tool_subset, build_compact_tool_prompt,
)

SCHEMAS = {
    "search_web": {
        "description": "Search the web for information",
        "category": "search",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "default": 5},
        },
        "required": ["query"],
    },
    "play_audio": {
        "description": "Play an audio file",
        "category": "audio",
        "properties": {
            "file_path": {"type": "string", "description": "Path to audio file"},
            "volume": {"type": "number", "default": 1.0},
        },
        "required": ["file_path"],
    },
    "speak_text": {
        "description": "Convert text to speech using TTS",
        "category": "audio",
        "properties": {
            "text": {"type": "string", "description": "Text to speak"},
            "voice": {"type": "string", "default": "default"},
        },
        "required": ["text"],
    },
}


def test_repair_pipeline():
    """Test the full repair pipeline on realistic malformed output."""
    # Trailing comma + wrong key name + string type
    bad = '{"tool": "search_web", "arguments": {"query": "hello world", "maxResults": "10",}}'
    args, name, fixes = repair_tool_call(bad, SCHEMAS)
    assert name == "search_web", f"Expected search_web, got {name}"
    assert args["query"] == "hello world"
    assert args["max_results"] == 10, f"Expected 10 (int), got {args['max_results']}"
    assert isinstance(args["max_results"], int), "max_results should be int"
    assert any("Renamed" in f for f in fixes), f"Should have renamed key, fixes: {fixes}"
    assert any("Converted" in f for f in fixes), f"Should have coerced type, fixes: {fixes}"
    print(f"  [PASS] repair_pipeline: {fixes}")


def test_repair_injects_defaults():
    """Missing optional args get defaults from schema."""
    raw = '{"tool": "search_web", "arguments": {"query": "test"}}'
    args, name, fixes = repair_tool_call(raw, SCHEMAS)
    assert args.get("max_results") == 5, f"Default not injected: {args}"
    print(f"  [PASS] repair_injects_defaults: max_results={args['max_results']}")


def test_repair_markdown_fences():
    """Model wraps output in ```json fences — should be stripped."""
    raw = '```json\n{"tool": "play_audio", "arguments": {"file_path": "/a/b.wav"}}\n```'
    args, name, fixes = repair_tool_call(raw, SCHEMAS)
    assert name == "play_audio"
    assert args["file_path"] == "/a/b.wav"
    print(f"  [PASS] repair_markdown_fences")


def test_confidence_high_for_good_call():
    """Valid call with good history should score high."""
    harness = create_harness_session()
    harness.tool_success_counts["search_web"] = 10
    harness.tool_failure_counts["search_web"] = 0

    score = score_tool_call_confidence(
        "search_web", {"query": "test"}, SCHEMAS["search_web"], harness=harness
    )
    assert score.overall >= 0.8, f"Expected >= 0.8, got {score.overall}"
    assert not score.should_escalate
    print(f"  [PASS] confidence_high: {score.overall:.3f}")


def test_confidence_low_for_avoided_tool():
    """Tool in avoided list should score very low."""
    harness = create_harness_session()
    harness.avoided_tools.add("search_web")

    score = score_tool_call_confidence(
        "search_web", {"query": "test"}, SCHEMAS["search_web"], harness=harness
    )
    assert score.history_score <= 0.1, f"Expected <= 0.1, got {score.history_score}"
    print(f"  [PASS] confidence_low_avoided: history={score.history_score:.3f}")


def test_confidence_low_for_heavy_repairs():
    """Many repairs = low repair_score."""
    score = score_tool_call_confidence(
        "search_web", {"query": 123}, SCHEMAS["search_web"],
        repairs_applied=["Fixed JSON", "Coerced type", "Renamed key", "Injected defaults"],
    )
    assert score.repair_cost < 0.6, f"Expected < 0.6, got {score.repair_cost}"
    print(f"  [PASS] confidence_low_repairs: repair_cost={score.repair_cost:.3f}")


def test_tool_disclosure_ranks_by_query():
    """Audio query should rank audio tools higher."""
    ranked = rank_tools("play some music", SCHEMAS, top_n=2)
    names = [ts.name for ts in ranked]
    assert "play_audio" in names, f"Expected play_audio in top 2, got {names}"
    print(f"  [PASS] disclosure_ranks: {names}")


def test_tool_disclosure_history_boost():
    """Successful history should boost a tool's rank."""
    harness = create_harness_session()
    harness.tool_success_counts["speak_text"] = 20
    harness.tool_failure_counts["speak_text"] = 0

    ranked = rank_tools("say something", SCHEMAS, harness=harness, top_n=2)
    assert ranked[0].name == "speak_text", f"Expected speak_text first, got {ranked[0].name}"
    print(f"  [PASS] disclosure_history_boost: {ranked[0].name} (score={ranked[0].score:.3f})")


def test_tool_disclosure_avoided_demoted():
    """Avoided tool should be pushed down in ranking."""
    harness = create_harness_session()
    harness.avoided_tools.add("play_audio")

    ranked = rank_tools("play some music", SCHEMAS, harness=harness, top_n=3)
    names = [ts.name for ts in ranked]
    if "play_audio" in names:
        idx = names.index("play_audio")
        assert idx > 0, f"play_audio should not be first in {names}"
    print(f"  [PASS] disclosure_avoided_demoted: {names}")


def test_compact_prompt_size():
    """Compact prompt should show only N tools, not all."""
    prompt = build_compact_tool_prompt("search the web", SCHEMAS, top_n=2)
    tool_lines = [l for l in prompt.split("\n") if l.startswith("- ")]
    assert len(tool_lines) == 2, f"Expected 2 tool lines, got {len(tool_lines)}"
    assert "search_web" in prompt
    print(f"  [PASS] compact_prompt: {len(tool_lines)} tools shown")


def test_full_session_lifecycle():
    """Simulate a full session: create, record, steer, detect loops, compact."""
    harness = create_harness_session(n_ctx=4096)

    # Record some executions
    for i in range(5):
        rec = ExecutionRecord(
            timestamp=f"2026-01-01T00:00:0{i}Z",
            tool_name="search_web",
            arguments={"query": f"test {i}"},
            status="completed",
            result=f"Result {i}",
            duration_ms=100 + i * 10,
        )
        harness.record_execution(rec)

    # Two failures (threshold is 2 before steering kicks in)
    for i in range(2):
        fail_rec = ExecutionRecord(
            timestamp=f"2026-01-01T00:00:0{6+i}Z",
            tool_name="play_audio",
            arguments={"file_path": "/bad.wav"},
            status="failed",
            error="File not found",
            duration_ms=50,
        )
        harness.record_execution(fail_rec)

    # Check stats
    assert harness.total_tool_calls == 7
    assert harness.successful_calls == 5
    assert harness.failed_calls == 2
    assert harness.tool_success_counts["search_web"] == 5
    assert harness.tool_failure_counts["play_audio"] == 2

    # Steering hints should mention play_audio
    hints = harness.steering_hints
    assert any("play_audio" in h for h in hints), f"Expected play_audio in steering hints: {hints}"

    # Context pressure
    harness.update_context_pressure(3500, 4096)
    assert harness.is_context_warned, f"Expected context warned at 3500/4096"

    # Loop detection (list of tool name strings)
    harness.recent_tool_calls = ["search_web", "search_web", "search_web"]
    assert detect_failure_loop(harness.recent_tool_calls) == "search_web"

    # Response compaction — need a response that exceeds max_tokens
    big_data = {
        "audio_file": "/very/long/path/segment1/segment2/segment3/final_mix.wav",
        "file_path": "/very/long/path/segment1/segment2/segment3/final_mix.wav",
        "duration": 300.5,
        "sample_rate": 44100,
        "channels": 2,
        "format": "wav",
        "size_bytes": 26460016,
        "effects_applied": ["eq", "compress", "limit", "normalize", "reverb", "chorus", "delay"],
        "metadata": {"encoder": "ffmpeg", "bitrate": "1411k", "profile": "lossless", "title": "Final Mix"},
    }
    big = json.dumps(big_data)
    compacted, was_compacted = compact_tool_response("play_audio", big, max_tokens=10)
    assert was_compacted, f"Expected compaction for {len(big)} char response at max_tokens=10"
    assert len(compacted) < len(big)

    # Audit trail
    trail = format_audit_trail(harness)
    assert "search_web" in trail
    assert "play_audio" in trail

    # Summary
    summary = generate_session_summary(harness)
    assert "7 calls" in summary
    assert "71%" in summary  # 5/7 = ~71%
    assert "play_audio" in summary  # in "avoiding: play_audio"

    print(f"  [PASS] full_session_lifecycle")


def test_schema_generation():
    """Generate OpenAI-compatible tool schemas."""
    s1 = tool_schema("search_web", "Search the web", {
        "query": {"type": "string", "description": "Search query"},
    })
    assert s1["type"] == "function"
    assert s1["function"]["name"] == "search_web"
    assert "query" in s1["function"]["parameters"]["properties"]

    from pydantic import BaseModel, Field
    class SearchArgs(BaseModel):
        query: str = Field(description="Search query")
        max_results: int = Field(default=5)

    s2 = pydantic_tool_schema(SearchArgs, "search_web", "Search the web")
    assert s2["type"] == "function"
    assert "query" in s2["function"]["parameters"]["properties"]

    print(f"  [PASS] schema_generation")


def test_escalation_decision():
    """should_escalate_to_cloud returns correct bool."""
    good = ConfidenceScore(overall=0.9, schema_match=1.0, history_score=0.9,
                           completeness=1.0, repair_cost=1.0, should_escalate=False, reason="ok")
    bad = ConfidenceScore(overall=0.3, schema_match=0.5, history_score=0.1,
                          completeness=0.5, repair_cost=0.3, should_escalate=True, reason="bad")
    assert should_escalate_to_cloud(good) is False
    assert should_escalate_to_cloud(bad) is True
    print(f"  [PASS] escalation_decision")


if __name__ == "__main__":
    print("Running integration tests...")
    test_repair_pipeline()
    test_repair_injects_defaults()
    test_repair_markdown_fences()
    test_confidence_high_for_good_call()
    test_confidence_low_for_avoided_tool()
    test_confidence_low_for_heavy_repairs()
    test_tool_disclosure_ranks_by_query()
    test_tool_disclosure_history_boost()
    test_tool_disclosure_avoided_demoted()
    test_compact_prompt_size()
    test_full_session_lifecycle()
    test_schema_generation()
    test_escalation_decision()
    print("\nALL 13 INTEGRATION TESTS PASSED")
