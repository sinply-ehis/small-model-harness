"""Tests for progressive tool disclosure module."""
import pytest

from small_model_harness import HarnessState, create_harness_session
from small_model_harness.tool_disclosure import (
    rank_tools,
    get_tool_subset,
    build_compact_tool_prompt,
)


TOOL_SCHEMAS = {
    "search_web": {
        "description": "Search the web for information",
        "category": "search",
        "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
    },
    "play_audio": {
        "description": "Play an audio file",
        "category": "audio",
        "properties": {"file_path": {"type": "string"}, "volume": {"type": "number"}},
    },
    "speak_text": {
        "description": "Convert text to speech using TTS engine",
        "category": "audio",
        "properties": {"text": {"type": "string"}, "voice": {"type": "string"}},
    },
    "read_file": {
        "description": "Read contents of a file from disk",
        "category": "file",
        "properties": {"path": {"type": "string"}},
    },
    "write_file": {
        "description": "Write content to a file on disk",
        "category": "file",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
    },
    "master_audio": {
        "description": "Master audio to LUFS target with compression and limiting",
        "category": "audio",
        "properties": {"file_path": {"type": "string"}, "target_lufs": {"type": "number"}},
    },
    "analyze_audio": {
        "description": "Analyze audio file for loudness and spectral properties",
        "category": "analysis",
        "properties": {"file_path": {"type": "string"}},
    },
}


class TestRankTools:
    def test_audio_query_ranks_audio_tools_higher(self):
        ranked = rank_tools("play some music", TOOL_SCHEMAS, top_n=3)
        names = [ts.name for ts in ranked]
        assert "play_audio" in names
        assert ranked[0].score >= ranked[-1].score

    def test_search_query_ranks_search_tools_higher(self):
        ranked = rank_tools("search the web for news", TOOL_SCHEMAS, top_n=3)
        names = [ts.name for ts in ranked]
        assert "search_web" in names

    def test_file_query_ranks_file_tools_higher(self):
        ranked = rank_tools("read my file", TOOL_SCHEMAS, top_n=3)
        names = [ts.name for ts in ranked]
        assert "read_file" in names

    def test_top_n_limits_results(self):
        ranked = rank_tools("do something", TOOL_SCHEMAS, top_n=3)
        assert len(ranked) == 3

    def test_all_tools_if_top_n_exceeds(self):
        ranked = rank_tools("do something", TOOL_SCHEMAS, top_n=100)
        assert len(ranked) == len(TOOL_SCHEMAS)

    def test_scores_are_sorted_descending(self):
        ranked = rank_tools("search the web", TOOL_SCHEMAS, top_n=7)
        scores = [ts.score for ts in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_scores_between_0_and_1(self):
        ranked = rank_tools("anything", TOOL_SCHEMAS, top_n=7)
        for ts in ranked:
            assert 0.0 <= ts.score <= 1.0

    def test_history_boosts_successful_tool(self):
        harness = create_harness_session()
        harness.tool_success_counts["search_web"] = 20
        harness.tool_failure_counts["search_web"] = 0

        ranked = rank_tools("search the web for news", TOOL_SCHEMAS, harness=harness, top_n=3)
        assert ranked[0].name == "search_web"

    def test_avoided_tool_demoted(self):
        harness = create_harness_session()
        harness.avoided_tools.add("play_audio")

        ranked = rank_tools("play music", TOOL_SCHEMAS, harness=harness, top_n=5)
        names = [ts.name for ts in ranked]
        # play_audio should not be first
        if "play_audio" in names:
            idx = names.index("play_audio")
            assert idx > 0  # not first

    def test_empty_query_returns_all(self):
        ranked = rank_tools("", TOOL_SCHEMAS, top_n=5)
        assert len(ranked) == 5  # all returned, scores may be low


class TestGetToolSubset:
    def test_returns_dict_subset(self):
        subset = get_tool_subset("search the web", TOOL_SCHEMAS, top_n=3)
        assert isinstance(subset, dict)
        assert len(subset) == 3
        assert all(name in TOOL_SCHEMAS for name in subset)

    def test_schemas_preserved(self):
        subset = get_tool_subset("search", TOOL_SCHEMAS, top_n=2)
        for name, schema in subset.items():
            assert schema == TOOL_SCHEMAS[name]


class TestBuildCompactPrompt:
    def test_includes_top_tools(self):
        prompt = build_compact_tool_prompt("search the web", TOOL_SCHEMAS, top_n=3)
        assert "search_web" in prompt
        assert "You have access to the following tools:" in prompt

    def test_includes_descriptions(self):
        prompt = build_compact_tool_prompt("search the web", TOOL_SCHEMAS, top_n=3)
        assert "Search the web" in prompt

    def test_includes_param_info(self):
        prompt = build_compact_tool_prompt("search the web", TOOL_SCHEMAS, top_n=3)
        assert "query" in prompt

    def test_does_not_include_all_tools(self):
        prompt = build_compact_tool_prompt("search", TOOL_SCHEMAS, top_n=3)
        # Should NOT include all 7 tools
        assert prompt.count("- ") == 3  # only 3 tool lines

    def test_top_n_5(self):
        prompt = build_compact_tool_prompt("do something", TOOL_SCHEMAS, top_n=5)
        assert prompt.count("- ") == 5

    def test_includes_instruction(self):
        prompt = build_compact_tool_prompt("test", TOOL_SCHEMAS, top_n=3)
        assert "Select the most appropriate tool" in prompt
