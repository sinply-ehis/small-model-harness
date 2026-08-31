"""Tests for thinking model detection utilities."""
from small_model_harness import (
    is_thinking_model,
    split_thinking,
    NON_THINKING_MODEL_MARKERS,
)


class TestIsThinkingModel:
    def test_qwen_is_thinking(self):
        assert is_thinking_model("qwen3.5-4b") is True

    def test_chatterbox_not_thinking(self):
        assert is_thinking_model("chatterbox-v1") is False

    def test_piper_not_thinking(self):
        assert is_thinking_model("piper-tts") is False

    def test_whisper_not_thinking(self):
        assert is_thinking_model("whisper-base") is False

    def test_bark_not_thinking(self):
        assert is_thinking_model("bark-small") is False

    def test_tts_prefix_not_thinking(self):
        assert is_thinking_model("tts-english") is False

    def test_stt_prefix_not_thinking(self):
        assert is_thinking_model("stt-whisper") is False

    def test_case_insensitive(self):
        assert is_thinking_model("ChatterBox") is False
        assert is_thinking_model("PIPER-TTS") is False

    def test_unknown_model_is_thinking(self):
        assert is_thinking_model("some-random-model") is True

    def test_empty_string(self):
        assert is_thinking_model("") is True


class TestSplitThinking:
    def test_no_thinking_block(self):
        thinking, visible = split_thinking("Hello world")
        assert thinking == ""
        assert visible == "Hello world"

    def test_thinking_block_extracted(self):
        text = "<think>Let me think about this...</think>The answer is 42."
        thinking, visible = split_thinking(text)
        assert "Let me think" in thinking
        assert visible == "The answer is 42."

    def test_multiple_thinking_blocks(self):
        text = "<think>First thought</think>Step 1. <think>Second thought</think>Step 2."
        thinking, visible = split_thinking(text)
        assert "First thought" in thinking
        assert "Second thought" in thinking
        assert "Step 1." in visible
        assert "Step 2." in visible

    def test_thinking_at_start(self):
        text = "<think>reasoning</think>result"
        thinking, visible = split_thinking(text)
        assert "reasoning" in thinking
        assert visible == "result"

    def test_thinking_at_end(self):
        text = "<think>reasoning</think>"
        thinking, visible = split_thinking(text)
        assert "reasoning" in thinking
        assert visible == ""

    def test_empty_thinking_block(self):
        text = "<think>x</think>actual response"
        thinking, visible = split_thinking(text)
        assert thinking == "x"
        assert visible == "actual response"

    def test_multiline_thinking(self):
        text = "<think>Line 1\nLine 2</think>answer"
        thinking, visible = split_thinking(text)
        assert "Line 1" in thinking
        assert "Line 2" in thinking
        assert visible == "answer"

    def test_realistic_qwen_output(self):
        text = """<think>The user wants me to search for something. Let me use the search_web tool.</think>I'll search for that information."""
        thinking, visible = split_thinking(text)
        assert "search_web" in thinking
        assert visible == "I'll search for that information."

    def test_empty_string(self):
        thinking, visible = split_thinking("")
        assert thinking == ""
        assert visible == ""

    def test_only_visible(self):
        text = "Just a normal response with no thinking."
        thinking, visible = split_thinking(text)
        assert thinking == ""
        assert visible == text


class TestMarkers:
    def test_markers_are_tuple(self):
        assert isinstance(NON_THINKING_MODEL_MARKERS, tuple)

    def test_markers_include_common_tts(self):
        assert "piper" in NON_THINKING_MODEL_MARKERS
        assert "whisper" in NON_THINKING_MODEL_MARKERS
        assert "bark" in NON_THINKING_MODEL_MARKERS
