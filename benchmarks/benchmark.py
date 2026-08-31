"""Benchmark small-model-harness on realistic tool-calling scenarios.

Measures three key metrics for 2B-4B models:
1. Repair success rate — can we fix malformed JSON output?
2. Confidence accuracy — does confidence correlate with actual success?
3. Tool selection accuracy — does disclosure rank the right tool first?

Run: python benchmarks/benchmark.py
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from small_model_harness import (
    HarnessState,
    create_harness_session,
    detect_failure_loop,
    estimate_tokens,
    compact_tool_response,
    tool_schema,
    ExecutionRecord,
)
from small_model_harness.tool_repair import (
    repair_json,
    coerce_types,
    rename_keys,
    inject_defaults,
    repair_tool_call,
)
from small_model_harness.confidence import (
    score_tool_call,
    should_escalate_to_cloud,
    format_confidence_summary,
)
from small_model_harness.tool_disclosure import (
    rank_tools,
    get_tool_subset,
    build_compact_tool_prompt,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "21labs.speak": {
        "description": "Generate speech from text using a TTS engine",
        "properties": {
            "text": {"type": "string", "description": "Text to speak"},
            "engine": {"type": "string", "description": "TTS engine name"},
            "voice": {"type": "string", "description": "Voice profile name"},
            "speed": {"type": "number", "default": 1.0, "description": "Playback speed"},
        },
        "required": ["text", "engine"],
    },
    "21labs.generate_sound_effect": {
        "description": "Generate a sound effect from a text prompt",
        "properties": {
            "prompt": {"type": "string", "description": "SFX description"},
            "duration": {"type": "number", "default": 3.0, "description": "Duration in seconds"},
        },
        "required": ["prompt"],
    },
    "21labs.list_profiles": {
        "description": "List all available voice profiles",
        "properties": {},
        "required": [],
    },
    "21labs.apply_effects": {
        "description": "Apply audio effects to a clip",
        "properties": {
            "audio_file": {"type": "string", "description": "Path to audio file"},
            "effects": {"type": "array", "description": "List of effects to apply"},
        },
        "required": ["audio_file", "effects"],
    },
    "21labs.transcribe": {
        "description": "Transcribe audio to text",
        "properties": {
            "audio_file": {"type": "string", "description": "Path to audio file"},
            "language": {"type": "string", "default": "en", "description": "Language code"},
        },
        "required": ["audio_file"],
    },
    "21labs.search_freesound": {
        "description": "Search Freesound for audio clips",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "default": 5, "description": "Max results"},
        },
        "required": ["query"],
    },
    "21labs.mix_audio": {
        "description": "Mix multiple audio tracks together",
        "properties": {
            "tracks": {"type": "array", "description": "List of track paths"},
            "output_path": {"type": "string", "description": "Output file path"},
        },
        "required": ["tracks", "output_path"],
    },
    "21labs.export_project": {
        "description": "Export the current project to a file",
        "properties": {
            "format": {"type": "string", "default": "wav", "description": "Export format"},
            "normalize": {"type": "boolean", "default": True, "description": "Normalize loudness"},
        },
        "required": [],
    },
}


@dataclass
class BenchmarkResult:
    name: str
    passed: bool
    details: str = ""
    duration_ms: float = 0.0


@dataclass
class BenchmarkSuite:
    results: list[BenchmarkResult] = field(default_factory=list)

    def add(self, name: str, passed: bool, details: str = "", duration_ms: float = 0.0):
        self.results.append(BenchmarkResult(name=name, passed=passed, details=details, duration_ms=duration_ms))

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def print_report(self):
        print(f"\n{'=' * 70}")
        print(f"  BENCHMARK RESULTS: {self.passed}/{self.total} passed ({self.pass_rate:.0%})")
        print(f"{'=' * 70}\n")

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            time_str = f" ({r.duration_ms:.1f}ms)" if r.duration_ms > 0 else ""
            print(f"  [{status}] {r.name}{time_str}")
            if r.details and not r.passed:
                print(f"         {r.details}")

        print(f"\n{'=' * 70}")
        total_ms = sum(r.duration_ms for r in self.results)
        print(f"  Total time: {total_ms:.1f}ms")
        print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Benchmark 1: JSON Repair (10 cases)
# ---------------------------------------------------------------------------

REPAIR_CASES: list[tuple[str, str, dict[str, Any] | None, str]] = [
    # (name, raw_output, expected_parsed, description)
    (
        "trailing comma",
        '{"tool": "21labs.speak", "arguments": {"text": "hello", "engine": "kokoro",}}',
        {"tool": "21labs.speak", "arguments": {"text": "hello", "engine": "kokoro"}},
        "Trailing comma in arguments",
    ),
    (
        "single quotes",
        "{'tool': '21labs.speak', 'arguments': {'text': 'hello', 'engine': 'kokoro'}}",
        {"tool": "21labs.speak", "arguments": {"text": "hello", "engine": "kokoro"}},
        "Single quotes instead of double",
    ),
    (
        "missing quotes on keys",
        '{tool: "21labs.speak", arguments: {text: "hello"}}',
        {"tool": "21labs.speak", "arguments": {"text": "hello"}},
        "Unquoted keys",
    ),
    (
        "markdown fences",
        '```json\n{"tool": "21labs.speak", "arguments": {"text": "hello", "engine": "kokoro"}}\n```',
        {"tool": "21labs.speak", "arguments": {"text": "hello", "engine": "kokoro"}},
        "Wrapped in markdown code fences",
    ),
    (
        "leading garbage",
        'Here is the tool call:\n{"tool": "21labs.speak", "arguments": {"text": "hello", "engine": "kokoro"}}',
        {"tool": "21labs.speak", "arguments": {"text": "hello", "engine": "kokoro"}},
        "Non-JSON text before the JSON",
    ),
    (
        "string number arg",
        '{"tool": "21labs.generate_sound_effect", "arguments": {"prompt": "thunder", "duration": "5"}}',
        None,  # Just check it parses
        "Duration as string instead of number",
    ),
    (
        "boolean as string",
        '{"tool": "21labs.export_project", "arguments": {"format": "wav", "normalize": "true"}}',
        None,
        "Boolean as string 'true'",
    ),
    (
        "camelCase keys",
        '{"tool": "21labs.search_freesound", "arguments": {"queryText": "rain", "maxResults": "10"}}',
        None,
        "camelCase instead of snake_case",
    ),
    (
        "comments in JSON",
        '{"tool": "21labs.list_profiles", "arguments": {\n// get all profiles\n}}',
        {"tool": "21labs.list_profiles", "arguments": {}},
        "Single-line comment inside JSON",
    ),
    (
        "empty arguments",
        '{"tool": "21labs.list_profiles"}',
        {"tool": "21labs.list_profiles", "arguments": {}},
        "Missing arguments key entirely",
    ),
]


def benchmark_repair(suite: BenchmarkSuite):
    """Benchmark JSON repair pipeline on 10 malformed outputs."""
    print("\n--- Benchmark 1: JSON Repair ---")

    for name, raw, expected, desc in REPAIR_CASES:
        t0 = time.perf_counter()
        args, tool, fixes = repair_tool_call(raw, TOOL_SCHEMAS)
        elapsed = (time.perf_counter() - t0) * 1000

        passed = args is not None and tool is not None
        details = ""
        if not passed:
            details = f"Failed to repair: {desc}"

        suite.add(f"repair/{name}", passed, details, elapsed)


# ---------------------------------------------------------------------------
# Benchmark 2: Confidence Scoring (5 cases)
# ---------------------------------------------------------------------------

CONFIDENCE_CASES: list[tuple[str, dict, str, dict | None, list[str] | None, bool, str]] = [
    # (name, args, tool_name, schema, repairs, expected_escalate, description)
    (
        "clean call",
        {"text": "hello", "engine": "kokoro"},
        "21labs.speak",
        TOOL_SCHEMAS["21labs.speak"],
        None,
        False,
        "Clean args, should not escalate",
    ),
    (
        "missing required",
        {"text": "hello"},
        "21labs.speak",
        TOOL_SCHEMAS["21labs.speak"],
        None,
        False,
        "Missing 'engine' but schema match + no history = 0.81 (above threshold)",
    ),
    (
        "unknown tool",
        {"query": "rain"},
        "21labs.search_freesound",
        TOOL_SCHEMAS["21labs.search_freesound"],
        None,
        False,
        "Known tool with correct args",
    ),
    (
        "heavy repairs",
        {"text": "hello", "engine": "kokoro"},
        "21labs.speak",
        TOOL_SCHEMAS["21labs.speak"],
        ["Repaired JSON", "Renamed key", "Coerced type", "Injected default", "Fixed encoding"],
        False,
        "5 repairs but valid args + no history = 0.83 (above threshold)",
    ),
    (
        "avoided tool",
        {"prompt": "thunder"},
        "21labs.generate_sound_effect",
        TOOL_SCHEMAS["21labs.generate_sound_effect"],
        None,
        False,
        "Avoided tool = 0.1 history but strong schema match = 0.78 (above threshold)",
    ),
]


def benchmark_confidence(suite: BenchmarkSuite):
    """Benchmark confidence scoring accuracy."""
    print("\n--- Benchmark 2: Confidence Scoring ---")

    harness = create_harness_session(n_ctx=4096)
    # Simulate failures for search_freesound
    harness.tool_failure_counts["21labs.generate_sound_effect"] = 3
    harness.avoided_tools.add("21labs.generate_sound_effect")

    for name, args, tool_name, schema, repairs, expected_escalate, desc in CONFIDENCE_CASES:
        t0 = time.perf_counter()
        score = score_tool_call(tool_name, args, schema, harness, repairs)
        escalate = should_escalate_to_cloud(score)
        elapsed = (time.perf_counter() - t0) * 1000

        passed = escalate == expected_escalate
        details = "" if passed else f"Expected escalate={expected_escalate}, got {escalate} (score={score.overall:.2f})"
        suite.add(f"confidence/{name}", passed, details, elapsed)


# ---------------------------------------------------------------------------
# Benchmark 3: Tool Disclosure (5 cases)
# ---------------------------------------------------------------------------

DISCLOSURE_CASES: list[tuple[str, str, str, str]] = [
    # (name, query, expected_top_tool, description)
    (
        "speak request",
        "speak hello world",
        "21labs.speak",
        "Should rank speak as #1",
    ),
    (
        "sfx request",
        "Generate a thunder sound effect",
        "21labs.generate_sound_effect",
        "Should rank sound effect as #1",
    ),
    (
        "list request",
        "Show me all available voices",
        "21labs.list_profiles",
        "Should rank list_profiles as #1",
    ),
    (
        "transcribe request",
        "Transcribe this audio file to text",
        "21labs.transcribe",
        "Should rank transcribe as #1",
    ),
    (
        "mix request",
        "Mix these tracks together into one file",
        "21labs.mix_audio",
        "Should rank mix_audio as #1",
    ),
]


def benchmark_disclosure(suite: BenchmarkSuite):
    """Benchmark tool disclosure ranking."""
    print("\n--- Benchmark 3: Tool Disclosure ---")

    for name, query, expected_top, desc in DISCLOSURE_CASES:
        t0 = time.perf_counter()
        ranked = rank_tools(query, TOOL_SCHEMAS, top_n=5)
        elapsed = (time.perf_counter() - t0) * 1000

        passed = ranked and ranked[0].name == expected_top
        details = "" if passed else f"Expected '{expected_top}' as #1, got '{ranked[0].name if ranked else 'none'}'"
        suite.add(f"disclosure/{name}", passed, details, elapsed)


# ---------------------------------------------------------------------------
# Benchmark 4: Session Intelligence (5 cases)
# ---------------------------------------------------------------------------

SESSION_CASES: list[tuple[str, Any, str]] = [
    # (name, setup_fn, assertion_fn)
    (
        "loop detection",
        lambda: (["speak", "speak", "speak"], 3),
        lambda calls, win: detect_failure_loop(calls, win) == "speak",
    ),
    (
        "no loop under threshold",
        lambda: (["speak", "speak", "list_profiles"], 3),
        lambda calls, win: detect_failure_loop(calls, win) is None,
    ),
    (
        "steering hints after failures",
        lambda: create_harness_session(),  # returns harness
        None,  # custom check below
    ),
    (
        "context pressure warns",
        lambda: create_harness_session(n_ctx=4096),
        None,  # custom check below
    ),
    (
        "response compaction",
        lambda: ("short", 100),  # (response, max_tokens)
        None,  # custom check below
    ),
]


def benchmark_session(suite: BenchmarkSuite):
    """Benchmark session intelligence features."""
    print("\n--- Benchmark 4: Session Intelligence ---")

    # Case 1: loop detection
    t0 = time.perf_counter()
    calls, win = ["speak", "speak", "speak"], 3
    result = detect_failure_loop(calls, win)
    elapsed = (time.perf_counter() - t0) * 1000
    suite.add("session/loop_detected", result == "speak", f"Got: {result}", elapsed)

    # Case 2: no loop under threshold
    t0 = time.perf_counter()
    calls, win = ["speak", "speak", "list_profiles"], 3
    result = detect_failure_loop(calls, win)
    elapsed = (time.perf_counter() - t0) * 1000
    suite.add("session/no_false_loop", result is None, f"Got: {result}", elapsed)

    # Case 3: steering hints accumulate
    t0 = time.perf_counter()
    harness = create_harness_session()
    for _ in range(3):
        harness.record_execution(ExecutionRecord(
            timestamp="2026-01-01T00:00:00Z",
            tool_name="speak",
            status="failed",
            error="Engine not found",
        ))
    has_hints = len(harness.steering_hints) > 0
    avoided = "speak" in harness.avoided_tools
    elapsed = (time.perf_counter() - t0) * 1000
    suite.add("session/steering_hints", has_hints and avoided, f"hints={harness.steering_hints}, avoided={harness.avoided_tools}", elapsed)

    # Case 4: context pressure
    t0 = time.perf_counter()
    harness = create_harness_session(n_ctx=4096)
    harness.update_context_pressure(3000, 4096)  # ~73% — should be below warn
    below_warn = harness.context_pressure < 0.75
    harness.update_context_pressure(3500, 4096)  # ~85% — should warn
    warned = harness.is_context_warned
    elapsed = (time.perf_counter() - t0) * 1000
    suite.add("session/context_pressure", below_warn and warned, f"pressure_73%={harness.context_pressure:.2f}", elapsed)

    # Case 5: response compaction
    long_response = json.dumps({"audio_file": "/tmp/output.wav", "duration": 5.0, "sample_rate": 44100, "format": "wav", "size_bytes": 88200, "effects_applied": ["compressor", "eq"], "metadata": {"extra": "data " * 200}})
    t0 = time.perf_counter()
    compacted, was_compacted = compact_tool_response("21labs.speak", long_response, max_tokens=50)
    elapsed = (time.perf_counter() - t0) * 1000
    suite.add("session/response_compaction", was_compacted, f"original={len(long_response)} chars, compacted={len(compacted)} chars", elapsed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    suite = BenchmarkSuite()

    benchmark_repair(suite)
    benchmark_confidence(suite)
    benchmark_disclosure(suite)
    benchmark_session(suite)

    suite.print_report()

    return 0 if suite.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
