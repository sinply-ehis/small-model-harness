"""Basic usage example for small-model-harness.

Demonstrates the full pipeline that a real assistant integration uses:
1. Create a session with context-aware budgeting
2. Rank tools by query relevance (progressive disclosure)
3. Repair malformed JSON output from a small model
4. Score confidence and decide whether to escalate
5. Track session state (failures, loops, steering hints)
6. Compact large tool responses to save context

Run: python examples/basic_usage.py
"""

from __future__ import annotations

import json

from small_model_harness import (
    create_harness_session,
    detect_failure_loop,
    estimate_tokens,
    compact_tool_response,
    compact_text,
    tool_schema,
    ExecutionRecord,
)
from small_model_harness.tool_repair import repair_tool_call
from small_model_harness.confidence import score_tool_call, should_escalate_to_cloud, format_confidence_summary
from small_model_harness.tool_disclosure import rank_tools, build_compact_tool_prompt


# ---------------------------------------------------------------------------
# 1. Define your tool schemas
# ---------------------------------------------------------------------------

TOOLS = {
    "21labs.speak": tool_schema(
        name="21labs.speak",
        description="Generate speech from text using a TTS engine",
        parameters={
            "text": {"type": "string", "description": "Text to speak"},
            "engine": {"type": "string", "description": "TTS engine name"},
            "voice": {"type": "string", "description": "Voice profile name"},
            "speed": {"type": "number", "default": 1.0, "description": "Playback speed"},
        },
    )["function"],

    "21labs.generate_sound_effect": tool_schema(
        name="21labs.generate_sound_effect",
        description="Generate a sound effect from a text prompt",
        parameters={
            "prompt": {"type": "string", "description": "SFX description"},
            "duration": {"type": "number", "default": 3.0},
        },
    )["function"],

    "21labs.list_profiles": tool_schema(
        name="21labs.list_profiles",
        description="List all available voice profiles",
        parameters={},
    )["function"],

    "21labs.transcribe": tool_schema(
        name="21labs.transcribe",
        description="Transcribe audio to text",
        parameters={
            "audio_file": {"type": "string", "description": "Path to audio file"},
            "language": {"type": "string", "default": "en"},
        },
    )["function"],

    "21labs.mix_audio": tool_schema(
        name="21labs.mix_audio",
        description="Mix multiple audio tracks together",
        parameters={
            "tracks": {"type": "array", "description": "List of track paths"},
            "output_path": {"type": "string", "description": "Output file path"},
        },
    )["function"],
}


def main():
    print("=" * 60)
    print("  small-model-harness: Basic Usage Example")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 2. Create a session (budget auto-derived from context window)
    # ------------------------------------------------------------------

    harness = create_harness_session(n_ctx=4096)
    print(f"\n[Session] ID: {harness.session_id}")
    print(f"[Session] Budget: {harness.budget_remaining} tool calls")
    print(f"[Session] Context: {harness.n_ctx} tokens")

    # ------------------------------------------------------------------
    # 3. Progressive tool disclosure — show only relevant tools
    # ------------------------------------------------------------------

    user_query = "Say hello in a warm voice"
    print(f"\n[Query] {user_query}")

    prompt = build_compact_tool_prompt(user_query, TOOLS, harness, top_n=3)
    print(f"\n[Tool Prompt]\n{prompt}")

    # ------------------------------------------------------------------
    # 4. Repair malformed model output
    # ------------------------------------------------------------------

    # Simulate what a small model actually outputs (trailing comma, string duration)
    raw_output = '{"tool": "21labs.speak", "arguments": {"text": "Hello, welcome to the studio!", "engine": "kokoro", "speed": "1.2",}}'

    print(f"\n[Raw Model Output] {raw_output}")

    args, tool_name, fixes = repair_tool_call(raw_output, TOOLS)

    print(f"[Repaired] tool={tool_name}")
    print(f"[Repaired] args={json.dumps(args, indent=2)}")
    print(f"[Fixes] {fixes}")

    # ------------------------------------------------------------------
    # 5. Score confidence
    # ------------------------------------------------------------------

    score = score_tool_call(tool_name, args, TOOLS.get(tool_name), harness, fixes)
    print(f"\n{format_confidence_summary(score)}")

    if should_escalate_to_cloud(score):
        print("[Action] Escalating to cloud model — low confidence")
    else:
        print("[Action] Proceeding with local execution")

    # ------------------------------------------------------------------
    # 6. Record execution and track session state
    # ------------------------------------------------------------------

    harness.record_execution(ExecutionRecord(
        timestamp="2026-01-01T00:00:00Z",
        tool_name=tool_name,
        arguments=args,
        status="completed",
        duration_ms=450.0,
        confidence=score.overall,
    ))

    # Simulate a failure
    harness.record_execution(ExecutionRecord(
        timestamp="2026-01-01T00:00:01Z",
        tool_name="21labs.generate_sound_effect",
        arguments={"prompt": "thunder"},
        status="failed",
        error="Engine not loaded",
    ))

    # Second failure triggers avoidance
    harness.record_execution(ExecutionRecord(
        timestamp="2026-01-01T00:00:02Z",
        tool_name="21labs.generate_sound_effect",
        arguments={"prompt": "rain"},
        status="failed",
        error="Engine not loaded",
    ))

    print(f"\n[Session State]")
    print(f"  Success rate: {harness.success_rate:.0%}")
    print(f"  Avoided tools: {sorted(harness.avoided_tools)}")
    print(f"  Steering hints: {harness.steering_hints}")

    steering = harness.get_steering_prompt()
    if steering:
        print(f"\n[Steering Prompt for System Message]{steering}")

    # ------------------------------------------------------------------
    # 7. Detect loops
    # ------------------------------------------------------------------

    harness.recent_tool_calls = ["speak", "speak", "speak"]
    loop = detect_failure_loop(harness.recent_tool_calls)
    if loop:
        print(f"\n[Loop Detected] {loop} — model is stuck in a loop!")

    # ------------------------------------------------------------------
    # 8. Compact a large tool response
    # ------------------------------------------------------------------

    large_response = json.dumps({
        "audio_file": "/tmp/output.wav",
        "duration": 5.0,
        "sample_rate": 44100,
        "format": "wav",
        "size_bytes": 88200,
        "effects_applied": ["compressor", "eq"],
        "metadata": {"extra": "data " * 200},
    })

    compacted, was_compacted = compact_tool_response("21labs.speak", large_response, max_tokens=60)
    print(f"\n[Compaction]")
    print(f"  Original: {len(large_response)} chars ({estimate_tokens(large_response)} tokens)")
    print(f"  Compacted: {len(compacted)} chars ({estimate_tokens(compacted)} tokens)")
    print(f"  Result: {compacted}")

    # ------------------------------------------------------------------
    # 9. Compact file content
    # ------------------------------------------------------------------

    long_text = "This is a sentence about audio production. " * 200
    compacted_text, truncated = compact_text(long_text, n_ctx=4096)
    print(f"\n[Text Compaction]")
    print(f"  Original: {len(long_text)} chars")
    print(f"  Compacted: {len(compacted_text)} chars (truncated={truncated})")

    # ------------------------------------------------------------------
    # 10. Audit trail
    # ------------------------------------------------------------------

    from small_model_harness import format_audit_trail
    print(f"\n{format_audit_trail(harness)}")

    print("\n" + "=" * 60)
    print("  Done! All pipeline stages demonstrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
