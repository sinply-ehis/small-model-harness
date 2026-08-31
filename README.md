<p align="center">
  <img src="logo.svg" width="120" alt="small-model-harness logo">
</p>

<h1 align="center">small-model-harness</h1>

<p align="center">
  <strong>Session-level intelligence for small LLMs (2B–4B parameters)</strong>
</p>

<p align="center">
  <a href="#install">Install</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#features">Features</a> ·
  <a href="#api">API</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#research">Research</a>
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/pydantic-v2-ff6b6b.svg" alt="Pydantic v2">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License MIT">
  <img src="https://img.shields.io/badge/142-tests-passing-brightgreen.svg" alt="142 Tests">
  <img src="https://img.shields.io/badge/size-<10KB-lightgrey.svg" alt="Size">
</p>

<p align="center">
  Small models fail on tool calling in predictable ways. This harness<em> compensates</em> for those failures
  deterministically — no extra LLM calls, no retries, no cloud dependency.
</p>

---

## Why this exists

A 2B–4B parameter model will:

1. **Skip tool calls entirely** — acknowledge uncertainty, then confabulate an answer instead of calling the tool
2. **Produce malformed JSON** — trailing commas, wrong types, camelCase instead of snake_case
3. **Select the wrong tool** — when given 20+ options, small models pick randomly
4. **Get stuck in loops** — call the same tool repeatedly hoping for a different result
5. **Run out of context** — verbose tool responses fill the window before the task completes

This harness fixes all five — deterministically, at the session level, without modifying the model.

## Install

```bash
pip install small-model-harness
```

With optional extras:

```bash
pip install "small-model-harness[full]"       # PDF + EPUB support
pip install "small-model-harness[pydantic-deep]"  # Full agent framework
pip install "small-model-harness[dev]"        # Testing tools
```

## Quick start

```python
from small_model_harness import (
    create_harness_session,
    ExecutionRecord,
    repair_tool_call,
    score_tool_call_confidence,
    rank_tools,
    compact_tool_response,
)

# 1. Create a session
session = create_harness_session(n_ctx=4096)

# 2. Before showing tools, rank by relevance (show top 5, not all 211)
ranked = rank_tools("play some music", tool_schemas, top_n=5)
prompt = build_compact_tool_prompt("play some music", tool_schemas, top_n=5)

# 3. Model outputs malformed JSON — repair it deterministically
args, name, fixes = repair_tool_call(
    '{"tool": "play_audio", "arguments": {"filePath": "/a/b.wav", "volume": "0.8"}}',
    tool_schemas,
)
# name = "play_audio"
# args = {"file_path": "/a/b.wav", "volume": 0.8}  # renamed + coerced
# fixes = ["Renamed 'filePath' → 'file_path'", "Converted 'volume' from string to number"]

# 4. Score confidence — should we escalate to cloud?
score = score_tool_call_confidence(name, args, tool_schemas[name], harness=session)
if score.should_escalate:
    escalate_to_cloud()

# 5. Record the execution
session.record_execution(ExecutionRecord(
    timestamp="2026-08-31T12:00:00Z",
    tool_name=name,
    arguments=args,
    status="completed",
    result="Playing audio...",
    duration_ms=340.0,
))

# 6. Get steering hints for the next system prompt
steering = session.get_steering_prompt()
# → "Avoid: play_audio (failed 3 times this session)"

# 7. Compact verbose responses to save context
compacted, was_compacted = compact_tool_response("list_voices", huge_json, max_tokens=200)
```

## Features

### Deterministic tool repair

Fixes malformed output without re-prompting. The model's *intent* is usually correct — only the serialization drifted.

| What breaks | How it's fixed |
|---|---|
| Trailing commas: `{"a": 1,}` | JSON repair strips them |
| Wrong types: `"5"` instead of `5` | Type coercion from schema |
| Key naming: `maxResults` vs `max_results` | Key renaming (camelCase/snake_case) |
| Missing defaults: model forgets optional args | Default injection from schema |
| Markdown fences: ` ```json ... ``` ` | Stripped before parsing |

```python
from small_model_harness import repair_tool_call

args, name, fixes = repair_tool_call(malformed_output, tool_schemas)
# Always returns (args, tool_name, fixes_applied) or (None, None, error)
```

### Confidence scoring

Multi-signal scoring to know when to trust the model vs escalate to cloud.

| Signal | Weight | What it measures |
|---|---|---|
| Schema match | 35% | Are args valid against the tool schema? |
| History | 25% | Has this tool succeeded before in this session? |
| Completeness | 25% | Are required arguments present? |
| Repair cost | 15% | How much did we have to fix? |

```python
from small_model_harness import score_tool_call_confidence

score = score_tool_call_confidence(
    "search_web", args, schema, harness=session, repairs_applied=fixes
)
print(score.overall)        # 0.85
print(score.should_escalate)  # False
print(format_confidence_summary(score))
# Confidence: 85% (HIGH)
#   Schema: 100%
#   History: 90%
#   Completeness: 100%
#   Repair cost: 90%
```

### Progressive tool disclosure

Don't show the model all 211 tools — show the 5 most relevant ones. Reduces selection errors dramatically.

```python
from small_model_harness import rank_tools, build_compact_tool_prompt

# Rank tools by query relevance
ranked = rank_tools("play some music", tool_schemas, top_n=5)
# Returns: [ToolScore(name="play_audio", score=0.85, ...), ...]

# Build a compact prompt with only relevant tools
prompt = build_compact_tool_prompt("play some music", tool_schemas, top_n=5)
# "You have access to the following tools:
#  - play_audio: Play an audio file | params: file_path: string*, volume: number
#  - speak_text: Convert text to speech | params: text: string*, voice: string
#  ..."
```

### Adaptive steering

After repeated failures, inject "avoid this" hints into the system prompt.

```python
session.record_execution(ExecutionRecord(
    timestamp="...", tool_name="play_audio", status="failed", ...
))
session.record_execution(ExecutionRecord(
    timestamp="...", tool_name="play_audio", status="failed", ...
))

steering = session.get_steering_prompt()
# → "Avoid: play_audio (failed 2 times this session)"
```

### Loop detection

Catch when the model calls the same tool repeatedly.

```python
from small_model_harness import detect_failure_loop

calls = ["search", "search", "search"]
tool = detect_failure_loop(calls)
# → "search" (loop detected)
```

### Context pressure monitoring

Track token usage and trigger compaction before overflow.

```python
session.update_context_pressure(estimated_tokens=3500, n_ctx=4096)
print(session.context_pressure)    # 0.85
print(session.is_context_warned)   # True  (≥75%)
print(session.is_context_critical) # False (<90%)
```

### Response compaction

Shrink verbose tool responses to save context space.

```python
from small_model_harness import compact_tool_response

compacted, was_compacted = compact_tool_response(
    "play_audio",
    '{"audio_file": "/long/path/file.wav", "duration": 300, "sample_rate": 44100, ...}',
    max_tokens=50,
)
# → '{"audio_file": "/long/path/file.wav", "duration": 300}' (compacted)
```

## API

### Core functions

| Function | Purpose |
|---|---|
| `create_harness_session(session_id, budget, n_ctx)` | Create a new session |
| `repair_tool_call(raw_output, tool_schemas)` | Fix malformed model output |
| `score_tool_call_confidence(name, args, schema, harness, repairs)` | Score tool call confidence |
| `rank_tools(query, tool_schemas, harness, top_n)` | Rank tools by relevance |
| `build_compact_tool_prompt(query, tool_schemas, harness, top_n)` | Build compact tool prompt |
| `compact_tool_response(tool_name, response, max_tokens)` | Shrink verbose responses |
| `detect_failure_loop(recent_calls, window)` | Detect same-tool loops |
| `estimate_tokens(text)` | Rough token estimate |
| `tool_schema(name, description, properties)` | Generate OpenAI tool schema |
| `pydantic_tool_schema(model, name, description)` | Generate schema from pydantic model |

### HarnessState

| Method | Purpose |
|---|---|
| `record_execution(record)` | Record a tool call, update stats |
| `update_context_pressure(tokens, n_ctx)` | Update pressure estimate |
| `get_steering_prompt()` | Get hints for system prompt |
| `get_avoided_tools_prompt()` | List tools to avoid |
| `format_audit_trail()` | Human-readable execution log |
| `generate_session_summary()` | One-line session summary |

| Property | Type | Meaning |
|---|---|---|
| `success_rate` | float | 0.0–1.0 |
| `is_context_warned` | bool | pressure ≥ 0.75 |
| `is_context_critical` | bool | pressure ≥ 0.90 |
| `is_budget_exhausted` | bool | no calls remaining |
| `steering_hints` | list[str] | Current steering hints |
| `avoided_tools` | set[str] | Tools marked as avoided |

### Models (pydantic v2)

| Model | Fields |
|---|---|
| `HarnessState` | session_id, started_at, budget, execution_history, tool stats, steering, pressure |
| `ExecutionRecord` | timestamp, tool_name, arguments, status, result, error, duration_ms, confidence |
| `FileInput` | path, content, mime_type, size_bytes |
| `ConfidenceScore` | overall, schema_match, history_score, completeness, repair_cost, should_escalate |
| `ToolScore` | name, score, reason, schema |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Your application                                            │
│  (streaming loop, agent, CLI, whatever)                      │
├──────────────────────────────────────────────────────────────┤
│  small-model-harness                                         │
│                                                              │
│  ┌─────────────┐ ┌──────────────┐ ┌────────────────────┐    │
│  │ Tool Repair  │ │  Confidence  │ │  Tool Disclosure   │    │
│  │ JSON fix     │ │  Schema      │ │  Rank by relevance │    │
│  │ Type coerce  │ │  History     │ │  Top-N selection   │    │
│  │ Key rename   │ │  Completeness│ │  Compact prompts   │    │
│  │ Defaults     │ │  Repair cost │ │                    │    │
│  └─────────────┘ └──────────────┘ └────────────────────┘    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │ Session Intelligence                                  │    │
│  │ Failure tracking · Adaptive steering · Loop detection │    │
│  │ Context pressure · Response compaction · Audit trail  │    │
│  └──────────────────────────────────────────────────────┘    │
├──────────────────────────────────────────────────────────────┤
│  llama.cpp / vLLM / OpenAI-compatible endpoint              │
└──────────────────────────────────────────────────────────────┘
```

### How the pieces fit together

```python
# The full pipeline — from raw model output to repaired, scored, tracked call

# 1. Model outputs something broken
raw = model.generate(prompt)

# 2. Repair it
args, name, fixes = repair_tool_call(raw, tool_schemas)

# 3. Score confidence
score = score_tool_call_confidence(name, args, tool_schemas[name], session, fixes)

# 4. If confidence is low, escalate
if score.should_escalate:
    raw = cloud_model.generate(prompt)
    args, name, fixes = repair_tool_call(raw, tool_schemas)

# 5. Execute the tool
result = execute_tool(name, args)

# 6. Compact the response
compacted, _ = compact_tool_response(name, result, max_tokens=200)

# 7. Record execution
session.record_execution(ExecutionRecord(
    timestamp=now(), tool_name=name, arguments=args,
    status="completed", result=compacted, confidence=score.overall,
))

# 8. Feed back to model
messages.append({"role": "tool", "content": compacted})
```

## Research

This harness is informed by:

- **Cho et al. (2026)** — "It's Not the Size: Harness Design Determines Operational Stability in Small Language Models" (arXiv:2605.12129). Found that a 4-stage scaffold takes 2B models from 58% to 95% task success.
- **ManiFreeBird (2026)** — Deterministic tool repair eliminates retry loops. Key insight: "structural failures are not reasoning failures."
- **Gorilla (Patil et al., 2023)** — Tool retrieval reduces selection hallucination. Show 3–5 relevant tools, not all.
- **NVIDIA (2025)** — SLM-first routing with confidence-based cloud escalation keeps 80–90% of steps local.

## Pydantic-deep integration (optional)

For full agent capabilities (context compaction, subagents, skills, memory, planning):

```bash
pip install "small-model-harness[pydantic-deep]"
```

```python
from small_model_harness.pydantic_deep_integration import (
    build_small_model_agent,
    run_with_harness,
)

result = build_small_model_agent(
    model_url="http://localhost:8080/v1",
    model_name="qwen3.5-4b",
    n_ctx=4096,
)

run_result = await run_with_harness(result, "Plan a 3-step audio mixing task")
```

## Running tests

```bash
pip install "small-model-harness[dev]"
pytest                      # Run all 142 tests
pytest tests/test_harness.py  # Core harness tests only
pytest -v                   # Verbose output
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
