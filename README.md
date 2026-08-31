# small-model-harness

Session-level intelligence for small LLMs (2B–4B parameters).

Makes small models behave like bigger ones by tracking what works, avoiding what doesn't, and compressing context before it overflows.

## What it does

| Capability | What it solves |
|---|---|
| **Failure tracking** | Model keeps retrying broken tools → now it learns to avoid them |
| **Adaptive steering** | Injects "do NOT use X" hints into the system prompt after failures |
| **Loop detection** | Same tool called N times in a row → breaks the loop with a warning |
| **Context pressure** | Monitors token usage, warns at 75%, triggers compaction at 90% |
| **Response compaction** | Extracts essentials from verbose tool responses (audio metadata, list summaries, JSON fields) |
| **Audit trail** | Full execution history with timing, confidence, and retry counts |

## Install

```bash
pip install small-model-harness
```

With optional PDF/EPUB support:

```bash
pip install "small-model-harness[all]"
```

## Quick start

```python
from small_model_harness import create_harness_session, compact_tool_response

# Create a session
session = create_harness_session(n_ctx=4096)

# After each tool call, record it
from small_model_harness import ExecutionRecord, ExecutionStatus

record = ExecutionRecord(
    timestamp="2026-08-31T12:00:00Z",
    tool_name="search_web",
    arguments={"query": "python async"},
    status=ExecutionStatus.COMPLETED,
    result="Found 10 results...",
    duration_ms=340.0,
)
session.record_execution(record)

# Get steering hints for the system prompt
steering = session.get_steering_prompt()
if steering:
    system_prompt += steering

# Compact verbose tool responses
compacted, was_compacted = compact_tool_response(
    "list_voices",
    '{"profiles": [{"id": "v1", "name": "Alice", ...}, ...100 more...]}',
    max_tokens=200,
)
```

## API

### Session management

```python
create_harness_session(session_id=None, budget=None, n_ctx=None)
```

- `session_id`: Unique ID (auto-generated if omitted)
- `budget`: Max tool calls per session (default: `min(100, max(5, n_ctx // 500))`)
- `n_ctx`: Context window size for pressure monitoring

### HarnessState

The main state object. Key methods:

| Method | Purpose |
|---|---|
| `record_execution(record)` | Record a tool call, update stats |
| `update_context_pressure(tokens, n_ctx)` | Update pressure estimate (0.0–1.0) |
| `get_steering_prompt()` | Get hints to inject into system prompt |
| `get_avoided_tools_prompt()` | List tools to avoid |
| `to_dict()` | Serialize full state |

Key properties:

| Property | Type | Meaning |
|---|---|---|
| `success_rate` | float | 0.0–1.0 |
| `is_context_warned` | bool | pressure ≥ 0.75 |
| `is_context_critical` | bool | pressure ≥ 0.90 |
| `is_budget_exhausted` | bool | no calls remaining |

### Response compaction

```python
compact_tool_response(tool_name, response, max_tokens=500)
```

Returns `(compacted_response, was_compacted)`. Strategies:
- **Audio files**: Extract path + metadata only
- **Lists**: Summarize count + first 10 items
- **JSON**: Extract key fields, drop verbose metadata
- **Generic**: Truncate with `[... truncated ...]`

### Loop detection

```python
detect_failure_loop(recent_calls, window=3)
```

Returns tool name if same tool called `window` times consecutively, else `None`.

### Token estimation

```python
estimate_tokens(text)           # ~4 chars per token
estimate_messages_tokens(msgs)  # Estimate for message list
```

### File handling

```python
process_file_input(path, n_ctx=None)       # Single file
process_multiple_files(paths, n_ctx=None)  # Batch
```

Supported: `.md`, `.txt`, `.pdf`, `.epub`

## Design philosophy

This is a **session intelligence layer**, not an agent framework. It sits between your tool-calling loop and the model, providing:

- **Zero model dependency**: Works with any OpenAI-compatible endpoint
- **Zero framework dependency**: Pure Python, no pydantic-ai, no langchain
- **Composable**: Use `HarnessState` in your existing streaming loop
- **Observable**: Full audit trail for debugging small model behavior

## License

MIT
