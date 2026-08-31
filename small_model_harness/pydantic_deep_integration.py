"""Pydantic-deep integration — optional bridge for full agent capabilities.

Combines small-model-harness session intelligence with pydantic-deep's
agent framework (context compaction, subagents, skills, memory, planning).

Requires: pip install small-model-harness[pydantic-deep]

Usage:
    from small_model_harness.pydantic_deep_integration import build_small_model_agent

    agent, deps, harness = build_small_model_agent(
        model_url="http://localhost:8080/v1",
        model_name="qwen3.5-4b",
        n_ctx=4096,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import (
    HarnessState,
    create_harness_session,
    estimate_messages_tokens,
)

logger = logging.getLogger(__name__)


def _check_pydantic_deep():
    """Lazy import — only fails if user actually tries to use this module."""
    try:
        from pydantic_deep import create_deep_agent, create_default_deps
        from pydantic_deep import create_sliding_window_processor

        return create_deep_agent, create_default_deps, create_sliding_window_processor
    except ImportError as err:
        raise ImportError(
            "pydantic-deep is required for this module. "
            "Install with: pip install small-model-harness[pydantic-deep]"
        ) from err


# ---------------------------------------------------------------------------
# Small-model-optimized agent builder
# ---------------------------------------------------------------------------


@dataclass
class SmallModelAgentResult:
    """Result of build_small_model_agent — agent + deps + harness bundled."""

    agent: Any  # pydantic-deep Agent
    deps: Any  # DeepAgentDeps
    harness: HarnessState  # our session intelligence
    paths: Any | None = None  # ProjectPaths if memory_dir was given


def build_small_model_agent(
    *,
    model_url: str = "http://localhost:8080/v1",
    model_name: str = "qwen3.5-4b",
    model_api_key: str = "not-needed",
    instructions: str | None = None,
    n_ctx: int = 4096,
    session_id: str | None = None,
    # pydantic-deep feature toggles (all default to small-model-friendly)
    include_memory: bool = True,
    include_todo: bool = True,
    include_skills: bool = True,
    include_subagents: bool = True,
    include_plan: bool = False,  # planning costs tokens — off by default for small models
    include_checkpoints: bool = True,
    # storage paths (None = in-memory only)
    memory_dir: str | Path | None = None,
    plans_dir: str | Path | None = None,
    skills_dir: str | Path | None = None,
    checkpoints_dir: str | Path | None = None,
    # context management
    sliding_window_messages: int = 50,  # keep last N messages (free compaction)
    eviction_token_limit: int | None = None,  # evict tool outputs above this
    # MCP
    mcp_servers: list[Any] | None = None,
) -> SmallModelAgentResult:
    """Build a pydantic-deep agent optimized for small local models.

    Combines pydantic-deep's agent framework with our harness's session
    intelligence. Uses sliding-window compaction (free, no LLM call) instead
    of LLM-based summarization (expensive, wasteful for small models).

    Args:
        model_url: OpenAI-compatible endpoint URL.
        model_name: Model name as reported by the server.
        model_api_key: API key (default "not-needed" for llama.cpp).
        instructions: System prompt. Defaults to a small-model-friendly prompt.
        n_ctx: Context window size. Drives budget + compaction thresholds.
        session_id: Custom session ID for the harness.
        include_memory: Persistent MEMORY.md across sessions.
        include_todo: Task planning with subtasks.
        include_skills: Domain-specific skills from SKILL.md files.
        include_subagents: Multi-agent delegation via task() tool.
        include_plan: Structured planning before execution (token-heavy).
        include_checkpoints: Save/rewind conversation state.
        memory_dir: Where to store MEMORY.md. None = in-memory.
        plans_dir: Where to store plans. None = in-memory.
        skills_dir: Where to find SKILL.md files.
        checkpoints_dir: Where to store checkpoints. None = in-memory.
        sliding_window_messages: Keep last N messages (free compaction).
        eviction_token_limit: Evict tool outputs above this token count.
        mcp_servers: List of MCPToolset objects for external tools.

    Returns:
        SmallModelAgentResult with agent, deps, harness, and paths.
    """
    create_deep_agent, create_default_deps, create_sliding_window_processor = (
        _check_pydantic_deep()
    )

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    if instructions is None:
        instructions = (
            "You are a helpful assistant running locally on a small model. "
            "Plan before acting on multi-step tasks. Keep responses concise. "
            "Use your TODO list to track progress across turns."
        )

    # Build model
    model = OpenAIChatModel(model_name, provider=OpenAIProvider(base_url=model_url, api_key=model_api_key))

    # Harness — our session intelligence
    harness = create_harness_session(session_id=session_id, n_ctx=n_ctx)

    # Directory setup
    _memory_dir = str(memory_dir) if memory_dir else None
    _plans_dir = str(plans_dir) if plans_dir else None
    _checkpoints_dir = str(checkpoints_dir) if checkpoints_dir else None

    for d in (_memory_dir, _plans_dir, _checkpoints_dir):
        if d:
            Path(d).mkdir(parents=True, exist_ok=True)

    # Skills — only if directory exists and has content
    skill_dirs = None
    if skills_dir and Path(skills_dir).exists() and any(Path(skills_dir).iterdir()):
        skill_dirs = [str(skills_dir)]

    # Sliding window processor — FREE compaction, no LLM call
    sliding_window = create_sliding_window_processor(
        trigger=("messages", sliding_window_messages * 2),  # trigger at 2x
        keep=("messages", sliding_window_messages),
    )

    # Context pressure callback — feeds into our HarnessState
    def on_context_update(percentage: float, current_tokens: int, max_tokens: int) -> None:
        harness.update_context_pressure(current_tokens, max_tokens)

    # Build agent with small-model-optimized settings
    kwargs: dict[str, Any] = dict(
        model=model,
        instructions=instructions,
        memory_dir=_memory_dir,
        plans_dir=_plans_dir,
        skill_directories=skill_dirs,
        include_checkpoints=include_checkpoints,
        include_memory=include_memory,
        include_todo=include_todo,
        include_skills=include_skills,
        include_subagents=include_subagents,
        include_plan=include_plan,
        # Context management — sliding window is free, no LLM cost
        history_processors=[sliding_window],
        on_context_update=on_context_update,
        # Web search — DuckDuckGo fallback, no API key needed
        web_search=True,
        web_fetch=True,
    )

    if _checkpoints_dir:
        from pydantic_deep import FileCheckpointStore

        kwargs["checkpoint_store"] = FileCheckpointStore(_checkpoints_dir)

    if eviction_token_limit is not None:
        kwargs["eviction_token_limit"] = eviction_token_limit

    if mcp_servers:
        kwargs["mcp_servers"] = mcp_servers

    agent = create_deep_agent(**kwargs)
    deps = create_default_deps()

    return SmallModelAgentResult(
        agent=agent,
        deps=deps,
        harness=harness,
    )


# ---------------------------------------------------------------------------
# Run helper — wraps agent.run() with harness integration
# ---------------------------------------------------------------------------


async def run_with_harness(
    result: SmallModelAgentResult,
    prompt: str,
    message_history: list[dict] | None = None,
) -> Any:
    """Run a prompt through the agent with harness tracking.

    Records tool executions, updates context pressure, and returns
    the result. The harness's steering hints are available via
    result.harness.get_steering_prompt() for the next turn.

    Args:
        result: From build_small_model_agent().
        prompt: The user message.
        message_history: Optional prior messages.

    Returns:
        pydantic-deep RunResult with .output and .new_messages().
    """
    from pydantic_ai.exceptions import ModelAPIError

    try:
        run_result = await result.agent.run(
            prompt,
            deps=result.deps,
            message_history=message_history,
        )
    except ModelAPIError as exc:
        raise ConnectionError(
            f"Can't reach model server. Check that llama.cpp is running.\n"
            f"Original error: {exc}"
        ) from exc

    return run_result
