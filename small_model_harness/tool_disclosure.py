"""Progressive tool disclosure — show the model fewer, better tools.

Small models (2B-4B) fail on tool selection when presented with too many
options. This module scores tools by relevance to the current query and
returns only the top N, reducing selection confusion.

Gorilla (Patil et al., 2023) and BuildingEffectiveAgents (2026) both
identify this as a core mitigation for hallucinated tool selection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from . import HarnessState


@dataclass
class ToolScore:
    """A tool with its relevance score."""

    name: str
    score: float
    reason: str
    schema: dict[str, Any]


def rank_tools(
    query: str,
    tool_schemas: dict[str, dict[str, Any]],
    harness: HarnessState | None = None,
    top_n: int = 5,
) -> list[ToolScore]:
    """Rank tools by relevance to the current query.

    Scoring signals:
    - Keyword overlap: do tool description/name match query terms?
    - Historical success: has this tool worked before?
    - Avoided penalty: is this tool in the avoided list?
    - Category match: does the tool category match the query intent?

    Args:
        query: The user's message/query.
        tool_schemas: Map of tool_name → schema (must include "description").
        harness: Session state for history lookups.
        top_n: Number of tools to return.

    Returns:
        List of top_n ToolScore objects, sorted by relevance.
    """
    scores: list[ToolScore] = []
    query_lower = query.lower()
    query_tokens = set(_tokenize(query_lower))

    for name, schema in tool_schemas.items():
        score, reason = _score_tool(name, schema, query_lower, query_tokens, harness)
        scores.append(ToolScore(name=name, score=score, reason=reason, schema=schema))

    # Sort by score descending
    scores.sort(key=lambda x: x.score, reverse=True)

    return scores[:top_n]


def get_tool_subset(
    query: str,
    tool_schemas: dict[str, dict[str, Any]],
    harness: HarnessState | None = None,
    top_n: int = 5,
) -> dict[str, dict[str, Any]]:
    """Get a subset of tool schemas most relevant to the query.

    Returns only the top_n tools as a dict suitable for passing to
    repair_tool_call or for generating a compact tool prompt.
    """
    ranked = rank_tools(query, tool_schemas, harness, top_n)
    return {ts.name: ts.schema for ts in ranked}


def build_compact_tool_prompt(
    query: str,
    tool_schemas: dict[str, dict[str, Any]],
    harness: HarnessState | None = None,
    top_n: int = 5,
) -> str:
    """Build a compact tool prompt showing only relevant tools.

    For small models: instead of listing all tools, show only the
    top_n most relevant ones. This dramatically reduces selection errors.
    """
    ranked = rank_tools(query, tool_schemas, harness, top_n)

    lines = ["You have access to the following tools:"]
    for ts in ranked:
        desc = ts.schema.get("description", "No description")
        params = ts.schema.get("properties", {})
        required = ts.schema.get("required", [])

        param_parts = []
        for pname, pschema in params.items():
            ptype = pschema.get("type", "any")
            default = pschema.get("default")
            req = "*" if pname in required else ""
            if default is not None:
                param_parts.append(f"{pname}: {ptype}{req} (default: {default})")
            else:
                param_parts.append(f"{pname}: {ptype}{req}")

        params_str = ", ".join(param_parts) if param_parts else "no parameters"
        lines.append(f"- {ts.name}: {desc} | params: {params_str}")

    lines.append("")
    lines.append("Select the most appropriate tool for the task.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scoring internals
# ---------------------------------------------------------------------------


def _score_tool(
    name: str,
    schema: dict[str, Any],
    query_lower: str,
    query_tokens: set[str],
    harness: HarnessState | None,
) -> tuple[float, str]:
    """Score a single tool's relevance to the query."""
    score = 0.0
    reasons: list[str] = []

    description = schema.get("description", "").lower()
    name_lower = name.lower()

    # --- Keyword overlap ---
    desc_tokens = set(_tokenize(description))
    name_tokens = set(_tokenize(name_lower))

    # Name match (highest signal)
    for token in query_tokens:
        if token in name_tokens:
            score += 0.3
            reasons.append(f"name match: {token}")

    # Description match
    overlap = query_tokens & desc_tokens
    if overlap:
        boost = min(0.4, len(overlap) * 0.08)
        score += boost
        reasons.append(f"desc match: {len(overlap)} tokens")

    # --- Historical success ---
    if harness:
        successes = harness.tool_success_counts.get(name, 0)
        failures = harness.tool_failure_counts.get(name, 0)
        total = successes + failures

        if total > 0:
            success_rate = successes / total
            score += success_rate * 0.2
            reasons.append(f"history: {success_rate:.0%}")

        # Penalty for avoided tools
        if name in harness.avoided_tools:
            score -= 0.5
            reasons.append("avoided")

    # --- Category hints from query ---
    category = _detect_query_category(query_lower)
    tool_category = schema.get("category", "")
    if category and tool_category:
        if category == tool_category:
            score += 0.15
            reasons.append(f"category match: {category}")

    # Clamp
    score = max(0.0, min(1.0, score))

    reason = "; ".join(reasons) if reasons else "no signals"
    return score, reason


def _tokenize(text: str) -> list[str]:
    """Split text into meaningful tokens."""
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "both", "each", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "just", "don", "now", "and", "but", "or",
        "if", "this", "that", "these", "those", "i", "me", "my", "you", "your",
        "it", "its", "we", "our", "they", "them", "their", "what", "which",
        "who", "whom", "up", "about", "get", "got", "make", "made",
    }

    # Split on non-alphanumeric
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if t and t not in stop_words and len(t) > 1]


def _detect_query_category(query_lower: str) -> str:
    """Detect the intent category of the query."""
    # Audio-related
    audio_keywords = {"speak", "voice", "audio", "music", "sound", "tts", "stt", "speech", "sing", "play", "mix", "master"}
    if any(kw in query_lower for kw in audio_keywords):
        return "audio"

    # File-related
    file_keywords = {"file", "read", "write", "save", "load", "open", "folder", "directory", "path"}
    if any(kw in query_lower for kw in file_keywords):
        return "file"

    # Search-related
    search_keywords = {"search", "find", "look", "query", "web", "internet", "google"}
    if any(kw in query_lower for kw in search_keywords):
        return "search"

    # Analysis-related
    analysis_keywords = {"analyze", "analyse", "measure", "check", "verify", "inspect", "test"}
    if any(kw in query_lower for kw in analysis_keywords):
        return "analysis"

    return ""
