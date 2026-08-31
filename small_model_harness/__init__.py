"""Small-model harness — session memory, context pressure, adaptive steering.

Provides budget tracking, audit trail, result quality scoring, file handling
(compaction + type filtering + transcription), and session-level intelligence
for small on-device models (2B--4B).

Key capabilities for tiny models:
- Session memory: track what worked/failed, prevent repetition
- Context pressure: monitor token usage, trigger compaction
- Adaptive steering: inject "avoid this" hints after failures
- Loop detection: break same-tool-same-args repetition
- Response compaction: extract essentials from tool responses

Built on pydantic v2 for validated models, JSON schema generation, and
clean serialization.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__version__ = "0.2.0"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".epub"}
SUPPORTED_MIME_TYPES = {
    "text/markdown",
    "text/plain",
    "application/pdf",
    "application/epub+zip",
    "application/octet-stream",
}
MAX_FILE_SIZE = 10 * 1024 * 1024

# Absolute ceiling for compacted tokens (used when n_ctx is unknown)
_ABSOLUTE_MAX_TOKENS = 8000

# Small-model thresholds
_FAILURE_STEER_THRESHOLD = 2  # failures before marking tool as "avoid"
_LOOP_DETECTION_WINDOW = 3  # consecutive same-tool calls to trigger loop break
_CONTEXT_PRESSURE_WARN = 0.75  # warn when context is 75% full
_CONTEXT_PRESSURE_CRITICAL = 0.9  # trigger aggressive compaction at 90%
_RECENT_HISTORY_SIZE = 10  # last N tool calls for loop detection
_MAX_STEERING_HINTS = 5  # cap hints to avoid prompt bloat


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

ExecutionStatus = Literal["planned", "running", "completed", "failed", "skipped"]


class FileInput(BaseModel):
    """Represents a file input to the assistant."""

    model_config = ConfigDict(frozen=True)

    path: str
    filename: str
    extension: str
    mime_type: str
    size_bytes: int = Field(ge=0)
    content: str = ""
    compacted: bool = False
    truncated: bool = False
    error: str | None = None

    @property
    def is_supported(self) -> bool:
        return self.extension.lower() in SUPPORTED_EXTENSIONS

    @property
    def is_text_based(self) -> bool:
        return self.extension.lower() in {".md", ".txt"}

    @property
    def needs_transcription(self) -> bool:
        return self.extension.lower() in {".pdf", ".epub"}


class ExecutionRecord(BaseModel):
    """Audit trail entry for a single tool execution."""

    model_config = ConfigDict(frozen=True)

    timestamp: str
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: ExecutionStatus
    result: str | None = None
    error: str | None = None
    duration_ms: float = Field(default=0.0, ge=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    retry_count: int = Field(default=0, ge=0)


class HarnessState(BaseModel):
    """Complete harness state for a conversation session.

    Tracks session-level intelligence for small models: tool success/failure
    stats, recent call history for loop detection, context pressure, and
    dynamic steering hints injected into the system prompt.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        json_schema_extra={
            "examples": [
                {
                    "session_id": "abc123",
                    "started_at": "2026-01-01T00:00:00Z",
                    "budget_remaining": 8,
                    "n_ctx": 4096,
                }
            ]
        },
    )

    session_id: str
    started_at: str
    total_tool_calls: int = Field(default=0, ge=0)
    successful_calls: int = Field(default=0, ge=0)
    failed_calls: int = Field(default=0, ge=0)
    total_duration_ms: float = Field(default=0.0, ge=0.0)
    budget_remaining: int = Field(default=100, ge=0)
    execution_history: list[ExecutionRecord] = Field(default_factory=list)
    file_inputs: list[FileInput] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # --- Small-model session memory ---
    tool_failure_counts: dict[str, int] = Field(default_factory=dict)
    tool_success_counts: dict[str, int] = Field(default_factory=dict)
    recent_tool_calls: list[str] = Field(default_factory=list)
    avoided_tools: set[str] = Field(default_factory=set)
    steering_hints: list[str] = Field(default_factory=list)
    context_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    n_ctx: int = Field(default=0, ge=0)

    def record_execution(self, record: ExecutionRecord) -> None:
        """Record a tool execution and update session intelligence."""
        self.execution_history.append(record)
        self.total_tool_calls += 1
        self.total_duration_ms += record.duration_ms
        self.budget_remaining -= 1

        if record.status == "completed":
            self.successful_calls += 1
            self.tool_success_counts[record.tool_name] = self.tool_success_counts.get(record.tool_name, 0) + 1
        elif record.status == "failed":
            self.failed_calls += 1
            self.tool_failure_counts[record.tool_name] = self.tool_failure_counts.get(record.tool_name, 0) + 1

        # Track recent calls for loop detection
        self.recent_tool_calls.append(record.tool_name)
        if len(self.recent_tool_calls) > _RECENT_HISTORY_SIZE:
            self.recent_tool_calls = self.recent_tool_calls[-_RECENT_HISTORY_SIZE:]

        # Update steering hints
        self._update_steering_hints()

    def _update_steering_hints(self) -> None:
        """Regenerate steering hints from current session state."""
        hints: list[str] = []

        # Tools that failed too many times
        for tool, count in sorted(self.tool_failure_counts.items(), key=lambda x: -x[1]):
            if count >= _FAILURE_STEER_THRESHOLD:
                self.avoided_tools.add(tool)
                success = self.tool_success_counts.get(tool, 0)
                hints.append(
                    f"Do NOT use {tool} — it failed {count} time(s) ({success} success, {count} failures this session)."
                )

        # Loop detection
        loop_tool = detect_failure_loop(self.recent_tool_calls)
        if loop_tool:
            hints.append(f"You are repeating {loop_tool} in a loop. Stop and try a completely different approach.")

        # Context pressure
        if self.context_pressure >= _CONTEXT_PRESSURE_CRITICAL:
            hints.append("Context is nearly full. Give a concise final answer now — do not call more tools.")
        elif self.context_pressure >= _CONTEXT_PRESSURE_WARN:
            hints.append("Context is getting full. Minimize tool calls and wrap up soon.")

        self.steering_hints = hints[:_MAX_STEERING_HINTS]

    def update_context_pressure(self, estimated_tokens: int, n_ctx: int) -> None:
        """Update context pressure estimate (0.0 to 1.0)."""
        self.n_ctx = n_ctx
        if n_ctx > 0:
            self.context_pressure = min(1.0, estimated_tokens / n_ctx)
        self._update_steering_hints()

    def get_steering_prompt(self) -> str:
        """Get steering hints to inject into the system prompt.

        Returns empty string when no hints are active.
        """
        if not self.steering_hints:
            return ""
        return "\n\n# Session Guidance\n" + "\n".join(f"- {h}" for h in self.steering_hints)

    def get_avoided_tools_prompt(self) -> str:
        """List tools the model should avoid in this session."""
        if not self.avoided_tools:
            return ""
        names = ", ".join(sorted(self.avoided_tools))
        return f"\n\nAvoid these tools (they failed this session): {names}"

    @property
    def success_rate(self) -> float:
        if self.total_tool_calls == 0:
            return 1.0
        return self.successful_calls / self.total_tool_calls

    @property
    def is_budget_exhausted(self) -> bool:
        return self.budget_remaining <= 0

    @property
    def is_context_critical(self) -> bool:
        return self.context_pressure >= _CONTEXT_PRESSURE_CRITICAL

    @property
    def is_context_warned(self) -> bool:
        return self.context_pressure >= _CONTEXT_PRESSURE_WARN


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------


def create_harness_session(
    session_id: str | None = None,
    budget: int | None = None,
    n_ctx: int | None = None,
) -> HarnessState:
    """Create a new harness session with a fresh budget.

    ``budget`` caps tool calls per session.  When omitted, defaults to 100.
    The caller should derive a tighter budget from n_ctx for small models:
    ``min(100, max(5, n_ctx // 500))``.

    ``n_ctx`` sets the context window size for pressure monitoring.
    """
    if session_id is None:
        session_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:12]

    if budget is None:
        if n_ctx is not None:
            budget = min(100, max(5, n_ctx // 500))
        else:
            budget = 100

    return HarnessState(
        session_id=session_id,
        started_at=datetime.now(timezone.utc).isoformat(),
        budget_remaining=budget,
        n_ctx=n_ctx or 0,
    )


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------


def detect_failure_loop(
    recent_calls: list[str],
    window: int = _LOOP_DETECTION_WINDOW,
) -> str | None:
    """Detect if the model is stuck calling the same tool repeatedly.

    Returns the tool name if a loop is detected, None otherwise.
    """
    if len(recent_calls) < window:
        return None
    last_n = recent_calls[-window:]
    if len(set(last_n)) == 1:
        return last_n[0]
    return None


# ---------------------------------------------------------------------------
# Context pressure estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English)."""
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """Estimate total token count for a message list."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if content:
            total += estimate_tokens(content)
        # Tool calls add overhead
        if "tool_calls" in msg:
            total += estimate_tokens(json.dumps(msg["tool_calls"]))
    return total


# ---------------------------------------------------------------------------
# Smart response compaction
# ---------------------------------------------------------------------------


def compact_tool_response(
    tool_name: str,
    response: str,
    max_tokens: int = 500,
) -> tuple[str, bool]:
    """Compact a tool response to essential information.

    Different strategies per tool type:
    - JSON responses: extract key fields, drop verbose metadata
    - List responses: summarize count + first few items
    - Error responses: preserve the error message
    - Audio file responses: keep path + metadata only

    Returns (compacted_response, was_compacted).
    """
    if not response:
        return response, False

    estimated = estimate_tokens(response)
    if estimated <= max_tokens:
        return response, False

    # Error responses — always keep full
    if response.startswith(("Error", "Tool execution failed")):
        return response, False

    # Try JSON extraction
    try:
        data = json.loads(response)
        compacted = _compact_json_response(tool_name, data, max_tokens)
        if compacted != response:
            return compacted, True
    except (json.JSONDecodeError, TypeError):
        pass

    # List-like responses — summarize
    if response.startswith(("[", "List of", "Available")):
        lines = response.split("\n")
        if len(lines) > 20:
            summary = "\n".join(lines[:10]) + f"\n\n[... {len(lines) - 10} more items ...]"
            return summary, True

    # Generic truncation as fallback
    max_chars = max_tokens * 4
    if len(response) > max_chars:
        return response[:max_chars] + "\n[... truncated ...]", True

    return response, False


def _compact_json_response(tool_name: str, data: Any, max_tokens: int) -> str:
    """Extract essential info from a JSON tool response."""
    if isinstance(data, dict):
        # Audio file responses — keep path + key metadata
        if "audio_file" in data or "file_path" in data:
            essential = {}
            for key in ("audio_file", "file_path", "duration", "sample_rate", "format", "size_bytes"):
                if key in data:
                    essential[key] = data[key]
            if "effects_applied" in data:
                essential["effects_applied"] = data["effects_applied"]
            compacted = json.dumps(essential, indent=None)
            if estimate_tokens(compacted) <= max_tokens:
                return compacted

        # Profile/voice responses — keep names and IDs
        if "profiles" in data or "voices" in data or "items" in data:
            list_key = next((k for k in ("profiles", "voices", "items") if k in data), None)
            if list_key and isinstance(data[list_key], list):
                items = data[list_key]
                compacted_items = []
                for item in items[:10]:
                    if isinstance(item, dict):
                        compacted_items.append(
                            {k: item[k] for k in ("id", "name", "type", "language", "status") if k in item}
                        )
                    else:
                        compacted_items.append(str(item)[:100])
                data[list_key] = compacted_items
                if len(items) > 10:
                    data[f"{list_key}_count"] = len(items)
                compacted = json.dumps(data, indent=None)
                if estimate_tokens(compacted) <= max_tokens:
                    return compacted

        # Loudness/analysis responses — keep numbers
        if any(k in data for k in ("lufs", "loudness", "peak", "rms", "pitch")):
            essential = {k: v for k, v in data.items() if isinstance(v, (int, float, str))}
            compacted = json.dumps(essential, indent=None)
            if estimate_tokens(compacted) <= max_tokens:
                return compacted

    # List responses
    if isinstance(data, list) and len(data) > 10:
        compacted = json.dumps(data[:10]) + f" [+{len(data) - 10} more]"
        if estimate_tokens(compacted) <= max_tokens:
            return compacted

    # Fallback: full JSON
    return json.dumps(data, indent=None)


# ---------------------------------------------------------------------------
# File handling: compaction, type filtering, transcription
# ---------------------------------------------------------------------------


def validate_file_input(file_path: str | Path) -> FileInput:
    """Validate a file input, check type and size limits."""
    path = Path(file_path)
    stat = path.stat()
    extension = path.suffix.lower()
    mime_type, _ = mimetypes.guess_type(str(path))

    file_input = FileInput(
        path=str(path),
        filename=path.name,
        extension=extension,
        mime_type=mime_type or "application/octet-stream",
        size_bytes=stat.st_size,
    )

    if not file_input.is_supported:
        error_msg = f"Unsupported file type: {extension}. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        logger.warning("Rejected file: %s (%s)", path.name, error_msg)
        return file_input.model_copy(update={"error": error_msg})

    if stat.st_size > MAX_FILE_SIZE:
        error_msg = f"File too large: {stat.st_size} bytes (max {MAX_FILE_SIZE})"
        logger.warning("Rejected file: %s (%s)", path.name, error_msg)
        return file_input.model_copy(update={"error": error_msg})

    return file_input


def read_text_file(path: Path) -> str:
    """Read a text file with encoding fallback."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def compact_text(
    content: str,
    max_tokens: int | None = None,
    n_ctx: int | None = None,
) -> tuple[str, bool]:
    """Compact text content by removing excessive whitespace and truncating.

    When ``n_ctx`` is provided, the token budget is derived as
    ``min(_ABSOLUTE_MAX_TOKENS, n_ctx // 4)`` — file content should never
    exceed a quarter of the context window.  ``max_tokens`` overrides this
    when given.

    Returns (compacted_content, was_truncated).
    """
    if max_tokens is None:
        if n_ctx is not None:
            max_tokens = min(_ABSOLUTE_MAX_TOKENS, max(512, n_ctx // 4))
        else:
            max_tokens = _ABSOLUTE_MAX_TOKENS

    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r"[ \t]+\n", "\n", content)

    estimated_tokens = len(content) // 4
    if estimated_tokens <= max_tokens:
        return content, False

    max_chars = max_tokens * 4
    keep_start = int(max_chars * 0.7)
    keep_end = max_chars - keep_start

    start_part = content[:keep_start]
    end_part = content[-keep_end:] if keep_end > 0 else ""
    return start_part + "\n\n[... content truncated ...]\n\n" + end_part, True


def transcribe_pdf(path: Path) -> str:
    """Extract text from a PDF file."""
    text_parts: list[str] = []

    try:
        import fitz  # type: ignore

        doc = fitz.open(str(path))
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)
    except ImportError:
        pass
    except Exception as e:
        logger.warning("PyMuPDF failed for %s: %s", path, e)

    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n\n".join(text_parts)
    except ImportError:
        pass
    except Exception as e:
        logger.warning("pdfplumber failed for %s: %s", path, e)

    raise RuntimeError("PDF transcription requires PyMuPDF or pdfplumber. Install with: pip install pymupdf")


def transcribe_epub(path: Path) -> str:
    """Extract text from an EPUB file."""
    try:
        import ebooklib  # type: ignore
        from bs4 import BeautifulSoup  # type: ignore
        from ebooklib import epub  # type: ignore
    except ImportError as err:
        raise RuntimeError(
            "EPUB transcription requires ebooklib and beautifulsoup4. Install with: pip install ebooklib beautifulsoup4"
        ) from err

    book = epub.read_epub(str(path))
    text_parts: list[str] = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        if text:
            text_parts.append(text)

    return "\n\n".join(text_parts)


def process_file_input(file_path: str | Path, n_ctx: int | None = None) -> FileInput:
    """Process a file input: validate, read, compact, transcribe if needed."""
    file_input = validate_file_input(file_path)
    if file_input.error:
        return file_input

    path = Path(file_path)

    try:
        if file_input.is_text_based:
            content = read_text_file(path)
        elif file_input.extension.lower() == ".pdf":
            content = transcribe_pdf(path)
        elif file_input.extension.lower() == ".epub":
            content = transcribe_epub(path)
        else:
            return file_input.model_copy(update={"error": f"Unhandled file type: {file_input.extension}"})

        compacted_content, was_truncated = compact_text(content, n_ctx=n_ctx)

        logger.info(
            "Processed file: %s (%d chars, compacted=%s, truncated=%s)",
            path.name,
            len(content),
            True,
            was_truncated,
        )

        return file_input.model_copy(
            update={
                "content": compacted_content,
                "compacted": True,
                "truncated": was_truncated,
            }
        )

    except Exception as e:
        logger.exception("File processing failed for %s", path)
        return file_input.model_copy(update={"error": f"Failed to process file: {e}"})


def process_multiple_files(
    file_paths: list[str | Path],
    n_ctx: int | None = None,
) -> list[FileInput]:
    """Process multiple file inputs, returning only successfully processed ones."""
    results: list[FileInput] = []
    for fp in file_paths:
        file_input = process_file_input(fp, n_ctx=n_ctx)
        if file_input.error:
            logger.warning("Skipping file %s: %s", fp, file_input.error)
        else:
            results.append(file_input)
    return results


# ---------------------------------------------------------------------------
# Harness execution helpers
# ---------------------------------------------------------------------------


def validate_result_quality(
    tool_name: str,
    result: str,
    expected_type: str | None = None,
) -> tuple[bool, list[str]]:
    """Validate result quality after tool execution.

    Returns (is_valid, issues). Issues is a list of quality concerns.
    """
    issues: list[str] = []

    if not result or result.strip() == "":
        issues.append("Empty result returned")
        return False, issues

    if result.startswith(("Error ", "Tool execution")):
        issues.append("Result indicates execution error")
        return False, issues

    if expected_type == "json":
        try:
            json.loads(result)
        except json.JSONDecodeError:
            issues.append("Expected JSON result but got invalid JSON")
            return False, issues

    if expected_type == "audio" and "audio_file" not in result and "bytes" not in result:
        issues.append("Expected audio file metadata but got unexpected format")
        return False, issues

    return len(issues) == 0, issues


def format_audit_trail(harness: HarnessState) -> str:
    """Format the execution audit trail as a readable summary."""
    lines = [
        "=== Harness Audit Trail ===",
        f"Session: {harness.session_id}",
        f"Started: {harness.started_at}",
        f"Tool calls: {harness.total_tool_calls} ({harness.successful_calls} ok, {harness.failed_calls} failed)",
        f"Success rate: {harness.success_rate:.1%}",
        f"Total time: {harness.total_duration_ms:.0f}ms",
        f"Budget remaining: {harness.budget_remaining}",
        f"Context pressure: {harness.context_pressure:.0%}",
        "",
    ]

    if harness.file_inputs:
        lines.append("File inputs:")
        for f in harness.file_inputs:
            status = "OK" if not f.error else f"ERROR: {f.error}"
            lines.append(f"  - {f.filename} ({f.extension}, {f.size_bytes} bytes) [{status}]")
        lines.append("")

    if harness.execution_history:
        lines.append("Execution history:")
        for i, record in enumerate(harness.execution_history, 1):
            status_marker = "ok" if record.status == "completed" else "FAIL"
            conf = f" (conf={record.confidence:.2f})" if record.confidence else ""
            lines.append(f"  {i}. {status_marker} {record.tool_name} ({record.duration_ms:.0f}ms{conf})")
            if record.error:
                lines.append(f"     ERROR: {record.error}")

    if harness.avoided_tools:
        lines.append("")
        lines.append(f"Avoided tools: {', '.join(sorted(harness.avoided_tools))}")

    if harness.steering_hints:
        lines.append("")
        lines.append("Steering hints:")
        for h in harness.steering_hints:
            lines.append(f"  -> {h}")

    if harness.warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in harness.warnings:
            lines.append(f"  ! {w}")

    return "\n".join(lines)


def generate_session_summary(harness: HarnessState) -> str:
    """Generate a concise session summary for the model/user."""
    avoid = f", avoiding: {', '.join(sorted(harness.avoided_tools))}" if harness.avoided_tools else ""
    return (
        f"Session {harness.session_id}: "
        f"{harness.total_tool_calls} calls, "
        f"{harness.success_rate:.0%} success, "
        f"{harness.total_duration_ms:.0f}ms total, "
        f"budget: {harness.budget_remaining} remaining, "
        f"pressure: {harness.context_pressure:.0%}{avoid}"
    )


# ---------------------------------------------------------------------------
# JSON Schema generation (for small model tool definitions)
# ---------------------------------------------------------------------------


def tool_schema(
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """Generate an OpenAI-compatible tool schema from parameters.

    Useful for generating tool definitions for small models that need
    compact, well-structured schemas.

    Example:
        schema = tool_schema(
            name="search_web",
            description="Search the web for information",
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 5},
            }
        )
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": [k for k, v in parameters.items() if "default" not in v],
            },
        },
    }


def pydantic_tool_schema(model: type[BaseModel], name: str, description: str) -> dict[str, Any]:
    """Generate a tool schema from a pydantic BaseModel.

    Uses the model's JSON schema to generate OpenAI-compatible tool definition.

    Example:
        class SearchArgs(BaseModel):
            query: str = Field(description="Search query")
            max_results: int = Field(default=5, description="Max results")

        schema = pydantic_tool_schema(SearchArgs, "search_web", "Search the web")
    """
    schema = model.model_json_schema()
    # Remove pydantic-specific keys
    for key in ("title", "$schema", "definitions"):
        schema.pop(key, None)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": schema,
        },
    }


# ---------------------------------------------------------------------------
# Tool repair (deterministic, no LLM call)
# ---------------------------------------------------------------------------

from .tool_repair import (
    repair_json,
    coerce_types,
    rename_keys,
    inject_defaults,
    repair_tool_call,
)

# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

from .confidence import (
    ConfidenceScore,
    score_tool_call as score_tool_call_confidence,
    should_escalate_to_cloud,
    format_confidence_summary,
)

# ---------------------------------------------------------------------------
# Progressive tool disclosure
# ---------------------------------------------------------------------------

from .tool_disclosure import (
    ToolScore,
    rank_tools,
    get_tool_subset,
    build_compact_tool_prompt,
)


# ---------------------------------------------------------------------------
# Thinking model detection
# ---------------------------------------------------------------------------

import re as _re

NON_THINKING_MODEL_MARKERS: tuple[str, ...] = (
    "chatterbox", "piper", "whisper", "bark", "tts-", "stt-",
)


def is_thinking_model(model_id: str) -> bool:
    """Detect whether a model is a thinking model based on its ID.

    Thinking models emit ``<think>`` blocks that would be suppressed by
    a grammar constraint.  Returns True unless the model ID matches a known
    non-thinking model marker.
    """
    lower = model_id.lower()
    return not any(marker in lower for marker in NON_THINKING_MODEL_MARKERS)


def split_thinking(text: str) -> tuple[str, str]:
    """Split a model response into ``(thinking, visible)`` parts.

    Extracts ``<think>...</think>`` blocks as silent reasoning. If no
    thinking block is present the whole text is returned as visible with an
    empty thinking part.
    """
    think_pattern = _re.compile(r"<think>(.*?)</think>", _re.DOTALL)
    thinking_parts: list[str] = []
    last_end = 0
    visible_parts: list[str] = []

    for match in think_pattern.finditer(text):
        before = text[last_end : match.start()]
        if before:
            visible_parts.append(before)
        thinking_parts.append(match.group(1).strip())
        last_end = match.end()

    tail = text[last_end:]
    if tail:
        visible_parts.append(tail)

    thinking = "\n\n".join(p for p in thinking_parts if p)
    visible = "".join(visible_parts).strip()
    if not thinking:
        visible = text.strip()
    return thinking, visible
