"""Confidence scoring — know when to trust the model and when to escalate.

Scores tool call confidence from multiple signals:
- Schema match: are args valid against the schema?
- Historical success: has this tool worked before in this session?
- Argument completeness: are required args present?
- Repair cost: how much did we have to fix?

Low confidence = escalate to cloud or ask user for clarification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import HarnessState


@dataclass
class ConfidenceScore:
    """Detailed confidence breakdown for a tool call."""

    overall: float  # 0.0 to 1.0
    schema_match: float  # 0.0 to 1.0 — args valid against schema?
    history_score: float  # 0.0 to 1.0 — has this tool succeeded before?
    completeness: float  # 0.0 to 1.0 — required args present?
    repair_cost: float  # 0.0 to 1.0 — 1.0 = no repairs needed

    should_escalate: bool  # True if confidence is below threshold
    reason: str  # Human-readable reason for low confidence

    def to_dict(self) -> dict:
        return {
            "overall": round(self.overall, 3),
            "schema_match": round(self.schema_match, 3),
            "history_score": round(self.history_score, 3),
            "completeness": round(self.completeness, 3),
            "repair_cost": round(self.repair_cost, 3),
            "should_escalate": self.should_escalate,
            "reason": self.reason,
        }


# Thresholds
_LOW_CONFIDENCE = 0.5
_MEDIUM_CONFIDENCE = 0.7


def score_tool_call(
    tool_name: str,
    args: dict[str, Any],
    schema: dict[str, Any] | None,
    harness: HarnessState | None = None,
    repairs_applied: list[str] | None = None,
) -> ConfidenceScore:
    """Score confidence for a tool call.

    Args:
        tool_name: Name of the tool being called.
        args: Arguments the model wants to pass.
        schema: Tool schema (for validation). None = skip schema checks.
        harness: Session state (for history lookups). None = skip history.
        repairs_applied: List of fixes applied by tool_repair. None = none.

    Returns:
        ConfidenceScore with breakdown and escalation recommendation.
    """
    reasons: list[str] = []

    # --- Schema match ---
    schema_score = _score_schema_match(args, schema)
    if schema_score < 1.0:
        reasons.append(f"Schema match: {schema_score:.0%}")

    # --- History ---
    history_score = _score_history(tool_name, harness)
    if history_score < 1.0:
        reasons.append(f"History: {history_score:.0%}")

    # --- Completeness ---
    completeness = _score_completeness(args, schema)
    if completeness < 1.0:
        reasons.append(f"Completeness: {completeness:.0%}")

    # --- Repair cost ---
    repair_score = _score_repair_cost(repairs_applied)
    if repair_score < 1.0:
        reasons.append(f"Repairs needed: {len(repairs_applied or [])}")

    # --- Weighted overall ---
    # Schema match and completeness are weighted higher than history
    weights = {"schema": 0.35, "history": 0.25, "completeness": 0.25, "repair": 0.15}
    overall = (
        weights["schema"] * schema_score
        + weights["history"] * history_score
        + weights["completeness"] * completeness
        + weights["repair"] * repair_score
    )

    # Escalation decision
    should_escalate = overall < _LOW_CONFIDENCE
    reason = "; ".join(reasons) if reasons else "All checks passed"

    return ConfidenceScore(
        overall=overall,
        schema_match=schema_score,
        history_score=history_score,
        completeness=completeness,
        repair_cost=repair_score,
        should_escalate=should_escalate,
        reason=reason,
    )


def _score_schema_match(args: dict[str, Any], schema: dict[str, Any] | None) -> float:
    """Score how well args match the schema."""
    if schema is None:
        return 1.0  # No schema = can't check = assume OK

    properties = schema.get("properties", {})
    if not properties:
        return 1.0  # No properties defined = can't check

    if not args:
        return 0.5  # No args at all

    score = 1.0
    total = len(properties)

    for key, value in args.items():
        if key not in properties:
            score -= 0.1  # Unknown key
            continue

        prop = properties[key]
        expected_type = prop.get("type")

        if expected_type == "string" and not isinstance(value, str):
            score -= 0.15
        elif expected_type == "integer" and not isinstance(value, int):
            score -= 0.15
        elif expected_type == "number" and not isinstance(value, (int, float)):
            score -= 0.15
        elif expected_type == "boolean" and not isinstance(value, bool):
            score -= 0.15
        elif expected_type == "array" and not isinstance(value, list):
            score -= 0.15
        elif expected_type == "object" and not isinstance(value, dict):
            score -= 0.15

    return max(0.0, min(1.0, score))


def _score_history(tool_name: str, harness: HarnessState | None) -> float:
    """Score based on historical success rate of this tool."""
    if harness is None:
        return 0.75  # No history = moderate confidence

    # Check avoided list first (even if no recorded history)
    if tool_name in harness.avoided_tools:
        return 0.1

    successes = harness.tool_success_counts.get(tool_name, 0)
    failures = harness.tool_failure_counts.get(tool_name, 0)
    total = successes + failures

    if total == 0:
        return 0.75  # Never tried = moderate confidence

    return successes / total


def _score_completeness(args: dict[str, Any], schema: dict[str, Any] | None) -> float:
    """Score how many required arguments are present."""
    if schema is None:
        return 1.0

    required = schema.get("required", [])
    properties = schema.get("properties", {})

    if not required:
        # No explicit required — check for fields without defaults
        required = [
            k for k, v in properties.items()
            if "default" not in v
        ]

    if not required:
        return 1.0

    present = sum(1 for k in required if k in args)
    return present / len(required)


def _score_repair_cost(repairs_applied: list[str] | None) -> float:
    """Score based on how much repair was needed."""
    if not repairs_applied:
        return 1.0

    # More repairs = lower confidence
    count = len(repairs_applied)
    if count == 1:
        return 0.9  # Minor fix
    elif count == 2:
        return 0.75  # Moderate
    elif count <= 4:
        return 0.5  # Significant
    else:
        return 0.3  # Heavy repair — model struggled


# ---------------------------------------------------------------------------
# Escalation helpers
# ---------------------------------------------------------------------------


def should_escalate_to_cloud(score: ConfidenceScore, threshold: float = _LOW_CONFIDENCE) -> bool:
    """Decide whether to escalate this turn to a cloud model.

    Escalation triggers:
    - Overall confidence below threshold
    - Tool is in avoided list (history failures)
    - Required args missing and no repair could fix them
    """
    return score.should_escalate or score.overall < threshold


def format_confidence_summary(score: ConfidenceScore) -> str:
    """Format confidence score as a readable summary."""
    level = "HIGH" if score.overall >= _MEDIUM_CONFIDENCE else "LOW" if score.overall >= _LOW_CONFIDENCE else "VERY LOW"
    escalate = " [ESCALATE]" if score.should_escalate else ""

    return (
        f"Confidence: {score.overall:.0%} ({level}){escalate}\n"
        f"  Schema: {score.schema_match:.0%}\n"
        f"  History: {score.history_score:.0%}\n"
        f"  Completeness: {score.completeness:.0%}\n"
        f"  Repair cost: {score.repair_cost:.0%}\n"
        f"  Reason: {score.reason}"
    )
