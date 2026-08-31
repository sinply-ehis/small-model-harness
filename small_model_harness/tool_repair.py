"""Deterministic tool call repair — fix malformed output without re-prompting.

Small models (2B-4B) reliably produce certain classes of structural errors:
- Malformed JSON (trailing commas, missing quotes, unescaped characters)
- Type drift (numbers as strings, booleans as text, arrays as escaped strings)
- Key naming mismatch (camelCase vs snake_case, paraphrased parameter names)
- Missing required arguments (model forgets a param it should have included)

This module repairs these deterministically — no LLM call, no retry loop.
The model's *intent* is usually correct; only the serialization drifted.
"""

from __future__ import annotations

import json
import re
from typing import Any


# ---------------------------------------------------------------------------
# JSON repair
# ---------------------------------------------------------------------------


def repair_json(raw: str) -> tuple[str, bool]:
    """Repair common JSON malformation from small models.

    Fixes:
    - Trailing commas before } or ]
    - Single quotes instead of double quotes
    - Unescaped newlines/tabs in strings
    - Missing quotes around keys
    - Comments (// and /* */)
    - leading/trailing garbage around JSON

    Returns (repaired_json, was_repaired).
    """
    if not raw:
        return raw, False

    original = raw
    fixed = raw.strip()

    # Remove markdown code fences
    if fixed.startswith("```"):
        lines = fixed.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        fixed = "\n".join(lines).strip()

    # Strip leading/trailing non-JSON garbage
    # Find first { or [ and last } or ]
    first_brace = -1
    first_bracket = -1
    for i, c in enumerate(fixed):
        if c == "{":
            first_brace = i
            break
        if c == "[":
            first_bracket = i
            break

    if first_brace >= 0 and (first_bracket < 0 or first_brace < first_bracket):
        last = fixed.rfind("}")
        if last > first_brace:
            fixed = fixed[first_brace : last + 1]
    elif first_bracket >= 0:
        last = fixed.rfind("]")
        if last > first_bracket:
            fixed = fixed[first_bracket : last + 1]

    # Remove single-line comments
    fixed = re.sub(r"//[^\n]*", "", fixed)

    # Remove multi-line comments
    fixed = re.sub(r"/\*.*?\*/", "", fixed, flags=re.DOTALL)

    # Fix trailing commas: ,} → } and ,] → ]
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)

    # Fix single quotes → double quotes (simple cases only)
    # Only do this if the string doesn't already have double quotes
    if '"' not in fixed and "'" in fixed:
        fixed = fixed.replace("'", '"')

    # Fix unescaped newlines inside string values
    # This is tricky — only fix if we detect a broken JSON parse
    try:
        json.loads(fixed)
        return fixed, fixed != original.strip()
    except json.JSONDecodeError:
        pass

    # More aggressive: try to fix common issues
    # Fix missing quotes around keys: {key: "value"} → {"key": "value"}
    fixed = re.sub(r'(\{|,)\s*([a-zA-Z_]\w*)\s*:', r'\1"\2":', fixed)

    # Fix escaped quotes that shouldn't be escaped
    fixed = fixed.replace('\\"', '"')

    # Fix tabs and newlines in string values
    fixed = fixed.replace("\t", " ").replace("\n", " ")

    # Collapse multiple spaces
    fixed = re.sub(r"  +", " ", fixed)

    # Try parsing again
    try:
        json.loads(fixed)
        return fixed, fixed != original.strip()
    except json.JSONDecodeError:
        pass

    # Last resort: try to extract just the JSON object/array
    # by finding balanced braces
    result = _extract_balanced(fixed)
    if result:
        try:
            json.loads(result)
            return result, result != original.strip()
        except json.JSONDecodeError:
            pass

    return original, False


def _extract_balanced(text: str) -> str | None:
    """Extract a balanced JSON object or array from potentially broken text."""
    # Try object first
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    # Try array
    start = text.find("[")
    if start >= 0:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

    return None


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------


def coerce_types(args: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Coerce argument types to match the schema.

    Common small model drift:
    - "5" → 5 (string number → int)
    - "true" → True (string bool → bool)
    - "1.5" → 1.5 (string float → float)
    - "['a','b']" → ["a","b"] (string array → list)

    Returns (coerced_args, list_of_fixes_applied).
    """
    properties = schema.get("properties", {})
    fixes: list[str] = []
    coerced = dict(args)

    for key, prop_schema in properties.items():
        if key not in coerced:
            continue

        value = coerced[key]
        expected_type = prop_schema.get("type")

        if expected_type == "integer" and isinstance(value, str):
            try:
                coerced[key] = int(value)
                fixes.append(f"Converted '{key}' from string to integer")
                continue
            except (ValueError, TypeError):
                pass

        if expected_type == "number" and isinstance(value, str):
            try:
                coerced[key] = float(value)
                fixes.append(f"Converted '{key}' from string to number")
                continue
            except (ValueError, TypeError):
                pass

        if expected_type == "boolean" and isinstance(value, str):
            lower = value.lower().strip()
            if lower in ("true", "yes", "1", "on"):
                coerced[key] = True
                fixes.append(f"Converted '{key}' from string to boolean true")
                continue
            elif lower in ("false", "no", "0", "off", ""):
                coerced[key] = False
                fixes.append(f"Converted '{key}' from string to boolean false")
                continue

        if expected_type == "array" and isinstance(value, str):
            # Try to parse JSON array string
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    coerced[key] = parsed
                    fixes.append(f"Converted '{key}' from string to array")
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

        if expected_type == "object" and isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    coerced[key] = parsed
                    fixes.append(f"Converted '{key}' from string to object")
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

    return coerced, fixes


# ---------------------------------------------------------------------------
# Key renaming
# ---------------------------------------------------------------------------


def rename_keys(args: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Rename keys to match the schema — handles camelCase/snake_case drift.

    Common patterns:
    - queryText → query
    - maxResults → max_results
    - toolName → tool_name

    Also handles paraphrased keys by fuzzy matching.

    Returns (renamed_args, list_of_renames).
    """
    properties = schema.get("properties", {})
    if not properties:
        return args, []

    valid_keys = set(properties.keys())
    fixes: list[str] = []
    renamed = {}

    for key, value in args.items():
        if key in valid_keys:
            renamed[key] = value
            continue

        # Try snake_case conversion
        snake = _to_snake_case(key)
        if snake in valid_keys:
            renamed[snake] = value
            fixes.append(f"Renamed '{key}' → '{snake}'")
            continue

        # Try camelCase conversion
        camel = _to_camel_case(key)
        if camel in valid_keys:
            renamed[camel] = value
            fixes.append(f"Renamed '{key}' → '{camel}'")
            continue

        # Try fuzzy match (prefix match)
        matched = False
        for valid_key in valid_keys:
            if valid_key.startswith(key[:3]) or key.startswith(valid_key[:3]):
                renamed[valid_key] = value
                fixes.append(f"Renamed '{key}' → '{valid_key}' (fuzzy)")
                matched = True
                break

        if not matched:
            renamed[key] = value

    return renamed, fixes


def _to_snake_case(s: str) -> str:
    """Convert camelCase or PascalCase to snake_case."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def _to_camel_case(s: str) -> str:
    """Convert snake_case to camelCase."""
    parts = s.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


# ---------------------------------------------------------------------------
# Missing defaults
# ---------------------------------------------------------------------------


def inject_defaults(args: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Inject default values for missing required-adjacent fields.

    If a field has a default in the schema and is missing from args,
    inject it. This prevents "missing argument" errors for optional
    fields that the model simply forgot.

    Returns (args_with_defaults, list_of_injections).
    """
    properties = schema.get("properties", {})
    fixes: list[str] = []
    result = dict(args)

    for key, prop_schema in properties.items():
        if key in result:
            continue
        if "default" in prop_schema:
            result[key] = prop_schema["default"]
            fixes.append(f"Injected default for '{key}': {prop_schema['default']}")

    return result, fixes


# ---------------------------------------------------------------------------
# Full repair pipeline
# ---------------------------------------------------------------------------


def repair_tool_call(
    raw_output: str,
    tool_schemas: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    """Full deterministic repair pipeline for a model's tool call output.

    1. Parse JSON (repair if needed)
    2. Extract tool name and arguments
    3. Coerce types to match schema
    4. Rename keys to match schema
    5. Inject defaults

    Args:
        raw_output: The model's raw output (may be malformed JSON).
        tool_schemas: Map of tool_name → schema dict.

    Returns:
        (repaired_args, tool_name, fixes_applied) on success.
        (None, None, error_message) if unrecoverable.
    """
    fixes: list[str] = []

    # Step 1: Parse JSON
    repaired_json, json_fixed = repair_json(raw_output)
    if json_fixed:
        fixes.append("Repaired malformed JSON")

    try:
        data = json.loads(repaired_json)
    except json.JSONDecodeError as e:
        return None, None, [f"Unrecoverable JSON: {e}"]

    # Step 2: Extract tool name and args
    if isinstance(data, dict):
        tool_name = data.get("tool") or data.get("name") or data.get("function") or data.get("tool_name")
        args = data.get("arguments") or data.get("args") or data.get("parameters") or {}
    elif isinstance(data, list) and len(data) > 0:
        # Some models output tool calls as arrays
        first = data[0]
        tool_name = first.get("tool") or first.get("name") or first.get("function")
        args = first.get("arguments") or first.get("args") or {}
    else:
        return None, None, ["Could not extract tool name from output"]

    if not tool_name:
        return None, None, ["No tool name found in output"]

    if not isinstance(args, dict):
        return None, None, [f"Arguments not a dict: {type(args).__name__}"]

    # Step 3-5: Schema-aware repair if we have the schema
    if tool_name in tool_schemas:
        schema = tool_schemas[tool_name]

        # Rename keys FIRST (so coerce finds them by correct name)
        args, rename_fixes = rename_keys(args, schema)
        fixes.extend(rename_fixes)

        # Then coerce types (now matching correct key names)
        args, type_fixes = coerce_types(args, schema)
        fixes.extend(type_fixes)

        # Inject defaults last
        args, default_fixes = inject_defaults(args, schema)
        fixes.extend(default_fixes)

    return args, tool_name, fixes
