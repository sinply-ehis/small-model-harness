"""Tests for tool_repair module."""
import json

from small_model_harness.tool_repair import (
    repair_json,
    coerce_types,
    rename_keys,
    inject_defaults,
    repair_tool_call,
)


# --- repair_json ---

class TestRepairJson:
    def test_valid_json_passthrough(self):
        data = '{"tool": "search", "arguments": {"query": "hello"}}'
        result, was_repaired = repair_json(data)
        assert json.loads(result) == {"tool": "search", "arguments": {"query": "hello"}}
        assert was_repaired is False

    def test_trailing_comma(self):
        data = '{"tool": "search", "arguments": {"query": "hello",}}'
        result, was_repaired = repair_json(data)
        assert was_repaired is True
        assert json.loads(result) == {"tool": "search", "arguments": {"query": "hello"}}

    def test_single_quotes(self):
        data = "{'tool': 'search', 'arguments': {'query': 'hello'}}"
        result, was_repaired = repair_json(data)
        assert was_repaired is True
        assert json.loads(result) == {"tool": "search", "arguments": {"query": "hello"}}

    def test_markdown_fences(self):
        data = '```json\n{"tool": "search", "arguments": {"query": "hello"}}\n```'
        result, was_repaired = repair_json(data)
        assert was_repaired is True
        assert json.loads(result) == {"tool": "search", "arguments": {"query": "hello"}}

    def test_leading_garbage(self):
        data = 'Here is the tool call: {"tool": "search", "arguments": {"query": "hello"}} done'
        result, was_repaired = repair_json(data)
        assert was_repaired is True
        parsed = json.loads(result)
        assert parsed["tool"] == "search"

    def test_comments(self):
        data = '{\n"tool": "search", // this is the tool\n"arguments": {"query": "hello"}\n}'
        result, was_repaired = repair_json(data)
        assert was_repaired is True
        assert json.loads(result)["tool"] == "search"

    def test_completely_broken_returns_original(self):
        data = "not json at all"
        result, was_repaired = repair_json(data)
        assert result == data
        assert was_repaired is False

    def test_empty_string(self):
        result, was_repaired = repair_json("")
        assert result == ""
        assert was_repaired is False

    def test_nested_trailing_commas(self):
        data = '{"tool": "search", "args": {"query": "hi", "limit": 5,},}'
        result, was_repaired = repair_json(data)
        assert was_repaired is True
        parsed = json.loads(result)
        assert parsed["args"]["query"] == "hi"
        assert parsed["args"]["limit"] == 5

    def test_bracket_array(self):
        data = 'Here are the results: [{"name": "a"}, {"name": "b"}] and that is all'
        result, was_repaired = repair_json(data)
        assert was_repaired is True
        parsed = json.loads(result)
        assert len(parsed) == 2


# --- coerce_types ---

class TestCoerceTypes:
    def setup_method(self):
        self.schema = {
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
                "ratio": {"type": "number"},
                "verbose": {"type": "boolean"},
                "tags": {"type": "array"},
                "config": {"type": "object"},
            }
        }

    def test_string_to_integer(self):
        args, fixes = coerce_types({"limit": "5"}, self.schema)
        assert args["limit"] == 5
        assert len(fixes) == 1

    def test_string_to_number(self):
        args, fixes = coerce_types({"ratio": "0.75"}, self.schema)
        assert args["ratio"] == 0.75
        assert len(fixes) == 1

    def test_string_to_boolean_true(self):
        args, fixes = coerce_types({"verbose": "true"}, self.schema)
        assert args["verbose"] is True
        assert len(fixes) == 1

    def test_string_to_boolean_false(self):
        args, fixes = coerce_types({"verbose": "false"}, self.schema)
        assert args["verbose"] is False
        assert len(fixes) == 1

    def test_string_to_array(self):
        args, fixes = coerce_types({"tags": '["a", "b"]'}, self.schema)
        assert args["tags"] == ["a", "b"]
        assert len(fixes) == 1

    def test_string_to_object(self):
        args, fixes = coerce_types({"config": '{"key": "val"}'}, self.schema)
        assert args["config"] == {"key": "val"}
        assert len(fixes) == 1

    def test_no_coercion_needed(self):
        args, fixes = coerce_types({"limit": 5, "query": "hi"}, self.schema)
        assert args["limit"] == 5
        assert len(fixes) == 0

    def test_invalid_integer_keeps_string(self):
        args, fixes = coerce_types({"limit": "abc"}, self.schema)
        assert args["limit"] == "abc"
        assert len(fixes) == 0

    def test_missing_key_ignored(self):
        args, fixes = coerce_types({"other": "val"}, self.schema)
        assert args == {"other": "val"}
        assert len(fixes) == 0

    def test_multiple_coercions(self):
        args, fixes = coerce_types({"limit": "5", "verbose": "yes", "ratio": "1.0"}, self.schema)
        assert args["limit"] == 5
        assert args["verbose"] is True
        assert args["ratio"] == 1.0
        assert len(fixes) == 3

    def test_yes_as_boolean(self):
        args, fixes = coerce_types({"verbose": "yes"}, self.schema)
        assert args["verbose"] is True

    def test_off_as_boolean(self):
        args, fixes = coerce_types({"verbose": "off"}, self.schema)
        assert args["verbose"] is False


# --- rename_keys ---

class TestRenameKeys:
    def setup_method(self):
        self.schema = {
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
                "tool_name": {"type": "string"},
            }
        }

    def test_camel_to_snake(self):
        args, fixes = rename_keys({"maxResults": 5}, self.schema)
        assert args == {"max_results": 5}
        assert len(fixes) == 1

    def test_snake_to_camel(self):
        args, fixes = rename_keys({"toolName": "search"}, self.schema)
        assert args == {"tool_name": "search"}
        assert len(fixes) == 1

    def test_already_correct(self):
        args, fixes = rename_keys({"query": "hello"}, self.schema)
        assert args == {"query": "hello"}
        assert len(fixes) == 0

    def test_unknown_key_kept(self):
        args, fixes = rename_keys({"query": "hello", "extra": "val"}, self.schema)
        assert args["query"] == "hello"
        assert args["extra"] == "val"

    def test_empty_schema(self):
        args, fixes = rename_keys({"anything": "val"}, {"properties": {}})
        assert args == {"anything": "val"}
        assert len(fixes) == 0


# --- inject_defaults ---

class TestInjectDefaults:
    def setup_method(self):
        self.schema = {
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
                "format": {"type": "string", "default": "json"},
                "verbose": {"type": "boolean", "default": False},
            }
        }

    def test_injects_missing_defaults(self):
        args, fixes = inject_defaults({"query": "hello"}, self.schema)
        assert args["limit"] == 10
        assert args["format"] == "json"
        assert args["verbose"] is False
        assert len(fixes) == 3

    def test_does_not_overwrite_existing(self):
        args, fixes = inject_defaults({"query": "hello", "limit": 5}, self.schema)
        assert args["limit"] == 5
        assert len(fixes) == 2  # format and verbose get defaults (limit already present)

    def test_no_defaults(self):
        schema = {"properties": {"query": {"type": "string"}}}
        args, fixes = inject_defaults({}, schema)
        assert args == {}
        assert len(fixes) == 0


# --- repair_tool_call (full pipeline) ---

class TestRepairToolCall:
    def setup_method(self):
        self.schemas = {
            "search_web": {
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            "play_audio": {
                "properties": {
                    "file_path": {"type": "string"},
                    "volume": {"type": "number"},
                },
                "required": ["file_path"],
            },
        }

    def test_valid_call_passthrough(self):
        raw = json.dumps({"tool": "search_web", "arguments": {"query": "hello"}})
        args, name, fixes = repair_tool_call(raw, self.schemas)
        assert name == "search_web"
        assert args["query"] == "hello"

    def test_type_coercion_applied(self):
        raw = json.dumps({"tool": "search_web", "arguments": {"query": "hello", "max_results": "10"}})
        args, name, fixes = repair_tool_call(raw, self.schemas)
        assert args["max_results"] == 10
        assert isinstance(args["max_results"], int)

    def test_defaults_injected(self):
        raw = json.dumps({"tool": "search_web", "arguments": {"query": "hello"}})
        args, name, fixes = repair_tool_call(raw, self.schemas)
        assert args["max_results"] == 5  # default injected

    def test_key_rename_applied(self):
        raw = json.dumps({"tool": "search_web", "arguments": {"query": "hello", "maxResults": "10"}})
        args, name, fixes = repair_tool_call(raw, self.schemas)
        assert "max_results" in args
        assert args["max_results"] == 10

    def test_malformed_json_repaired(self):
        raw = '{"tool": "search_web", "arguments": {"query": "hello",}}'
        args, name, fixes = repair_tool_call(raw, self.schemas)
        assert name == "search_web"
        assert args["query"] == "hello"

    def test_array_format(self):
        raw = json.dumps([{"tool": "search_web", "arguments": {"query": "hello"}}])
        args, name, fixes = repair_tool_call(raw, self.schemas)
        assert name == "search_web"
        assert args["query"] == "hello"

    def test_no_tool_name(self):
        raw = json.dumps({"arguments": {"query": "hello"}})
        args, name, fixes = repair_tool_call(raw, self.schemas)
        assert args is None
        assert name is None

    def test_unknown_tool_still_returns_args(self):
        raw = json.dumps({"tool": "unknown_tool", "arguments": {"foo": "bar"}})
        args, name, fixes = repair_tool_call(raw, self.schemas)
        assert name == "unknown_tool"
        assert args == {"foo": "bar"}

    def test_completely_broken(self):
        args, name, fixes = repair_tool_call("not json at all", self.schemas)
        assert args is None
        assert name is None
