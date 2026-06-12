"""Tests for agent/tools.py — tool registration, discovery, schema generation, execution."""

import json

from tests.conftest import BaseTest
from agent.tools import (
    register_tool,
    get_registry,
    get_tool_schemas,
    execute_tool,
    discover_tools,
    _registry,
)


class TestToolRegistry(BaseTest):
    """Test tool registration and retrieval."""

    def test_register_and_retrieve(self):
        def handler(args):
            return {"result": "ok"}

        register_tool("test_tool", "A test tool", {"type": "object"}, handler)
        registry = get_registry()
        assert "test_tool" in registry
        assert registry["test_tool"]["description"] == "A test tool"

    def test_get_registry_returns_dict(self):
        assert isinstance(get_registry(), dict)


class TestToolSchemas(BaseTest):
    """Test OpenAI-format schema generation."""

    def test_schema_format(self):
        def handler(args):
            return "done"

        register_tool("schema_test", "Schema test", {"type": "object", "properties": {}}, handler)
        schemas = get_tool_schemas()
        matching = [s for s in schemas if s["function"]["name"] == "schema_test"]
        assert len(matching) == 1
        assert matching[0]["type"] == "function"
        assert "description" in matching[0]["function"]

    def test_empty_registry(self):
        # Clear registry to test empty state
        _registry.clear()
        schemas = get_tool_schemas()
        # After discover_tools runs, built-in tools should be present
        # Just verify it returns a list
        assert isinstance(schemas, list)


class TestExecuteTool(BaseTest):
    """Test tool execution."""

    def test_execute_returns_string(self):
        def handler(args):
            return "hello"

        register_tool("exec_test", "Exec test", {}, handler)
        result = execute_tool("exec_test", {})
        assert result == "hello"

    def test_execute_returns_dict_as_json(self):
        def handler(args):
            return {"key": "value"}

        register_tool("exec_dict", "Exec dict", {}, handler)
        result = execute_tool("exec_dict", {})
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_execute_unknown_tool(self):
        result = execute_tool("nonexistent_tool", {})
        parsed = json.loads(result)
        assert "error" in parsed

    def test_execute_with_arguments(self):
        def handler(args):
            return f"got: {args.get('input', 'nothing')}"

        register_tool("exec_args", "Exec args", {}, handler)
        result = execute_tool("exec_args", {"input": "test"})
        assert result == "got: test"


class TestDiscoverTools(BaseTest):
    """Test auto-discovery from the tools/ directory."""

    def test_discover_loads_built_in_tools(self):
        _registry.clear()
        from agent.tools import _discovered
        # Reset discovered flag via module attribute
        import agent.tools as tools_mod
        tools_mod._discovered = False

        discover_tools()
        registry = get_registry()
        # Basic tools should be registered
        assert len(registry) > 0

    def test_discover_is_idempotent(self):
        import agent.tools as tools_mod
        tools_mod._discovered = False
        _registry.clear()

        discover_tools()
        count_after_first = len(get_registry())

        # Second call should not add more
        discover_tools()
        count_after_second = len(get_registry())
        assert count_after_first == count_after_second
