"""Tests for agent/tools.py — registration, schema generation, execution."""

import json

import agent.tools as tools_mod
from tests.conftest import BaseTest
from agent.tools import (
    discover_tools,
    execute_tool,
    get_registry,
    get_tool_schemas,
    register_tool,
    _registry,
)


class TestRegistry(BaseTest):
    def test_registration_reaches_the_schema_list(self):
        register_tool(
            "schema_test", "Schema test", {"type": "object", "properties": {}},
            lambda args: "done",
        )
        assert get_registry()["schema_test"]["description"] == "Schema test"
        (schema,) = [
            s for s in get_tool_schemas() if s["function"]["name"] == "schema_test"
        ]
        assert schema["type"] == "function"
        assert "description" in schema["function"]

    def test_execute_returns_strings_and_serializes_dicts(self):
        register_tool("exec_test", "d", {}, lambda args: "hello")
        register_tool("exec_dict", "d", {}, lambda args: {"key": "value"})
        register_tool("exec_args", "d", {}, lambda args: f"got: {args.get('input')}")
        assert execute_tool("exec_test", {}) == "hello"
        assert json.loads(execute_tool("exec_dict", {}))["key"] == "value"
        assert execute_tool("exec_args", {"input": "test"}) == "got: test"
        assert "error" in json.loads(execute_tool("nonexistent_tool", {}))

    def test_discovery_loads_the_built_in_tools_once(self):
        tools_mod._discovered = False
        _registry.clear()
        discover_tools()
        first = len(get_registry())
        assert first > 0
        discover_tools()
        assert len(get_registry()) == first
