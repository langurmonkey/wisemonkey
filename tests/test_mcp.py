"""Tests for MCP client transports and config parsing."""

import os
import unittest

from agent.mcp import (
    HttpTransport,
    MCPClient,
    MCPServerConnection,
    StdioTransport,
    _HTTP_TYPES,
    _resolve_env_map,
    _resolve_env_value,
)
from tests.conftest import BaseTest


class TestEnvResolution(BaseTest):
    def test_values_may_reference_the_environment(self):
        os.environ["WM_TEST_VAR"] = "hello"
        try:
            assert _resolve_env_value("literal") == "literal"
            assert _resolve_env_value("${WM_TEST_VAR}") == "hello"
            # An unset variable resolves to empty rather than leaking the name.
            assert _resolve_env_value("${WM_DOES_NOT_EXIST}") == ""
            assert _resolve_env_map({"A": "${WM_TEST_VAR}", "B": "static"}) == {
                "A": "hello", "B": "static"
            }
        finally:
            del os.environ["WM_TEST_VAR"]


class TestHttpTransport(BaseTest):
    def test_start_stop_and_headers(self):
        transport = HttpTransport("srv", "https://example.com/mcp",
                                  headers={"Authorization": "sk-123"})
        assert not transport.is_running
        assert transport.start() and transport.is_running
        transport.session_id = "sess-1"
        headers = transport._build_headers()
        assert headers["Authorization"] == "sk-123"
        assert headers["Mcp-Session-Id"] == "sess-1"
        assert "text/event-stream" in headers["Accept"]
        transport.stop()
        assert transport.session_id is None and not transport.is_running

    def test_headers_resolve_env_and_http_aliases_are_accepted(self):
        os.environ["WM_MCP_TOKEN"] = "tok"
        try:
            transport = HttpTransport("srv", "https://x/mcp",
                                      headers={"Authorization": "${WM_MCP_TOKEN}"})
            assert transport._build_headers()["Authorization"] == "tok"
        finally:
            del os.environ["WM_MCP_TOKEN"]
        for alias in ("http", "streamable-http", "streamablehttp", "sse"):
            assert alias in _HTTP_TYPES


class TestBuildConnection(BaseTest):
    def test_command_or_url_picks_the_transport(self):
        client = MCPClient()
        stdio = client._build_connection("fs", {"command": "npx", "args": ["-y", "x"]})
        assert stdio is not None and isinstance(stdio.transport, StdioTransport)
        for spec in ({"type": "http", "url": "https://x/mcp"},
                     {"type": "streamable-http", "url": "https://x/mcp"},
                     {"url": "https://x/mcp"}):
            conn = client._build_connection("srv", spec)
            assert conn is not None and isinstance(conn.transport, HttpTransport), spec

    def test_no_command_or_url_and_an_unknown_type_are_refused(self):
        client = MCPClient()
        assert client._build_connection("srv", {}) is None
        assert client._build_connection(
            "srv", {"type": "bogus", "url": "https://x/mcp"}) is None


class TestLoadConfig(BaseTest):
    def test_loads_both_kinds_and_ignores_a_missing_file(self):
        path = self._write_json("mcp.json", {"mcpServers": {
            "filesystem": {"command": "npx", "args": ["-y", "srv", "/tmp"]},
            "langsearch": {"type": "http", "url": "https://x/mcp"},
        }})
        client = MCPClient()
        client.load_config(path)
        assert isinstance(client.servers["filesystem"].transport, StdioTransport)
        assert isinstance(client.servers["langsearch"].transport, HttpTransport)

        client = MCPClient()
        client.load_config(self._tmpdir / "nope.json")
        assert client.servers == {}
        client.load_config(self._write_json("e.json", {"mcpServers": {}}))
        assert client.servers == {}


class TestGetStatus(BaseTest):
    def test_status_names_each_server_and_its_state(self):
        client = MCPClient()
        assert client.get_status() == "No MCP servers configured."
        client.servers["fs"] = MCPServerConnection("fs", StdioTransport("fs", "npx"))
        status = client.get_status()
        assert "fs" in status and "stopped" in status
