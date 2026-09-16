"""Tests for MCP client transports and config parsing."""

import json
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
    def test_plain_value_passthrough(self):
        self.assertEqual(_resolve_env_value("literal"), "literal")

    def test_var_substitution(self):
        import os
        os.environ["WM_TEST_VAR"] = "hello"
        try:
            self.assertEqual(_resolve_env_value("${WM_TEST_VAR}"), "hello")
        finally:
            del os.environ["WM_TEST_VAR"]

    def test_missing_var_is_empty(self):
        self.assertEqual(_resolve_env_value("${WM_DOES_NOT_EXIST}"), "")

    def test_map_resolution(self):
        import os
        os.environ["WM_TEST_MAP"] = "value"
        try:
            result = _resolve_env_map({"A": "${WM_TEST_MAP}", "B": "static"})
            self.assertEqual(result, {"A": "value", "B": "static"})
        finally:
            del os.environ["WM_TEST_MAP"]


class TestStdioTransport(BaseTest):
    def test_full_command(self):
        transport = StdioTransport("srv", "npx", ["-y", "server"])
        self.assertEqual(transport.full_command, ["npx", "-y", "server"])

    def test_not_running_before_start(self):
        transport = StdioTransport("srv", "npx")
        self.assertFalse(transport.is_running)


class TestHttpTransport(BaseTest):
    def test_is_running_after_start(self):
        transport = HttpTransport("srv", "https://example.com/mcp")
        self.assertFalse(transport.is_running)
        self.assertTrue(transport.start())
        self.assertTrue(transport.is_running)

    def test_stop_clears_session(self):
        transport = HttpTransport("srv", "https://example.com/mcp")
        transport.start()
        transport.session_id = "abc"
        transport.stop()
        self.assertIsNone(transport.session_id)
        self.assertFalse(transport.is_running)

    def test_headers_include_auth_and_session(self):
        transport = HttpTransport(
            "srv",
            "https://example.com/mcp",
            headers={"Authorization": "sk-123"},
        )
        transport.session_id = "sess-1"
        headers = transport._build_headers()
        self.assertEqual(headers["Authorization"], "sk-123")
        self.assertEqual(headers["Mcp-Session-Id"], "sess-1")
        self.assertIn("application/json", headers["Accept"])
        self.assertIn("text/event-stream", headers["Accept"])

    def test_headers_resolve_env(self):
        import os
        os.environ["WM_MCP_TOKEN"] = "tok"
        try:
            transport = HttpTransport(
                "srv",
                "https://example.com/mcp",
                headers={"Authorization": "${WM_MCP_TOKEN}"},
            )
            headers = transport._build_headers()
            self.assertEqual(headers["Authorization"], "tok")
        finally:
            del os.environ["WM_MCP_TOKEN"]


class TestHttpTypeAliases(BaseTest):
    def test_known_types(self):
        for alias in ("http", "streamable-http", "streamablehttp", "sse"):
            self.assertIn(alias, _HTTP_TYPES)


class TestBuildConnection(BaseTest):
    def test_command_yields_stdio(self):
        client = MCPClient()
        conn = client._build_connection("fs", {"command": "npx", "args": ["-y", "x"]})
        self.assertIsNotNone(conn)
        assert conn is not None
        self.assertIsInstance(conn.transport, StdioTransport)

    def test_url_http_yields_http(self):
        client = MCPClient()
        conn = client._build_connection(
            "langsearch",
            {
                "type": "http",
                "url": "https://mcp.langsearch.com/mcp",
                "headers": {"Authorization": "sk-abc"},
            },
        )
        self.assertIsNotNone(conn)
        assert conn is not None
        self.assertIsInstance(conn.transport, HttpTransport)

    def test_url_streamable_http_alias(self):
        client = MCPClient()
        for alias in ("streamable-http", "streamableHttp"):
            conn = client._build_connection(
                "srv", {"type": alias, "url": "https://example.com/mcp"}
            )
            self.assertIsNotNone(conn, f"alias {alias} should be accepted")
            assert conn is not None
            self.assertIsInstance(conn.transport, HttpTransport)

    def test_url_without_type_defaults_to_http(self):
        client = MCPClient()
        conn = client._build_connection("srv", {"url": "https://example.com/mcp"})
        self.assertIsNotNone(conn)
        assert conn is not None
        self.assertIsInstance(conn.transport, HttpTransport)

    def test_missing_command_and_url(self):
        client = MCPClient()
        self.assertIsNone(client._build_connection("srv", {}))

    def test_unsupported_type(self):
        client = MCPClient()
        self.assertIsNone(
            client._build_connection("srv", {"type": "bogus", "url": "https://x/mcp"})
        )


class TestLoadConfig(BaseTest):
    def test_loads_stdio_and_http(self):
        config = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                },
                "langsearch": {
                    "type": "http",
                    "url": "https://mcp.langsearch.com/mcp",
                    "headers": {"Authorization": "sk-abc"},
                },
            }
        }
        path = self._write_json("mcp.json", config)
        client = MCPClient()
        client.load_config(path)

        self.assertIn("filesystem", client.servers)
        self.assertIn("langsearch", client.servers)
        self.assertIsInstance(client.servers["filesystem"].transport, StdioTransport)
        self.assertIsInstance(client.servers["langsearch"].transport, HttpTransport)

    def test_missing_file_is_noop(self):
        client = MCPClient()
        client.load_config(self._tmpdir / "nope.json")
        self.assertEqual(client.servers, {})

    def test_empty_servers(self):
        path = self._write_json("mcp.json", {"mcpServers": {}})
        client = MCPClient()
        client.load_config(path)
        self.assertEqual(client.servers, {})


class TestGetStatus(BaseTest):
    def test_no_servers(self):
        client = MCPClient()
        self.assertEqual(client.get_status(), "No MCP servers configured.")

    def test_status_lists_servers(self):
        client = MCPClient()
        client.servers["fs"] = MCPServerConnection(
            "fs", StdioTransport("fs", "npx")
        )
        status = client.get_status()
        self.assertIn("fs", status)
        self.assertIn("stopped", status)


if __name__ == "__main__":
    unittest.main()