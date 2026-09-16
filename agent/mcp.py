"""MCP (Model Context Protocol) client for Wisemonkey.

Manages MCP server connections and dynamically registers their tools
into the global tool registry.

Two transports are supported:

- **stdio**    — spawns a local subprocess and speaks JSON-RPC over
                 stdin/stdout (config key: ``command``).
- **http**     — talks to a remote server via the Streamable HTTP
                 transport (config key: ``url``). The ``type`` may be
                 ``http``, ``streamable-http`` or ``streamableHttp``.

Configuration in mcp.json in $XDG_CONFIG_HOME/wisemonkey/
"""

import json
import os
import subprocess
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from agent.console import print, warn
from agent.tools import register_tool


def _resolve_env_value(value: str) -> str:
    """Resolve a single value, supporting ``${VAR}`` substitution."""
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    return value


def _resolve_env_map(mapping: dict[str, str]) -> dict[str, str]:
    """Resolve every value in a mapping via ``${VAR}`` substitution."""
    return {key: _resolve_env_value(value) for key, value in mapping.items()}


class Transport(ABC):
    """Abstract wire transport for an MCP server connection."""

    @abstractmethod
    def start(self) -> bool:
        """Establish the transport. Return True on success."""

    @abstractmethod
    def stop(self) -> None:
        """Tear the transport down."""

    @abstractmethod
    def send_request(self, method: str, params: dict | None = None) -> dict | None:
        """Send a JSON-RPC request and return the parsed response."""

    @abstractmethod
    def send_notification(self, method: str, params: dict | None = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Whether the transport is currently usable."""


class StdioTransport(Transport):
    """JSON-RPC over a subprocess' stdin/stdout (newline-delimited)."""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.process: subprocess.Popen | None = None
        self._request_id = 0

    @property
    def full_command(self) -> list[str]:
        return [self.command] + self.args

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def start(self) -> bool:
        env = os.environ.copy()
        env.update(_resolve_env_map(self.env))

        try:
            self.process = subprocess.Popen(
                self.full_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
            )
            print(f"  MCP server [bold]{self.name}[/] started: {' '.join(self.full_command)}")
            return True
        except FileNotFoundError:
            warn(f"MCP server '{self.name}': command not found: {self.command}")
            return False
        except Exception as e:
            warn(f"MCP server '{self.name}': failed to start: {e}")
            return False

    def stop(self) -> None:
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            except Exception:
                pass
            self.process = None

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def send_request(self, method: str, params: dict | None = None) -> dict | None:
        if not self.process or self.process.stdin is None or self.process.stdout is None:
            return None

        request: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
        }
        if params:
            request["params"] = params

        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()

            line = self.process.stdout.readline()
            if not line:
                return None
            return json.loads(line)
        except Exception as e:
            warn(f"MCP server '{self.name}': request failed: {e}")
            return None

    def send_notification(self, method: str, params: dict | None = None) -> None:
        if not self.process or self.process.stdin is None:
            return

        notification: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            notification["params"] = params
        try:
            self.process.stdin.write(json.dumps(notification) + "\n")
            self.process.stdin.flush()
        except Exception:
            pass


class HttpTransport(Transport):
    """Streamable HTTP transport for remote MCP servers.

    Each JSON-RPC message is POSTed to the server URL. The response is
    either a single JSON body or a Server-Sent Events stream carrying
    the reply. An ``Mcp-Session-Id`` header returned by the server is
    echoed back on subsequent requests.
    """

    def __init__(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ):
        self.name = name
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout
        self.session_id: str | None = None
        self._request_id = 0
        self._started = False

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def start(self) -> bool:
        # Nothing to spawn; the transport is "started" once configured.
        self._started = True
        print(f"  MCP server [bold]{self.name}[/] configured: {self.url}")
        return True

    def stop(self) -> None:
        self._started = False
        self.session_id = None

    @property
    def is_running(self) -> bool:
        return self._started

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        headers.update(_resolve_env_map(self.headers))
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        return headers

    def _post(self, payload: dict, expect_response: bool) -> dict | None:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=data,
            headers=self._build_headers(),
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                session_id = response.headers.get("Mcp-Session-Id")
                if session_id:
                    self.session_id = session_id

                if not expect_response:
                    return None

                content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" in content_type:
                    return self._read_sse(response, payload.get("id"))

                body = response.read().decode("utf-8")
                if not body.strip():
                    return None
                return json.loads(body)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            warn(f"MCP server '{self.name}': HTTP {e.code}: {detail[:200]}")
            return None
        except Exception as e:
            warn(f"MCP server '{self.name}': request failed: {e}")
            return None

    def _read_sse(self, response, request_id: Any) -> dict | None:
        """Read an SSE stream and return the reply matching request_id."""
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data:
                continue
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict) and message.get("id") == request_id:
                return message
        return None

    def send_request(self, method: str, params: dict | None = None) -> dict | None:
        request: dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
        }
        if params:
            request["params"] = params
        return self._post(request, expect_response=True)

    def send_notification(self, method: str, params: dict | None = None) -> None:
        notification: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            notification["params"] = params
        self._post(notification, expect_response=False)


class MCPServerConnection:
    """Logical MCP server: owns a transport and its discovered tools."""

    def __init__(self, name: str, transport: Transport):
        self.name = name
        self.transport = transport
        self.tools: list[dict] = []

    def start(self) -> bool:
        return self.transport.start()

    def stop(self) -> None:
        self.transport.stop()

    @property
    def is_running(self) -> bool:
        return self.transport.is_running

    def initialize(self) -> bool:
        """Perform MCP handshake and discover tools."""
        response = self.transport.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "wisemonkey", "version": "0.1.0"},
        })
        if not response or "error" in response:
            warn(f"MCP server '{self.name}': initialization failed")
            return False

        self.transport.send_notification("notifications/initialized")

        response = self.transport.send_request("tools/list")
        if response and "result" in response:
            self.tools = response["result"].get("tools", [])
            print(f"  MCP server [bold]{self.name}[/]: {len(self.tools)} tools discovered")
            return True

        return False

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool on this MCP server."""
        response = self.transport.send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })

        if not response:
            return json.dumps({"error": f"No response from MCP server '{self.name}'"})

        if "error" in response:
            error = response["error"]
            return json.dumps({"error": f"MCP error: {error.get('message', error)}"})

        result = response.get("result", {})
        content = result.get("content", [])
        if content:
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
            return "\n".join(texts) if texts else json.dumps(result)

        return json.dumps(result)


# HTTP transport type aliases (per the MCP spec these are equivalent).
_HTTP_TYPES = {"http", "streamable-http", "streamablehttp", "sse"}


class MCPClient:
    """Manages multiple MCP servers and registers their tools dynamically."""

    def __init__(self):
        self.servers: dict[str, MCPServerConnection] = {}
        self._tool_to_server: dict[str, str] = {}  # tool_name -> server_name

    def _build_connection(self, name: str, server_conf: dict) -> MCPServerConnection | None:
        """Create the right connection/transport for a server definition."""
        command = server_conf.get("command", "")
        url = server_conf.get("url", "")
        server_type = str(server_conf.get("type", "")).lower()

        if command:
            return MCPServerConnection(
                name=name,
                transport=StdioTransport(
                    name=name,
                    command=command,
                    args=server_conf.get("args", []),
                    env=server_conf.get("env", {}),
                ),
            )

        if url and (not server_type or server_type in _HTTP_TYPES):
            return MCPServerConnection(
                name=name,
                transport=HttpTransport(
                    name=name,
                    url=url,
                    headers=server_conf.get("headers", {}),
                ),
            )

        if url:
            warn(f"MCP server '{name}': unsupported type '{server_type}'")
        else:
            warn(f"MCP server '{name}': missing 'command' or 'url'")
        return None

    def load_config(self, config: Path | None = None):
        """Load MCP server definitions from a JSON config file."""
        if not config or not os.path.exists(config):
            return

        with open(config, "r") as mcp_config:
            data = json.load(mcp_config)

        servers = data.get("mcpServers", {})
        for name, server_conf in servers.items():
            connection = self._build_connection(name, server_conf)
            if connection is not None:
                self.servers[name] = connection

    def start_all(self):
        """Start all configured MCP servers and register their tools."""
        for server in self.servers.values():
            if server.start() and server.initialize():
                self._register_server_tools(server)

    def stop_all(self):
        """Stop all MCP servers."""
        for server in self.servers.values():
            server.stop()

    def _register_server_tools(self, server: MCPServerConnection):
        """Register an MCP server's tools into the global registry."""
        for tool_def in server.tools:
            tool_name = tool_def["name"]
            registered_name = f"mcp_{server.name}_{tool_name}"

            description = tool_def.get("description", f"MCP tool: {tool_name}")
            parameters = tool_def.get("inputSchema", {"type": "object", "properties": {}})

            def make_handler(s, t):
                def handler(args):
                    return s.call_tool(t, args)
                return handler

            register_tool(
                name=registered_name,
                description=f"[MCP:{server.name}] {description}",
                parameters=parameters,
                handler=make_handler(server, tool_name),
            )
            self._tool_to_server[registered_name] = server.name

        print(f"  Registered {len(server.tools)} MCP tools from '{server.name}'")

    def get_status(self) -> str:
        """Return a status string for all MCP servers."""
        if not self.servers:
            return "No MCP servers configured."
        lines = []
        for name, server in self.servers.items():
            status = "running" if server.is_running else "stopped"
            lines.append(f"  {name}: {status} ({len(server.tools)} tools)")
        return "\n".join(lines)
