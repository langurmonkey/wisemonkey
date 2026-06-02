"""MCP (Model Context Protocol) client for Wisemonkey.

Manages MCP server processes and dynamically registers their tools
into the global tool registry.

Configuration (config.yaml):
    mcp_servers:
      filesystem:
        command: npx
        args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/user"]
        env:
          MY_TOKEN: "${MY_TOKEN}"
      github:
        command: npx
        args: ["-y", "@modelcontextprotocol/server-github"]
"""

import json
import os
import subprocess
from pathlib import Path

from agent.console import print, warn
from agent.tools import register_tool


class MCPServerConnection:
    """Manages a single MCP server process and its tool registration."""

    def __init__(self, name: str, command: str, args: list[str] | None = None,
                 env: dict[str, str] | None = None):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}
        self.process: subprocess.Popen | None = None
        self.tools: list[dict] = []
        self._request_id = 0

    @property
    def full_command(self) -> list[str]:
        return [self.command] + self.args

    def _resolve_env(self) -> dict[str, str]:
        """Resolve environment variables (supports ${VAR} substitution)."""
        resolved = {}
        for key, value in self.env.items():
            if value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
                resolved[key] = os.environ.get(var_name, "")
            else:
                resolved[key] = value
        return resolved

    def start(self) -> bool:
        """Start the MCP server process."""
        env = os.environ.copy()
        env.update(self._resolve_env())

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

    def stop(self):
        """Stop the MCP server process."""
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

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send_request(self, method: str, params: dict | None = None) -> dict | None:
        """Send a JSON-RPC request and return the response."""
        if not self.process or self.process.stdin is None:
            return None

        request = {
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

    def initialize(self) -> bool:
        """Perform MCP handshake and discover tools."""
        # Initialize
        response = self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "wisemonkey", "version": "0.1.0"},
        })
        if not response or "error" in response:
            warn(f"MCP server '{self.name}': initialization failed")
            return False

        # Send initialized notification
        if self.process and self.process.stdin:
            notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
            try:
                self.process.stdin.write(json.dumps(notification) + "\n")
                self.process.stdin.flush()
            except Exception:
                pass

        # List tools
        response = self._send_request("tools/list")
        if response and "result" in response:
            self.tools = response["result"].get("tools", [])
            print(f"  MCP server [bold]{self.name}[/]: {len(self.tools)} tools discovered")
            return True

        return False

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool on this MCP server."""
        response = self._send_request("tools/call", {
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
            # Extract text from MCP content blocks
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
            return "\n".join(texts) if texts else json.dumps(result)

        return json.dumps(result)


class MCPClient:
    """Manages multiple MCP servers and registers their tools dynamically."""

    def __init__(self):
        self.servers: dict[str, MCPServerConnection] = {}
        self._tool_to_server: dict[str, str] = {}  # tool_name -> server_name

    def load_config(self, config: Path = None):
        """Load MCP server definitions from config dict."""
        if config and os.path.exists(config):
            with open(config, 'r') as mcp_config:
                data = json.load(mcp_config)
                servers = data['mcpServers']
                for name, server_conf in servers.items():
                    command = server_conf.get("command", "")
                    args = server_conf.get("args", [])
                    env = server_conf.get("env", {})
                    if command:
                        self.servers[name] = MCPServerConnection(
                            name=name,
                            command=command,
                            args=args,
                            env=env,
                        )

    def start_all(self):
        """Start all configured MCP servers and register their tools."""
        for name, server in self.servers.items():
            if server.start():
                if server.initialize():
                    self._register_server_tools(server)

    def stop_all(self):
        """Stop all MCP servers."""
        for server in self.servers.values():
            server.stop()

    def _register_server_tools(self, server: MCPServerConnection):
        """Register an MCP server's tools into the global registry."""
        for tool_def in server.tools:
            tool_name = tool_def["name"]
            # Prefix with server name to avoid collisions
            registered_name = f"mcp_{server.name}_{tool_name}"

            description = tool_def.get("description", f"MCP tool: {tool_name}")
            parameters = tool_def.get("inputSchema", {"type": "object", "properties": {}})

            # Create a closure that captures server and tool name
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
            status = "running" if server.process and server.process.poll() is None else "stopped"
            lines.append(f"  {name}: {status} ({len(server.tools)} tools)")
        return "\n".join(lines)
