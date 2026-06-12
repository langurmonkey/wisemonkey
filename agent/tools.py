"""Tool registry and execution.

Each tool is a function with:
  - name: identifier
  - description: for the LLM to decide when to use it
  - parameters: JSON schema describing inputs
  - handler: the actual function to call

Tools are discovered from the tools/ directory on startup.
"""

import json
from importlib import util
from pathlib import Path

from agent.console import warn

# Global registry
_registry = {}
_discovered = False


def get_registry():
    """Return the tool registry dict."""
    return _registry


def register_tool(name, description, parameters, handler):
    """Register a tool in the global registry."""
    _registry[name] = {
        "name": name,
        "description": description,
        "parameters": parameters,
        "handler": handler,
    }


def tool(name, description, parameters):
    """Decorator to register tools."""

    def inner(func):
        register_tool(name, description, parameters, func)

    return inner


def get_tool_schemas():
    """Return all registered tools as OpenAI-format function schemas."""
    # Ensure tools are discovered before building schemas
    discover_tools()
    schemas = []
    for name, tool in _registry.items():
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        })
    return schemas


def execute_tool(name, arguments) -> str:
    """Execute a registered tool by name with parsed arguments."""
    if name not in _registry:
        return json.dumps({"error": f"Unknown tool: {name}"})

    tool = _registry[name]
    result = tool["handler"](arguments)

    # Handlers can return dict, str, or any JSON-serializable value
    if isinstance(result, dict):
        return json.dumps(result)
    if isinstance(result, str):
        return result
    return json.dumps(result)

def _tool_str(name, tool):
    result = f"⚙ [list-item]{name}[/]\n"
    result += f"[list-desc]{tool['description']}[/]\n"
    return result

def get_tools_str(prefix: str | None = None, contains: bool = True):
    """
    Auto-discover tools and log them to terminal
    Parameters:
    prefix:str    - string prefix to filter tools
    contains:bool - whether to filter tools that
                    contain (True) or do not contain (False) the prefix
    """
    discover_tools()
    result = ""
    for name, tool in _registry.items():
        if prefix:
            if contains and name.startswith(prefix):
                # Add tools that contain the prefix
                result += _tool_str(name, tool)
            if not contains and not name.startswith(prefix):
                # Add tools that don't contain the prefix
                result += _tool_str(name, tool)
        else:
            # Add all
            result += _tool_str(name, tool)

    return result

def discover_tools(tools_dir=None):
    """Auto-discover tools from the tools/ directory.

    Each .py file in the directory is imported. It should call
    register_tool() to add itself to the registry.

    Idempotent: tools are only loaded once per process.
    """
    global _discovered

    if _discovered:
        return

    tools_path = Path(tools_dir) if tools_dir else None

    if not tools_path:
        # Default: look in the package's tools/ directory
        tools_path = Path(__file__).parent.parent / "tools"

    if not tools_path.exists():
        _discovered = True
        return

    for py_file in tools_path.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
        try:
            spec = util.spec_from_file_location(py_file.stem, py_file)
            if spec is None or spec.loader is None:
                warn(f"Could not load spec for tool {py_file.name}")
                continue

            module = util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            warn(f"Failed to load tool {py_file.name}: {e}")

    _discovered = True


def get_all_tools():
    """Get all tools (discover + return registry)."""
    discover_tools()
    return list(_registry.values())
