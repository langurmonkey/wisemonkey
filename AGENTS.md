# Wisemonkey — Project Guide

## What is this project?

Wisemonkey is a simple, extensible CLI AI agent for Linux and macOS terminals. It connects to any OpenAI/Anthropic/Ollama-compatible endpoint and provides session management, persistent memory, vector store document embedding, native + MCP tools, and skills.

## Project Structure

```
wisemonkey/
├── agent/                  # Core agent code
│   ├── __main__.py         # CLI entry point, argument parsing
│   ├── agent.py            # Main agent loop, prompt handling, key bindings
│   ├── commands.py         # Slash commands (/embed, /quit, /config, etc.)
│   ├── config.py           # Configuration loading and handling
│   ├── console.py          # Rich console output with themed formatting
│   ├── core.py             # Core logic: system prompt, LLM calls, tool dispatch
│   ├── mcp.py              # MCP server support
│   ├── memory.py           # Session memory (user profile, notes, chat history)
│   ├── router.py           # API router (OpenAI, Ollama, Anthropic)
│   ├── skills.py           # Skill loading from markdown files
│   ├── tools.py            # Tool definitions and registration
│   ├── utils.py            # Utility functions
│   └── vectorstore.py      # Vector store wrapper
├── tools/                  # Tool implementations available to the model
│   ├── basic.py            # Basic/example tools
│   ├── files.py            # File read/write tools
│   ├── memory.py            # search_knowledge tool
│   ├── network.py          # URL fetching
│   ├── terminal.py         # Shell command execution
│   └── vectorstore.py      # Vector store tool handler
├── skills/                 # Skill definitions (.md files with YAML frontmatter)
│   ├── example.md
│   └── rolldice.md
├── config.yaml             # Default configuration file
├── pyproject.toml          # Project metadata and dependencies
├── install.sh              # One-line installer script
└── .env.example            # Example environment variables
```

## Key Architectural Patterns

### System Prompt Construction (`agent/core.py`)

The system prompt is built in `Core._build_system_prompt()` each turn. It assembles, in order:
1. Base system prompt from config
2. `AGENTS.md` workspace instructions (if found)
3. Formatted memory (user profile, notes)
4. Chat history
5. Loaded skills

### Tool System (`agent/tools.py` + `tools/`)

Tools are defined using the `@tool(name, description, parameters)` decorator. They are auto-discovered on startup. Each tool file in `tools/` contains one or more decorated handler functions.

### Slash Commands (`agent/commands.py`)

Commands use the `@cmd(name, description, aliases)` decorator and are auto-registered. Each returns `(ok: bool, msg: str, content: str, markdown: str)`.

### Skills (`agent/skills.py` + `skills/`)

Skills are `.md` files with YAML frontmatter (`name`, `description`). The body is injected into the system prompt. Follows the agentskills.io standard.

### Memory (`agent/memory.py`)

- **User profile** (`user_profile.json`) — set via `set_user_profile` tool
- **Notes** (`notes.json`) — added via `save_note` tool
- **Chat history** (`chat_history.json`) — rolling window of recent exchanges
- All stored per-session under `~/.local/share/wisemonkey/sessions/$SESSION_NAME/`

### Sessions

Sessions are directories under `~/.local/share/wisemonkey/sessions/`. Each session has its own memory, chat history, and vector store. Session name defaults to `default`.

## How to Extend

### Adding a Tool

1. Create or edit a file in `tools/`
2. Decorate a function with `@tool(name, description, parameters)`
3. The tool is auto-discovered — no registration needed

### Adding a Slash Command

1. Add a function in `agent/commands.py`
2. Decorate with `@cmd(name, description, aliases=[])`
3. Return `(ok, msg, content, markdown)`

### Adding a Skill

1. Create a `.md` file in `skills/` with YAML frontmatter (`name`, `description`)
2. The skill body is injected into the system prompt automatically

## Configuration

Configuration lives in `$XDG_CONFIG_HOME/wisemonkey/config.yaml` (created on first run). Key sections:
- `model` — provider, name, base_url, temperature, reasoning
- `embedding` — embedding model name and endpoint
- `agent` — max_turns, system_prompt, max_chat_history, vi_mode

Run `wisemonkey --onboard` for interactive configuration.

## Development

- Requires Python 3.13+ and `uv`
- Dependencies: `uv sync`
- Run from source: `uv run wisemonkey`
- Build: `uv build`
- Entry point: `agent.__main__:main` → `wisemonkey` CLI command

## Testing

Tests use the standard library `unittest` framework. Test files live in `tests/` at the project root, each mirroring the source module it tests.

### Test Structure

```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures (mock config, temp dirs, singleton resets)
├── test_config.py       # Config singleton, load/save, dot-notation get/set
├── test_core.py         # Workspace root finding, context file loading, prompt building
├── test_memory.py       # Memory, ChatMemory persistence and trimming
├── test_skills.py       # SkillLoader frontmatter parsing, load_all
└── test_tools.py        # Tool registration, discovery, execution
```

### Running Tests

```bash
# Run all tests
python -m unittest discover -s tests -v

# Run a specific test file
python -m unittest tests.test_core -v

# Run a specific test class
python -m unittest tests.test_core.TestFindWorkspaceRoot -v

# Run a single test
python -m unittest tests.test_core.TestFindWorkspaceRoot.test_finds_agents_md_in_parent -v
```

### Key Patterns

- **Reset singletons** — `Config` and `Memory` are singletons; reset them in `setUp`/`tearDown` or via fixtures (`Config._instance = None`, `Memory._instance = None`).
- **Use `tempfile.mkdtemp()`** — each test gets its own session directory for isolation.
- **Mock the LLM router** — never hit real API endpoints in tests.
- **`conftest.py`** — place shared fixtures here (temp dirs, mock config, singleton resets).

## Conventions

- Use `pathlib.Path` for filesystem operations
- Follow XDG Base Directory spec for data/config paths
- Use `rich` for all console output (via `agent.console`)
- Tools return plain dicts or strings; the agent serializes as needed
- Keep the agent loop in `agent/agent.py` separate from core logic in `agent/core.py`
