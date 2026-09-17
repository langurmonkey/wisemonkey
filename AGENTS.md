# Wisemonkey — Project Guide

## What is this project?

Wisemonkey is a simple, extensible CLI AI agent for Linux and macOS terminals. It connects to any OpenAI/Anthropic/Ollama-compatible endpoint and provides session management, persistent memory, vector store document embedding, native + MCP tools, and skills.

## Project Structure

```
wisemonkey/
├── agent/                  # Core agent code.
│   ├── agent.py            # Main agent loop, prompt handling, key bindings.
│   ├── commands.py         # Slash commands (e.g. /embed, /quit).
│   ├── config.py           # Configuration loading and handling.
│   ├── console.py          # Rich console output with themed formatting.
│   ├── core.py             # Core agent functions, like API connection and tool calls.
│   ├── emitter.py          # Turn event emitter: core callbacks -> IPC events.
│   ├── ipc.py              # Client/server protocol: messages, payloads, transports.
│   ├── mcp.py              # MCP server support.
│   ├── memory.py           # Session memory, paste file creation.
│   ├── router.py           # API router implementation for OpenAI, Ollama, and Anthropic.
│   ├── skills.py           # Skill loading and management.
│   ├── tools.py            # Tool definitions.
│   ├── update.py           # Update management.
│   ├── utils.py            # Utility functions.
│   └── vectorstore.py      # Vector store wrapper.
├── tools/                  # Tool implementations available to the model.
│   ├── basic.py            # Basic and example tools.
│   ├── files.py            # File read/write tools.
│   ├── memory.py            # search_knowledge tool.
│   ├── network.py          # URL fetching.
│   ├── terminal.py         # Shell command execution.
│   └── vectorstore.py      # Vector store tool handler.
├── skills/                 # Skill definitions. Add new skills here.
│   ├── example.md
│   └── rolldice.md
├── tests/                  # Contains all `unittest` tests.
│   └── [...]
├── config.yaml             # Default config file.
├── README.md
├── pyproject.toml
├── install.sh              # Installer script.
└── .env.example
```

## Key Architectural Patterns

### System Prompt Construction (`agent/core.py`)

The system prompt is built in `Core._build_system_prompt()` each turn. It assembles, in order:
1. Base system prompt from config
2. Soul files (`Core._load_soul_files()`: global `$XDG_CONFIG_HOME/wisemonkey/SOUL.md`, then workspace `SOUL.md`) — identity/persona
3. `AGENTS.md` workspace instructions (if found)
4. Formatted memory (user profile, notes)
5. Chat history
6. Loaded skills

### IPC Protocol (`agent/ipc.py`)

Phase 0 of the client/server split (see the `CLIENT_SERVER` session plan). Defines the wire protocol that lets a client (REPL/TUI) talk to a server that owns `Core`/`Memory`/`Config`:

- `Message` envelope: `kind` (`event`/`request`/`response`/`error`), `name`, `payload`, `id`, `reply_to`, `turn_id`, `ts`.
- Framing: newline-delimited JSON via `encode_message()` / `decode_message()`.
- `PROTOCOL_VERSION` + `check_protocol_version()` (major must match).
- Events (`Event`), client requests (`ClientRequest`), server requests (`ServerRequest`, mirrors `OutputAdapter`), `InjectWhen` for mid-turn/between-turn injection.
- `OutputPayload` carries pre-rendered output (`format`, `level`, `indent`, panel/rule fields) so the server needs no Rich console.
- `Transport` Protocol (`send`/`recv(timeout)`/`close`/`closed`) with an in-process `LoopbackTransport` / `loopback_pair()` for phase-1 work.
- `payload_as()` reconstructs a payload dataclass, ignoring unknown keys (forward compatibility).

### Turn results and cancellation (`agent/core.py`, `agent/emitter.py`)

`Core.run_turn()` returns a `TurnResult` (`response`, `total_tokens`, `n_tools`, `gen_time`, plus `cancelled` and `error`). It **unpacks as the historical 4-tuple**, so existing callers keep working, but new drivers should read `.cancelled` instead of string-matching on `"[Cancelled]"`.

Cancellation is **observable state, not an exception**: `_stream_handler()` sets `self._turn_cancelled` and stops the stream when `poll()` returns True, and `run_turn()` returns `TurnResult(cancelled=True)` without persisting a partial answer. `cancel_callback` remains supported (legacy `raise_on_cancel` emitters still raise `TurnCancelled`, which `run_turn()` converts into a cancelled result).

`run_turn()` accepts an optional `tool_result_callback` invoked as `(tool_id, tool_name, content, is_error, duration[, image_base64, mime_type])` after each tool finishes, so tool results can be streamed as events.

### Tool System (`agent/tools.py` + `tools/`)

Tools are defined using the `@tool(name, description, parameters)` decorator. They are auto-discovered on startup. Each tool file in `tools/` contains one or more decorated handler functions.

`read_file` accepts an optional `max_lines` parameter to read only the first N lines from the top (like `head`). This is useful for large files: it keeps the tool result small and avoids flooding the chat history. When set, the result includes `truncated`, `total_lines`, and `shown_lines` metadata. Omit it or set it to `0` to read the whole file.

#### Chat memory accounting

`ChatMemory` tracks `total_chars` and triggers `/session-chat-compact` when it exceeds `agent.max_chat_history`. To avoid premature compaction, `ChatMemory._entry_len()` counts tool results exactly as they are injected into the prompt — truncated to `agent.chat_history_tool_result_max_chars` unless `agent.chat_history_full_tool_results` is `true` (see `get_formatted()`). Keep this accounting in sync with `get_formatted()` if the formatting logic changes.

### Slash Commands (`agent/commands.py`)

Commands use the `@cmd(name, description, aliases)` decorator and are auto-registered. Each returns `(ok: bool, msg: str, content: str, markdown: str)`.

### Skills (`agent/skills.py` + `skills/`)

Skills are `.md` files with YAML frontmatter (`name`, `description`). The body is injected into the system prompt. Follows the agentskills.io standard. By default, `.md` files in the wisemonkey `skills/` directory, as well as `.md` files in `./skills/` (current working directory) are loaded.

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
- `agent` — max_turns, system_prompt, max_chat_history, vi_mode, soul_file, global_soul_file, context_files

Run `wisemonkey --onboard` for interactive configuration.

## Development

- Requires Python 3.13+ and `uv`
- Dependencies: `uv sync`
- Run from source: `uv run wisemonkey`
- Build: `uv build`
- Entry point: `agent.__main__:main` → `wisemonkey` CLI command

## Type checking

Whenever you make changes, always run the type checker. To run the type checker for all files in the project, execute:

```bash
just checkall
```

If you want to run the type checker for a particular file or directory:
```bash
# A single file
just check agent/agent.py

# A directory
just check agent
```


## Testing

Tests use the standard library `unittest` framework. Test files live in `tests/` at the project root, each mirroring the source module it tests.

### Test Structure

```
tests/
├── __init__.py
├── conftest.py          # Shared fixtures (mock config, temp dirs, singleton resets)
├── test_config.py       # Config singleton, load/save, dot-notation get/set
├── test_core.py         # Workspace root finding, context file loading, prompt building
├── test_core_turn.py    # run_turn: TurnResult, cancellation state, tool result callbacks
├── test_memory.py       # Memory, ChatMemory persistence and trimming
├── test_skills.py       # SkillLoader frontmatter parsing, load_all
├── test_files.py        # read_file handler (full read + head-style max_lines)
├── test_ipc.py          # IPC protocol: message factories, serialization, loopback transport
├── test_emitter.py      # TurnEmitter: core callbacks -> events, cancel state
└── test_tools.py        # Tool registration, discovery, execution
```

### Running Tests

```bash
# Run all tests
uv run python -m unittest discover -s tests -v

# Run a specific test file
uv run python -m unittest tests.test_core -v

# Run a specific test class
uv run python -m unittest tests.test_core.TestFindWorkspaceRoot -v

# Run a single test
uv run python -m unittest tests.test_core.TestFindWorkspaceRoot.test_finds_agents_md_in_parent -v
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
