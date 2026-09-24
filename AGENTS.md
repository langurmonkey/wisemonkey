# Wisemonkey — Project Guide

## What is this project?

Wisemonkey is a simple, extensible CLI AI agent for Linux and macOS terminals. It connects to any OpenAI/Anthropic/Ollama-compatible endpoint and provides session management, persistent memory, vector store document embedding, native + MCP tools, and skills.

## Project Structure

```
wisemonkey/
├── agent/                  # Core agent code.
│   ├── agent.py            # REPL frontend: agent loop, prompt handling, key bindings.
│   ├── tui.py              # Textual TUI frontend.
│   ├── client.py           # Client-side connection to a daemon server.
│   ├── server.py           # Daemon server (UDS) owning the session Core.
│   ├── ipc.py              # Client/server protocol: messages, payloads, transports.
│   ├── emitter.py          # Turn event emitter: core callbacks -> IPC events.
│   ├── commands.py         # Slash commands (e.g. /embed, /quit).
│   ├── completion.py       # Smart path completion for the prompt.
│   ├── at_files.py         # @file/@dir reference expansion.
│   ├── shellcmd.py         # `!` shell command handling.
│   ├── startup.py          # Startup banner and info rendering.
│   ├── tokens.py           # Token counting (tiktoken).
│   ├── history.py          # Input history handling.
│   ├── config.py           # Configuration loading and handling.
│   ├── console.py          # Rich console output with themed formatting.
│   ├── output.py           # Output adapter protocol (REPL, TUI, IPC).
│   ├── core.py             # Core agent functions, like API connection and tool calls.
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
5. Loaded skills
6. Chat history (last — it changes every turn, keeping the static prefix cacheable)

### IPC Protocol (`agent/ipc.py`)

Phase 0 of the client/server split (see the `CLIENT_SERVER` session plan). Defines the wire protocol that lets a client (REPL/TUI) talk to a server that owns `Core`/`Memory`/`Config`:

- `Message` envelope: `kind` (`event`/`request`/`response`/`error`), `name`, `payload`, `id`, `reply_to`, `turn_id`, `ts`.
- Framing: newline-delimited JSON via `encode_message()` / `decode_message()`.
- `PROTOCOL_VERSION` + `check_protocol_version()` (major must match).
- Events (`Event`), client requests (`ClientRequest`), server requests (`ServerRequest`, mirrors `OutputAdapter`), `InjectWhen` for mid-turn/between-turn injection.
- `OutputPayload` carries pre-rendered output (`format`, `level`, `indent`, panel/rule fields) so the server needs no Rich console.
- `Transport` Protocol (`send`/`recv(timeout)`/`close`/`closed`) with an in-process `LoopbackTransport` / `loopback_pair()` for phase-1 work.
- `payload_as()` reconstructs a payload dataclass, ignoring unknown keys (forward compatibility).
- **Full wire specification:** `docs/PROTOCOL.md` — message envelope, all requests/events/RPCs with payloads, handshake flow, transports, and the single-reader-thread rule. Keep it in sync when changing `agent/ipc.py`.

### Client/server modes (`agent/server.py`, `agent/client.py`, `agent/agent.py`, `agent/tui.py`)

Phase 2/3 of the client/server split. The **server owns the session authoritatively**: `Core`, `ChatMemory`, profile/notes, vector store, tool execution, and persistence all live in the daemon process.

- `wmk --server [session]` runs the daemon: binds a UDS at `$XDG_RUNTIME_DIR/wisemonkey/$SESSION.sock`, serves requests until stopped.
- Clients (REPL `agent.py`, TUI `tui.py`) try `ServerConnection.connect(session, spawn=False)` at startup. On success they run as thin remote clients (turns via `remote.prompt`, commands via `remote.command`, cancel via `remote.cancel`, memory stats via `MEMORY_STATS`); on failure they fall back to local mode (in-process `Core` + loopback emitter).
- `ServerConnection` uses a **single reader thread** that dispatches all incoming messages: replies → per-request queues, events → the current turn's `on_event`, server requests → RPC handlers. Never `recv()` from multiple threads.
- Client requests: `PROMPT`, `COMMAND`, `CANCEL`, `PING`, `MEMORY_STATS`, `RECORD` (records a client-side exchange, e.g. a locally run `!` shell command, into the server's chat history), `SHUTDOWN`.
- `ServerRequest` RPCs (confirmations, questions, subprocess runs) are relayed to the attached client and answered there; `run_subprocess` executes client-side with full terminal control.
- In remote mode, frontends build a lightweight stub `Core` (`SimpleNamespace(config=local Config, memory=session-scoped Memory)`) for purely client-side UI concerns (prompt session, completions, paste files, startup banner). The stub never runs turns or persists chat history — the server does.
- Session matching is by name: a client only attaches to a daemon running the **same session name**; otherwise it starts its own independent session.

### Turn results and cancellation (`agent/core.py`, `agent/emitter.py`)

`Core.run_turn()` returns a `TurnResult` (`response`, `total_tokens`, `n_tools`, `gen_time`, plus `cancelled` and `error`). It **unpacks as the historical 4-tuple**, so existing callers keep working, but new drivers should read `.cancelled` instead of string-matching on `"[Cancelled]"`.

Cancellation is **observable state, not an exception**: `_stream_handler()` sets `self._turn_cancelled` and stops the stream when `poll()` returns True, and `run_turn()` returns `TurnResult(cancelled=True)` without persisting a partial answer. `cancel_callback` remains supported (legacy `raise_on_cancel` emitters still raise `TurnCancelled`, which `run_turn()` converts into a cancelled result).

`run_turn()` accepts an optional `tool_result_callback` invoked as `(tool_id, tool_name, content, is_error, duration[, image_base64, mime_type])` after each tool finishes, so tool results can be streamed as events.

### Tool System (`agent/tools.py` + `tools/`)

Tools are defined using the `@tool(name, description, parameters)` decorator. They are auto-discovered on startup. Each tool file in `tools/` contains one or more decorated handler functions.

`read_file` accepts an optional `max_lines` parameter to read only the first N lines from the top (like `head`). This is useful for large files: it keeps the tool result small and avoids flooding the chat history. When set, the result includes `truncated`, `total_lines`, and `shown_lines` metadata. Omit it or set it to `0` to read the whole file.

#### Chat memory accounting

`ChatMemory` tracks `total_tokens` and triggers `/session-chat-compact` when it exceeds `agent.max_chat_history`. `total_tokens` is computed by tokenizing the exact rendered history returned by `get_formatted(timestamps=False, width=0)`, including tool-result truncation and compact adjacent tool call/result blocks. Recount after changes to stored exchanges, loading, trimming, or clearing. Keep this accounting in sync with `get_formatted()` if its rendering changes.

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

Run `wmk --onboard` for interactive configuration.

## Development

- Requires Python 3.13+ and `uv`
- Dependencies: `uv sync`
- Run from source: `uv run wmk`
- Build: `uv build`
- Entry point: `agent.__main__:main` → `wmk` CLI command (with a deprecated `wisemonkey` alias)

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
