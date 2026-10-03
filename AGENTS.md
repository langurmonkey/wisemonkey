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
│   ├── footer.py           # Sticky REPL footer via ANSI scroll regions (DECSTBM).
│   ├── commands.py         # Slash commands (e.g. /embed, /quit).
│   ├── completion.py       # Smart path completion for the prompt.
│   ├── keys.py             # Terminal key protocol (modified Enter, kitty keys).
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
│   ├── mdstream.py         # Streaming markdown renderer for the REPL.
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
│   ├── screenshot.py       # Screen capture tool (base64 JPEG, user-confirmed).
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

Frontends must not print a line per successful tool. The `TOOL_CALL` event ("Activating tool: X") already says the tool ran, so a "finished in 0.0s" line under every call is pure noise: `Agent.tool_result_callback` and `Tui._append_tool_result` stay silent below `agent.tool_slow_threshold` (default `1.0` s, inclusive) and print the duration only above it. Errors are always reported. The tool's own result text is still streamed to the model and stored in chat history; this is only about what the human sees.

### Modified Enter (`agent/keys.py`)

A terminal sends the *same* byte (`0x0D`) for Enter, Shift+Enter and Ctrl+Enter, so "Shift+Enter inserts a newline" is only possible when the terminal sends something distinguishable. Two mechanisms do that, and both decode to `c-m` — the raw sequence survives in `event.data`, which is what the bindings test:

- **xterm `modifyOtherKeys`** (`CSI 27 ; mods ; 13 ~`) — already in prompt_toolkit's table.
- **Kitty keyboard protocol** (`CSI > 1 u` to push the disambiguate flag, `CSI < u` to pop) — kitty, Ghostty, Alacritty, WezTerm, foot, Rio. `enable_kitty_keyboard()` writes the push and registers an `atexit` pop; `_pop_kitty_keyboard()` runs it on the normal exit path too, because a terminal left in disambiguate mode breaks the shell that takes over. Terminals that ignore the sequence degrade silently, so nothing is gated on TERM on the send path.

prompt_toolkit 3.0.52 does not parse the `CSI u` form, so `extend_ansi_sequences()` adds those sequences to its table (idempotent, never clobbering an existing mapping). `install_enter_bindings(kb)` is the single definition of the rule — **Enter submits, a modified Enter inserts a newline** — including an explicit `c-j` binding: prompt_toolkit's default for Ctrl+J is a newline, but the `enter` binding resolves the same key on some terminals and a bound handler wins, which made Ctrl+J submit the prompt instead of continuing the line.

### Sticky footer (`agent/footer.py`)

During an assistant turn the REPL pins a status line (model, session, memory usage) to the bottom of the terminal using the ANSI scroll region (`DECSTBM`, `\x1b[1;<H-4>r`): the bottom four rows (separator, status, steering input / spinner, key hints) are excluded from scrolling, so they stay put while output streams. No cursor-position queries, no Rich file swapping — output code is untouched. Both modules degrade to no-ops when stdout is not a TTY or the terminal is too small.

Two invariants matter here:

- **One writer at a time.** `Footer.start()` swaps `sys.stdout` for a `_LockedWriter` proxy guarded by the module-level `WRITE_LOCK`, and `stop()` restores it. Once the footer is armed there are two independent writers on fd 1 (the footer and whatever is streaming the response), and without serialization a redraw can be split mid-escape-sequence by a streamed chunk. Rich resolves `sys.stdout` lazily (`Console.file` is a property), so the proxy is picked up without recreating the console.
- **Never write the footer through a Rich `FileProxy`.** `console.status()` builds a `Live` display, and `Live` replaces `sys.stdout` with a `FileProxy` for its lifetime. A `FileProxy` *interprets* ANSI escapes: the footer's `\x1b[24;1H` cursor addressing would be decoded, markup-parsed and re-emitted as the literal text `[24;1H`. All footer write sites therefore go through `real_stream()`, which follows `rich_proxied_file` to skip any proxy while stopping at our own `_LockedWriter` (so the lock is still taken). `Agent._spinner()` avoids the conflict at the source: while `output.footer_active()` it returns `None` and hands the label to `OutputAdapter.footer_spinner()` instead of starting a `Live` at all.
- **The footer animates its own spinner.** `Footer.set_spinner(label)` starts a daemon thread that ticks the braille frames (`SPINNER_FRAMES`, every `SPINNER_INTERVAL`) and redraws the *transient* row — the third row, which otherwise shows the steering input. Each tick takes `WRITE_LOCK` and only ever addresses rows inside the reserved zone, so it cannot disturb the streaming response. Typed input wins: `update_status()` records whether the transient row holds a caller-supplied line (`_input_is_spinner`), and a tick that finds a foreign owner on the row does nothing rather than overwriting what the user is typing. `set_spinner("")` joins the thread before blanking. `stop()` calls it first, so no thread can redraw into a torn-down footer.
- **Re-read the geometry.** `update_status()` re-reads the terminal size on every redraw, so a resize mid-turn cannot leave stale row numbers behind. Every line is passed through `clip_ansi()` so it can never wrap into the row below.

Set `agent.footer_debug_bytes: true` to append every byte written to the footer (repr'd) to `$XDG_STATE_HOME/wisemonkey/footer.log` — useful when a terminal renders something unexpected.

**Never blank the reserved rows with `\x1b[2K\r\n`.** That is the obvious spelling — go to the first footer row, erase, CR, LF, repeat — but the final `\n` puts the cursor *below the last screen row*, and leaving the bottom margin scrolls the viewport up by one. In `stop()` the scroll region has just been reset to the full screen, so the whole visible screen shifts and the REPL's turn header is the line that gets eaten (it looks like "Prompt processed" overwriting the banner; the two are unrelated). `_blank_footer_rows()` addresses each row explicitly (`CUP` + `EL`) so no write ever lands past the last row. `tests/test_footer.py` pins the invariant: no newline in the blanking output, and no row addressed beyond the screen height.

### Mid-turn steering (`agent/steer.py`)

While a turn is running, `agent/steer.py` puts the tty in **cbreak** mode and reads keystrokes on a daemon thread. Enter queues the line; the REPL main loop drains the queue *before* showing the prompt again (`Agent._steer_drain`), so a queued line goes through the exact same `@`-expansion, command dispatch and turn path as a typed prompt.

This is deliberately **not** a line editor: nothing is echoed to the terminal, the partial line is only surfaced on the footer's dedicated input row (`⤷ <text>` while typing, `↳ queued: <text>` once submitted; the footer draws its spinner animation on that same row when the row is otherwise free), and there is no history or completion. Keeping it out of stdout means no second writer on the cursor and no risk of corrupting the stream.

It clears `ICANON` and `ECHO` but deliberately **keeps `ISIG`**. With `ISIG` off the kernel never turns Ctrl+C into SIGINT: the reader thread would just see a literal `0x03` byte and set a flag, which the turn does not check while it is blocked in `router.chat()` or inside a tool — so Ctrl+C would silently do nothing during exactly the phases it is most needed. Keeping `ISIG` on means Ctrl+C raises `KeyboardInterrupt` in the main thread, which `core.py` already unwinds (`TurnResult(cancelled=True)`). `0x03` is dropped in `_handle_byte` as a belt-and-braces measure.

Lifecycle hooks in `agent/output.py`:

- `PromptHistory` (in `agent/history.py`) wraps `prompt_toolkit`'s `FileHistory` and filters lines whose first character is `/` out of `load_history_strings()`. Commands stay on disk; only what <kbd>↑</kbd> navigates is filtered, because commands cluster at the ends of a session and push real prompts out of reach.
- `RichOutputAdapter.steer_start/stop/take/take_all/ack/line/pending` — the real implementation; the protocol, `TuiOutputAdapter` and `IpcOutputAdapter` get no-op defaults. `take_all` drains everything at once (injection seam); `ack(text)` removes one line by value, which is how the remote path confirms delivery.
- `footer_update(status_line, input_line)` takes the status line and the steering input as two separate arguments; they are drawn on two separate rows by `agent/footer.py`.
- `footer_spinner(text)` starts/stops the footer's own animation for a stage label. `_spinner()` returns `None` while the footer is armed, so `spinner_prompt`/`spinner_thinking` stay `None` in that case — ownership of the footer row is tracked by `_footer_spinner_mine` instead, so only the stage that started a spinner clears it.
- `steer_paused()` — a context manager that stops the reader while the agent asks the user something (`ask_*`) or hands the terminal to a subprocess (`run_subprocess`, which additionally tears down and restores the scroll region). Queued lines survive a pause. It re-arms the reader **only if it was armed on entry**: between turns the agent calls `steer_stop()`, so a command like `/config edit` or `/edit` runs with the reader already disarmed and prompt_toolkit owning the tty. Re-arming unconditionally would leave two threads reading the same fd, leave the tty in cbreak mode after the editor exited, and silently drop most keystrokes into a steering buffer nobody drains.
- `Footer` writes are serialized with a `threading.Lock`, since status updates now come from the main thread and the steer reader.
- Disabled with `agent.steer_midturn: false` in `config.yaml`; a silent no-op when stdin is not a tty.

### Injection at the tool seam

A submitted line normally reaches the model at the *next tool-loop seam*, not at the end of the turn. The seam is the single point in `Core.run_turn` where the loop is about to call the LLM again after a tool batch; `inject_callback` is polled there, and every pending line is drained at once and appended as a `user` message plus a `memory.add_chat_exchange`, so it persists and counts tokens like any other user message.

Ownership differs by mode, because the queue lives in the client but the turn lives wherever it was started:

- **local** — `Agent._steer_inject` pops the client's FIFO (`SteerInput.take_all`) and announces each line as it leaves.
- **remote** — the client sends `INJECT(after_tool)`; the server queues it (`WisemonkeyServer._injections`) and its injection seam emits `Event.INJECTED`; the client pops its own copy on that event (`Agent._handle_injected`). `BETWEEN_TURNS` and `INTERRUPT` are rejected server-side — the first needs no queue, the second is unimplemented.

In both modes the line stays in the client FIFO until delivery is confirmed, so a turn that ends before another tool call loses nothing: the main loop's `_steer_drain` runs the line as an ordinary next prompt. A new turn also clears the server queue, so injections never leak across turns. A callback that raises is swallowed by `Core._drain_injections` — a UI failure to collect typed text must not abort a working turn.

### Chat memory rolling window (`agent/memory.py`)

`ChatMemory` supports a turn-based rolling window via `agent.memory_rolling_window_turns` (default `0` = disabled). When > 0, only the last n exchanges (a user message plus everything after it until the next user message) are kept; older exchanges are destructively removed. Trimming happens after each `add_exchange()` and at load time (reconciling a lowered setting), before the token-cap compaction, which remains as a backstop.

### Tool System (`agent/tools.py` + `tools/`)

Tools are defined using the `@tool(name, description, parameters)` decorator. They are auto-discovered on startup. Each tool file in `tools/` contains one or more decorated handler functions.

`read_file` reads a *window* of a file. Three optional parameters:

- `max_lines` — at most N lines, like `head`. Keeps the tool result small and avoids flooding the chat history.
- `offset` — a 1-based line to start at, like `tail -n +N`. The skipped prefix is streamed past, never buffered, so a window near the end of a large file does not load it.
- `show_line_numbers` — prepend numbers, always at their **real** positions in the file and padded to the width of the file's line count. Numbering a window 1..n would make the numbers useless as an anchor for a later edit.

`offset` and `max_lines` compose as a window (start, then how many): `offset: 700, max_lines: 90` is lines 700-789, not 0-89 and not an empty range. Either may be omitted for "no limit" on that side.

Every result carries `start_line` and `end_line`. When the file was not returned whole it also carries `truncated`, `total_lines` (the file's real length, not the window's) and `shown_lines`. `offset` accepts a string, since models send numbers as strings; a negative `offset` is ignored rather than read as `tail -n -N`.

#### Chat memory accounting

`ChatMemory` tracks `total_tokens` and triggers `/history-compact` when it exceeds `agent.max_chat_history`. `total_tokens` is computed by tokenizing the exact rendered history returned by `get_formatted(timestamps=False, width=0)`, including tool-result truncation and compact adjacent tool call/result blocks. Recount after changes to stored exchanges, loading, trimming, or clearing. Keep this accounting in sync with `get_formatted()` if its rendering changes.

### Slash Commands (`agent/commands.py`)

Commands use the `@cmd(name, description, aliases)` decorator and are auto-registered. Each returns `(ok: bool, msg: str, content: str, markdown: str)`.

A command **cannot run a turn**: the frontend owns the turn loop, the emitter, the footer and the spinners. Commands that need a prompt to run instead set `Core.pending_prompt` and return; `Agent._take_pending_prompt` (local) or `CommandResultPayload.pending_prompt` (remote) picks it up and sends it as the next prompt, through exactly the same path as a typed one.

### Correcting a turn (`/edit`, `/retry`, `/undo`)

`ChatMemory.drop_last(n)` removes the *n* most recent **exchanges** from the tail, where an exchange starts at a user message and runs to the next one (so tool calls and results go with their turn). It never removes a fragment: a tool result left without its matching call would be malformed in the prompt. `n <= 0` clears everything. `last_user_prompt()` returns the most recent user message, which is what `/retry` and `/edit` re-run.

`/edit` opens that prompt in `$EDITOR` on a temporary `.md` file, runs the editor through `OutputAdapter.run_subprocess` (so the sticky footer and steering are torn down around it), and replaces the old exchange. An unchanged, empty, or aborted edit re-runs nothing. `/retry --drop` is the same re-run from a clean context.

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
├── test_files.py        # read_file handler (full read, head-style max_lines, offset window)
├── test_ipc.py          # IPC protocol: message factories, serialization, loopback transport
├── test_emitter.py      # TurnEmitter: core callbacks -> events, cancel state
├── test_keys.py         # Terminal key protocol: modified Enter, newline bindings
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
