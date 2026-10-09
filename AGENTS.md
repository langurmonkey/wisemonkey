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
│   ├── palette.py         # The one source of truth for every colour drawn.
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
│   ├── jobs.py              # Background commands: spawn / poll_job / kill_job.
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

**Not every reader understands the protocol.** The flag is a *terminal* mode, so anything else reading the tty inherits it, and only prompt_toolkit has been taught the sequences. `rich.prompt` (`ask_string`/`ask_float`/`ask_confirm`) goes through the builtin `input()`, i.e. readline, which knows only `CSI D`; under the flag a Left arrow arrives as `CSI 1;129D` and is **typed into the line as escape characters** instead of moving the cursor. `legacy_key_encoding()` (in `agent/keys.py`) pops the flag for the duration and re-pushes on exit, and it wraps exactly those three plus `run_subprocess` (a `$EDITOR` that never asked for the protocol has the same problem). It deliberately does **not** wrap `ask_choice`, which is prompt_toolkit and parses them itself — dropping the flag there would only cost Shift+Enter. The pushed state lives in `keys.py` (`_protocol_pushed`), not on whoever pushed it, because the flag outlives the prompt that set it; the atexit pop is registered once rather than per push, so repeated push/pop cycles cannot unbalance it.

**`rich.prompt` also needs readline imported, and that was the actual arrow-key bug.** CPython only gives the builtin `input()` a line editor when the `readline` module has been *imported*; it imports it for an interactive interpreter, never for a script. Wisemonkey is a script, so every question it asked degraded to a bare canonical read: the kernel echoed each byte it received, and Right/Left typed `^[[C`/`^[[D` into the line. Popping the kitty flag was necessary but not sufficient — the bytes arrived in their *legacy* encoding and still had no line editor to interpret them. `ensure_readline()` (`agent/keys.py`) imports it once, called both at REPL start and before each of the three prompts, and returns False rather than raising when readline is genuinely absent. Do not remove it on the grounds that the protocol work covers arrow keys.

**A line editor is not enough: readline must be told what the prompt is.** `rich.prompt` prints the question itself and then calls `input()` with an *empty* prompt, so readline believes the line buffer begins at column 1. Every redraw it performs — Backspace above all, and equally an insert in the middle of the line or any character typed after a Left — then rewrites from column 1 and issues an erase-to-end-of-line, which wipes the question. The symptom is precise: press Backspace once and the whole question disappears, leaving only the typed text. Typing and even arrow keys look fine, because readline can patch those in place; only a redraw exposes it.

`_ReadlinePromptMixin` (`agent/output.py`) overrides Rich's `get_input` to hand readline the prompt instead, and `readline_prompt()` (`agent/keys.py`) builds it. Three things in there are load-bearing:

- **ANSI escapes must be marked non-printing.** readline counts the *characters* of its prompt to work out where the buffer starts, so an SGR escape it does not know about is counted as if it were visible and every column afterwards is wrong. `mark_prompt_nonprinting()` wraps each escape in `\001`/`\002` (`RL_PROMPT_START_IGNORE`/`END_IGNORE`, honoured by both GNU readline and libedit).
- **The prompt must not be wrapped.** Rich would break a long question at the console width; readline does its own wrapping and knows how to undo its own, whereas an embedded newline is read as a character of the prompt and skews the arithmetic permanently. `Text.render()` is used rather than `Console.render()` for exactly this reason — the latter applies the width and appends a line break.
- **A non-terminal console gets plain text with no delimiters.** `input()` prints a prompt verbatim when readline is not in play, so the raw SOH/STX bytes would reach the output.

Do not "simplify" this back to `RichPrompt.ask()`: the prompt looks right until the first Backspace. `tests/test_readline_prompt.py` drives a real pty and replays the bytes with a cursor tracker, because no mocked test can see this; it also asserts that Rich's own behaviour still reproduces the bug, so the fix cannot pass for the wrong reason.

### Sticky footer (`agent/footer.py`)

During an assistant turn the REPL pins a status line (model, session, memory usage) to the bottom of the terminal using the ANSI scroll region (`DECSTBM`, `\x1b[1;<H-4>r`): the bottom four rows (separator, status, steering input, key hints with the stage/thinking indicators on its right) are excluded from scrolling, so they stay put while output streams. No cursor-position queries, no Rich file swapping — output code is untouched. Both modules degrade to no-ops when stdout is not a TTY or the terminal is too small.

Two invariants matter here:

- **One writer at a time.** `Footer.start()` swaps `sys.stdout` for a `_LockedWriter` proxy guarded by the module-level `WRITE_LOCK`, and `stop()` restores it. Once the footer is armed there are two independent writers on fd 1 (the footer and whatever is streaming the response), and without serialization a redraw can be split mid-escape-sequence by a streamed chunk. Rich resolves `sys.stdout` lazily (`Console.file` is a property), so the proxy is picked up without recreating the console.
- **Never write the footer through a Rich `FileProxy`.** `console.status()` builds a `Live` display, and `Live` replaces `sys.stdout` with a `FileProxy` for its lifetime. A `FileProxy` *interprets* ANSI escapes: the footer's `\x1b[24;1H` cursor addressing would be decoded, markup-parsed and re-emitted as the literal text `[24;1H`. All footer write sites therefore go through `real_stream()`, which follows `rich_proxied_file` to skip any proxy while stopping at our own `_LockedWriter` (so the lock is still taken). `Agent._spinner()` avoids the conflict at the source: while `output.footer_active()` it returns `None` and hands the label to `OutputAdapter.footer_spinner()` instead of starting a `Live` at all.
- **The footer animates its own spinner.** `Footer.set_spinner(label)` starts a daemon thread that ticks the braille frames (`SPINNER_FRAMES`, every `SPINNER_INTERVAL`) and redraws the right of the *key-hints* row — the bottom one, where the stage label sits left of the `💡 Thinking...` indicator. Each indicator carries its own frame and the same thread drives both, so `set_spinner("")` leaves the animation running while `Footer.set_thinking` is still on (and vice versa: reasoning alone starts the thread). Each tick takes `WRITE_LOCK` and only ever addresses rows inside the reserved zone, so it cannot disturb the streaming response. That row belongs to the footer alone, so a tick never competes with the typed steering line on the row above it; both indicators are clipped to the terminal width so the row cannot wrap. `set_spinner("")` joins the thread before blanking. `stop()` clears the indicator first and then calls it, so no thread can redraw into a torn-down footer.
- **Re-read the geometry.** `update_status()` re-reads the terminal size on every redraw, so a resize mid-turn cannot leave stale row numbers behind. Every line is passed through `clip_ansi()` so it can never wrap into the row below.

**Arm the footer *before* printing anything for the turn.** `start()` blanks the bottom `FOOTER_LINES` rows (`_blank_footer_rows`). Output printed first therefore vanishes whenever the screen was already full and it landed on one of those rows — which is how the `Wisemonkey ⇒ model` turn header used to disappear intermittently. The header itself is gone now (the model lives in the footer; printing it every turn was repeated chrome), but the invariant is not: `run_interactive()` and `_run_interactive_remote()` still call `output.footer_start()` before the `newline()` that separates the turn from the prompt above it, and that blank line would vanish the same way. The one-shot paths (`_run_once_local`, `_run_once_remote`) keep the header — they are for scripting, where each run is self-describing. `tests/test_agent_stages.py::TestTurnHeaderOrdering` pins the ordering in the source.

Set `agent.footer_debug_bytes: true` to append every byte written to the footer (repr'd) to `$XDG_STATE_HOME/wisemonkey/footer.log` — useful when a terminal renders something unexpected.

**Never blank the reserved rows with `\x1b[2K\r\n`.** That is the obvious spelling — go to the first footer row, erase, CR, LF, repeat — but the final `\n` puts the cursor *below the last screen row*, and leaving the bottom margin scrolls the viewport up by one. In `stop()` the scroll region has just been reset to the full screen, so the whole visible screen shifts and the REPL's turn header is the line that gets eaten (it looks like "Prompt processed" overwriting the banner; the two are unrelated). `_blank_footer_rows()` addresses each row explicitly (`CUP` + `EL`) so no write ever lands past the last row. `tests/test_footer.py` pins the invariant: no newline in the blanking output, and no row addressed beyond the screen height.

### Live token counter (`agent/agent.py`)

The token count is shown twice, on purpose:

- **Live in the footer, during the turn.** `_footer_status()` appends a `Δ+…` segment while `_turn_in_progress` is set. The number is `used - _turn_start_mem` plus `_live_chars // 4` — the streamed response text converted with the chars/token estimate. It is refreshed by `_footer_live_tick()`, called from `content_callback` on every chunk but **throttled to ~4 Hz** (`_last_footer_tick`): a redraw per chunk would fight the footer's spinner thread for `WRITE_LOCK` and burn CPU on a fast stream. The estimate is deliberately cheap; the exact figure replaces it at turn end.
- **Exactly in the status line, at turn end.** `_statusline()` shows `Δ+…` (chat memory now minus `_turn_start_mem`, snapshotted at `footer_start()` time in both the local and remote turn loops) and `p:…` — provider-reported prompt tokens, summed over the turn's LLM rounds in `Core.run_turn` (`total_prompt_tokens`). Each round re-sends the whole conversation, so this grows superlinearly with tool use; it is the number that answers "why did this turn cost so much".

**Provider usage plumbing.** `_StreamChunk` carries a `usage` field. The OpenAI adapter sends `stream_options: {include_usage: true}` (servers that predate the option ignore it); the Anthropic adapter surfaces `final_msg.usage` on the final chunk; Ollama surfaces `prompt_eval_count`/`eval_count` on the `done` part. `_stream_handler()` picks the usage chunk up (a choices-less chunk is `continue`d before the delta access) and `_finish_inference()` prefers it over the tiktoken estimate, which covers only the response text and undercounts by the prompt. `TurnResult.prompt_tokens` is 0 when the provider reports nothing, and the UI **omits** `p:` and cost rather than showing a wrong number.

**Cost is opt-in.** `_turn_cost()` reads `model.pricing` (`input`/`output` per million tokens) from config. Prices change without notice, so they are never hard-coded: no config, no cost figure. `config.yaml` documents the key, commented out.

`tests/test_agent_stages.py::TestLiveTokenCounter` pins the delta, the prompt-token and cost gating, the char accumulation and the throttle.

### Thinking status in the footer (`agent/agent.py`)

Visible reasoning is **not** marked in the chat output. Instead, the footer's bottom row — the key-hints row — shows a `💡 Thinking...` indicator on its right, after the stage spinner, while the model is reasoning (`Agent._reasoning_active`). The indicator animates with the same braille frames as the stage spinner, so reasoning is visibly *live* rather than a static label. This keeps the response area clean — no "Thinking..." / "Done thinking" markers — while still making the model's reasoning state visible at a glance.

`_reasoning_start` sets the flag and calls `output.footer_thinking(True)`; `_reasoning_stop` clears it and calls `output.footer_thinking(False)`. `Footer.set_thinking` redraws the hints row on the change; `Footer.start` clears the indicator, so a turn that ended mid-reasoning cannot leave it stuck on the next one. The stage spinner (`Footer.set_spinner`) and the thinking indicator share the right of the hints row, clipped to the terminal width; neither ever touches the transient row above, so typed steering text is never overwritten.

The TUI needs no equivalent: it flushes the reasoning buffer without a marker and has its own status-bar indicator. `tests/test_agent_stages.py::TestReasoningFooterIndicator` pins the flag lifecycle and the absence of chat-output markers; `tests/test_footer.py::TestFooterHintsRow` pins which row the indicators are drawn on.

### Mid-turn steering (`agent/steer.py`)

While a turn is running, `agent/steer.py` puts the tty in **cbreak** mode and reads keystrokes on a daemon thread. Enter queues the line; the REPL main loop drains the queue *before* showing the prompt again (`Agent._steer_drain`), so a queued line goes through the exact same `@`-expansion, command dispatch and turn path as a typed prompt.

This is deliberately **not** a line editor: nothing is echoed to the terminal, the partial line is only surfaced on the footer's dedicated input row (`⤷ <text>` while typing, `↳ queued: <text>` once submitted; the stage spinner and the thinking indicator live on the hints row below it, never on this one), and there is no history or completion. Keeping it out of stdout means no second writer on the cursor and no risk of corrupting the stream.

It clears `ICANON` and `ECHO` but deliberately **keeps `ISIG`**. With `ISIG` off the kernel never turns Ctrl+C into SIGINT: the reader thread would just see a literal `0x03` byte and set a flag, which the turn does not check while it is blocked in `router.chat()` or inside a tool — so Ctrl+C would silently do nothing during exactly the phases it is most needed. Keeping `ISIG` on means Ctrl+C raises `KeyboardInterrupt` in the main thread, which `core.py` already unwinds (`TurnResult(cancelled=True)`). `0x03` is dropped in `_handle_byte` as a belt-and-braces measure.

Lifecycle hooks in `agent/output.py`:

- `PromptHistory` (in `agent/history.py`) wraps `prompt_toolkit`'s `FileHistory` and filters lines whose first character is `/` out of `load_history_strings()`. Commands stay on disk; only what <kbd>↑</kbd> navigates is filtered, because commands cluster at the ends of a session and push real prompts out of reach.
- `RichOutputAdapter.steer_start/stop/take/take_all/ack/line/pending` — the real implementation; the protocol, `TuiOutputAdapter` and `IpcOutputAdapter` get no-op defaults. `take_all` drains everything at once (injection seam); `ack(text)` removes one line by value, which is how the remote path confirms delivery.
- `footer_update(status_line, input_line)` takes the status line and the steering input as two separate arguments; they are drawn on two separate rows by `agent/footer.py`.
- `footer_spinner(text)` starts/stops the footer's own animation for a stage label, drawn on the right of the key-hints row. `_spinner()` returns `None` while the footer is armed, so `spinner_prompt`/`spinner_thinking` stay `None` in that case — ownership of the label is tracked by `_footer_spinner_mine` instead, so only the stage that started a spinner clears it.
- `footer_thinking(active)` shows/hides the `💡 Thinking...` indicator, on the right of the same row. `Footer.set_thinking` redraws it on change and drives the animation thread when it is the only indicator around, and `Footer.start` clears it, so a turn that ended mid-reasoning cannot leave it stuck on the next one.
- `steer_paused()` — a context manager that stops the reader while the agent asks the user something (`ask_*`) or hands the terminal to a subprocess (`run_subprocess`, which additionally tears down and restores the scroll region). Queued lines survive a pause. It re-arms the reader **only if it was armed on entry**: between turns the agent calls `steer_stop()`, so a command like `/config edit` or `/edit` runs with the reader already disarmed and prompt_toolkit owning the tty. Re-arming unconditionally would leave two threads reading the same fd, leave the tty in cbreak mode after the editor exited, and silently drop most keystrokes into a steering buffer nobody drains.
- `Footer` writes are serialized with a `threading.Lock`, since status updates now come from the main thread and the steer reader.
- Disabled with `agent.steer_midturn: false` in `config.yaml`; a silent no-op when stdin is not a tty.

### Injection at the tool seam

A submitted line normally reaches the model at the *next tool-loop seam*, not at the end of the turn. The seam is the single point in `Core.run_turn` where the loop is about to call the LLM again after a tool batch; `inject_callback` is polled there, and every pending line is drained at once and appended as a `user` message plus a `memory.add_chat_exchange`, so it persists and counts tokens like any other user message.

Ownership differs by mode, because the queue lives in the client but the turn lives wherever it was started:

- **local** — `Agent._steer_inject` pops the client's FIFO (`SteerInput.take_all`) and announces each line as it leaves.
- **remote** — the client sends `INJECT(after_tool)`; the server queues it (`WisemonkeyServer._injections`) and its injection seam emits `Event.INJECTED`; the client pops its own copy on that event (`Agent._handle_injected`). `BETWEEN_TURNS` and `INTERRUPT` are rejected server-side — the first needs no queue, the second is unimplemented.

In both modes the line stays in the client FIFO until delivery is confirmed, so a turn that ends before another tool call loses nothing: the main loop's `_steer_drain` runs the line as an ordinary next prompt. A new turn also clears the server queue, so injections never leak across turns. A callback that raises is swallowed by `Core._drain_injections` — a UI failure to collect typed text must not abort a working turn.

### Images from tools (`tools/screenshot.py`, `tools/read_image.py`)

`execute_tool()` serializes dict results to JSON, so what `Core._tool_calls()` receives is usually a **string** even when the handler returned a dict. The result is parsed back (`json.loads`, dicts only) before the image/error branches — without that, `isinstance(result, dict)` was never true, the multimodal branch was dead code, and images reached the model as a raw base64 string it could not see. A dict-shaped `{"error": ...}` was never flagged as an error either. A handler returning a plain string or a JSON list is untouched.

An image result (`image_base64` + `mime_type`) becomes a multimodal `tool` message with an `image_url` data-URL block. The routers translate it per provider:

- **OpenAI/LM Studio/generic** — sent as-is (`image_url` blocks).
- **Anthropic** — `_adapt_messages_for_anthropic()` converts `image_url` blocks to `{type: image, source: {type: base64, ...}}` in **tool results and user messages** (the pending-image path attaches to user prompts too).
- **Ollama** — has no image blocks at all: `_chat_ollama()` extracts the base64 data out of the blocks into a separate `images: [...]` list on the message and flattens content to plain text.

`tools/read_image.py` loads an image file from disk and returns the same shape as `screenshot`, so both reach the vision decoder through the identical path. `tests/test_core_turn.py::TestImageToolResults` pins the parse-back, the multimodal block, the error flag and the plain-string pass-through.

### Chat memory rolling window (`agent/memory.py`)

`ChatMemory` supports a turn-based rolling window via `agent.memory_rolling_window_turns` (default `0` = disabled). When > 0, only the last n exchanges (a user message plus everything after it until the next user message) are kept; older exchanges leave the live history. Trimming happens after each `add_exchange()` and at load time (reconciling a lowered setting), before the token-cap compaction, which remains as a backstop.

Trimmed entries are **never destroyed**: both `_trim_to_window()` and `_trim()` append what they drop to `chat_archive.jsonl` in the session directory (`ChatMemory._archive`). That file is append-only, is never read back into the prompt, and never affects token accounting or window state — it exists purely so the user can recover anything that aged out. Archive write failures are swallowed: the archive is a convenience, not state the agent depends on. Note that `_clear()` / `/history-clear` and `drop_last()` (used by `/edit`, `/retry`, `/undo`) are deliberate user actions and still remove entries without archiving.

### Tool System (`agent/tools.py` + `tools/`)

Tools are defined using the `@tool(name, description, parameters)` decorator. They are auto-discovered on startup. Each tool file in `tools/` contains one or more decorated handler functions.

**Malformed tool arguments are a tool error, never a crashed turn.** A model can emit a truncated or invalid `arguments` string (a bare `{` is common). `Core._tool_calls()` parses the arguments *before* calling the handler; a `JSONDecodeError` becomes a `tool` message carrying the error plus the raw text, recorded in chat history like any other result, and the loop continues so the model can re-issue the call. The router's Ollama adapter uses the same tolerance when re-serializing past calls. Without this, the parse error propagated to the turn handler and printed a misleading `Error sending prompt: Expecting property name enclosed in double quotes...` — the prompt was never the problem, the model's output was. `tests/test_core_turn.py::TestMalformedToolArguments` pins that the turn completes, the handler is never called, and the error reaches both history and the result callback.

`read_file` reads a *window* of a file. Three optional parameters:

- `max_lines` — at most N lines, like `head`. Keeps the tool result small and avoids flooding the chat history.
- `offset` — a 1-based line to start at, like `tail -n +N`. The skipped prefix is streamed past, never buffered, so a window near the end of a large file does not load it.
- `show_line_numbers` — prepend numbers, always at their **real** positions in the file and padded to the width of the file's line count. Numbering a window 1..n would make the numbers useless as an anchor for a later edit.

`offset` and `max_lines` compose as a window (start, then how many): `offset: 700, max_lines: 90` is lines 700-789, not 0-89 and not an empty range. Either may be omitted for "no limit" on that side.

Every result carries `start_line` and `end_line`. When the file was not returned whole it also carries `truncated`, `total_lines` (the file's real length, not the window's) and `shown_lines`. `offset` accepts a string, since models send numbers as strings; a negative `offset` is ignored rather than read as `tail -n -N`.

`patch_file` also has two modes, both line-based:

- **Text mode** (default) — `old_string` + `new_string`. Matching is whitespace-tolerant: *trailing* whitespace is ignored, leading indentation is not, so a block retyped by hand still matches. It must occur exactly once; if it occurs more than once, `start_line`/`end_line` narrow the search to disambiguate.
- **Line mode** — `start_line` (+ optional `end_line`) with no `old_string`, and `new_string` replaces exactly those lines. This is the mode that removes the re-read round trip: the characters do not have to be right, only the line numbers. `start_line = total_lines + 1` appends.

In text mode `start_line`/`end_line` are a search *window*, not a replacement range; the two are different parameters for different jobs, and `end_line` is inclusive. A window that excludes the real match reports where the text actually is rather than claiming it is gone. Both modes write atomically and report the line numbers changed; `new_string: ""` deletes.

`search_content` takes a literal substring by default and a **Python regex** with `regex: true`, so a pattern like `^def ` or `self\.\w+ = ` needs no shell quoting, and `^`/`$` are per-line anchors (`re.MULTILINE`). Metacharacters in a default literal query stay literal, so `config.yaml` matches itself. An invalid regex is an error, never a silent fallback to a literal search: a query that looks like a pattern and fails to compile is a mistake worth reporting, not something to quietly reinterpret. `multiline: true` matches across newlines and requires `regex`.

`max_results` (default 500) is reported in `omitted`, and the rendered `content` says how many matches were dropped. The cap used to slice silently, so "Found 500" could not be told apart from "Found 5000" — a caller would conclude the pattern is rarer than it is.

`spawn` / `poll_job` / `kill_job` (`tools/jobs.py`) run a command in the background. Use them for work that outlasts the turn — a test suite, a build — and `run_command` for anything short, since its `timeout` is all-or-nothing: either the command finishes inside it or you get nothing and have to guess how long to sleep.

Output goes to a **temp file, not a pipe**. A pipe is a fixed buffer that blocks the child when nobody is reading it, so a job meant to outlive the call that started it would deadlock against its own output. `poll_job` returns only what appeared since the previous poll, and a finished job stays pollable afterwards — the common shape is "is it done? yes — what did it say?", and a registry that forgets the job on exit answers the second half with "no such job". `all_output: true` re-reads from the start.

`output_bytes` is what the job has written **in total**, not what this poll returned; `omitted_bytes` is what is still unread and waiting for the next poll. A per-poll cap of 20k characters is what makes polling a chatty job affordable, and the two fields keep a truncated tail from reading like a complete one.

The child gets its own **process group**, so `kill_job` signals the whole group: a shell pipeline does not leave its second half running after the first half is dead. `kill_job` returns the output produced before dying (a killed test run still says which test failed) and, with `force: true`, uses `SIGKILL` for a command that traps `SIGTERM`. A process that ignores `SIGTERM` is reported as still running rather than claimed as killed. Jobs live in the agent's process and are killed by an `atexit` hook, so a job cannot outlive the session; the last 8 finished jobs are kept (logs included) and older ones reaped.

#### Chat memory accounting

`ChatMemory` tracks `total_tokens` and triggers `/history-compact` when it exceeds `agent.max_chat_history`. `total_tokens` is computed by tokenizing the exact rendered history returned by `get_formatted(timestamps=False, width=0)`, including tool-result truncation and compact adjacent tool call/result blocks. Recount after changes to stored exchanges, loading, trimming, or clearing. Keep this accounting in sync with `get_formatted()` if its rendering changes.

### API key resolution (`agent/router.py`)

Wisemonkey is typically driven through routers (OpenRouter, OpenCode, a local vLLM) that all speak the OpenAI dialect but each want their own key. A single global key variable is therefore the wrong shape: it must be rewritten on every provider switch (so it ends up in shell history), and one endpoint's key would be sent to another the moment the provider changed. `_get_api_key()` is instead an ordered, per-provider chain, first non-empty wins:

1. `model.api_key_env` — an environment variable **name** from `config.yaml`. Explicit always wins. It is a name and never the key itself, so `config.yaml` stays safe to share or commit; the key lives in the environment or `.env`.
2. `WM_API_KEY_<PROVIDER>` — namespaced per provider (`WM_API_KEY_GENERIC`, `WM_API_KEY_OPENAI`, `WM_API_KEY_ANTHROPIC`). This is what actually solves the router case: several endpoint keys coexist without any of them moving.
3. The provider's conventional variable — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`. This step is what keeps the `openai`/`anthropic` SDKs' own auto-discovery and every existing user setup working.
4. `WM_API_KEY` — a generic fallback, tried **last** for every key-bearing provider.

Local providers (`OLLAMA`, `LMSTUDIO`) resolve to no chain at all and get a `dummy` key. `api_key_candidates()` is the single definition of the order and is also what the missing-key error and `get_status()` render, so the message names every variable actually tried instead of guessing one. Two invariants the tests pin: only step 4 is shared (so `WM_API_KEY_GENERIC` never satisfies an Anthropic request), and an empty value anywhere in the chain is skipped rather than treated as a stop.

Do not collapse the chain into one variable, and do not cache the resolved key on the router: the point of the design is that switching `model.provider` / `model.base_url` re-resolves correctly.

### The endpoint is one decision (`agent/commands.py`)

`/model` configures a whole endpoint: URL → provider → key variable → model, with `/url` as an alias for the same flow. URL, provider, key and model are not four independent settings. OpenRouter, OpenCode and a local vLLM are all `generic` but need three *different* keys, and the key belongs to the endpoint, not to the model — switching from `deepseek-v4` to `gpt-5.1` on OpenRouter must never touch `api_key_env`. Asking for the URL first also decides what else is relevant: a local endpoint (`api_key_candidates` returns `[]`) asks no key question at all.

Two details worth preserving:

- `_match_provider_url()` returns `None` for an unknown host, while `_detect_provider()` collapses to `GENERIC`. The command needs the distinction: a **known** URL (`api.anthropic.com`, `localhost:11434`) decides the provider with no question — asking is how an Anthropic key ends up being offered to a generic host — while an **unknown** host must ask, defaulting to `generic`.
- `@cmd` registers the handler and returns `None`, so no command function name survives the decorator. Both `/model` and `/url` therefore delegate to the plain `_configure_endpoint()`; calling the decorated name raises `TypeError: 'NoneType' object is not callable`.

`/config` runs the merged flow exactly once (it used to run `/url` and then `/model`, asking for the URL twice).

The key step is a **choice list**, not free text. For an OpenAI-compatible host the variable name is genuinely arbitrary — one per endpoint — and having to recall it is the failure mode, so the options are the host's conventional name first (`suggest_key_env()`, matched longest-fragment-first over `_HOST_KEY_HINTS`), then the resolution chain, then whatever is already configured; variables actually present in the environment are marked `(set)`. `suggest_key_env()` is advisory only and never enters `api_key_candidates()`, which is the single definition of the order. "Auto-detect" (empty) stays the default so existing setups are unaffected.

### Reasoning / thinking effort across providers

One config vocabulary (`model.thinking.effort`: `none, minimal, low, medium, high, xhigh, max`) meets three providers with different dialects, so each path translates it separately:

- **OpenAI / generic** — `reasoning_effort` in `extra_body`, sent only when thinking is actually on. `none` is a valid value in the OpenAI spec (gpt-5.1+), but it is a 400 on older models and on strict OpenAI-compatible proxies, and the same config key feeds LM Studio/vLLM/OpenCode, so the field is **omitted** instead of sent as `"none"` — omitting is the only value every endpoint accepts.
- **Ollama** — a *top-level* `think` param (`bool | "low" | "medium" | "high"`), **not** an entry in `options`. `options` is serialized through the Pydantic `Options` model, which has no such field, so putting `reasoning_effort` there is silently dropped and Ollama thinking never turns on.
- **Anthropic** — no ladder at all: an absolute `budget_tokens`, and `max_tokens` is *derived* from it (`max(8192, budget + 4096)`) because the API requires `budget_tokens < max_tokens`. A fixed `max_tokens` makes every budget above it a guaranteed 400.

`_clamp_effort(effort, allowed)` snaps a *known* config level to the nearest one a provider accepts, up or down — asking for more than a provider has caps at its ceiling rather than silently meaning "off". An unknown value (a vendor extension like `ultra`) passes through untouched so an endpoint that does understand it still receives it.

`reasoning_content` must be echoed back on the assistant message for DeepSeek-style thinking mode (`Core._tool_calls`), and stripped on the Anthropic path (`_adapt_messages_for_anthropic`), whose SDK rejects unknown message keys and whose thinking blocks are signed and cannot be reconstructed from text.

### Colour (`agent/palette.py`)

`agent/palette.py` is the single source of truth for every colour the agent draws. Four frontends consume it, and **each is generated from the same fields**, so they cannot disagree:

| consumer | method | why it is not shared directly |
|---|---|---|
| Rich output | `rich_theme_dict()` | style *tags* (`[weak]`, `[patch-add]`, ...), not values |
| REPL prompt | `prompt_toolkit_dict()` | prompt_toolkit needs literal hex, not Rich style names |
| Sticky footer | `footer_sgr()` | the footer writes raw escape sequences and never goes through Rich |
| Textual TUI | `textual_theme()` | Textual resolves `$primary` etc. from its own theme object |

The rules the palette encodes, each of which fixed a real defect:

- **One identity hue, stepped by weight, not by hue.** `blue` / `blue_mid` / `blue_dim` are what distinguish `user` from `accent` from `tool`. They used to be three near-identical blues (`deep_sky_blue3` for both `user` and `accent`), so your prompt and the model name were visually equivalent for no reason.
- **Orange means warning and nothing else.** `agent` used to be `orange3` while `warn` was `orange_red1`, so a warning read as though the agent itself were complaining. `agent` is now dim blue: it is chrome, not a speaker.
- **`patch-add`/`patch-remove` are the same green/red as `ok`/`error`.** They were `green` and `red` against `chartreuse4` and `bold red` — two greens and two reds doing one job each.
- **`path` is foreground-only.** It was `#999999 on #252525`, which drew a dark rectangle around every path and every `$ command` in a light terminal.
- **Yellow is keycaps only.** Keycaps are the one place a hue that means nothing semantic is worth the contrast.
- **Reasoning text has its own role, `thinking`.** It was `[weak]` in the REPL and `[dim]` in the TUI — two different greys for one thing, and the brighter of the two competed with the answer. `thinking` is its own field, `think` (darker than both `weak` and `faint`), in both frontends, since the model musing about its plan is not part of the answer.

Roles (`agent`, `user`, `tool`, `path`, ...) are **properties**, not fields, so they cannot be assigned independently and drift apart.

Two failure modes this structure prevents, both of which were live:

1. **An undefined style tag renders silently.** `[server]` is used in `agent.py`, `tui.py` and `server.py` but was never in the theme; Rich drops unknown tags, so those lines rendered unstyled. `tests/test_palette.py` pins that every referenced tag resolves.
2. **A hard-coded colour outside the palette.** A repo-wide test rejects any stray `steel_blue3` / `grey39` / `deep_sky_blue3` in `agent/`. It found four real ones in `tui.py`, `output.py` and `commands.py` that the theme did not control.

Colours are truecolor hex throughout; Rich, Textual and the footer all downsample to the terminal's actual palette when it is not a truecolor terminal, so a hex value means the same thing in all three.

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

`@cmd` returns `None`, so another command cannot call a decorated handler by name. Put the logic in a plain function and let each `@cmd` entry point delegate to it (as `/model` and `/url` do via `_configure_endpoint`).

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
├── test_core_turn.py    # run_turn: TurnResult, cancellation state, tool result callbacks, malformed tool arguments
├── test_memory.py       # Memory, ChatMemory persistence and trimming
├── test_skills.py       # SkillLoader frontmatter parsing, load_all
├── test_files.py        # read_file (full read, max_lines, offset window), patch_file (text/line modes), search_content (literal/regex)
├── test_jobs.py         # spawn / poll_job / kill_job: incremental output, process-group kill, reaping
├── test_palette.py      # The shared colour source: tag coverage, frontend agreement, no stray colours
├── test_ipc.py          # IPC protocol: message factories, serialization, loopback transport
├── test_emitter.py      # TurnEmitter: core callbacks -> events, cancel state
├── test_keys.py         # Terminal key protocol: modified Enter, newline bindings, readline prompt
├── test_output_prompts.py # Adapter prompts: readline loaded, kitty flag dropped, prompt passed to input()
├── test_readline_prompt.py # A Rich prompt survives a readline redraw (real pty + cursor replay)
├── test_tools.py        # Tool registration, discovery, execution
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
