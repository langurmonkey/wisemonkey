"""The agent loop of Wisemonkey.

The agent orchestrates the 'user-assistant' turns and delegates the actual turn
handling to the core.
"""

import threading
import time
from collections.abc import Callable
from typing import Any, cast

from pubsub import pub
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt

from agent.at_files import expand_at_references
from agent.commands import registry
from agent.completion import SmartPathCompleter
from agent.console import err, info, newline, ok, print
from agent.core import Core
from agent.emitter import TurnEmitter
from agent.footer import _ACCENT, _DIM, _LABEL, _RESET
from agent.ipc import (
    ContentPayload,
    Event,
    InjectedPayload,
    InjectWhen,
    LoopbackTransport,
    ProtocolError,
    ReasoningPayload,
    StageKind,
    StagePayload,
    ToolCallPayload,
    ToolResultPayload,
    TransportClosed,
    loopback_pair,
    payload_as,
)
from agent.mdstream import MarkdownStreamRenderer
from agent.history import PromptHistory
from agent.keys import (
    disable_kitty_keyboard,
    enable_kitty_keyboard,
    install_enter_bindings,
)
from agent.output import RichOutputAdapter, set_output
from agent.startup import startup_info
from agent.utils import add_command, collapse_none_dicts, format_tool_args, term_width

# Try to import prompt_toolkit for rich input; fall back to plain input.
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.clipboard import InMemoryClipboard
    from prompt_toolkit.completion import NestedCompleter
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.lexers import PygmentsLexer
    from prompt_toolkit.styles import Style
    from pygments.lexers.markup import MarkdownLexer

    _HAS_PROMPT_TOOLKIT = True

except ImportError:
    err("Could not initialize prompt toolkit")
    _HAS_PROMPT_TOOLKIT = False


# Constants
txt_goodbye = "\n[accent-bold]Goodbye![/accent-bold]"
PASTE_THRESHOLD = 1500


class Agent:
    def __init__(self, config_path=None, session="default"):
        self.session = session
        self.spinner_prompt = None
        self.spinner_thinking = None
        # While the sticky footer is armed, stage spinners are drawn in the
        # footer input row rather than as a Rich `Live` display (see _spinner).
        self._footer_spinner: str | None = None
        # Whether the spinner currently on the footer row is one we started.
        # `_spinner()` returns None while the footer owns the row, so
        # `spinner_prompt`/`spinner_thinking` cannot be used to tell whether
        # there is something to clear.
        self._footer_spinner_mine = False
        self._last_ctrl_c_time = (
            0  # Timestamp of last Control+C for double-tap detection
        )
        self.output = RichOutputAdapter()
        set_output(self.output)
        self._turn_in_progress = False
        self._md_stream = None
        self._prompt_reported = False
        # Whether we pushed the kitty keyboard protocol in _create_prompt_session
        # and still owe the terminal a pop. See agent/keys.py.
        self._kitty_enabled = False
        # Number of LLM rounds already requested this turn. `Core.run_turn`
        # loops back to the model once per tool batch, and each pass fires
        # STAGE.START again, so this is what tells "processing the user's
        # prompt" (round 1) apart from "waiting for the model after tools".
        self._prompt_rounds = 0

        # Client/server phase 3: if a daemon server is already running for
        # this session, attach to it and run as a thin remote client. If not,
        # fall back to the local mode (Core in-process, loopback emitter).
        self.remote = None
        try:
            from agent.client import ServerConnection

            conn = ServerConnection.connect(
                session, spawn=False, config_path=config_path
            )
            conn.attach(client="cli")
            self.remote: ServerConnection | None = conn
        except Exception:
            self.remote = None

        if self.remote is not None:
            # Remote mode: the server owns the real Core. The client still
            # needs a config and a session-scoped Memory for purely local UI
            # concerns (prompt session, completions, paste files, history).
            from types import SimpleNamespace

            from agent.config import Config as _Config
            from agent.memory import Memory as _Memory

            cfg = _Config()
            cfg.load(config_path)
            self.core = cast(
                Core, SimpleNamespace(config=cfg, memory=_Memory(session=session))
            )
            self.emitter = cast(TurnEmitter, None)
            self._event_peer = cast(LoopbackTransport, None)
            self._event_thread = None
            self._events_running = False
            return

        self.core = Core(config_path, session)
        # Event emitter (phase 1 of the client/server split): core callbacks
        # are translated into IPC events on a loopback transport and dispatched
        # by an event pump running on a background thread.
        transport, peer = loopback_pair()
        self.emitter = TurnEmitter(transport=transport)
        self._event_peer = peer
        self._event_thread = None
        self._events_running = False
        pub.subscribe(self._create_prompt_session, "prompt-update")

    # ── event handling (IPC phase 1) ────────────────────────────────────────

    def start_events(self) -> None:
        """Start the event pump that renders IPC events from the emitter."""
        if self._event_thread is not None:
            return
        self._events_running = True
        self._event_thread = threading.Thread(
            target=self._event_loop, name="wisemonkey-events", daemon=True
        )
        self._event_thread.start()

    def stop_events(self, timeout: float = 2.0) -> None:
        """Stop the event pump and close the loopback transport."""
        self._events_running = False
        transport = self.emitter.transport
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass
        if self._event_thread is not None:
            self._event_thread.join(timeout=timeout)
            self._event_thread = None

    def _event_loop(self) -> None:
        """Drain events from the loopback peer until stopped.

        This runs on a dedicated thread; every UI mutation is funnelled back
        onto the main thread via ``call_soon_threadsafe``-style helpers.
        """
        peer = self._event_peer
        while self._events_running:
            try:
                message = peer.recv(timeout=0.1)
            except Exception:
                break
            if message is None:
                continue
            try:
                self._handle_event(message)
            except Exception:
                # A rendering error must never take down the event pump.
                pass

    def _handle_event(self, message) -> None:
        """Dispatch a single protocol event to the UI."""
        if message.name == Event.STAGE:
            payload = payload_as(StagePayload, message)
            if payload.name == "prompt":
                if payload.stage == StageKind.START:
                    self._prompt_start()
                elif payload.stage == StageKind.STOP:
                    self._prompt_stop()

        elif message.name == Event.REASONING:
            payload = payload_as(ReasoningPayload, message)
            if payload.stage == StageKind.START:
                self._reasoning_start(payload.visible)
            elif payload.stage == StageKind.PROCESS:
                if payload.text and payload.visible:
                    print(f"[weak]{escape(payload.text)}[/]", end="")
            elif payload.stage == StageKind.STOP:
                self._reasoning_stop()

        elif message.name == Event.CONTENT:
            self.content_callback(payload_as(ContentPayload, message).text)

        elif message.name == Event.TOOL_CALL:
            payload = payload_as(ToolCallPayload, message)
            self.tool_callback(payload.name, payload.arguments)

        elif message.name == Event.TOOL_RESULT:
            payload = payload_as(ToolResultPayload, message)
            self.tool_result_callback(
                payload.id,
                payload.name,
                payload.content,
                payload.is_error,
                payload.duration,
            )

        elif message.name == Event.INJECTED:
            payload = payload_as(InjectedPayload, message)
            self._handle_injected(payload.text, payload.when)
        elif message.name == Event.CANCELLED:
            print("[warn]⏹  Turn cancelled by user  ⏹[/warn]")

        # TURN_START / TURN_END / STATUS are consumed by the turn runner
        # (run_interactive), which reads the TurnResult directly in phase 1.

    # ── stage handlers (extracted from the old callbacks) ───────────────────

    def _spinner(self, text: str):
        """Start a status spinner on the active output adapter, if supported.

        A Rich ``Status`` is a ``Live`` display, and ``Live`` wraps
        ``sys.stdout`` in a ``FileProxy`` for its whole lifetime. While the
        footer is armed that proxy is in the way: the footer's
        cursor-addressing escapes would be *decoded and printed as literal
        text* by the proxy instead of being interpreted by the terminal.

        So while the sticky footer owns the bottom rows, the label is handed
        to :meth:`OutputAdapter.footer_spinner` instead, and the footer
        animates it on its own transient row. No ``Live`` is started at all.
        """
        if self.output.footer_active():
            self._footer_spinner = text
            self._footer_spinner_mine = True
            self.output.footer_spinner(text)
            return None
        console = getattr(self.output, "_console", None)
        return console.status(text) if console else None

    def _spinner_clear(self) -> None:
        """Clear the footer-borne spinner, if we are the one showing it.

        A real Rich spinner is left alone: those are stopped by their own
        ``spinner_*`` handle, not here. Spinners belonging to a later stage
        (a tool, say) are also left alone, since only their owner may clear
        them.
        """
        if self._footer_spinner_mine:
            self._footer_spinner = None
            self._footer_spinner_mine = False
            self.output.footer_spinner("")
            self._footer_refresh()

    def _prompt_start(self) -> None:
        # The prompt stage fires once per LLM round: once for the user's
        # prompt, then again after every tool batch. Only the first is the
        # user's prompt; the rest are the model working on tool results,
        # which needs no spinner — and repeating "Processing prompt..."
        # there was actively misleading.
        first = self._prompt_rounds == 0
        self._prompt_rounds += 1
        if not first:
            return
        self.spinner_prompt = self._spinner("⏳ Processing prompt...")
        if self.spinner_prompt:
            self.spinner_prompt.start()

    def _prompt_stop(self) -> None:
        if self.spinner_prompt:
            self.spinner_prompt.stop()
            self.spinner_prompt = None
        self._spinner_clear()
        # The prompt stage fires once per LLM round (each tool round
        # re-enters the LLM), so only report completion on the first
        # round of a turn.
        if not self._prompt_reported:
            self._prompt_reported = True
            ok("⏳ Prompt processed")

    def _reasoning_start(self, visible: bool) -> None:
        if visible:
            info("💡 Thinking...\n")
        else:
            self.spinner_thinking = self._spinner("💡 Thinking...")
            if self.spinner_thinking:
                self.spinner_thinking.start()

    def _reasoning_stop(self) -> None:
        if self.spinner_thinking:
            self.spinner_thinking.stop()
            self.spinner_thinking = None
        self._spinner_clear()
        ok("💡 Done thinking\n")

    def content_callback(self, content: str = ""):
        """Called when new chunks arrive in streaming mode."""
        if self._md_stream is not None:
            # Live streaming markdown rendering: complete lines are rendered
            # incrementally with theme styles; the trailing partial line is
            # held back until complete.
            self._md_stream.feed(content)
            return
        print(escape(content), end="")

    def _md_stream_start(self) -> None:
        """Start live markdown rendering if enabled by config."""
        # New turn: reset per-turn UI state.
        self._prompt_reported = False
        self._prompt_rounds = 0
        if self.core is None or not self.core.config.get("agent.markdown_stream", True):
            return
        console = getattr(self.output, "_console", None)
        if console is not None:
            self._md_stream = MarkdownStreamRenderer(console)

    def _md_stream_stop(self) -> None:
        """Finish live markdown rendering."""
        if self._md_stream is not None:
            self._md_stream.flush()
            self._md_stream = None

    def _flush_md_stream(self) -> None:
        """Emit any held-back partial line before non-content output.

        Narration between tool calls usually has no trailing newline, so
        the streaming renderer holds it back. Tool lines must not appear
        above it — flush first so interleaving stays in order.
        """
        if self._md_stream is not None:
            self._md_stream.flush()

    def tool_callback(self, tool_name: str, tool_args):
        self._flush_md_stream()
        newline()
        args_str = format_tool_args(tool_args)
        if args_str:
            info(
                f"🛠️ [weak]Activating tool:[/weak]  [tool]{tool_name}[/tool]  [weak]({escape(args_str)})[/weak]"
            )
        else:
            info(f"🛠️ [weak]Activating tool:[/weak]  [tool]{tool_name}[/tool]")

    def tool_result_callback(self, tool_id, tool_name, content, is_error, duration):
        """Called after a tool finishes (event: TOOL_RESULT).

        A successful tool that finished fast prints nothing: its activation
        line is already on screen, and a "finished in 0.0s" line under every
        single call turns a fifteen-tool turn into fifteen lines of noise.
        Errors are always reported, and a tool slow enough to have made you
        wait gets its duration so it can be spotted as the culprit.
        """
        if is_error:
            self.output.err(
                f"Tool {tool_name} failed ({duration:.1f}s): {content}")
            return
        if duration >= self._tool_slow_threshold():
            self.output.ok(f"Tool {tool_name} finished in {duration:.1f}s")

    def _tool_slow_threshold(self) -> float:
        """Seconds after which a successful tool earns a result line.

        Set ``agent.tool_slow_threshold`` to 0 to always print, or to a large
        value to never print anything but errors.
        """
        value = self.core.config.get("agent.tool_slow_threshold", 1.0)
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            return 1.0
        return max(0.0, threshold)

    def _statusline(self, total_tokens, ntools, total_gen_time):
        if self.remote is not None:
            length, max, rate = self.remote.memory_stats()
        else:
            length, max, rate = self.core.memory.get_chat_stats()
        title = f"  {total_gen_time:.1f}s   |   {total_tokens} tokens   |   {ntools} tools   |   Mem: {length}/{max} tks ({rate:.2f}%)  "
        self.output.rule(title=title, style="status")
        # Refresh the sticky footer with the final memory stats.
        self._footer_refresh()

    def _footer_status(self) -> str:
        """Build the sticky footer status line (model, session, memory).

        The line is always the same during a turn: the steering input has
        its own row, below this one (see :meth:`_footer_input`).
        """
        model = self.core.config.get("model.name")
        session = self.core.memory.session
        if self.remote is not None:
            used, limit, rate = self.remote.memory_stats()
        else:
            used, limit, rate = self.core.memory.get_chat_stats()
        return (
            f" {_ACCENT}⇒{_RESET} {_LABEL}{model}{_RESET}  {_DIM}|{_RESET}  "
            f"{_LABEL}session:{_RESET} {_ACCENT}{session}{_RESET}  {_DIM}|{_RESET}  "
            f"{_LABEL}Mem:{_RESET} {used}/{limit} tks ({rate:.1f}%)"
        )

    def _footer_input(self) -> str:
        """Build the steering input row shown under the status line.

        Priority: what is being typed, then the oldest queued follow-up.
        Empty when there is nothing to show, which is the common case -- the
        footer then falls back to its own spinner animation on this same row
        (see :meth:`Footer.set_spinner`), so a partially typed line is never
        overwritten by an animation frame.
        """
        if not self._turn_in_progress:
            return ""
        line = self.output.steer_line()
        queued = self.output.steer_pending()
        if line:
            # Show the tail of a long line, so the caret stays visible.
            room = max(20, term_width() - 8)
            if len(line) > room:
                line = "…" + line[-(room - 1) :]
            return f" {_ACCENT}⤷{_RESET} {_LABEL}{line}{_RESET}"
        if queued:
            # A submitted line is handed to the running turn at its next tool
            # seam; if the turn ends before another tool call, the main loop
            # runs it as an ordinary next prompt.
            return (
                f" {_ACCENT}↳{_RESET} {_LABEL}queued: {queued}{_RESET}"
                f"  {_DIM}(next tool result){_RESET}"
            )
        # No typed text and nothing queued: leave the row for the footer,
        # which draws the spinner frame there if a stage is active.
        return ""

    # ── mid-turn steering ───────────────────────────────────────────────────

    def _steer_submit(self, line: str) -> None:
        """A line was submitted into the footer (reader thread).

        The line itself already sits in the adapter's FIFO. In remote mode the
        running turn lives in the daemon, so the line is forwarded to it here
        and the local copy is kept until the server confirms delivery with an
        ``injected`` event (see :meth:`_handle_injected`). In local mode the
        turn's injection seam drains the FIFO directly, so there is nothing to
        do but redraw the footer.
        """
        if self.remote is not None and self._turn_in_progress:
            try:
                self.remote.inject(line, when=InjectWhen.AFTER_TOOL)
            except (TransportClosed, ProtocolError) as e:
                # The daemon is gone or refused the line; it stays queued
                # locally and runs as an ordinary next prompt.
                self.output.steer_ack(line)
                self.output.err(f"Could not steer this turn: {e}")
        self._footer_refresh()

    def _steer_refresh(self) -> None:
        """Redraw the footer when the steering line changes (reader thread)."""
        if self._turn_in_progress:
            self._footer_refresh()

    def _steer_start(self) -> None:
        """Arm mid-turn steering for the duration of a turn."""
        if not self.core.config.get("agent.steer_midturn", True):
            return
        self.output.steer_start(self._steer_submit, self._steer_refresh)

    def _handle_injected(self, text: str, when: str = "") -> None:
        """The server delivered a submitted line mid-turn: confirm it.

        The local copy is dropped here and nowhere else, so a line is echoed
        on stdout exactly once, at the moment the model actually receives it.
        """
        self.output.steer_ack(text)
        self._announce_steered(text, when=when or "mid-turn")
        self._footer_refresh()

    def _steer_inject(self) -> list[str]:
        """Injection seam for a local turn: hand queued lines to the model.

        Called by ``Core.run_turn`` at every tool-loop seam, so a line
        submitted while a tool is running reaches the model on the *next*
        request instead of waiting for the whole turn to end.

        The lines are announced as they leave the queue, so the
        `↳ queued: <text>` row in the footer has its visible counterpart at
        the moment of delivery rather than minutes later.
        """
        lines = self.output.steer_take_all()
        for line in lines:
            self._announce_steered(line, when="mid-turn")
        return lines

    def _take_pending_prompt(self) -> str | None:
        """Take a prompt a slash command asked to run (`/retry`, `/edit`).

        The frontend owns the turn loop, so a command cannot start a turn
        itself: it stores the text on the core and returns, and the loop picks
        it up here. Reading it clears it, so a request can never be run twice
        nor leak into the next prompt.
        """
        prompt = getattr(self.core, "pending_prompt", None)
        if not prompt:
            return None
        self.core.pending_prompt = None
        return str(prompt)

    def _steer_drain(self) -> str | None:
        """Pop the next queued steering line, if the user sent one.

        The line is announced on stdout as it leaves the queue, so the
        `↳ queued: <text>` row in the footer has a visible counterpart once
        the line is actually fed to the turn below it. Any lines still queued
        behind it are picked up by the next iteration of the main loop.
        """
        line = self.output.steer_take()
        if line is None:
            return None
        return line

    def _announce_steered(self, line: str, when: str = "mid-turn") -> None:
        """Echo a steering line as it enters the turn path.

        Called the moment the line leaves the queue, so the `↳ queued:
        <text>` row in the footer always has a visible counterpart on
        stdout once the line is really fed to the turn.
        """
        remaining = self.output.steer_pending()
        note = ", 1 more queued" if remaining else ""
        self.output.newline()
        self.output.print(
            f"[user]⤷ {escape(line)}[/user]"
            f"{'  ' + _DIM if note else ''}"
            f"[steered mid-turn{': ' + note if note else ''}]"
            f"{_RESET if note else ''}"
        )

    def _footer_refresh(self) -> None:
        """Redraw the footer (status line + steering input row)."""
        self.output.footer_update(self._footer_status(), self._footer_input())

    def _cancel_all_spinners(self):
        self._footer_spinner = None
        self._footer_spinner_mine = False
        if self.spinner_prompt:
            self.spinner_prompt.stop()
            self.spinner_prompt = None
        if self.spinner_thinking:
            self.spinner_thinking.stop()
            self.spinner_thinking = None

    def _create_prompt_session(self):
        # Key bindings:
        kb = KeyBindings()

        # Ask the terminal to stop sending a bare 0x0D for Shift/Ctrl+Enter.
        # Terminals that do not understand it ignore the sequence, so this is
        # safe to attempt unconditionally; the atexit hook registered inside
        # pops it again if we are killed.
        self._kitty_enabled = enable_kitty_keyboard()

        # Enter submits; a modified Enter inserts a newline. Installed from
        # agent/keys.py so the rule lives in one place.
        install_enter_bindings(kb)

        @kb.add("c-c")
        def _(event):
            """Control+C: first press clears input, second press (within 1s) quits."""
            buffer = event.current_buffer
            now = time.time()

            if buffer.text:
                # First press with text: clear the buffer
                buffer.reset()
                self._last_ctrl_c_time = int(now)
            else:
                # Buffer is empty: check for double-tap
                if now - self._last_ctrl_c_time < 1.0:
                    # Double Control+C: quit
                    raise KeyboardInterrupt
                else:
                    # Single press on empty: just reset and record time
                    buffer.reset()
                    self._last_ctrl_c_time = int(now)

        def _handle_paste(text):
            """Intercept large pastes and save them to a file."""
            if len(text) > PASTE_THRESHOLD:
                file_path = self.core.memory.create_pasted_file(text)
                return f"*Pasted file: {file_path}*\n"
            return text

        # Bracketed paste: catches middle-click, Shift+Insert, and
        # Control+Shift+V
        @kb.add(Keys.BracketedPaste)
        def _(event):
            """Bracketed paste: intercept large pastes from any paste method."""
            text = event.data
            event.current_buffer.insert_text(_handle_paste(text))

        # Create prompt session now
        style = Style.from_dict(
            {
                "prompt": "#0087d7",
                "frame.border": "#0087d7",
                "bottom-toolbar": "#ffffff bg:#262626 noreverse",
                "kbd": "#ffd787 bold",
                "model": "#005faf",
                "weak": "#393939",
                "unsafe-warn": "bold bg:#8b0000 #ffffff",
            }
        )

        # Vi mode
        vi_mode = self.core.config.get("agent.vi_mode", False)

        # Build slash command dict
        commands = [cmd.name for cmd in registry.list_commands()]
        commands_dict = {}
        for command in commands:
            add_command(commands_dict, command)
        commands_dict = collapse_none_dicts(commands_dict)
        slash_completer = NestedCompleter.from_nested_dict(commands_dict)

        # Smart path completer: completes the last token, handles ~/ expansion
        # and paths embedded mid-sentence (see agent/completion.py).
        path_completer = SmartPathCompleter()

        # Hybrid completer: slash commands when the line starts with '/',
        # smart path completion otherwise.
        from prompt_toolkit.completion import Completer

        class HybridCompleter(Completer):
            def __init__(self, slash_comp, path_comp):
                self.slash_completer = slash_comp
                self.path_completer = path_comp

            def get_completions(self, document, complete_event):
                text = document.text_before_cursor

                # Slash commands: always check first when line starts with '/'
                if text.startswith("/"):
                    cmds = list(
                        self.slash_completer.get_completions(document, complete_event)
                    )
                    if cmds:
                        return cmds
                    # If no slash completions match, fall through to path check

                return self.path_completer.get_completions(document, complete_event)

        completer = HybridCompleter(slash_completer, path_completer)

        # History path
        history_path = self.core.memory.session_dir / "history.txt"
        history_path.parent.mkdir(parents=True, exist_ok=True)

        # Toolbar
        def prompt_toolbar():
            return HTML(
                "  <kbd>Alt</kbd>/<kbd>Shift</kbd>+<kbd>↵</kbd>: new line"
                " | <kbd>Ctrl</kbd>+<kbd>C</kbd>: clear / double-tap to quit"
            )

        model = self.core.config.get("model.name")
        unsafe = self.core.config.get("agent.unsafe", False)
        unsafe_warn = "  <unsafe-warn> ⚠ UNSAFE </unsafe-warn>" if unsafe else ""
        self._session = PromptSession(
            style=style,
            message=HTML(
                f"⩥ You ⩤   <weak>model:</weak> <model>{model}</model>  <weak>session:</weak> <model>{self.core.memory.session}</model>{unsafe_warn}\n❯ "
            ),
            history=PromptHistory(history_path),
            show_frame=True,
            multiline=True,
            key_bindings=kb,
            vi_mode=vi_mode,
            clipboard=InMemoryClipboard(),
            enable_open_in_editor=vi_mode,
            complete_while_typing=True,
            complete_in_thread=True,
            completer=completer,
            auto_suggest=AutoSuggestFromHistory(),
            lexer=PygmentsLexer(MarkdownLexer),
            bottom_toolbar=prompt_toolbar,
        )

    def run_interactive(self):
        """Run the agent in interactive mode."""

        if self.remote is not None:
            self._run_interactive_remote()
            return

        startup_info(self.core, self.output)

        if _HAS_PROMPT_TOOLKIT:
            self._create_prompt_session()

            def get_input():
                return str(self._session.prompt()).strip()
        else:
            # Rich
            def get_input():
                return Prompt.ask(
                    prompt="[user]⩥ [bold]You[/bold] ⩤[/user]\n❯",
                    console=self.output._console,
                )

        # Phase 1 (client/server): the turn is driven through the event
        # emitter. The core's callbacks are wired to the emitter, whose events
        # are rendered by the event pump (self._event_loop). Cancellation is
        # state: Ctrl+C calls emitter.cancel(), which poll() surfaces to the
        # core between streamed chunks. The TurnResult reports the outcome.
        self.emitter.reset()
        self.start_events()

        # Main loop
        while True:
            # A line typed into the footer during the previous turn (mid-turn
            # steering) is consumed before the prompt is shown again, so it
            # goes through exactly the same code path as a typed prompt.
            steered = self._steer_drain()
            if steered:
                self._announce_steered(steered)
            else:
                try:
                    user_input = get_input()
                except (EOFError, KeyboardInterrupt):
                    self.output.print(txt_goodbye)
                    break

                if not user_input:
                    continue

                user_input = str(user_input)

            # No cancel-on-empty-prompt branch here: Ctrl+C during a turn is
            # caught inside the turn handler above and sets
            # _turn_in_progress = False before the prompt is shown again, so
            # by the time we reach the prompt no turn is running. Cancelling
            # from here would be dead code.

            # Process user-invoked shell commands (`!` prefix)
            if user_input.startswith("!"):
                from agent.shellcmd import append_to_memory, run_shell_command

                command = user_input[1:].strip()
                if command:
                    result = run_shell_command(command, self.output)
                    append_to_memory(self.core, command, result)
                    self.output.newline()
                else:
                    self.output.err("Empty shell command")
                continue

            # Conflate ? with help command
            if user_input == "?":
                user_input = "/help"

            # Process slash commands
            if user_input.startswith("/"):
                tokens = user_input.split()
                command, params = registry.lookup(tokens)

                if command:
                    no_errors, msg, content, md, should_exit = registry.execute(
                        self.core, command, params, self.output
                    )

                    if should_exit:
                        self.output.print(txt_goodbye)
                        break

                    if no_errors:
                        # Content in rich or Markdown format
                        if content or md:
                            if params:
                                param_list = " ".join(params)
                            else:
                                param_list = ""

                            if content:
                                cont = content
                            elif md:
                                cont = Markdown(md)
                            self.output.print_rich(
                                Panel(
                                    cont,
                                    border_style="output-frame",
                                    title=f"{command.name} {param_list}",
                                    subtitle=f"{command.name} {param_list}",
                                    highlight=True,
                                )
                            )

                        # Short status message
                        if msg:
                            self.output.ok(msg)
                        self.output.newline()

                        # `/retry` and `/edit` cannot run a turn themselves
                        # (the frontend owns the turn loop); they leave the
                        # prompt here and it runs next, through exactly the
                        # same path as a typed prompt.
                        retry_prompt = self._take_pending_prompt()
                        if not retry_prompt:
                            continue
                        user_input = retry_prompt

                    else:
                        # Error
                        if msg:
                            self.output.err(f"{msg}")
                        continue

                else:
                    self.output.err(f"Command not found: {user_input}")
                    continue

            else:
                self.output.newline()
                self.output.rule(style="agent")
                self.output.print(
                    f"[agent]⩥ [bold]Wisemonkey[/bold] ⩤ [/agent]  [accent]⇒ {self.core.config.get('model.name')}[/accent]"
                )
                self.output.print("  [kbd]Ctrl[/kbd]+[kbd]C[/kbd]: Cancel turn\n")
                try:
                    self._turn_in_progress = True
                    # Reset the emitter so a previously cancelled turn does
                    # not poison this one (poll() must start out False).
                    self.emitter.reset()
                    self._md_stream_start()
                    # Arm the sticky footer for the duration of the turn.
                    self.output.footer_start()
                    self._footer_refresh()
                    # Capture mid-turn keystrokes so the user can steer.
                    self._steer_start()
                    # Expand @file references into attached context (model
                    # sees the content; the typed text stays as-is on screen).
                    max_at = self.core.config.get("agent.at_file_max_chars", 8000)
                    prompt = (
                        expand_at_references(user_input, max_at)
                        if max_at > 0
                        else user_input
                    )
                    result = self.core.run_turn(
                        prompt,
                        self.emitter.prompt,
                        self.emitter.reasoning,
                        self.emitter.content,
                        self.emitter.tool_call,
                        self.emitter.cancelled,
                        self.emitter.error,
                        tool_result_callback=self.emitter.tool_result,
                        inject_callback=self._steer_inject,
                        poll=self.emitter.poll,
                    )
                    response = result.response
                    total_tokens = result.total_tokens
                    ntools = result.n_tools
                    total_gen_time = result.gen_time
                    # Flush the streaming renderer before the statusline so
                    # the trailing partial line doesn't land after it.
                    self._md_stream_stop()
                    self.output.newline()

                    if result.cancelled:
                        continue  # skip status line, go straight back to prompt

                    if result.max_turns_reached:
                        self.output.err(
                            f"Turn stopped: maximum number of turns reached "
                            f"(agent.max_turns = "
                            f"{self.core.config.get('agent.max_turns', 50)}). "
                            "The model kept requesting tools without a final answer."
                        )

                    self._statusline(total_tokens, ntools, total_gen_time)
                    self.output.newline()

                    if (
                        self.core.config.get("agent.markdown", False)
                        and self._md_stream is None
                    ):
                        # Print markdown (skipped when the streaming renderer
                        # already rendered the response live).
                        md = self.core.memory.get_chat_history_unformatted()[-1][
                            "content"
                        ]
                        md = Panel(
                            Markdown(md),
                            border_style="output-frame",
                            title="Markdown",
                            subtitle="Markdown",
                            highlight=True,
                        )
                        self.output.print_rich(md)
                        self._statusline(total_tokens, ntools, total_gen_time)
                except Exception as e:
                    self._cancel_all_spinners()
                    self.emitter.cancel("error")
                    self.output.err(f"Error sending prompt: {e}")
                    # The turn's partial conversation has already been persisted
                    # by core.run_turn(), so we just continue to the next prompt.
                    self.output.print(
                        "  [dim]Partial response was saved to chat history.[/dim]"
                    )
                finally:
                    self._md_stream_stop()
                    self._turn_in_progress = False
                    self._cancel_all_spinners()
                    # Disarm mid-turn steering and the sticky footer before
                    # the prompt returns (queued lines survive the restart).
                    self.output.steer_stop()
                    self.output.footer_spinner("")
                    self.output.footer_stop()
                    self._footer_spinner = None
                    self._footer_spinner_mine = False

        # Persist memory, stop the event pump, and shut down core on exit
        self._pop_kitty_keyboard()
        if self.core:
            self.stop_events()
            self.core.save_memory()
            self.core.shutdown()
        if self.remote is not None:
            self.remote.close()

    # ── remote (daemon) mode ────────────────────────────────────────────────

    def run_once(self, prompt: str) -> str:
        """Run a single turn with the given prompt, then return the response.

        Non-interactive counterpart of run_interactive(): no prompt_toolkit
        session, no prompt loop, no mid-turn steering or footer (a one-shot
        run has no user at the keyboard). Works in both local and remote
        (daemon) mode, and cleans up events/connections before returning.
        """
        user_input = str(prompt).strip()
        if not user_input:
            return ""

        if self.remote is not None:
            response = self._run_once_remote(user_input)
        else:
            response = self._run_once_local(user_input)

        # Persist memory, stop the event pump, and shut down core.
        if self.core:
            self.stop_events()
            self.core.save_memory()
            self.core.shutdown()
        if self.remote is not None:
            self.remote.close()

        return response

    def _run_once_local(self, user_input: str) -> str:
        """One turn against the in-process Core."""
        # No startup_info(): a one-shot run is meant for scripting, so the
        # banner, update check and session report would just be noise.
        self.emitter.reset()
        self.start_events()

        self.output.newline()
        self.output.rule(style="agent")
        self.output.print(
            f"[agent]⩥ [bold]Wisemonkey[/bold] ⩤ [/agent]  "
            f"[accent]⇒ {self.core.config.get('model.name')}[/accent]"
        )
        self.output.print("  [kbd]Ctrl[/kbd]+[kbd]C[/kbd]: Cancel turn\n")

        response = ""
        try:
            self._turn_in_progress = True
            self.emitter.reset()
            self._md_stream_start()
            # Expand @file references into attached context, exactly as the
            # interactive loop does.
            max_at = self.core.config.get("agent.at_file_max_chars", 8000)
            prompt = (
                expand_at_references(user_input, max_at) if max_at > 0 else user_input
            )
            result = self.core.run_turn(
                prompt,
                self.emitter.prompt,
                self.emitter.reasoning,
                self.emitter.content,
                self.emitter.tool_call,
                self.emitter.cancelled,
                self.emitter.error,
                tool_result_callback=self.emitter.tool_result,
            )
            response = result.response
            self._md_stream_stop()
            self.output.newline()

            if result.max_turns_reached:
                self.output.err(
                    f"Turn stopped: maximum number of turns reached "
                    f"(agent.max_turns = "
                    f"{self.core.config.get('agent.max_turns', 50)}). "
                    "The model kept requesting tools without a final answer."
                )

            if not result.cancelled:
                self._statusline(result.total_tokens, result.n_tools, result.gen_time)
                self.output.newline()

            if (
                self.core.config.get("agent.markdown", False)
                and self._md_stream is None
            ):
                md = Panel(
                    Markdown(response),
                    border_style="output-frame",
                    title="Markdown",
                    subtitle="Markdown",
                    highlight=True,
                )
                self.output.print_rich(md)
        except Exception as e:
            self._cancel_all_spinners()
            self.emitter.cancel("error")
            self.output.err(f"Error sending prompt: {e}")
        finally:
            self._md_stream_stop()
            self._turn_in_progress = False
            self._cancel_all_spinners()

        return response

    def _run_once_remote(self, user_input: str) -> str:
        """One turn against a remote daemon server."""
        from agent.client import ServerConnection
        from agent.ipc import HandshakePayload

        remote = cast(ServerConnection, self.remote)
        handshake = cast(HandshakePayload, remote.handshake)
        self.output.print(
            f"[server]One-shot run against daemon server for session "
            f"[accent-bold]{handshake.session}[/accent-bold] "
            f"(pid {handshake.server_pid}, model [accent]{handshake.model}[/accent])[/server]"
        )
        self.output.newline()

        self._remote_stop = False
        remote.on_disconnect = self._on_server_lost
        rpc_thread = threading.Thread(
            target=self._remote_serve_requests, name="wisemonkey-rpc", daemon=True
        )
        rpc_thread.start()

        response = ""
        try:
            self.output.newline()
            self.output.rule(style="agent")
            self.output.print(
                f"[agent]⩥ [bold]Wisemonkey[/bold] ⩤ [/agent]  "
                f"[accent]⇒ {handshake.model}[/accent]"
            )
            self.output.print("  [kbd]Ctrl[/kbd]+[kbd]C[/kbd]: Cancel turn\n")
            self._turn_in_progress = True
            self._md_stream_start()
            end = remote.prompt(text=user_input, on_event=self._handle_event)
            response = end.response
            self._md_stream_stop()
            self.output.newline()
            if not end.cancelled:
                if end.max_turns_reached:
                    self.output.err(
                        f"Turn stopped: maximum number of turns reached "
                        f"(agent.max_turns = "
                        f"{self.core.config.get('agent.max_turns', 50)}). "
                        "The model kept requesting tools without a final answer."
                    )
                self._statusline(end.total_tokens, end.n_tools, end.gen_time)
                self.output.newline()
        except Exception as e:
            self._cancel_all_spinners()
            self.output.err(f"Error sending prompt: {e}")
        finally:
            self._md_stream_stop()
            self._turn_in_progress = False
            self._cancel_all_spinners()
            self._remote_stop = True

        return response

    # ── remote (daemon) mode ─────────────────────────────────────────

    def _run_interactive_remote(self):
        """Run the REPL against a remote daemon server."""
        from agent.client import ServerConnection
        from agent.ipc import HandshakePayload

        remote = cast(ServerConnection, self.remote)
        handshake = cast(HandshakePayload, remote.handshake)
        # Startup banner: reuse the standard one, but with server-side
        # memory stats (the client's stub Memory has no chat history).
        startup_info(self.core, self.output)
        remote_stats = remote.memory_stats()
        self.output.print(
            f"[server]Attached to daemon server for session "
            f"[accent-bold]{handshake.session}[/accent-bold] "
            f"(pid {handshake.server_pid}, model [accent]{handshake.model}[/accent])[/server]"
        )
        self.output.newline()

        if _HAS_PROMPT_TOOLKIT:
            self._create_prompt_session()

            def get_input():
                return str(self._session.prompt()).strip()
        else:

            def get_input():
                return Prompt.ask(
                    prompt="[user]⨯ [bold]You[/bold] ⨯[/user]\n❯",
                    console=self.output._console,
                )

        # Background thread answering ServerRequest RPCs (confirmations,
        # questions, subprocess runs) relayed by the server.
        self._remote_stop = False
        remote.on_disconnect = self._on_server_lost
        rpc_thread = threading.Thread(
            target=self._remote_serve_requests, name="wisemonkey-rpc", daemon=True
        )
        rpc_thread.start()

        try:
            while True:
                # Drain a mid-turn steering line before prompting again.
                steered = self._steer_drain()
                if steered:
                    self._announce_steered(steered)
                else:
                    try:
                        user_input = get_input()
                    except (EOFError, KeyboardInterrupt):
                        self.output.print(txt_goodbye)
                        break

                    if not user_input:
                        continue

                if user_input.startswith("!"):
                    from agent.shellcmd import run_shell_command

                    command = user_input[1:].strip()
                    if command:
                        # Run locally with full terminal control, then record
                        # the exchange in the server's chat history.
                        result = run_shell_command(command, self.output)
                        try:
                            remote.record("user", result.get("_chat_content", ""))
                        except Exception as e:
                            self.output.err(
                                f"Could not record command in server history: {e}"
                            )
                        self.output.newline()
                    else:
                        self.output.err("Empty shell command")
                    continue

                if user_input == "?":
                    user_input = "/help"

                if user_input.startswith("/"):
                    try:
                        result = remote.command(user_input)
                    except TransportClosed:
                        self.output.err("Connection to server lost.")
                        self.output.print(txt_goodbye)
                        break
                    if result.should_exit:
                        self.output.print(txt_goodbye)
                        break
                    if result.ok:
                        if result.command == "/config-edit":
                            # The daemon reloads its authoritative config; refresh
                            # this client's UI-only config shadow as well.
                            self.core.config.reload()
                            if _HAS_PROMPT_TOOLKIT:
                                self._create_prompt_session()
                        if result.content or result.markdown:
                            cont = (
                                result.content
                                if result.content
                                else Markdown(result.markdown)
                            )
                            self.output.print_rich(
                                Panel(
                                    cont,
                                    border_style="output-frame",
                                    title=f"{result.command}",
                                    subtitle=f"{result.command}",
                                    highlight=True,
                                )
                            )
                        if result.msg:
                            self.output.ok(result.msg)
                        self.output.newline()

                        # `/retry` and `/edit` run in the daemon but need the
                        # client to drive the turn, so the server hands the
                        # prompt back here and it runs next.
                        if result.pending_prompt:
                            user_input = result.pending_prompt
                            continue
                    else:
                        if result.msg:
                            self.output.err(result.msg)
                    continue

                self.output.newline()
                self.output.rule(style="agent")
                self.output.print(
                    f"[agent]⨯ [bold]Wisemonkey[/bold] ⨯ [/agent]  "
                    f"[accent]⇒ {handshake.model}[/accent]"
                )
                self.output.print("  [kbd]Ctrl[/kbd]+[kbd]C[/kbd]: Cancel turn\n")
                self._turn_in_progress = True
                self._md_stream_start()
                # Arm the sticky footer for the duration of the turn.
                self.output.footer_start()
                self._footer_refresh()
                # Capture mid-turn keystrokes so the user can steer.
                self._steer_start()
                try:
                    end = remote.prompt(text=user_input, on_event=self._handle_event)
                    # Flush the streaming renderer before the statusline so
                    # the trailing partial line doesn't land after it.
                    self._md_stream_stop()
                    self.output.newline()
                    if not end.cancelled:
                        if end.max_turns_reached:
                            self.output.err(
                                f"Turn stopped: maximum number of turns reached "
                                f"(agent.max_turns = "
                                f"{self.core.config.get('agent.max_turns', 50)}). "
                                "The model kept requesting tools without a final answer."
                            )
                        self._statusline(end.total_tokens, end.n_tools, end.gen_time)
                        self.output.newline()

                        # Markdown summary (same as local mode), using the
                        # response carried in the turn-end payload. Skipped
                        # when the streaming renderer already rendered it live.
                        if (
                            self.core.config.get("agent.markdown", False)
                            and self._md_stream is None
                        ):
                            md = Panel(
                                Markdown(end.response),
                                border_style="output-frame",
                                title="Markdown",
                                subtitle="Markdown",
                                highlight=True,
                            )
                            self.output.print_rich(md)
                            self._statusline(
                                end.total_tokens, end.n_tools, end.gen_time
                            )
                            self.output.newline()
                except Exception as e:
                    self._cancel_all_spinners()
                    self.output.err(f"Error sending prompt: {e}")
                finally:
                    self._md_stream_stop()
                    self._turn_in_progress = False
                    self._cancel_all_spinners()
                    # Disarm mid-turn steering and the sticky footer before
                    # the prompt returns (queued lines survive the restart).
                    self.output.steer_stop()
                    self.output.footer_spinner("")
                    self.output.footer_stop()
                    self._footer_spinner = None
                    self._footer_spinner_mine = False
        finally:
            self._remote_stop = True
            remote.close()
            self._pop_kitty_keyboard()

    def _pop_kitty_keyboard(self) -> None:
        """Restore the terminal's default key mode before we hand it back.

        A terminal left in disambiguate mode keeps prefixing key codes with
        the ``CSI 27 ; mods ;`` wrapper, which would break whatever the user
        runs next -- in the common case, their shell. Idempotent, and called
        from both interactive loops plus ``run_once``.
        """
        if getattr(self, "_kitty_enabled", False):
            self._kitty_enabled = False
            disable_kitty_keyboard()

    def _on_server_lost(self):
        """Called on the reader thread when the daemon connection drops."""
        self._remote_stop = True
        self._cancel_all_spinners()
        self.output.err("Connection to server lost (server exited or was killed).")
        self._turn_in_progress = False

    def _remote_serve_requests(self):
        """Answer ServerRequest RPCs from the daemon until shutdown."""
        from agent.client import ServerConnection
        from agent.ipc import ServerRequest

        def handle_confirm(payload):
            return self.output.ask_confirm(payload.message, payload.default)

        def handle_ask_string(payload):
            return self.output.ask_string(payload.message, payload.default)

        def handle_ask_float(payload):
            return self.output.ask_float(payload.message, payload.default)

        def handle_ask_choice(payload):
            return self.output.ask_choice(
                payload.message,
                [(o[0], o[1]) for o in payload.options],
                payload.default or None,
            )

        def handle_run_subprocess(payload):
            # Client-local: run with full terminal control here.
            import subprocess

            return subprocess.run(payload.cmd, check=False)

        handlers: dict[ServerRequest, Callable[[Any], Any]] = {
            ServerRequest.CONFIRM: handle_confirm,
            ServerRequest.ASK_STRING: handle_ask_string,
            ServerRequest.ASK_FLOAT: handle_ask_float,
            ServerRequest.ASK_CHOICE: handle_ask_choice,
            ServerRequest.RUN_SUBPROCESS: handle_run_subprocess,
        }
        remote = cast(ServerConnection, self.remote)
        remote.serve_requests(handlers, should_stop=lambda: self._remote_stop)
