"""
Simple file-based session memory system.

Stores user profile and persistent notes as JSON.
Follows XDG Base Directory spec:
- Data: $XDG_DATA_HOME/wisemonkey/session/$SESSION_NAME

Design: memory is buffered in memory. Changes are persisted to disk
when save() is called. On init, state is loaded from disk.
"""

import json

from agent.tokens import count_tokens
import datetime
import os

from textwrap import shorten
from rich.markup import escape
from pathlib import Path
from xdg_base_dirs import xdg_data_home


# Roles that represent the human-readable conversation, as opposed to the
# machinery around it. A recap of "what did we discuss" wants these; the
# system prompt wants everything.
CONVERSATIONAL_ROLES = ("user", "assistant", "summary")

# Sentence-final punctuation, for the (continued) heuristic.
_SENTENCE_END = ".!?\"')]}`*_"


def _ends_mid_sentence(text: str) -> bool:
    """Whether *text* looks like it was cut off rather than finished.

    Used only to decide whether to append ``(continued)`` to the last entry of
    the startup recap, so it is deliberately cheap and conservative: anything
    that is not clearly finished counts as unfinished. A false positive costs
    eight characters; a false negative shows a half-sentence as if it were the
    whole answer.
    """
    stripped = (text or "").rstrip()
    if not stripped:
        return False
    return stripped[-1] not in _SENTENCE_END


def _load_vectorstore(session_dir):
    """Lazily initialize the vector store. Returns None if dependencies are missing."""
    try:
        from agent.vectorstore import VectorStore

        return VectorStore(session_dir)
    except Exception:
        return None


SESSIONS_DIR = xdg_data_home() / "wisemonkey" / "sessions"
SESSION_METADATA_FILE = ".session-metadata"


def is_valid_session_name(name: str) -> bool:
    """Whether *name* is usable as a session directory name.

    Session names become directory names under :data:`SESSIONS_DIR`, so
    anything that could escape that directory (a separator, ``..``, a NUL)
    is rejected rather than sanitised -- a surprising session name is better
    than a surprising directory.
    """
    if not name or name in (".", ".."):
        return False
    if any(c in name for c in ("/", "\\", "\0")):
        return False
    return name == Path(name).name


def session_info(session_dir: Path) -> dict:
    """Summarize one session directory without opening its memory.

    Reads only ``.session-metadata`` and counts the entries in
    ``chat_history.json``, so listing sessions never loads a profile or a
    vector store. A session created by hand (no metadata) still reports,
    using the directory mtime as a fallback for both timestamps.
    """
    name = Path(session_dir).name
    info = {
        "name": name,
        "dir": session_dir,
        "created": None,
        "accessed": None,
        "messages": 0,
    }

    metadata_file = session_dir / SESSION_METADATA_FILE
    raw = {}
    if metadata_file.is_file():
        try:
            for line in metadata_file.read_text(encoding="utf-8").splitlines():
                key, sep, value = line.partition(":")
                if sep:
                    raw[key.strip()] = value.strip()
        except OSError:
            raw = {}
    for key in ("created", "accessed"):
        try:
            info[key] = datetime.datetime.fromisoformat(raw[key])
        except (KeyError, TypeError, ValueError):
            info[key] = None

    if info["accessed"] is None:
        try:
            info["accessed"] = datetime.datetime.fromtimestamp(
                session_dir.stat().st_mtime
            ).astimezone()
        except OSError:
            info["accessed"] = None

    history = session_dir / "chat_history.json"
    if history.is_file():
        try:
            data = json.loads(history.read_text(encoding="utf-8"))
            info["messages"] = len(data.get("exchanges", []))
        except (OSError, json.JSONDecodeError, AttributeError):
            info["messages"] = 0

    return info


def list_sessions() -> list[dict]:
    """Return :func:`session_info` for every session, newest first.

    The current session sorts to the top, then everything else by last
    access, falling back to creation time and then to name so the order is
    always stable.
    """
    sessions = []
    if SESSIONS_DIR.is_dir():
        for entry in sorted(SESSIONS_DIR.iterdir()):
            if not entry.is_dir():
                continue
            sessions.append(session_info(entry))

    sessions.sort(
        key=lambda s: (
            s["accessed"] or s["created"] or datetime.datetime.min,
            s["name"],
        ),
        reverse=True,
    )
    return sessions

# Default maximum number of characters kept per tool result when formatting
# chat history for the system prompt. 0 means "no truncation".
DEFAULT_TOOL_RESULT_MAX_CHARS = 500


def _tool_result_limit() -> int:
    """Return the max chars to keep per tool result when formatting history.

    Returns 0 when the user has opted into full tool results via the
    ``agent.chat_history_full_tool_results`` config flag.
    """
    try:
        from agent.config import get_config

        cfg = get_config()
        if cfg.get("agent.chat_history_full_tool_results", False):
            return 0
        return int(
            cfg.get(
                "agent.chat_history_tool_result_max_chars",
                DEFAULT_TOOL_RESULT_MAX_CHARS,
            )
        )
    except Exception:
        return DEFAULT_TOOL_RESULT_MAX_CHARS


# Singleton instance
_instance = None


class Memory:
    """Persistent per-session memory with in-memory buffering.

    Singleton: all Memory() calls return the same instance, so in-memory
    state is shared between the agent and tool handlers.
    """

    metadata_file = Path()
    _user_profile_path = Path()
    _notes_path = Path()
    _notes = []

    def __new__(
        cls, max_chat_history=80000, session_dir=None, session="default", window_turns=0
    ):
        global _instance
        if _instance is None:
            _instance = super().__new__(cls)
        return _instance

    def __init__(
        self,
        max_chat_history=80000,
        session_dir=None,
        session="default",
        window_turns=0,
    ):
        # Only initialize on first creation
        if hasattr(self, "_initialized"):
            return

        self._bind(session, session_dir, max_chat_history, window_turns)
        self._initialized = True

    def _bind(self, session, session_dir, max_chat_history, window_turns):
        """Point this instance at *session* and load its state from disk.

        Called once from ``__init__`` and again from :meth:`switch_session`,
        so switching a session and starting one run exactly the same code.
        """
        # Session name
        self.session = session

        # Memory directory:
        # - if `session_dir` is present, use that
        # - else, use SESSIONS_DIR/session
        self.session_dir = Path(session_dir) if session_dir else SESSIONS_DIR / session
        self.session_is_new = not os.path.exists(self.session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        # Update session metadata file
        self.metadata_file = self.session_dir / SESSION_METADATA_FILE
        metadata_exists = os.path.exists(self.metadata_file)
        now = datetime.datetime.now()
        if self.session_is_new or not metadata_exists:
            # New session or file does not exist
            self.session_created = now
            self.session_accessed = now
            # Write 'created' and 'accessed'
            md = {"created": now.isoformat(), "accessed": now.isoformat()}
            self._write_metadata(md)

        elif metadata_exists:
            # Restored session
            # Read 'created' and 'accessed'
            md = self._read_metadata()
            if md and "created" in md:
                self.session_created = datetime.datetime.fromisoformat(
                    md["created"]
                ).astimezone()
            else:
                self.session_created = None
            if md and "accessed" in md:
                self.session_accessed = datetime.datetime.fromisoformat(
                    md["accessed"]
                ).astimezone()
            else:
                self.session_accessed = None

            # Update 'accessed'
            if self.session_created:
                # Write 'created' and 'accessed'
                md["accessed"] = now.isoformat()
                self._write_metadata(md)
        else:
            raise RuntimeError(
                "Invalid session state: new session but metadata already exists?"
            )

        # User profile
        self._user_profile_path = self.session_dir / "user_profile.json"
        # Persistent notes
        self._notes_path = self.session_dir / "notes.json"
        # Chat history
        self._chat_history = ChatMemory(
            self.session_dir, max_tokens=max_chat_history, window_turns=window_turns
        )
        # Document vector store (lazy, optional). Dropped rather than carried
        # over: it is scoped to a session directory, and the new session may
        # not have one yet. `_load_vectorstore` rebuilds it on next use.
        self.vectorstore = None

        # Load from disk into memory buffers
        self._user_profile = self._load_json(self._user_profile_path, {})
        self._notes = self._load_json(self._notes_path, [])

    def switch_session(
        self,
        session,
        session_dir=None,
        max_chat_history=None,
        window_turns=None,
    ) -> str:
        """Rebind this singleton to *session* and return the previous name.

        The old session's buffered state is flushed to disk first, so
        switching never silently discards unsaved notes or chat history.
        Everything else is replaced by :meth:`_bind`, which reloads the
        target's profile, notes, chat history and metadata; the vector store
        is dropped and rebuilt lazily on next use.

        This mutates the singleton in place. Callers that cached session state
        (the prompt message, the footer status line, the TUI status bar) must
        refresh, and callers that hold a session name of their own -- the
        remote client's, the daemon's -- are unaffected, because they never
        go through this path: the daemon owns its session.
        """
        if not is_valid_session_name(session):
            raise ValueError(f"invalid session name: {session!r}")

        previous = self.session
        if previous == session and session_dir is None:
            return previous

        if hasattr(self, "_initialized"):
            self.save()

        self._bind(
            session,
            session_dir,
            self._chat_history.max_tokens if max_chat_history is None else max_chat_history,
            self._chat_history.window_turns if window_turns is None else window_turns,
        )
        return previous

    def _read_metadata(self):
        """Read a .session-metadata file as a dict, or return empty dict."""
        metadata = {}
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if ":" in line:
                            key, _, value = line.partition(":")
                            metadata[key.strip()] = value.strip()
            except OSError:
                pass
        return metadata

    def _write_metadata(self, metadata):
        """Write a dict to a .session-metadata file in key: value format."""
        with open(self.metadata_file, "w") as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")

    def _load_json(self, path, default):
        """Load JSON from file, returning default if not found or invalid."""
        if path.exists():
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return default
        return default

    def save(self):
        """Persist in-memory state to disk.

        This is the authoritative write — all changes are buffered
        in memory and only written here.
        """
        with open(self._user_profile_path, "w") as f:
            json.dump(self._user_profile, f, indent=2)

        with open(self._notes_path, "w") as f:
            json.dump(self._notes, f, indent=2)

        self._chat_history.save()

    def get_user_profile(self):
        """Return the in-memory user profile."""
        return self._user_profile

    def set_user_profile(self, data):
        """Update the in-memory user profile. Call save() to persist."""
        if isinstance(data, dict):
            self._user_profile = {**self._user_profile, **data}
        else:
            self._user_profile = {**self._user_profile, **data}

    def get_notes(self):
        """Return the in-memory notes list."""
        return self._notes

    def add_note(self, content, category="general"):
        """Add a note to in-memory buffer. Call save() to persist."""
        note = {
            "category": category,
            "content": content,
            "id": len(self._notes) + 1,
        }
        self._notes.append(note)
        return note

    def get_user_profile_formatted(self, user_profile=True, notes=True):
        """Return all memory formatted for the system prompt."""
        lines = []

        if not user_profile and not notes:
            return ""

        if user_profile and self._user_profile:
            lines.append("## User Profile")
            for key, value in self._user_profile.items():
                lines.append(f"- {key}: {value}")

        if notes and self._notes:
            lines.append("\n## Persistent Notes")
            for note in self._notes:
                lines.append(f"- [{note['category']}] {note['content']}")

        return "\n".join(lines) if lines else None

    def reset_chat_memory(self, content):
        """
        Resets the chat memory with the given content.
        """
        self._chat_history.set_exchanges(content)
        self.save()

    def get_chat_history_unformatted(self):
        return self._chat_history.get_unformatted()

    def conversational_count(self, num_exchanges: int = 0) -> int:
        """How many turns a recap of *num_exchanges* would cover.

        Lets the caller put the real number in a heading without rendering the
        history twice or reaching into ``_chat_history``.
        """
        return self._chat_history.conversational_count(num_exchanges)

    def get_chat_history_formatted(
        self,
        num_exchanges: int = 0,
        timestamps: bool = False,
        collapse_tools: bool = False,
        width: int = 0,
        roles: tuple[str, ...] | None = None,
        assistant_width: int = 0,
        mark_incomplete: bool = False,
    ):
        """
        Returns the chat history as a formatted string.

        Parameters:
        - num_exchanges: int    - The number of most recent exchanges to add (0 for all)
        - timestamps: bool      - Add timestamps to the output
        - collapse_tools: bool  - Collapse tool calls
        - width: int            - Maximum width of each entry's content (0 to not truncate)
        - roles: tuple[str,...] - Keep only these roles, slicing after filtering
        - assistant_width: int  - Wider truncation for assistant entries (0 = use width)
        - mark_incomplete: bool - Mark the last assistant entry if it looks cut off
        """
        return self._chat_history.get_formatted(
            num_exchanges,
            timestamps,
            collapse_tools,
            width,
            roles=roles,
            assistant_width=assistant_width,
            mark_incomplete=mark_incomplete,
        )

    def add_chat_exchange(self, core, role, content, **extra):
        self._chat_history.add_exchange(core, role, content, **extra)

    def get_chat_stats(self):
        """
        Returns the current chat memory size in tokens, the maximum size,
        and the fill percentage
        """
        curr = self._chat_history.total_tokens
        max_tokens = self._chat_history.max_tokens
        fill_rate = float(curr) * 100.0 / float(max_tokens)

        return curr, max_tokens, fill_rate

    def create_pasted_file(self, content):
        """Save pasted content to a file in the session's pasted directory.

        Creates session_dir/pasted/paste_$TIMESTAMP.md with the given content.
        Returns the file path as a string.
        """
        pasted_dir = self.session_dir / "pasted"
        pasted_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        paste_file = pasted_dir / f"paste_{timestamp}.md"
        paste_file.write_text(content, encoding="utf-8")

        return str(paste_file)

    def clear_chat(self, n=5):
        """
        Clear the n last chat exchanges from the memory.

        Return: The number of exchanges actually cleared.
        """
        return self._chat_history._clear(n)

    def drop_last_exchanges(self, n=1):
        """Remove the *n* most recent exchanges. See ChatMemory.drop_last."""
        return self._chat_history.drop_last(n)

    def last_user_prompt(self):
        """Return the most recent user message, or "" if there is none."""
        return self._chat_history.last_user_prompt()


class ChatMemory:
    """Rolling chat memory that stores recent exchanges.

    Maintains a rolling window of recent user input/assistant output pairs,
    limited by token count from configuration and (optionally) by a
    turn-based rolling window (`window_turns`: keep only the last n
    exchanges, where an exchange starts at a user message). Automatically
    trimmed when a limit is exceeded. Persisted to disk.
    """

    def __init__(self, session_dir, max_tokens=80000, window_turns=0):
        """Initialize chat memory.

        Args:
            max_tokens: Maximum total tokens to keep in memory (default: 80000)
            window_turns: If > 0, keep only the last n exchanges
                (an exchange = user message and everything after it until
                the next user message). 0 disables the turn window.
        """
        self._exchanges = []  # list of {"role": "user"|"assistant"|"summary", "content": str}
        self.total_tokens = 0
        self.max_tokens = max_tokens
        self.window_turns = window_turns

        # Set up persistence
        self._chat_path = Path(session_dir) / "chat_history.json"

        # Load from disk
        self._load()
        # Reconcile with the turn window in case the setting changed
        # since the last session (destructive trim).
        self._trim_to_window()

    def set_exchanges(self, content):
        self._exchanges = content
        self._recount_tokens()

    def _recount_tokens(self) -> None:
        """Count exactly the rendered history text injected into the prompt."""
        formatted = self.get_formatted(
            0, timestamps=False, collapse_tools=False, width=0
        )
        self.total_tokens = count_tokens(formatted or "")

    @staticmethod
    def _tool_entries_match(call: dict, result: dict) -> bool:
        """Return whether adjacent tool-call/result records belong together."""
        if call.get("role") != "tool_call" or result.get("role") != "tool_result":
            return False
        call_id = call.get("tool_call_id")
        result_id = result.get("tool_call_id")
        if call_id is not None or result_id is not None:
            return call_id is not None and call_id == result_id
        return call.get("name", "unknown") == result.get("name", "unknown")

    def _load(self):
        """Load chat history from disk."""
        if self._chat_path.exists():
            try:
                with open(self._chat_path, "r") as f:
                    data = json.load(f)
                    self._exchanges = data.get("exchanges", [])
                    self._recount_tokens()
            except (json.JSONDecodeError, IOError):
                pass

    def save(self):
        """Persist in-memory state to disk."""
        self._chat_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._chat_path, "w") as f:
            json.dump({"exchanges": self._exchanges}, f, indent=2)

    def add_exchange(self, core, role, content, **extra):
        """
        Add a user input, assistant output, or tool step to memory.

        Parameters:
            role:str    - "user", "assistant", "summary", "tool_call", or
                          "tool_result"
            content:str - The text content
            **extra     - Optional structured fields merged into the exchange
                          (e.g. tool_calls for assistant messages, or name /
                          arguments / tool_call_id for tool steps)
        """
        if content is None:
            content = ""

        # Truncate tool results at storage time so persisted JSON stays
        # bounded. Rendered-history token accounting below uses the same
        # formatter that injects history into the prompt.
        if role == "tool_result":
            limit = _tool_result_limit()
            if limit > 0 and len(content) > limit:
                content = content[:limit]

        now_utc = datetime.datetime.now(datetime.UTC)
        # Add the new exchange
        entry = {
            "role": role,
            "utc": str(now_utc),
            "content": content,
        }
        entry.update(extra)
        self._exchanges.append(entry)
        # The formatter can compact adjacent tool call/result pairs, so count
        # the complete rendered history rather than summing entry estimates.
        self._recount_tokens()

        # Turn-based rolling window (destructive trim of old exchanges).
        self._trim_to_window()

        # Compact if exceeded
        if self.total_tokens > self.max_tokens:
            from agent.commands import registry

            _, _, _, _, _ = registry.run_command(core, "/history-compact")

        # Persist immediately
        self.save()

    def _trim(self):
        """Remove oldest exchanges until under the token limit."""

        self._recount_tokens()
        while self.total_tokens > self.max_tokens and self._exchanges:
            self._exchanges.pop(0)
            self._recount_tokens()

        # Save after trimming
        self.save()

    def _exchange_starts(self) -> list[int]:
        """Return the indices of entries that start a new exchange.

        An exchange starts at each user entry and includes every entry
        after it until the next user entry (assistant responses, tool
        calls and results).
        """
        starts = [i for i, e in enumerate(self._exchanges) if e.get("role") == "user"]
        return starts

    def _trim_to_window(self) -> int:
        """Destructively trim to the last `window_turns` exchanges.

        A no-op when `window_turns` is 0 (disabled). Returns the number of
        exchanges removed.
        """
        if self.window_turns <= 0 or not self._exchanges:
            return 0

        starts = self._exchange_starts()
        if len(starts) <= self.window_turns:
            return 0

        # Keep everything from the start of the (window_turns)-th from last
        # exchange onwards; also drop any leading non-user entries (e.g.
        # stray tool results) before that point.
        cutoff = starts[-self.window_turns]
        removed = cutoff
        self._exchanges = self._exchanges[cutoff:]
        self._recount_tokens()
        self.save()
        return removed

    def _clear(self, n=5):
        """
        Removes the n oldest exchanges from the chat memory.
        If n <= 0, all chat exchanges are cleared.

        Return: The number of exchanges actually cleared.
        """

        cleared = 0
        if n <= 0:
            n = len(self._exchanges)

        for _ in range(n):
            if self._exchanges:
                self._exchanges.pop(0)
                cleared += 1
            else:
                break

        self._recount_tokens()
        self.save()

        return cleared

    def drop_last(self, n=1) -> int:
        """Remove the *n* most recent exchanges (not *n* entries).

        An exchange starts at a user message and runs until the next user
        message, so dropping the tail of the history has to drop whole
        exchanges: leaving an orphaned tool result behind would put a result
        in the prompt with no matching call.

        ``n <= 0`` drops everything. Returns the number of entries removed.
        """
        if not self._exchanges:
            return 0
        if n <= 0:
            removed = len(self._exchanges)
            self._exchanges = []
            self._recount_tokens()
            self.save()
            return removed

        starts = self._exchange_starts()
        if len(starts) <= n:
            # The request reaches past the first exchange: there is nothing
            # coherent left to keep.
            removed = len(self._exchanges)
            self._exchanges = []
            self._recount_tokens()
            self.save()
            return removed

        cutoff = starts[-n]
        removed = len(self._exchanges) - cutoff
        self._exchanges = self._exchanges[:cutoff]
        self._recount_tokens()
        self.save()
        return removed

    def last_user_prompt(self) -> str:
        """Return the content of the most recent user message, or ""."""
        for entry in reversed(self._exchanges):
            if entry.get("role") == "user":
                return entry.get("content", "")
        return ""

    def get_unformatted(self):
        return self._exchanges

    def conversational_turns(
        self, roles: tuple[str, ...] = CONVERSATIONAL_ROLES, num_turns: int = 0
    ) -> list[dict]:
        """The last *num_turns* conversational turns, oldest first.

        A turn is a ``user`` message plus everything up to the next one, and
        each is reduced to three entries: the question, an optional ``summary``,
        and the *final* assistant message.

        Both reductions matter. Without the role filter a tool-heavy turn
        contributes nothing but tool output, so a recap can show no question at
        all. Without the "last assistant message only" rule a turn that ran
        several tools contributes one message per narration step, and a recap
        asked for one exchange shows two ``Assistant:`` blocks and no ``User:``
        -- the same machinery noise as the tool results, one level up.

        Entries before the first ``user`` message are dropped: there is no
        question to attach them to.
        """
        if "user" not in roles or "assistant" not in roles:
            # Grouping is defined by the user/assistant alternation; without
            # both, fall back to a plain filter (see conversational_entries).
            return []

        turns: list[dict] = []
        current: dict | None = None
        for entry in self._exchanges:
            role = entry.get("role")
            if role == "user":
                if current is not None:
                    turns.append(current)
                current = {"user": entry, "summary": None, "assistant": None}
                continue
            if current is None or role not in roles:
                continue
            # Narration emitted before a tool call is stored as its own
            # assistant message. Only the last one is the answer.
            current["summary" if role == "summary" else "assistant"] = entry
        if current is not None:
            turns.append(current)

        return turns[-num_turns:] if num_turns > 0 else turns

    @staticmethod
    def _flatten_turns(turns: list[dict]) -> list[dict]:
        """Turn dicts back to a flat entry list, in reading order."""
        flat: list[dict] = []
        for turn in turns:
            for key in ("user", "summary", "assistant"):
                if turn.get(key) is not None:
                    flat.append(turn[key])
        return flat

    def conversational_entries(self, num_exchanges: int = 0) -> list[dict]:
        """The entries of the last *num_exchanges* conversational turns.

        See :meth:`conversational_turns`: counting turns, not stored entries,
        is what makes ``num_exchanges`` mean what the name says.
        """
        return self._flatten_turns(self.conversational_turns(num_turns=num_exchanges))

    def conversational_count(self, num_exchanges: int = 0) -> int:
        """How many turns a recap of *num_exchanges* would cover."""
        return len(self.conversational_turns(num_turns=num_exchanges))

    def get_formatted(
        self,
        num_exchanges: int,
        timestamps: bool = False,
        collapse_tools: bool = False,
        width: int = 0,
        roles: tuple[str, ...] | None = None,
        assistant_width: int = 0,
        mark_incomplete: bool = False,
    ):
        """Return chat history formatted for the system prompt.

        Handles all exchange roles: user, assistant, summary, tool_call,
        and tool_result. Tool results are truncated (see
        ``_tool_result_limit``) unless full tool results are enabled.

        Parameters beyond the prompt's own use:

        - roles: keep only these roles, and slice *after* filtering. The
          default ``None`` keeps every role, which is what the system prompt
          needs -- tool results are part of the conversation the model
          continues. A human-facing recap is the opposite case.
        - assistant_width: a wider truncation for assistant entries, which are
          typically longer than the question that prompted them. 0 means "use
          ``width``".
        - mark_incomplete: append ``(continued)`` to the final assistant entry
          when it looks cut off mid-sentence.

        Returns:
            Formatted string of recent exchanges, or None if empty
        """
        if not self._exchanges:
            return None

        tool_limit = _tool_result_limit()

        lines = []
        # Show most recent exchanges (num_exchanges == 0 -> all)
        if roles is None:
            history = (
                self._exchanges[-num_exchanges:]
                if num_exchanges > 0
                else self._exchanges
            )
        elif "user" in roles and "assistant" in roles:
            # A human-facing recap: count *turns*, so `num_exchanges` is a
            # number of question/answer pairs. Slicing the filtered list
            # instead would let one turn's narration fill the whole panel --
            # two Assistant blocks and no User, which is what the recap showed
            # before this.
            history = self._flatten_turns(
                self.conversational_turns(roles=roles, num_turns=num_exchanges)
            )
        else:
            filtered = [e for e in self._exchanges if e.get("role") in roles]
            history = (
                filtered[-num_exchanges:] if num_exchanges > 0 else filtered
            )

        # The (continued) marker belongs on the *last assistant* entry, which
        # is not necessarily the last entry: a recap that ends with the user's
        # pending question must not mark the question, and must not skip the
        # marker because a user turn happens to follow the answer.
        last_assistant = max(
            (i for i, e in enumerate(history) if e.get("role") == "assistant"),
            default=-1,
        )

        def is_tool(turn) -> bool:
            return turn.get("role") in ("tool_call", "tool_result")

        i = 0
        while i < len(history):
            turn = history[i]
            role = turn.get("role", "")
            t = f"`({turn['utc']})`" if timestamps and "utc" in turn else ""

            # Collapse runs of consecutive tool exchanges into one line.
            if collapse_tools and is_tool(turn):
                start = i
                while i < len(history) and is_tool(history[i]):
                    i += 1
                count = i - start
                lines.append(f"## Tool calls: #{count}\n{t}\n\n")
                continue

            # Render adjacent matching tool call/result entries together.
            if i + 1 < len(history) and self._tool_entries_match(turn, history[i + 1]):
                result = history[i + 1]
                name = turn.get("name", "unknown")
                args = escape(str(turn.get("arguments", "")))
                result_content = escape(result.get("content") or "")
                if tool_limit > 0 and len(result_content) > tool_limit:
                    result_content = result_content[:tool_limit] + " …[truncated]"
                lines.append(
                    f"## Tool: {name}\n{t}\nArgs: {args}\nResult: {result_content}\n\n"
                )
                i += 2
                continue

            content = escape(turn.get("content") or "")
            if role == "tool_result":
                name = turn.get("name", "unknown")
                if tool_limit > 0 and len(content) > tool_limit:
                    content = content[:tool_limit] + " …[truncated]"
                lines.append(f"## Tool Result ({name}):\n{t}\n{content}\n\n")
            elif role == "tool_call":
                name = turn.get("name", "unknown")
                args = turn.get("arguments", "")
                lines.append(f"## Tool Call ({name}):\n{t}\n{escape(str(args))}\n\n")
            else:
                # An assistant answer is usually several times longer than the
                # question that prompted it, so `assistant_width` lets a recap
                # give it more room than `width` allows for a question.
                limit = width
                if role == "assistant" and assistant_width > 0:
                    limit = assistant_width
                truncated = False
                if limit > 0 and len(content) > limit:
                    content = shorten(content, width=limit)
                    truncated = True
                # Mark the tail when the last answer was cut off -- either by
                # this truncation or because the session ended mid-sentence.
                # A recap that ends in a full stop is honest about being
                # complete; one that does not needs saying so.
                if mark_incomplete and role == "assistant" and i == last_assistant:
                    if truncated or _ends_mid_sentence(content):
                        content += " (continued)"
                lines.append(f"## {role.capitalize()}:\n{t}\n{content}\n\n")
            i += 1

        return "\n".join(lines)
