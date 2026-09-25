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


def _load_vectorstore(session_dir):
    """Lazily initialize the vector store. Returns None if dependencies are missing."""
    try:
        from agent.vectorstore import VectorStore
        return VectorStore(session_dir)
    except Exception:
        return None

SESSIONS_DIR = xdg_data_home() / "wisemonkey" / "sessions"
SESSION_METADATA_FILE = ".session-metadata"

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
        return int(cfg.get("agent.chat_history_tool_result_max_chars",
                           DEFAULT_TOOL_RESULT_MAX_CHARS))
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

    def __new__(cls, max_chat_history=80000, session_dir=None, session='default'):
        global _instance
        if _instance is None:
            _instance = super().__new__(cls)
        return _instance

    def __init__(self, max_chat_history=80000, session_dir=None, session='default'):
        # Only initialize on first creation
        if hasattr(self, "_initialized"):
            return

        # Session name
        self.session = session

        # Memory directory:
        # - if `session_dir` is present, use that
        # - else, use SESSIONS_DIR/session
        self.session_dir = Path(session_dir) if session_dir else  SESSIONS_DIR / session
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
            md = {
                "created": now.isoformat(),
                "accessed": now.isoformat()
            }
            self._write_metadata(md)

        elif metadata_exists:
            # Restored session
            # Read 'created' and 'accessed'
            md = self._read_metadata()
            if md and 'created' in md:
                self.session_created = datetime.datetime.fromisoformat(md['created'])
            else:
                self.session_created = None
            if md and 'accessed' in md:
                self.session_accessed = datetime.datetime.fromisoformat(md['accessed'])
            else:
                self.session_accessed = None

            # Update 'accessed'
            if self.session_created:
                # Write 'created' and 'accessed'
                md['accessed'] = now.isoformat()
                self._write_metadata(md)
        else:
            raise RuntimeError("Invalid session state: new session but metadata already exists?")
            

        # User profile
        self._user_profile_path = self.session_dir / "user_profile.json"
        # Persistent notes
        self._notes_path = self.session_dir / "notes.json"
        # Chat history
        self._chat_history = ChatMemory(self.session_dir, max_tokens=max_chat_history)
        # Document vector store (lazy, optional)
        self.vectorstore = None

        # Load from disk into memory buffers
        self._user_profile = self._load_json(self._user_profile_path, {})
        self._notes = self._load_json(self._notes_path, [])
        self._initialized = True

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

    def get_chat_history_formatted(self,
                           num_exchanges: int = 0,
                           timestamps: bool = False,
                           collapse_tools: bool = False,
                           width: int = 0):
        """
        Returns the chat history as a formatted string.

        Parameters:
        - num_exchanges: int    - The number of most recent exchanges to add (0 for all)
        - timestamps: bool      - Add timestamps to the output
        - collapse_tools: bool  - Collapse tool calls
        - width: int            - Maximum width of each entry's content (0 to not truncate)
        """
        return self._chat_history.get_formatted(num_exchanges, timestamps, collapse_tools, width)

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


class ChatMemory:
    """Rolling chat memory that stores recent exchanges.
    
    Maintains a rolling window of recent user input/assistant output pairs,
    limited by token count from configuration. Automatically trimmed
    when the window is exceeded. Persisted to disk.
    """
    
    def __init__(self, session_dir, max_tokens=80000):
        """Initialize chat memory.

        Args:
            max_tokens: Maximum total tokens to keep in memory (default: 80000)
        """
        self._exchanges = []  # list of {"role": "user"|"assistant"|"summary", "content": str}
        self.total_tokens = 0
        self.max_tokens = max_tokens
        
        # Set up persistence
        self._chat_path = Path(session_dir) / "chat_history.json"
        
        # Load from disk
        self._load()

    def set_exchanges(self, content):
        self._exchanges = content
        self._recount_tokens()

    def _recount_tokens(self) -> None:
        """Count exactly the rendered history text injected into the prompt."""
        formatted = self.get_formatted(0, timestamps=False, collapse_tools=False, width=0)
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
        
        # Compact if exceeded
        if self.total_tokens > self.max_tokens:
            from agent.commands import registry
            _, _, _, _, _ = registry.run_command(core, "/session-chat-compact")
        
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

    def get_unformatted(self):
        return self._exchanges
    
    
    def get_formatted(self,
                      num_exchanges: int,
                      timestamps: bool = False,
                      collapse_tools: bool = False,
                      width: int = 0):
        """Return chat history formatted for the system prompt.

        Handles all exchange roles: user, assistant, summary, tool_call,
        and tool_result. Tool results are truncated (see
        ``_tool_result_limit``) unless full tool results are enabled.

        Returns:
            Formatted string of recent exchanges, or None if empty
        """
        if not self._exchanges:
            return None

        tool_limit = _tool_result_limit()

        lines = []
        # Show most recent exchanges (num_exchanges == 0 -> all)
        history = self._exchanges[-num_exchanges:] if num_exchanges > 0 else self._exchanges

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
            if (i + 1 < len(history)
                    and self._tool_entries_match(turn, history[i + 1])):
                result = history[i + 1]
                name = turn.get("name", "unknown")
                args = escape(str(turn.get("arguments", "")))
                result_content = escape(result.get("content") or "")
                if tool_limit > 0 and len(result_content) > tool_limit:
                    result_content = result_content[:tool_limit] + " …[truncated]"
                lines.append(
                    f"## Tool: {name}\n{t}\n"
                    f"Args: {args}\n"
                    f"Result: {result_content}\n\n"
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
                if width > 0:
                    content = shorten(content, width=width)
                lines.append(f"## {role.capitalize()}:\n{t}\n{content}\n\n")
            i += 1

        return "\n".join(lines)
