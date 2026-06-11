"""
Simple file-based session memory system.

Stores user profile and persistent notes as JSON.
Follows XDG Base Directory spec:
- Data: $XDG_DATA_HOME/wisemonkey/session/$SESSION_NAME

Design: memory is buffered in memory. Changes are persisted to disk
when save() is called. On init, state is loaded from disk.
"""

import json
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
    _notes = {}

    def __new__(cls, max_chat_history=300000, session_dir=None, session='default'):
        global _instance
        if _instance is None:
            _instance = super().__new__(cls)
        return _instance

    def __init__(self, max_chat_history=300000, session_dir=None, session='default'):
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
        self._chat_history = ChatMemory(self.session_dir, max_chars=max_chat_history)
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

    def get_formatted(self, user_profile=True, notes=True):
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

    def get_chat_unformatted(self):
        return self._chat_history.get_unformatted()

    def get_chat_formatted(self,
                           num_exchanges: int = 0,
                           timestamps: bool = False,
                           width: int = 0):
        """
        Returns the chat history as a formatted string.

        Parameters:
        - num_exchanges: int  - The number of most recent exchanges to add (0 for all)
        - timestamps: bool    - Add timestamps to the output
        - width: int          - Maximum width of each entry's content (0 to not truncate)
        """
        return self._chat_history.get_formatted(num_exchanges, timestamps, width)

    def add_chat_exchange(self, core, role, content):
        self._chat_history.add_exchange(core, role, content)

    def get_chat_stats(self):
        """
        Returns the current chat memory length, the maximum length, and the
        fill percentage
        """
        curr_length = self._chat_history.total_chars
        max_chars = self._chat_history.max_chars
        fill_rate = float(curr_length) * 100.0 / float(max_chars)

        return curr_length, max_chars, fill_rate

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
    limited by character count from configuration. Automatically trimmed
    when the window is exceeded. Persisted to disk.
    """
    
    def __init__(self, session_dir, max_chars=300000):
        """Initialize chat memory.
        
        Args:
            max_chars: Maximum total characters to keep in memory (default: 320000)
            memory_dir: Directory to persist chat history (default: ~/.local/share/wisemonkey/memory/)
        """
        self._exchanges = []  # list of {"role": "user"|"assistant"|"summary", "content": str}
        self.total_chars = 0
        self.max_chars = max_chars
        
        # Set up persistence
        self._chat_path = Path(session_dir) / "chat_history.json"
        
        # Load from disk
        self._load()

    def set_exchanges(self, content):
        self._exchanges = content
        self.total_chars = sum(len(e["content"]) for e in self._exchanges)

    def _load(self):
        """Load chat history from disk."""
        if self._chat_path.exists():
            try:
                with open(self._chat_path, "r") as f:
                    data = json.load(f)
                    self._exchanges = data.get("exchanges", [])
                    self.total_chars = sum(len(e["content"]) for e in self._exchanges)
            except (json.JSONDecodeError, IOError):
                pass
    
    def save(self):
        """Persist in-memory state to disk."""
        self._chat_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._chat_path, "w") as f:
            json.dump({"exchanges": self._exchanges}, f, indent=2)
    
    def add_exchange(self, core, role, content):
        """
        Add a user input or assistant output to memory.
        
        Parameters:
            role:str    - "user" or "assistant"
            content:str - The text content
        """
        char_count = len(content)
        
        now_utc = datetime.datetime.now(datetime.UTC)
        # Add the new exchange
        self._exchanges.append({
            "role": role,
            "utc": str(now_utc),
            "content": content,
        })
        self.total_chars += char_count
        
        # Compact if exceeded
        if self.total_chars > self.max_chars:
            from agent.commands import registry
            ok, msg, _, _, _ = registry.run_command(core, "/session-compact")
        
        # Persist immediately
        self.save()
    
    def _trim(self):
        """Remove oldest exchanges until under the character limit."""

        while self.total_chars > self.max_chars and self._exchanges:
            oldest = self._exchanges.pop(0)
            self.total_chars -= len(oldest["content"])
        
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

        for i in range(n):
            if self._exchanges:
                out = self._exchanges.pop(0)
                self.total_chars -= len(out["content"])
                cleared += 1
            else:
                # We ran out of items
                break

        # Save after clearing
        self.save()

        return cleared

    def get_unformatted(self):
        return self._exchanges
    
    
    def get_formatted(self, num_exchanges: int, timestamps: bool, width: int):
        """Return chat history formatted for the system prompt.
        
        Returns:
            Formatted string of recent exchanges, or None if empty
        """
        if not self._exchanges:
            return None
        
        lines = []
        # Show most recent exchanges
        history = self._exchanges[-num_exchanges:]
        for turn in history:
            t = f"`({turn['utc']})`" if timestamps and 'utc' in turn else ""
            content = escape(turn['content'])
            if width > 0:
                content = shorten(content, width=width)

            lines.append(f"## {turn['role'].capitalize()}:\n{t}\n")
            lines.append(f"{content}\n\n")
        
        return "\n".join(lines)
