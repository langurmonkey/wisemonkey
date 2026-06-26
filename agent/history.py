"""Per-session input history manager.

Stores submitted prompts in a structured text file at
``~/.local/share/wisemonkey/sessions/<session>/history.txt``.

Each block is a timestamp header (``#``) followed by one or more content
lines (``+``), separated by a blank line::

    # 2026-06-25 15:15:39.141343
    +Ok, now I see the drop down for the provider…

Designed for use with the TUI's alt+up/alt+down navigation.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

Timestamp = str  # ISO-8601-ish: "YYYY-MM-DD HH:MM:SS.ffffff"
Entry = tuple[Timestamp, str]  # (timestamp, content)


class History:
    """A per-session input-history store backed by a structured text file.

    The file is written immediately after every ``add()`` call so that
    history survives crashes.  Duplicate consecutive entries are skipped.
    """

    def __init__(self, session_dir: str | Path, max_entries: int = 1000) -> None:
        self._path = Path(session_dir) / "history.txt"
        self._entries: list[Entry] = []
        self._max_entries = max_entries
        self._index = 0  # 0 <= index <= len(entries); len = beyond newest
        self._draft: str = ""  # in-progress text, stashed when leaving the draft position
        self._load()

    def add(self, text: str) -> None:
        """Append *text* to the history (skipped if identical to last entry)."""
        if not text:
            return
        if self._entries and self._entries[-1][1] == text:
            self._index = len(self._entries)
            self._draft = ""
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        self._entries.append((ts, text))
        if len(self._entries) > self._max_entries:
            self._entries.pop(0)
        self._index = len(self._entries)
        self._draft = ""
        self._save()

    def up(self, current_text: str) -> str | None:
        """Move one step back into the past.

        Returns the entry at that position, or ``None`` if already at
        the oldest entry.
        """
        if self._index == len(self._entries):
            self._draft = current_text
        if self._index > 0:
            self._index -= 1
            return self._entries[self._index][1]
        return None

    def down(self) -> str | None:
        """Move one step forward toward the present.

        Returns the entry at that position, or ``None`` if already at
        the newest entry.  Once past the newest entry the cursor resets
        to the end-of-list position and ``None`` is returned.
        """
        if self._index >= len(self._entries):
            return None
        self._index += 1
        if self._index == len(self._entries):
            return self._draft
        return self._entries[self._index][1]

    def _load(self) -> None:
        """Parse the structured history file.

        Format per block::

            # YYYY-MM-DD HH:MM:SS.ffffff
            +line one
            +line two

            (next block)
        """
        self._entries = []
        if not self._path.exists():
            self._index = 0
            return

        raw = self._path.read_text()
        blocks = raw.strip().split("\n\n")

        for block in blocks:
            lines = block.splitlines()
            ts = ""
            content_parts: list[str] = []
            for ln in lines:
                if ln.startswith("# "):
                    ts = ln[2:].strip()
                elif ln.startswith("+"):
                    content_parts.append(ln[1:].strip())
            if content_parts and ts:
                self._entries.append((ts, "\n".join(content_parts)))

        self._index = len(self._entries)

    def _save(self) -> None:
        """Write entries in the structured format."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for ts, content in self._entries:
            lines.append(f"# {ts}")
            for content_line in content.split("\n"):
                lines.append(f"+{content_line}")
            lines.append("")  # blank line separator
        self._path.write_text("\n".join(lines) + "\n")
