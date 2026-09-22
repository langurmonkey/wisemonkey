"""`@file` references: inline file contents into the user prompt.

Writing ``@agent/agent.py`` (or the backtick-quoted ``@`agent/agent.py` ``)
in a prompt attaches the file's content to the message sent to the model:

* Files smaller than ``agent.at_file_max_chars`` are inlined as fenced
  blocks appended to the user message.
* Larger files inject only a stub (path, size, first lines) so the model
  knows the file exists and can read it itself with ``read_file``.
* Directories inject a shallow listing.

The user-visible text in the REPL/TUI keeps the bare ``@path`` reference;
expansion happens in :func:`expand_at_references` right before the turn is
run, so chat history stays compact (only the model sees the inlined block).
"""

from __future__ import annotations

import os
from pathlib import Path

# Matches @path or @`path` tokens. The path part must not contain
# whitespace or backticks; it may start with ~, ., / or be a bare word
# containing a path separator or an extension (to avoid eating @handles
# like "@langur" in prose).
import re

_AT_RE = re.compile(r"@(`[^`\s]+`|[^\s`]+)")

# Extensions/basename patterns that make a bare word look like a file.
_FILELIKE_RE = re.compile(
    r"\.(py|md|txt|json|yaml|yml|toml|sh|cfg|ini|js|ts|c|h|cpp|hpp|rs|go|"
    r"java|html|css|xml|csv|log|lock|env|ini|tex|rst)$"
)


def _strip_at(token: str) -> str:
    """Remove any leading ``@`` characters (handles accidental ``@@path``)."""
    return token.lstrip("@")


def _is_filelike(token: str) -> bool:
    """Heuristic: does the bare token look like a file path?"""
    token = _strip_at(token)
    if _FILELIKE_RE.search(token):
        return True
    # Paths with separators or home-relative are file-like.
    if os.sep in token or token.startswith("~"):
        return True
    # Bare word: file-like only if it exists on disk (file or directory).
    return Path(os.path.expanduser(token)).exists()


def find_at_references(text: str) -> list[str]:
    """Return the path tokens of all ``@...`` references in ``text``."""
    out = []
    for m in _AT_RE.finditer(text):
        token = m.group(1)
        if token.startswith("`") and token.endswith("`"):
            token = token[1:-1]
        if _is_filelike(token):
            out.append(token)
    return out


def _render_file(path: Path, max_chars: int) -> str:
    """Render one file reference: full content or a stub for big files."""
    try:
        raw = path.read_text(errors="replace")
    except OSError as e:
        return f"@{path}: **unreadable** ({e})"

    size = len(raw)
    if size <= max_chars:
        lang = path.suffix.lstrip(".") or "text"
        return f"`{path}`:\n```{lang}\n{raw}\n```"

    # Stub for large files: first chunk + hint to use read_file.
    head = "\n".join(raw.splitlines()[:30])
    return (
        f"`{path}` (**{size} chars — too large to inline; use `read_file`**):\n"
        f"```{path.suffix.lstrip('.') or 'text'}\n{head}\n… [truncated, {size} chars total]\n```"
    )


def _render_dir(path: Path) -> str:
    """Render a directory reference as a shallow listing."""
    try:
        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError as e:
        return f"@{path}: **unreadable** ({e})"
    lines = [f"`{path}` (directory):"]
    for e in entries[:100]:
        marker = "/" if e.is_dir() else ""
        lines.append(f"- {e.name}{marker}")
    if len(entries) > 100:
        lines.append(f"… and {len(entries) - 100} more")
    return "\n".join(lines)


def expand_at_references(text: str, max_chars: int = 8000) -> str:
    """Expand ``@path`` references in ``text`` into an augmented prompt.

    The original text is preserved verbatim at the top; attachments are
    appended in a clearly delimited section. Non-existent paths are left
    untouched (they may be handles, emails, etc.).
    """
    matches = _AT_RE.findall(text)
    if not matches:
        return text

    blocks = []
    seen = set()
    for token in matches:
        inner = _strip_at(token)  # group(1) excludes one @; tolerate more
        if inner.startswith("`") and inner.endswith("`") and len(inner) >= 2:
            inner = inner[1:-1]
        expanded = os.path.expanduser(inner)
        path = Path(expanded)
        if not path.exists():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        if path.is_dir():
            blocks.append(_render_dir(path))
        else:
            blocks.append(_render_file(path, max_chars))

    if not blocks:
        return text

    attachment = "\n\n---\n\n**Attached context (from @-references):**\n\n" + "\n\n".join(blocks)
    return text + attachment