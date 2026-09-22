"""Smart file-path completion shared by the REPL and the TUI.

The stock ``prompt_toolkit.PathCompleter`` is close to what we need, but it
has shortcomings that make it awkward inside a chat prompt:

* It does not expand ``~`` unless ``expand_user=True`` is set, and even then
  the completion text mixes the expanded home path into the buffer.
* It cannot complete a path that is embedded in a larger sentence (e.g.
  ``"look at agent/agent.py"``), because it validates the whole document.
* It has no notion of bare-word completion (``ag`` → ``agent/``).

This module provides:

* :class:`SmartPathCompleter` — a prompt_toolkit ``Completer`` for the REPL
  that completes the *last token* of the input, whatever it is, handling
  ``~`` expansion and rewriting completions so the ``~`` prefix survives.
* :func:`complete_path` — a framework-agnostic helper returning
  ``(completions, common_prefix)`` for the last token of a text string, used
  by the TUI (Textual has no completer protocol) for Tab completion.
"""

from __future__ import annotations

import os
from pathlib import Path

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

# Minimum length for bare-word (no path separator) completion. Below this we
# don't try, to avoid noisy suggestions while typing short words mid-sentence.
_MIN_BARE_WORD_LEN = 2


def _last_token(text: str) -> tuple[str, int]:
    """Return ``(token, offset)`` of the last whitespace-delimited token.

    ``offset`` is the index in ``text`` where the token starts.
    """
    stripped = text.rstrip()
    if not stripped or text[-1] in " \t\n":
        return "", len(text)
    pos = max(stripped.rfind(" "), stripped.rfind("\t")) + 1
    return stripped[pos:], pos


def _split_backticks(token: str) -> tuple[str, str, str]:
    r"""Split a token into ``(inner, lead, trail)`` markdown-quote parts.

    ``\`agent/agent.py\``` → ``("agent/agent.py", "\`", "\`")``.
    A bare token → ``(token, "", "")``. An unclosed opening backtick is
    treated as a quote too (the user is mid-typing).
    """
    if token.startswith("`") and token.endswith("`") and len(token) >= 2:
        return token[1:-1], "`", "`"
    if token.startswith("`"):
        return token[1:], "`", ""
    return token, "", ""


def _path_candidates(directory: Path, pattern: str) -> list[Path]:
    """List entries in ``directory`` whose names start with ``pattern``."""
    try:
        entries = sorted(
            directory.iterdir(),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
    except (OSError, PermissionError):
        return []
    if pattern:
        entries = [e for e in entries if e.name.startswith(pattern)]
    return entries


def _completions_for_token(token: str) -> list[str]:
    """Compute full replacement strings for the given path token.

    Returns a list of complete tokens (what should replace ``token`` in the
    input), including the ``~`` prefix when the user typed it.
    """
    if not token:
        return []

    expand_home = token.startswith("~")
    if expand_home:
        expanded = os.path.expanduser(token)
    else:
        expanded = token

    # Split into directory part and filename prefix.
    if expanded.endswith("/"):
        dirname, prefix = expanded, ""
    else:
        dirname, prefix = os.path.split(expanded)

    base = Path(dirname) if dirname else Path(".")
    entries = _path_candidates(base, prefix)

    results = []
    for entry in entries:
        full = str(entry)
        # Append a separator for directories so the user can keep drilling.
        if entry.is_dir():
            full += os.sep
        if expand_home:
            home = os.path.expanduser("~")
            if full == home:
                full = "~"
            elif full.startswith(home + os.sep):
                full = "~" + full[len(home):]
        results.append(full)
    return results


def complete_path(text: str) -> tuple[list[str], str]:
    r"""Complete the last token of ``text``.

    The token may be wrapped in markdown backticks (``\`agent/agent.py\```):
    the backticks are stripped for path lookup but preserved in the
    completions so they survive in the buffer.

    Returns ``(completions, common_prefix)`` where ``completions`` are full
    replacement tokens and ``common_prefix`` is the longest prefix shared by
    all of them (empty string if none). The caller replaces the last token
    with the chosen completion.
    """
    token, _offset = _last_token(text)
    # An @-prefix marks a file reference (@agent/agent.py); keep it in the
    # completions but strip it for path lookup.
    at = ""
    if token.startswith("@"):
        at, token = "@", token[1:]
    inner, lead, trail = _split_backticks(token)
    if not inner or (os.sep not in inner and inner[0] not in "~." and len(inner) < _MIN_BARE_WORD_LEN):
        return [], ""

    completions = _completions_for_token(inner)
    if not completions:
        return [], ""

    # Re-attach the @ prefix and markdown quotes.
    completions = [f"{at}{lead}{c}{trail}" for c in completions]
    common = os.path.commonprefix(completions)
    return completions, common


class SmartPathCompleter(Completer):
    """Path completer that works on the last token of any input line.

    Handles ``~/...`` expansion (keeping the ``~`` in the buffer), paths
    embedded mid-sentence, and bare-word completion (``ag`` → ``agent/``).
    """

    def __init__(self, min_bare_word_len: int = _MIN_BARE_WORD_LEN) -> None:
        self.min_bare_word_len = min_bare_word_len

    def get_completions(self, document: Document, complete_event) -> list[Completion]:
        text = document.text_before_cursor
        token, offset = _last_token(text)
        at = ""
        full_token = token
        if token.startswith("@"):
            at, token = "@", token[1:]
        inner, _lead, _trail = _split_backticks(token)

        if not inner:
            return []
        # Skip very short bare words to avoid noise mid-sentence.
        if (
            os.sep not in inner
            and inner[0] not in "~."
            and len(inner) < self.min_bare_word_len
        ):
            return []

        completions = _completions_for_token(inner)
        # Preserve the @ prefix and any backtick wrapping from the typed token.
        _, lead, trail = _split_backticks(token)
        completions = [f"{at}{lead}{c}{trail}" for c in completions]
        out = []
        for full in completions:
            # text to insert at cursor = full token minus what's typed,
            # adjusted by start_position so prompt_toolkit replaces the token.
            bare = full.strip("@`")
            display = os.path.basename(bare.rstrip(os.sep)) or bare
            out.append(
                Completion(
                    text=full,
                    start_position=-len(full_token),
                    display=display,
                )
            )
        return out
