"""Streaming markdown renderer for the REPL.

Renders streamed LLM output incrementally, without ``rich.live`` and
without Rich markup strings. The strategy is deliberately simple and cheap:

- All received text is kept in a buffer.
- On every chunk, only the **completed lines** (up to the last newline)
  are converted and emitted; the trailing partial line is held back until
  it is complete. This means a heading marker like ``##`` at the end of a
  line is only formatted once we know a newline follows, and no line is
  ever rendered twice.
- The conversion produces :class:`rich.text.Text` objects with **style
  spans** — never markup strings. Since Rich performs no markup parsing
  on ``Text`` output, malformed or adversarial model output can never
  raise a ``MarkupError``.
- The parser is a lightweight line/inline scanner, not a full markdown
  parser. It covers the constructs that matter visually while streaming:
  headings, code fences, lists, blockquotes, horizontal rules, and inline
  bold/italic/code/strikethrough/link formatting. Styles come from the
  console theme (``agent/console.py``), so the output matches the rest of
  the UI.
"""

from __future__ import annotations

import re

from rich.markup import escape
from rich.text import Text

from agent.console import theme_dict


def _style(name: str, fallback: str = "") -> str:
    """Look up a theme style, falling back to *fallback*."""
    return theme_dict.get(name, fallback)


# Combined inline scanner. Alternation order matters: code first, then
# bold (** before *), then italic, strikethrough, links.
_INLINE_RE = re.compile(
    r"`([^`]+)`"                                   # 1: code span
    r"|\*\*(.+?)\*\*"                              # 2: bold (**)
    r"|__(.+?)__"                                  # 3: bold (__)
    r"|(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)"   # 4: italic (*)
    r"|(?<![\w\\])_([^_\n]+?)_(?!\w)"              # 5: italic (_)
    r"|~~([^~\n]+?)~~"                             # 6: strikethrough
    r"|\[([^\]\n]+)\]\(([^)\n]+)\)"                # 7, 8: [label](url)
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_HR_RE = re.compile(r"^\s*([-*_]\s*){3,}$")
_UL_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_OL_RE = re.compile(r"^(\s*)(\d+[.)])\s+(.*)$")
_QUOTE_RE = re.compile(r"^(\s*)>\s?(.*)$")


def _render_inline(text: str, base: str = "") -> Text:
    """Convert inline markdown in *text* to a styled ``Text``.

    Literal text is appended as-is (no escaping needed — ``Text`` is not
    parsed as markup), and recognized constructs get style spans.
    """
    result = Text(style=base)
    pos = 0
    for m in _INLINE_RE.finditer(text):
        if m.start() > pos:
            result.append(text[pos:m.start()])
        if m.group(1) is not None:  # code
            result.append(m.group(1), style=_style("code"))
        elif m.group(2) is not None or m.group(3) is not None:  # bold
            result.append(m.group(2) or m.group(3), style="bold")
        elif m.group(4) is not None or m.group(5) is not None:  # italic
            result.append(m.group(4) or m.group(5), style="italic")
        elif m.group(6) is not None:  # strikethrough
            result.append(m.group(6), style="strike")
        else:  # link
            result.append(m.group(7), style=_style("link"))
            result.append(f" ({m.group(8)})", style=_style("weak", "grey39"))
        pos = m.end()
    if pos < len(text):
        result.append(text[pos:])
    return result


def render_line(line: str) -> Text:
    """Render a single complete markdown line as a styled ``Text``."""
    if not line.strip():
        return Text("")

    # Code fences: the fence line itself is dimmed.
    if line.lstrip().startswith("```"):
        lang = line.lstrip()[3:].strip()
        label = f" ({lang})" if lang else ""
        return Text(f"{'─' * 3}{label}", style=_style("weak", "grey39"))

    m = _HEADING_RE.match(line)
    if m:
        level = len(m.group(1))
        # H1 gets the accent style underlined; H2 just accent-bold; deeper
        # headings plain bold.
        if level == 1:
            style = f"{_style('accent-bold')} underline"
        elif level == 2:
            style = _style("accent-bold")
        else:
            style = "bold"
        return _render_inline(m.group(2), base=style)

    if _HR_RE.match(line):
        return Text("─" * 40, style=_style("weak", "grey39"))

    m = _UL_RE.match(line)
    if m:
        indent, text = m.group(1), m.group(2)
        result = Text(indent)
        result.append("•", style=_style("list-item"))
        result.append(" ")
        result.append(_render_inline(text))
        return result

    m = _OL_RE.match(line)
    if m:
        indent, num, text = m.group(1), m.group(2), m.group(3)
        result = Text(indent)
        result.append(num, style=_style("list-item"))
        result.append(" ")
        result.append(_render_inline(text))
        return result

    m = _QUOTE_RE.match(line)
    if m:
        indent, text = m.group(1), m.group(2)
        result = Text(indent)
        result.append("│", style=_style("weak", "grey39"))
        result.append(" ")
        result.append(_render_inline(text))
        return result

    return _render_inline(line)


class MarkdownStreamRenderer:
    """Incrementally render streamed markdown text to the console.

    Feed chunks with :meth:`feed`. Only complete lines are emitted, so
    output is stable and nothing is rendered twice. Call :meth:`flush`
    at the end of the stream to emit the trailing partial line.
    """

    def __init__(self, console) -> None:
        self._console = console
        self._buffer = ""
        self._emitted = 0  # chars of buffer already rendered
        self._in_code = False

    def feed(self, text: str) -> None:
        """Accept a new chunk of streamed text."""
        self._buffer += text
        self._drain(final=False)

    def flush(self) -> None:
        """Emit any remaining buffered text (end of stream)."""
        self._drain(final=True)

    # ── internals ───────────────────────────────────────────────────────────

    def _drain(self, final: bool) -> None:
        pending = self._buffer[self._emitted:]
        if not pending:
            return

        # Determine how much of the pending text is safe to render:
        # everything up to and including the last newline. On the final
        # flush, everything.
        if final:
            safe, pending = pending, ""
        else:
            idx = pending.rfind("\n")
            if idx < 0:
                return  # no complete line yet
            safe, pending = pending[:idx + 1], pending[idx + 1:]

        # safe always ends with "\n" (except on final flush), so drop the
        # trailing empty element to avoid emitting a spurious blank line
        # after every rendered line.
        lines = safe.split("\n")
        if not final:
            lines = lines[:-1]
        for line in lines:
            self._emit_line(line)
        self._emitted = len(self._buffer) - len(pending)

    def _emit_line(self, line: str) -> None:
        # Track fenced code blocks: while inside one, lines are rendered
        # verbatim in the code style (no inline markdown processing).
        if line.lstrip().startswith("```"):
            self._in_code = not self._in_code
            self._console.print(render_line(line), highlight=False)
            return
        if self._in_code:
            self._console.print(
                Text(line, style=_style("code")), highlight=False
            )
            return
        rendered = render_line(line)
        if rendered.plain:
            try:
                self._console.print(rendered, highlight=False)
            except Exception:
                # Belt and braces: a rendering bug must never crash the
                # turn. Plain escaped text is always safe.
                self._console.print(escape(line), highlight=False)
        else:
            self._console.print()
