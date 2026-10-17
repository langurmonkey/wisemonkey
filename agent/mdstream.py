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
- Tables are the one construct that cannot be rendered incrementally. The
  column widths and alignments are only known once every row has arrived,
  and the delimiter line that marks a table at all comes *after* its
  header, so a header candidate holds the renderer back -- nothing further
  is emitted until the table ends -- and the finished table is drawn in one
  piece as a Rich ``Table``.
"""

from __future__ import annotations

import re

from rich import box
from rich.console import JustifyMethod
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from agent.console import theme_dict


def _style(name: str, fallback: str = "") -> str:
    """Look up a theme style, falling back to *fallback*.

    *fallback* is a Rich style spec, not a theme tag. Every caller passes a
    tag that the palette always defines, so in practice it is never used --
    it exists so a missing tag degrades to something rather than to an
    unstyled span.
    """
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
            result.append(f" ({m.group(8)})", style=_style("weak"))
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
        return Text(f"{'─' * 3}{label}", style=_style("weak"))

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
        return Text("─" * 40, style=_style("weak"))

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
        result.append("│", style=_style("weak"))
        result.append(" ")
        result.append(_render_inline(text))
        return result

    return _render_inline(line)


# ── tables ────────────────────────────────────────────────────────────────
#
# A table cannot be rendered until it ends: the column widths come from every
# row, and the delimiter row that identifies a table at all arrives *after*
# its header. So a header candidate stops the renderer -- nothing is emitted
# again until the table does end -- and the finished table is drawn in one
# piece. See `MarkdownStreamRenderer`.

_ESCAPED_PIPE = "\\|"  # ``\|``: a literal pipe inside a cell
_PIPE_MARK = chr(0)  # stands in for it while the row is split
_DELIMITER_CELL_RE = re.compile(r"^:?-+:?$")


def _table_cells(line: str) -> list[str]:
    """Split a table row into its cells.

    The leading and trailing ``|`` are both optional in markdown, so each is
    dropped if present before splitting on the pipes that remain.
    """
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith(_ESCAPED_PIPE):
        body = body[:-1]
    cells = body.replace(_ESCAPED_PIPE, _PIPE_MARK).split("|")
    return [cell.replace(_PIPE_MARK, "|").strip() for cell in cells]


def _is_delimiter_row(cells: list[str], columns: int) -> bool:
    """Whether *cells* is the ``--- | :--: |`` line of a *columns*-wide table.

    The count has to match the header's: that is what separates a table from
    a thematic break (``---``) that happens to follow a line with a pipe in
    it, which is the only false positive that would otherwise be common.
    """
    return (
        bool(cells)
        and len(cells) == columns
        and all(_DELIMITER_CELL_RE.match(cell.strip()) for cell in cells)
    )


def _table_alignments(cells: list[str]) -> list[JustifyMethod]:
    """The column justifications a delimiter row names with its colons."""
    alignments: list[JustifyMethod] = []
    for cell in cells:
        cell = cell.strip()
        if cell.startswith(":") and cell.endswith(":"):
            alignments.append("center")
        elif cell.endswith(":"):
            alignments.append("right")
        else:
            alignments.append("left")
    return alignments


def render_table(header_line: str, delimiter_line: str, rows: list[str]) -> Table:
    """Render a complete markdown table as a Rich ``Table``.

    Cell contents are rendered inline (bold, code, links), and the column
    count is the widest of any row so a short or ragged row cannot misalign
    what follows it. Alignment is taken from the delimiter row, which is the
    only place markdown records it.
    """
    header = _table_cells(header_line)
    body = [_table_cells(row) for row in rows]
    columns = max([len(header), *(len(row) for row in body)])
    alignments = _table_alignments(_table_cells(delimiter_line))

    table = Table(box=box.SIMPLE, show_edge=False, border_style=_style("weak"))
    for index in range(columns):
        header_cell = header[index] if index < len(header) else ""
        table.add_column(
            _render_inline(header_cell, base=_style("accent-bold")),
            justify=alignments[index] if index < len(alignments) else "left",
        )
    for row in body:
        table.add_row(
            *[
                _render_inline(row[index] if index < len(row) else "")
                for index in range(columns)
            ]
        )
    return table


class MarkdownStreamRenderer:
    """Incrementally render streamed markdown text to the console.

    Feed chunks with :meth:`feed`. Only complete lines are emitted, so
    output is stable and nothing is rendered twice. A table is the one
    exception, and it is deliberate: its lines are held until the table
    ends (see the module docstring), then drawn in one piece. Call
    :meth:`flush` at the end of the stream to emit the trailing partial
    line and close a table still open.
    """

    def __init__(self, console) -> None:
        self._console = console
        self._pending = ""  # the trailing, still-incomplete line
        self._queue: list[str] = []  # complete lines not yet emitted
        self._in_code = False
        # The table being accumulated: its header, its delimiter row and the
        # raw rows seen so far. `_table_header` is None when there is none.
        self._table_header: str | None = None
        self._table_delimiter = ""
        self._table_rows: list[str] = []

    def feed(self, text: str) -> None:
        """Accept a new chunk of streamed text."""
        if not text:
            return
        self._pending += text
        # Everything up to the last newline is a complete line; whatever
        # follows it is still being written and waits for the next chunk.
        complete, separator, self._pending = self._pending.rpartition("\n")
        if separator:
            self._queue.extend(complete.split("\n"))
        self._drain(final=False)

    def flush(self) -> None:
        """Emit any remaining buffered text (end of stream)."""
        if self._pending:
            self._queue.append(self._pending)
            self._pending = ""
        self._drain(final=True)

    # ── internals ─────────────────────────────────────────────────────────

    def _drain(self, final: bool) -> None:
        index = 0
        total = len(self._queue)
        while index < total:
            line = self._queue[index]

            # Already inside a table: keep taking rows until one is not,
            # then lay the whole thing out. The line that ends the table is
            # rendered as an ordinary one.
            if self._table_header is not None:
                if "|" in line:
                    self._table_rows.append(line)
                    index += 1
                    continue
                self._close_table()
                continue

            # Fenced code is verbatim: a pipe inside it is not a table.
            if line.lstrip().startswith("```"):
                self._in_code = not self._in_code
                self._emit(render_line(line), line)
                index += 1
                continue
            if self._in_code:
                self._emit(Text(line, style=_style("code")), line)
                index += 1
                continue

            if "|" in line:
                # A pipe alone does not make a table: the *next* line has to
                # be a matching delimiter row. Until that line arrives this
                # one is held, so the header is never printed in a shape the
                # finished table would contradict.
                if index + 1 < total:
                    following = self._queue[index + 1]
                    if _is_delimiter_row(
                        _table_cells(following), len(_table_cells(line))
                    ):
                        self._table_header = line
                        self._table_delimiter = following
                        self._table_rows = []
                        index += 2
                        continue
                elif not final:
                    break  # the deciding line has not arrived yet

            self._emit(render_line(line), line)
            index += 1

        del self._queue[:index]
        if final:
            self._close_table()

    def _close_table(self) -> None:
        """Draw the table being accumulated, if there is one."""
        if self._table_header is None:
            return
        header = self._table_header
        delimiter = self._table_delimiter
        rows = self._table_rows
        self._table_header = None
        self._table_delimiter = ""
        self._table_rows = []
        try:
            self._console.print(render_table(header, delimiter, rows))
        except Exception:
            # A layout bug must not lose the content: show it as it arrived.
            for line in (header, delimiter, *rows):
                self._console.print(escape(line), highlight=False)

    def _emit(self, rendered: Text, raw: str) -> None:
        """Print one rendered line, never letting a render bug kill the turn."""
        if not rendered.plain:
            self._console.print()
            return
        try:
            self._console.print(rendered, highlight=False)
        except Exception:
            # Belt and braces: plain escaped text is always safe.
            self._console.print(escape(raw), highlight=False)
