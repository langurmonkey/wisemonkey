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
  headings, code fences (`` ``` `` and ``~~~``, run-length matched), lists
  and task lists, blockquotes, horizontal rules, images, and inline
  bold/italic/code/strikethrough/link formatting with backslash escapes.
  Styles come from the console theme (``agent/console.py``), so the output
  matches the rest of the UI.
- Three constructs cannot be rendered a line at a time, and all three are
  held back rather than printed as they arrive. A fenced code block is
  buffered until its closing fence and then drawn in one piece with real
  syntax highlighting (``rich.syntax.Syntax``), since highlighting is a
  property of the whole block. A table is buffered for a second reason as
  well: the column widths come from every row, and the delimiter line that
  marks a table at all arrives *after* its header, so a header candidate
  holds the renderer back -- nothing further is emitted until the table
  ends -- and the finished table is drawn in one piece as a Rich ``Table``.
  An ordinary paragraph line is held back *one line*, because a setext
  heading underline (``===`` or ``---``) turns the line above it into a
  heading, and a ``---`` cannot be told from a thematic break until the line
  under it arrives.
"""

from __future__ import annotations

import re
import string

from rich import box
from rich.console import JustifyMethod
from rich.markup import escape
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from agent.console import theme_dict
from agent.palette import PALETTE


def _style(name: str, fallback: str = "") -> str:
    """Look up a theme style, falling back to *fallback*.

    *fallback* is a Rich style spec, not a theme tag. Every caller passes a
    tag that the palette always defines, so in practice it is never used --
    it exists so a missing tag degrades to something rather than to an
    unstyled span.
    """
    return theme_dict.get(name, fallback)


#: The syntax theme, built once. It is derived from the palette rather than
#: named, so a fenced code block cannot introduce a colour the rest of the
#: UI has never seen (see `Palette.syntax_theme`).
_SYNTAX_THEME = None


def _syntax_theme():
    """Return the shared syntax theme, building it on first use."""
    global _SYNTAX_THEME
    if _SYNTAX_THEME is None:
        _SYNTAX_THEME = PALETTE.syntax_theme()
    return _SYNTAX_THEME


# ── backslash escapes ──────────────────────────────────────────────────────
#
# Markdown lets a backslash escape any ASCII punctuation, meaning the
# character is literal: ``\*`` is a star, not the start of emphasis. The
# inline scanner is a flat regex and cannot be told "ignore this one" after
# the fact, so an escaped character has to be taken out of its reach *before*
# scanning and put back afterwards. Each punctuation character is swapped for
# a private-use sentinel -- one character for one character, so nothing
# downstream shifts -- and restored in every piece of text that reaches the
# ``Text``.

_ESCAPE_RE = re.compile(r"\\([" + re.escape(string.punctuation) + r"])")
_SENTINEL = {ch: chr(0xE000 + ord(ch)) for ch in string.punctuation}
_UNSENTINEL = str.maketrans(
    {sentinel: ch for ch, sentinel in _SENTINEL.items()}
)


def _protect_escapes(text: str) -> str:
    """Hide backslash-escaped punctuation from the inline scanner."""
    return _ESCAPE_RE.sub(lambda m: _SENTINEL[m.group(1)], text)


def _restore_escapes(text: str) -> str:
    """Put back the punctuation that :func:`_protect_escapes` took out."""
    return text.translate(_UNSENTINEL)


# Combined inline scanner. Alternation order matters: code first, then
# bold (** before *), then italic, strikethrough, images (before links, so
# the leading ``!`` is consumed rather than left as a stray literal), links.
# Named groups rather than numbers: the image alternative added two groups,
# and a positional ``group(7)`` would silently start meaning something else
# the next time a construct is inserted.
_INLINE_RE = re.compile(
    r"`(?P<code>[^`]+)`"                                          # code span
    r"|\*\*(?P<bold_star>.+?)\*\*"                                # bold (**)
    r"|__(?P<bold_under>.+?)__"                                   # bold (__)
    r"|(?<!\*)\*(?!\*)(?P<italic_star>[^*\n]+?)(?<!\*)\*(?!\*)"   # italic (*)
    r"|(?<![\w\\])_(?P<italic_under>[^_\n]+?)_(?!\w)"             # italic (_)
    r"|~~(?P<strike>[^~\n]+?)~~"                                  # strikethrough
    r"|!\[(?P<image_alt>[^\]\n]*)\]\((?P<image_url>[^)\n]+)\)"    # image
    r"|\[(?P<link_label>[^\]\n]+)\]\((?P<link_url>[^)\n]+)\)"     # link
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_HR_RE = re.compile(r"^\s*([-*_]\s*){3,}$")
#: A setext heading's underline: a run of ``=`` (an H1) or ``-`` (an H2),
#: optionally indented up to three spaces and trailed by spaces. Note that a
#: ``---`` is therefore *both* a setext underline and a thematic break, and
#: which of the two it is depends on the line above it -- the reason setext is
#: a hold-back (see `MarkdownStreamRenderer`).
_SETEXT_RE = re.compile(r"^ {0,3}(?P<mark>=+|-+)\s*$")
_UL_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_OL_RE = re.compile(r"^(\s*)(\d+[.)])\s+(.*)$")
_QUOTE_RE = re.compile(r"^(\s*)>\s?(.*)$")
_TASK_RE = re.compile(r"^\[([ xX])\]\s+(.*)$")
#: Three or more backticks or tildes: a code fence. The run length matters on
#: close (a fence has to be closed by one *at least as long*), so both are
#: counted rather than merely detected.
_FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")


def _fence_match(line: str) -> tuple[str, int, str] | None:
    """Return ``(char, length, rest)`` if *line* opens or closes a fence.

    A fence is three or more backticks or tildes. A closing fence has to use
    the *same* character for *at least* the opening length, so a block opened
    with four backticks can contain a line of three without ending early --
    which is exactly why a code sample about code fences is written with four.
    """
    m = _FENCE_RE.match(line)
    if not m:
        return None
    return m.group(2)[0], len(m.group(2)), m.group(3)


def _setext_level(line: str) -> int:
    """Return 1 or 2 if *line* is a setext underline, otherwise 0.

    ``===`` underlines an H1, ``---`` an H2. Which one a line of dashes is --
    a setext underline or a thematic break -- is decided by its *caller*,
    against the line above: see :func:`MarkdownStreamRenderer._drain`.
    """
    m = _SETEXT_RE.match(line)
    if m is None:
        return 0
    return 1 if m.group("mark").startswith("=") else 2


def _could_be_setext_text(line: str) -> bool:
    """Whether *line* could be the text of a setext heading.

    A setext heading's text is a paragraph, so a line that opens some *other*
    block -- a fence, an ATX heading, a list, a quote, a rule, or an underline
    itself -- can never be one, and is emitted the moment it arrives. Only an
    ordinary paragraph line is worth holding back, because only it can be
    turned into a heading by the line that follows.
    """
    if not line.strip():
        return False
    if _fence_match(line) is not None:
        return False
    if _HEADING_RE.match(line) or _SETEXT_RE.match(line) or _HR_RE.match(line):
        return False
    if _UL_RE.match(line) or _OL_RE.match(line) or _QUOTE_RE.match(line):
        return False
    return True


def _heading_style(level: int) -> str:
    """The style for a heading of *level*: H1 underlined, H2 accent-bold."""
    if level == 1:
        return f"{_style('accent-bold')} underline"
    if level == 2:
        return _style("accent-bold")
    return "bold"


def _render_inline(text: str, base: str = "") -> Text:
    """Convert inline markdown in *text* to a styled ``Text``.

    Literal text is appended as-is (no escaping needed — ``Text`` is not
    parsed as markup), and recognized constructs get style spans. Backslash
    escapes are honoured: ``\\*`` is a literal asterisk, not emphasis.
    """
    result = Text(style=base)
    protected = _protect_escapes(text)

    def add(segment: str, style: str | None = None) -> None:
        result.append(_restore_escapes(segment), style=style)

    pos = 0
    for m in _INLINE_RE.finditer(protected):
        if m.start() > pos:
            add(protected[pos:m.start()])
        g = m.groupdict()
        if g["code"] is not None:  # code span
            add(g["code"], _style("code"))
        elif g["bold_star"] is not None or g["bold_under"] is not None:  # bold
            add(g["bold_star"] or g["bold_under"] or "", "bold")
        elif g["italic_star"] is not None or g["italic_under"] is not None:  # italic
            add(g["italic_star"] or g["italic_under"] or "", "italic")
        elif g["strike"] is not None:  # strikethrough
            add(g["strike"], "strike")
        elif g["image_alt"] is not None:  # image
            # Consume the ``!`` here rather than leave it as a stray literal:
            # alt text and source, styled like a link.
            add(g["image_alt"] or "image", _style("link"))
            add(f" ({g['image_url']})", _style("weak"))
        else:  # link
            add(g["link_label"] or "", _style("link"))
            add(f" ({g['link_url']})", _style("weak"))
        pos = m.end()
    if pos < len(protected):
        add(protected[pos:])
    return result


def render_line(line: str) -> Text:
    """Render a single complete markdown line as a styled ``Text``."""
    if not line.strip():
        return Text("")

    # Code fences: the fence line itself is dimmed, with its language label.
    fence = _fence_match(line)
    if fence is not None:
        label = f" ({fence[2].strip()})" if fence[2].strip() else ""
        return Text(f"{'─' * 3}{label}", style=_style("weak"))

    m = _HEADING_RE.match(line)
    if m:
        # H1 gets the accent style underlined; H2 just accent-bold; deeper
        # headings plain bold. A setext heading reaches the same styles via
        # `_heading_style` -- `===` is an H1, `---` an H2.
        return _render_inline(m.group(2), base=_heading_style(len(m.group(1))))

    if _HR_RE.match(line):
        return Text("─" * 40, style=_style("weak"))

    m = _UL_RE.match(line)
    if m:
        indent, text = m.group(1), m.group(2)
        result = Text(indent)
        # A task list item: ``- [x]`` / ``- [ ]``. The marker is recognised
        # before inline rendering, which would otherwise leave the ``[x]``
        # as a literal run of text.
        task = _TASK_RE.match(text)
        if task:
            checked = task.group(1).lower() == "x"
            result.append("☑" if checked else "☐", style=_style("list-item"))
            result.append(" ")
            result.append(_render_inline(task.group(2)))
            return result
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
    output is stable and nothing is rendered twice. Two exceptions are
    deliberate: a table's lines are held until the table ends (see the
    module docstring) and drawn in one piece, and an ordinary paragraph line
    is held *one line* to see whether a setext underline follows it. Call
    :meth:`flush` at the end of the stream to emit the trailing partial line,
    the paragraph line still held, and close a table or fence still open.
    """

    def __init__(self, console) -> None:
        self._console = console
        self._pending = ""  # the trailing, still-incomplete line
        self._queue: list[str] = []  # complete lines not yet emitted
        self._in_code = False
        # The open fence's character and run length, so a line of three
        # backticks cannot close a block opened with four (nor a tilde fence
        # an opened backtick one).
        self._fence_char = ""
        self._fence_len = 0
        # The open fence's language and the body lines seen so far. The body
        # is held until the block closes so it can be highlighted as a whole;
        # the language is known at the opening fence, so only the body waits.
        self._code_lang = ""
        self._code_lines: list[str] = []
        # A paragraph line held back one line, to see whether a setext
        # underline follows it. `None` when there is none held.
        self._para: str | None = None
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

            # A paragraph line held back to see whether a setext underline
            # would follow it. Resolve it against this line first: at most one
            # of the two can be pending, and the held line has to be drawn
            # before anything that comes after it.
            if self._para is not None:
                held = self._para
                self._para = None
                level = _setext_level(line)
                if level:
                    # The line above *was* a setext heading after all. The
                    # underline itself is consumed, not printed.
                    self._emit(
                        _render_inline(held, base=_heading_style(level)), held
                    )
                    index += 1
                    continue
                # Not an underline: ordinary prose. Emit it and re-process
                # this line from the top, which may hold *it* in turn.
                self._emit(render_line(held), held)
                continue

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

            # Fenced code is verbatim: a pipe inside it is not a table. A
            # fence is opened by three or more backticks or tildes and closed
            # only by the same character at least as long. The body is held
            # back rather than printed line by line, so the whole block can be
            # highlighted at once when the closing fence arrives.
            fence = _fence_match(line)
            if self._in_code:
                closes = (
                    fence is not None
                    and fence[0] == self._fence_char
                    and fence[1] >= self._fence_len
                    and not fence[2].strip()
                )
                if closes:
                    self._close_code()
                    self._emit(render_line(line), line)
                else:
                    self._code_lines.append(line)
                index += 1
                continue
            if fence is not None:
                self._in_code = True
                self._fence_char = fence[0]
                self._fence_len = fence[1]
                self._code_lang = fence[2].strip()
                self._emit(render_line(line), line)
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

            # An ordinary paragraph line is held back one line, because a
            # setext underline (``===`` or ``---``) turns the line *above* it
            # into a heading and cannot be recognised until it arrives. Every
            # other kind of line is emitted immediately.
            if _could_be_setext_text(line):
                self._para = line
            else:
                self._emit(render_line(line), line)
            index += 1

        del self._queue[:index]
        if final:
            # A stream can end mid-paragraph: a line held for a setext
            # underline that never came was ordinary prose.
            if self._para is not None:
                self._emit(render_line(self._para), self._para)
                self._para = None
            # A stream can also end with a fence still open. Close it first:
            # only one of the two can be open at a time, and the code block
            # has to be drawn before the table that follows it.
            self._close_code()
            self._close_table()

    def _close_code(self) -> None:
        """Draw the code block being accumulated, highlighted as a whole.

        The body is held until the fence closes for the same reason a table
        is: syntax highlighting is a property of the whole block, not of one
        line of it. The language, unlike a table's column widths, is known
        the moment the fence opens -- so the opening marker is printed
        immediately and only the body waits.

        The fence line itself is *not* printed here: the caller emits it as
        an ordinary line, before this runs for the opening fence and after it
        for the closing one, which is what keeps the marker above the code.
        """
        if not self._in_code:
            return
        lines = self._code_lines
        lang = self._code_lang
        self._in_code = False
        self._fence_char = ""
        self._fence_len = 0
        self._code_lang = ""
        self._code_lines = []
        if not lines:
            return
        try:
            self._console.print(
                Syntax(
                    "\n".join(lines),
                    lang or "text",
                    theme=_syntax_theme(),
                    line_numbers=False,
                    word_wrap=True,
                    # The same surface the inline `code` style sits on, so a
                    # fenced block and a code span read as the same thing.
                    background_color=PALETTE.bar_bg,
                )
            )
        except Exception:
            # An unknown lexer or a layout bug must not lose the code: show
            # it exactly as it arrived, one plain line at a time.
            for line in lines:
                self._console.print(Text(line, style=_style("code")))

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
