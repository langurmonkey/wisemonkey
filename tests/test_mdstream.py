"""Tests for the streaming markdown renderer."""

from __future__ import annotations

import io
import unittest

from rich.console import Console

from agent.console import monkee_theme
from agent.mdstream import (
    MarkdownStreamRenderer,
    _fence_match,
    _is_delimiter_row,
    _render_inline,
    _table_cells,
    render_table,
)


def _capture(width: int = 100) -> tuple[Console, io.StringIO]:
    stream = io.StringIO()
    console = Console(
        file=stream,
        width=width,
        theme=monkee_theme,
        force_terminal=False,
        highlight=False,
    )
    return console, stream


def _render(chunks: list[str], width: int = 100) -> str:
    """Feed *chunks* to a fresh renderer and return what reached the console."""
    console, stream = _capture(width)
    renderer = MarkdownStreamRenderer(console)
    for chunk in chunks:
        renderer.feed(chunk)
    renderer.flush()
    return stream.getvalue()


TABLE = [
    "Here is a table:\n",
    "| Name | Age |\n",
    "| --- | --- |\n",
    "| Ada | 36 |\n",
    "| Alan | 41 |\n",
]


class TestTableParsing(unittest.TestCase):
    def test_cells_and_the_delimiter_row(self):
        assert _table_cells("| a | b |") == ["a", "b"]
        assert _table_cells("a | b") == ["a", "b"]
        assert _table_cells(r"| a \| b | c |") == ["a | b", "c"]

        # The delimiter must match the header's cell count, or a thematic
        # break after any pipe-bearing line would become a one-column table.
        assert _is_delimiter_row(["---", ":--", "--:"], 3)
        assert not _is_delimiter_row(["---", ":--"], 3)
        assert not _is_delimiter_row(["oops"], 1)
        assert not _is_delimiter_row([], 0)


class TestRenderTable(unittest.TestCase):
    def test_alignment_comes_from_the_delimiter_row_and_nothing_is_lost(self):
        table = render_table("| a | b | c |", "| :-- | :-: | --: |", ["| 1 | 2 | 3 |"])
        assert [column.justify for column in table.columns] == [
            "left", "center", "right"
        ]
        assert table.row_count == 1  # the header is a header, not a data row

        # A ragged row gains a column rather than dropping the cell; a bold
        # cell keeps its markup out of the table body.
        console, stream = _capture()
        console.print(render_table("| a | b |", "| --- | --- |",
                                   ["| 1 |", "| 2 | 3 | 4 |"]))
        assert "4" in stream.getvalue()
        assert "**" not in _render(["| a |\n", "| --- |\n", "| **bold** |\n"])


class TestTableInStream(unittest.TestCase):
    def test_nothing_is_printed_while_it_streams_and_the_whole_arrives(self):
        """Widths come from every row, so a table cannot be drawn line by line."""
        console, stream = _capture()
        renderer = MarkdownStreamRenderer(console)
        for line in TABLE:
            renderer.feed(line)
            written = stream.getvalue()
            assert "Name" not in written and "Ada" not in written
        renderer.flush()
        assert "Ada" in stream.getvalue()

        out = _render(TABLE + ["Thanks.\n"])
        for text in ("Name", "Ada", "Alan", "Thanks."):
            assert text in out
        assert out.index("Here is a table") < out.index("Name")

    def test_a_table_open_at_the_end_is_drawn_and_a_blank_or_heading_ends_one(self):
        # flush() draws a table the stream never closed, newline or not.
        assert "Alan" in _render(TABLE)
        out = _render(["| a | b |\n", "| --- | --- |\n", "| 1 | 2 |"])
        assert "1" in out and "2" in out
        out = _render(TABLE + ["\n", "| x | y |\n", "| --- | --- |\n",
                               "| 1 | 2 |\n", "## Next\n", "end\n"])
        assert out.index("Ada") < out.index("Next") < out.index("end")
        assert "x" in out


class TestTablesAreNotOverEager(unittest.TestCase):
    def test_pipe_bearing_lines_and_fences_are_not_tables(self):
        for chunks, expected in (
            (["just a | pipe\n", "and another line\n"], ["just a | pipe", "another"]),
            (["a | b\n", "---\n", "text\n"], ["a | b", "text"]),       # a rule
            (["a | b\n", "|---|\n", "text\n"], ["a | b", "text"]),     # count mismatch
        ):
            out = _render(chunks)
            for text in expected:
                assert text in out, (chunks, text)
        # A trailing pipe line with no delimiter is ordinary text too.
        assert "| a | b |" in _render(["| a | b |\n"])
        # A pipe inside a fence stays verbatim.
        out = _render(["```\n", "| not | a | table |\n", "```\n", "after\n"])
        assert "| not | a | table |" in out and "after" in out


class TestBackslashEscapes(unittest.TestCase):
    def test_escaped_punctuation_is_literal(self):
        for source, plain in ((r"\*stars\*", "*stars*"), (r"\_under\_", "_under_"),
                              (r"a \\ b", r"a \ b")):
            text = _render_inline(source)
            assert text.plain == plain, source
            assert all(span.style != "italic" for span in text.spans), source
        assert _render_inline("*stars*").plain == "stars"  # unescaped still works
        # And no backslash survives into the output.
        out = _render([r"literal \*stars\* here" + "\n"])
        assert "literal *stars* here" in out and "\\" not in out


class TestInlineConstructs(unittest.TestCase):
    def test_images_links_and_task_lists(self):
        out = _render(["![alt text](http://x/y.png)\n"])
        assert "alt text" in out and "http://x/y.png" in out and "!alt" not in out
        assert "image" in _render(["![](http://x/y.png)\n"])
        assert _render_inline("[label](http://x)").plain == "label (http://x)"

        out = _render(["- [x] done\n", "- [X] also\n", "- [ ] todo\n", "- plain\n"])
        assert "☑ done" in out and "☑ also" in out
        assert "☐ todo" in out
        assert "• plain" in out


class TestCodeFences(unittest.TestCase):
    def test_fence_matching_and_a_shorter_run_does_not_close_a_longer_fence(self):
        assert _fence_match("```") == ("`", 3, "")
        assert _fence_match("~~~python") == ("~", 3, "python")
        assert _fence_match("`` only two") is None
        assert _fence_match("not a fence") is None
        # Which is why a code sample about fences is written with four.
        out = _render(["````\n", "text\n", "```\n", "still code\n", "````\n", "after\n"])
        assert "```" in out and "after" in out

    def test_a_tilde_fence_contains_a_backtick_fence_and_is_labelled(self):
        out = _render(["~~~python\n", "```\n", "x = 1\n", "~~~\n", "after\n"])
        assert "```" in out and "x = 1" in out and "(python)" in out and "after" in out


if __name__ == "__main__":
    unittest.main()
