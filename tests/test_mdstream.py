"""Tests for the streaming markdown renderer, tables above all.

A table is the one construct the renderer cannot emit as it arrives: the
delimiter row that identifies a table comes *after* its header, and the
column widths come from the rows. So these tests pin both halves of that --
nothing of the table is printed while it streams, and the whole of it is
printed once it ends.
"""

from __future__ import annotations

import io
import unittest

from rich.console import Console

from agent.console import monkee_theme
from agent.mdstream import (
    MarkdownStreamRenderer,
    _is_delimiter_row,
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


class TestTableCells(unittest.TestCase):
    def test_outer_pipes_are_optional(self):
        self.assertEqual(_table_cells("| a | b |"), ["a", "b"])
        self.assertEqual(_table_cells("a | b"), ["a", "b"])
        self.assertEqual(_table_cells("| a | b"), ["a", "b"])

    def test_an_escaped_pipe_stays_in_one_cell(self):
        self.assertEqual(_table_cells(r"| a \| b | c |"), ["a | b", "c"])

    def test_a_delimiter_row_must_match_the_header(self):
        self.assertTrue(_is_delimiter_row(["---", ":--", "--:"], 3))
        self.assertTrue(_is_delimiter_row(["---"], 1))
        self.assertFalse(_is_delimiter_row(["---", ":--"], 3))
        self.assertFalse(_is_delimiter_row(["oops"], 1))
        self.assertFalse(_is_delimiter_row([], 0))


class TestRenderTable(unittest.TestCase):
    def _table(self, *lines: str) -> str:
        console, stream = _capture()
        console.print(render_table(lines[0], lines[1], list(lines[2:])))
        return stream.getvalue()

    def test_header_and_rows_are_both_drawn(self):
        out = self._table("| Name | Age |", "| --- | --- |", "| Ada | 36 |")
        self.assertIn("Name", out)
        self.assertIn("Ada", out)
        self.assertIn("36", out)

    def test_alignment_comes_from_the_delimiter_row(self):
        table = render_table(
            "| a | b | c |", "| :-- | :-: | --: |", ["| 1 | 2 | 3 |"]
        )
        self.assertEqual(
            [column.justify for column in table.columns],
            ["left", "center", "right"],
        )

    def test_the_header_is_a_header_row_not_a_data_row(self):
        table = render_table("| a |", "| --- |", ["| 1 |", "| 2 |"])
        self.assertEqual(table.row_count, 2)

    def test_a_ragged_row_is_padded_rather_than_truncated(self):
        out = self._table(
            "| a | b |", "| --- | --- |", "| 1 |", "| 2 | 3 | 4 |"
        )
        self.assertIn("4", out)  # a fifth column appears rather than drop it

    def test_inline_markdown_inside_a_cell_is_rendered(self):
        out = self._table("| a |", "| --- |", "| **bold** |")
        self.assertIn("bold", out)
        self.assertNotIn("**", out)


class TestTableInStream(unittest.TestCase):
    TABLE = [
        "Here is a table:\n",
        "| Name | Age |\n",
        "| --- | --- |\n",
        "| Ada | 36 |\n",
        "| Alan | 41 |\n",
    ]

    def test_nothing_of_the_table_is_printed_while_it_streams(self):
        console, stream = _capture()
        renderer = MarkdownStreamRenderer(console)
        for line in self.TABLE:
            renderer.feed(line)
            written = stream.getvalue()
            self.assertIn("Here is a table", written)
            self.assertNotIn("Name", written)
            self.assertNotIn("Ada", written)

    def test_the_whole_table_arrives_when_the_table_ends(self):
        out = _render(self.TABLE + ["Thanks.\n"])
        self.assertIn("Name", out)
        self.assertIn("Ada", out)
        self.assertIn("Alan", out)
        self.assertIn("Thanks.", out)
        # ...and after the intro it was held behind, not before it.
        self.assertLess(out.index("Here is a table"), out.index("Name"))

    def test_a_table_open_at_the_end_of_the_stream_is_still_drawn(self):
        out = _render(self.TABLE)
        self.assertIn("Ada", out)
        self.assertIn("Alan", out)

    def test_a_table_at_the_very_end_without_a_newline_is_drawn(self):
        out = _render(["| a | b |\n", "| --- | --- |\n", "| 1 | 2 |"])
        self.assertIn("1", out)
        self.assertIn("2", out)

    def test_flush_between_narration_and_a_tool_line_draws_the_table(self):
        """`Agent._flush_md_stream` flushes mid-turn, before tool output."""
        console, stream = _capture()
        renderer = MarkdownStreamRenderer(console)
        for line in self.TABLE:
            renderer.feed(line)
        renderer.flush()
        self.assertIn("Ada", stream.getvalue())

    def test_a_second_table_after_the_first_is_drawn_too(self):
        out = _render(
            self.TABLE
            + [
                "\n",
                "| x | y |\n",
                "| --- | --- |\n",
                "| 1 | 2 |\n",
                "end\n",
            ]
        )
        self.assertIn("Ada", out)
        self.assertIn("x", out)
        self.assertIn("end", out)
        self.assertLess(out.index("Ada"), out.index("end"))


class TestTablesAreNotOverEager(unittest.TestCase):
    def test_a_pipe_line_that_is_not_a_table_is_printed_as_text(self):
        out = _render(["just a | pipe\n", "and another line\n"])
        self.assertIn("just a | pipe", out)
        self.assertIn("another line", out)

    def test_a_thematic_break_after_a_pipe_line_is_not_a_table(self):
        # A table would have split these into columns, so printing the line
        # back whole is exactly the evidence that none was drawn.
        out = _render(["a | b\n", "---\n", "text\n"])
        self.assertIn("a | b", out)
        self.assertIn("text", out)

    def test_a_delimiter_with_the_wrong_cell_count_is_not_a_table(self):
        out = _render(["a | b\n", "|---|\n", "text\n"])
        self.assertIn("a | b", out)
        self.assertIn("|---|", out)
        self.assertIn("text", out)

    def test_pipes_inside_a_code_fence_are_verbatim(self):
        out = _render(["```\n", "| not | a | table |\n", "```\n", "after\n"])
        self.assertIn("| not | a | table |", out)
        self.assertIn("after", out)

    def test_a_trailing_pipe_line_becomes_ordinary_text(self):
        out = _render(["| a | b |\n"])
        self.assertIn("| a | b |", out)


class TestTableEnds(unittest.TestCase):
    def test_a_blank_line_ends_the_table(self):
        out = _render(
            [
                "| a | b |\n",
                "| --- | --- |\n",
                "| 1 | 2 |\n",
                "\n",
                "After.\n",
            ]
        )
        self.assertIn("1", out)
        self.assertIn("After.", out)

    def test_a_heading_after_a_table_is_not_swallowed_as_a_row(self):
        out = _render(
            [
                "| a | b |\n",
                "| --- | --- |\n",
                "| 1 | 2 |\n",
                "## Next\n",
            ]
        )
        self.assertIn("1", out)
        self.assertIn("Next", out)


if __name__ == "__main__":
    unittest.main()
