"""Tests for tools/files.py — read_file handler (full read + head-style max_lines)."""

from typing import cast

from tests.conftest import BaseTest
from agent.tools import discover_tools, get_registry
from agent.output import set_output, OutputAdapter


class _FakeOutput:
    """Minimal OutputAdapter stub that swallows all output."""

    def print(self, *args, **kwargs):
        pass

    def err(self, *args, **kwargs):
        pass

    def ok(self, *args, **kwargs):
        pass

    def newline(self, *args, **kwargs):
        pass


class TestReadFile(BaseTest):
    """Test read_file_handler."""

    def setUp(self):
        super().setUp()
        # Ensure tools are discovered, then grab the real handler from the
        # registry (the @tool decorator does not return the function, so the
        # module-level name is None).
        discover_tools()
        self.handler = get_registry()["read_file"]["handler"]
        # read_file_handler calls get_output(); install a no-op adapter.
        set_output(cast(OutputAdapter, _FakeOutput()))
        self.addCleanup(set_output, None)

    def test_reads_full_file(self):
        path = self._write_file("full.txt", "line1\nline2\nline3\n")
        result = self.handler({"path": str(path)})
        assert result["content"] == "line1\nline2\nline3\n"
        assert "truncated" not in result

    def test_max_lines_reads_head(self):
        path = self._write_file("head.txt", "a\nb\nc\nd\ne\n")
        result = self.handler({"path": str(path), "max_lines": 2})
        assert result["content"] == "a\nb\n"
        assert result["truncated"] is True
        assert result["total_lines"] == 5
        assert result["shown_lines"] == 2

    def test_max_lines_larger_than_file_not_truncated(self):
        path = self._write_file("small.txt", "a\nb\n")
        result = self.handler({"path": str(path), "max_lines": 100})
        assert result["content"] == "a\nb\n"
        assert "truncated" not in result

    def test_max_lines_zero_reads_full(self):
        path = self._write_file("zero.txt", "a\nb\nc\n")
        result = self.handler({"path": str(path), "max_lines": 0})
        assert result["content"] == "a\nb\nc\n"
        assert "truncated" not in result

    def test_max_lines_invalid_string_reads_full(self):
        path = self._write_file("bad.txt", "a\nb\n")
        result = self.handler({"path": str(path), "max_lines": "not-a-number"})
        assert result["content"] == "a\nb\n"

    def test_show_line_numbers(self):
        path = self._write_file("num.txt", "x\ny\n")
        result = self.handler({"path": str(path), "show_line_numbers": True})
        assert "1: x" in result["content"]
        assert "2: y" in result["content"]

    def test_missing_file_returns_error(self):
        result = self.handler({"path": str(self._tmpdir / "nope.txt")})
        assert "error" in result

    def test_no_path_returns_error(self):
        result = self.handler({"path": ""})
        assert "error" in result


class TestReadFileOffset(BaseTest):
    """`offset` reads a window of a file instead of shelling out to `sed`.

    The two reasons this is not just `head` with extra steps: the skipped
    prefix is never held in memory (which is the whole point for a large
    file), and line numbers stay anchored to real positions, so two reads of
    the same region produce identical numbering.
    """

    def setUp(self):
        super().setUp()
        discover_tools()
        self.handler = get_registry()["read_file"]["handler"]
        set_output(cast(OutputAdapter, _FakeOutput()))
        self.addCleanup(set_output, None)

    def _lines_file(self, name="numbered.txt", n=20):
        path = self._tmpdir / name
        path.write_text("".join(f"line{i}\n" for i in range(1, n + 1)))
        return path

    def test_offset_reads_from_that_line(self):
        path = self._lines_file()
        result = self.handler({"path": str(path), "offset": 5, "max_lines": 3})
        assert result["content"] == "line5\nline6\nline7\n"
        assert result["start_line"] == 5
        assert result["end_line"] == 7
        assert result["truncated"] is True
        assert result["total_lines"] == 20
        assert result["shown_lines"] == 3

    def test_offset_zero_reads_from_the_top(self):
        path = self._lines_file()
        result = self.handler({"path": str(path), "offset": 0, "max_lines": 2})
        assert result["content"] == "line1\nline2\n"

    def test_offset_alone_reads_to_the_end(self):
        path = self._lines_file()
        result = self.handler({"path": str(path), "offset": 18})
        assert result["content"] == "line18\nline19\nline20\n"
        assert result["start_line"] == 18
        assert result["end_line"] == 20
        # Everything from 18 on was returned; the file is simply short there.
        assert result["total_lines"] == 20

    def test_line_numbers_reflect_real_positions(self):
        """Not the window's own numbering: 1..n would be useless as an anchor."""
        path = self._lines_file(n=200)
        result = self.handler(
            {"path": str(path), "offset": 98, "max_lines": 3, "show_line_numbers": True}
        )
        assert " 98: line98" in result["content"]
        assert " 99: line99" in result["content"]
        assert "100: line100" in result["content"]

    def test_line_number_width_is_stable_across_reads(self):
        """A narrow window at the end of a big file is padded like a full read.

        The padding comes from the width of the *file's* line count, not the
        window's, so the same line reads identically whether it was reached by a
        window or by reading the whole file.
        """
        path = self._lines_file(n=150)
        window = self.handler({"path": str(path), "offset": 148, "show_line_numbers": True})
        full = self.handler({"path": str(path), "show_line_numbers": True})
        # A full read has no trailing newline after its last line; a window
        # that reaches the end does. Compare the lines, not the raw strings.
        assert window["content"].splitlines() == full["content"].splitlines()[147:]
        assert window["content"].splitlines()[0] == "148: line148"

    def test_offset_past_the_end_is_an_empty_read(self):
        path = self._lines_file()
        result = self.handler({"path": str(path), "offset": 50})
        assert result["content"] == ""
        assert result["start_line"] == 50
        assert result["end_line"] == 50
        assert result["truncated"] is True

    def test_offset_past_the_end_with_line_numbers_is_empty(self):
        path = self._lines_file()
        result = self.handler(
            {"path": str(path), "offset": 99, "show_line_numbers": True}
        )
        assert result["content"] == ""

    def test_offset_far_past_max_lines_is_still_a_window(self):
        """`offset` is where to start, `max_lines` is how many: not a range.

        Reading ``offset=10, max_lines=3`` has to mean lines 10-12. Treating
        the pair as a range end would make the window empty for any offset
        beyond max_lines, which is the ordinary shape of a window near the end
        of a file.
        """
        path = self._lines_file()
        result = self.handler({"path": str(path), "offset": 10, "max_lines": 3})
        assert result["content"] == "line10\nline11\nline12\n"
        assert result["start_line"] == 10
        assert result["end_line"] == 12

    def test_negative_offset_is_ignored(self):
        """`tail -n -N` is far more likely a miscount than an intent."""
        path = self._lines_file()
        result = self.handler({"path": str(path), "offset": -5, "max_lines": 2})
        assert result["content"] == "line1\nline2\n"

    def test_offset_as_a_string_still_works(self):
        """Models send numbers as strings; a window must not fail on that."""
        path = self._lines_file()
        result = self.handler({"path": str(path), "offset": "5", "max_lines": "2"})
        assert result["content"] == "line5\nline6\n"

    def test_offset_invalid_string_reads_the_whole_file(self):
        path = self._lines_file(n=3)
        result = self.handler({"path": str(path), "offset": "not-a-number"})
        assert result["content"] == "line1\nline2\nline3\n"
        assert "truncated" not in result

    def test_offset_beyond_a_short_file_reports_the_real_total(self):
        path = self._lines_file(n=3)
        result = self.handler({"path": str(path), "offset": 2, "max_lines": 10})
        assert result["content"] == "line2\nline3\n"
        assert result["total_lines"] == 3

    def test_no_trailing_newline_is_handled(self):
        path = self._tmpdir / "no-newline.txt"
        path.write_text("a\nb\nc")
        result = self.handler(
            {"path": str(path), "offset": 2, "show_line_numbers": True}
        )
        assert result["content"] == "2: b\n3: c"
        assert result["end_line"] == 3

    def test_a_window_near_the_end_of_a_huge_file(self):
        """The prefix is skipped, not buffered -- that is the point of offset.

        50k lines is small enough to hold, so this does not measure memory; it
        pins the observable consequence that matters: a window near the end of
        a big file is read quickly and reports the file's real line count, so
        `total_lines` is not the number of lines returned.
        """
        path = self._tmpdir / "wide.txt"
        with open(path, "w") as f:
            for i in range(1, 50001):
                f.write(f"line{i}\n")

        result = self.handler({"path": str(path), "offset": 49999})
        assert result["content"] == "line49999\nline50000\n"
        assert result["total_lines"] == 50000
        assert result["start_line"] == 49999
        assert result["end_line"] == 50000
        assert result["truncated"] is True
