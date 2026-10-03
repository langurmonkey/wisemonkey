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


class TestPatchFile(BaseTest):
    """`patch_file` in both its text and line modes.

    The text mode is the legacy exact-match behaviour, with trailing
    whitespace forgiven. The line mode is the point: an edit anchored to line
    numbers does not care what the characters are, which removes the re-read
    round trip that an exact-match failure costs.
    """

    def setUp(self):
        super().setUp()
        discover_tools()
        self.handler = get_registry()["patch_file"]["handler"]
        set_output(cast(OutputAdapter, _FakeOutput()))
        self.addCleanup(set_output, None)

    def _file(self, text, name="f.py"):
        return self._write_file(name, text)

    # --- text mode, the legacy path ---

    def test_replaces_an_exact_block(self):
        path = self._file("alpha\nbeta\ngamma\n")
        result = self.handler(
            {"path": str(path), "old_string": "beta", "new_string": "BETA"}
        )
        assert result["success"] is True
        assert result["changed"] is True
        assert path.read_text() == "alpha\nBETA\ngamma\n"

    def test_multiline_block_replacement(self):
        path = self._file("def f():\n    return 1\n")
        result = self.handler(
            {
                "path": str(path),
                "old_string": "def f():\n    return 1",
                "new_string": "def f():\n    return 2",
            }
        )
        assert result["success"] is True
        assert path.read_text() == "def f():\n    return 2\n"

    def test_trailing_whitespace_in_the_match_is_forgiven(self):
        """The single most common reason an exact match failed by hand."""
        path = self._file("alpha\nbeta   \ngamma\n")
        result = self.handler(
            {"path": str(path), "old_string": "beta", "new_string": "BETA"}
        )
        assert result["success"] is True
        assert path.read_text() == "alpha\nBETA\ngamma\n"

    def test_trailing_whitespace_in_every_line_is_forgiven(self):
        path = self._file("a\nb   \nc  \nd   \n")
        result = self.handler(
            {
                "path": str(path),
                "old_string": "b\nc\nd",
                "new_string": "x\ny\nz",
            }
        )
        assert result["success"] is True
        assert path.read_text() == "a\nx\ny\nz\n"

    def test_indentation_is_not_forgiven(self):
        """Leading whitespace is meaning, not noise. Silently forgiving it
        would move a block out of (or into) a function body."""
        path = self._file("    beta\n")
        result = self.handler(
            {"path": str(path), "old_string": "beta", "new_string": "BETA"}
        )
        assert "error" in result
        assert path.read_text() == "    beta\n"

    def test_ambiguous_match_reports_the_line_numbers(self):
        path = self._file("x\ndup\ny\ndup\n")
        result = self.handler(
            {"path": str(path), "old_string": "dup", "new_string": "DUP"}
        )
        assert "error" in result
        assert "2, 4" in result["error"]
        # Unchanged, and it points at the fix.
        assert path.read_text() == "x\ndup\ny\ndup\n"

    def test_start_line_disambiguates(self):
        path = self._file("x\ndup\ny\ndup\n")
        result = self.handler(
            {
                "path": str(path),
                "old_string": "dup",
                "new_string": "DUP",
                "start_line": 4,
            }
        )
        assert result["success"] is True
        assert path.read_text() == "x\ndup\ny\nDUP\n"

    def test_scoped_search_reports_where_the_text_actually_is(self):
        """A window that excludes the real match must not read as 'gone'."""
        path = self._file("x\ndup\ny\ndup\n")
        result = self.handler(
            {
                "path": str(path),
                "old_string": "dup",
                "new_string": "DUP",
                "start_line": 3,
                "end_line": 3,
            }
        )
        assert "error" in result
        assert "within lines 3-3" in result["error"]
        assert "line(s) 2, 4" in result["error"]

    def test_a_window_containing_one_occurrence_succeeds(self):
        """The window has to be an honest disambiguator, not a stricter match."""
        path = self._file("x\ndup\ny\ndup\n")
        result = self.handler(
            {
                "path": str(path),
                "old_string": "dup",
                "new_string": "DUP",
                "start_line": 1,
                "end_line": 3,
            }
        )
        assert result["success"] is True
        assert path.read_text() == "x\nDUP\ny\ndup\n"

    def test_the_window_boundaries_are_inclusive(self):
        """`end_line` is inclusive: a match on the last line of the window is in."""
        path = self._file("a\nb\ndup\n")
        result = self.handler(
            {
                "path": str(path),
                "old_string": "dup",
                "new_string": "DUP",
                "start_line": 1,
                "end_line": 3,
            }
        )
        assert result["success"] is True
        assert path.read_text() == "a\nb\nDUP\n"

    def test_missing_text_names_the_line_mode_alternative(self):
        path = self._file("alpha\n")
        result = self.handler(
            {"path": str(path), "old_string": "nope", "new_string": "x"}
        )
        assert "start_line" in result["error"]

    def test_no_op_replacement_is_reported_not_rewritten(self):
        path = self._file("alpha\n")
        result = self.handler(
            {"path": str(path), "old_string": "alpha", "new_string": "alpha"}
        )
        assert result["success"] is True
        assert result["changed"] is False

    def test_empty_old_string_is_refused(self):
        path = self._file("alpha\n")
        result = self.handler({"path": str(path), "old_string": "", "new_string": ""})
        assert "error" in result

    def test_new_string_is_required(self):
        """Deleting is `new_string: ""`, so a missing key is a different thing."""
        path = self._file("alpha\n")
        result = self.handler({"path": str(path), "old_string": "alpha"})
        assert "error" in result
        assert path.read_text() == "alpha\n"

    # --- line mode ---

    def test_replaces_a_line_range(self):
        path = self._file("a\nb\nc\nd\ne\n")
        result = self.handler(
            {"path": str(path), "start_line": 2, "end_line": 3, "new_string": "X\nY"}
        )
        assert result["success"] is True
        assert result["changed"] is True
        assert path.read_text() == "a\nX\nY\nd\ne\n"

    def test_replaces_a_single_line(self):
        path = self._file("a\nb\nc\n")
        result = self.handler(
            {"path": str(path), "start_line": 2, "new_string": "B"}
        )
        assert result["success"] is True
        assert path.read_text() == "a\nB\nc\n"

    def test_replaces_a_single_line_without_a_trailing_newline(self):
        """The replacement must not swallow the following line."""
        path = self._file("a\nb\nc\n")
        result = self.handler(
            {"path": str(path), "start_line": 2, "new_string": "B"}
        )
        assert path.read_text() == "a\nB\nc\n"

    def test_line_mode_ignores_indentation_mismatches(self):
        """The point of the mode: the characters do not have to be right."""
        path = self._file("a\n        b\nc\n")
        result = self.handler(
            {"path": str(path), "start_line": 2, "new_string": "\tb"}
        )
        assert result["success"] is True
        assert path.read_text() == "a\n\tb\nc\n"

    def test_line_mode_does_not_need_the_text_at_all(self):
        path = self._file("a\nb\nc\n")
        result = self.handler(
            {"path": str(path), "start_line": 1, "end_line": 2, "new_string": "z"}
        )
        assert result["success"] is True
        assert path.read_text() == "z\nc\n"

    def test_empty_new_string_deletes_the_lines(self):
        path = self._file("a\nb\nc\nd\n")
        result = self.handler(
            {"path": str(path), "start_line": 2, "end_line": 3, "new_string": ""}
        )
        assert result["success"] is True
        assert path.read_text() == "a\nd\n"

    def test_deleting_everything_leaves_an_empty_file(self):
        path = self._file("a\nb\n")
        result = self.handler(
            {"path": str(path), "start_line": 1, "end_line": 2, "new_string": ""}
        )
        assert result["success"] is True
        assert path.read_text() == ""

    def test_reports_the_lines_it_changed(self):
        path = self._file("a\nb\nc\nd\ne\n")
        result = self.handler(
            {"path": str(path), "start_line": 2, "end_line": 3, "new_string": "X\nY\nZ"}
        )
        assert result["start_line"] == 2
        assert result["end_line"] == 4
        assert result["removed_lines"] == 2
        assert result["added_lines"] == 3

    def test_start_line_past_the_end_is_refused(self):
        path = self._file("a\nb\n")
        result = self.handler(
            {"path": str(path), "start_line": 50, "new_string": "x"}
        )
        assert "error" in result
        assert "2 lines" in result["error"]

    def test_end_line_past_the_end_is_clamped(self):
        """A line count from a stale read should still be usable."""
        path = self._file("a\nb\nc\n")
        result = self.handler(
            {"path": str(path), "start_line": 2, "end_line": 99, "new_string": "B"}
        )
        assert result["success"] is True
        assert result["removed_lines"] == 2
        assert path.read_text() == "a\nB\n"

    def test_end_line_before_start_line_is_refused(self):
        path = self._file("a\nb\nc\n")
        result = self.handler(
            {"path": str(path), "start_line": 3, "end_line": 2, "new_string": "x"}
        )
        assert "error" in result

    def test_neither_text_nor_line_is_refused(self):
        path = self._file("a\n")
        result = self.handler({"path": str(path), "new_string": "x"})
        assert "error" in result
        assert "start_line" in result["error"]

    def test_line_numbers_given_as_strings_still_work(self):
        path = self._file("a\nb\nc\n")
        result = self.handler(
            {"path": str(path), "start_line": "2", "end_line": "2", "new_string": "B"}
        )
        assert result["success"] is True
        assert path.read_text() == "a\nB\nc\n"

    def test_unparseable_line_number_does_not_edit_somewhere_random(self):
        """Failing to parse must not mean "line 0" and rewrite the whole file."""
        path = self._file("a\nb\nc\n")
        result = self.handler(
            {"path": str(path), "start_line": "not-a-number", "new_string": "x"}
        )
        assert "error" in result
        assert path.read_text() == "a\nb\nc\n"

    def test_insertion_at_the_end_of_a_file(self):
        path = self._file("a\nb\n")
        result = self.handler(
            {"path": str(path), "start_line": 3, "new_string": "c\n"}
        )
        assert result["success"] is True
        assert path.read_text() == "a\nb\nc\n"

    def test_a_file_without_a_trailing_newline_stays_that_way(self):
        path = self._file("a\nb")
        result = self.handler(
            {"path": str(path), "start_line": 2, "new_string": "B\nC"}
        )
        assert result["success"] is True
        # The replaced block had no trailing newline, so the new text does not
        # gain one: a whole-file trailing-newline change is a spurious diff.
        assert path.read_text() == "a\nB\nC"

    def test_text_mode_on_a_file_without_a_trailing_newline(self):
        path = self._file("a\nb")
        result = self.handler(
            {"path": str(path), "old_string": "b", "new_string": "B"}
        )
        assert result["success"] is True
        assert path.read_text() == "a\nB"


class TestSearchContentRegex(BaseTest):
    """`search_content` takes a regex, so a pattern does not need shell quoting.

    Literal search stays the default: a query that happens to contain regex
    metacharacters (`config.yaml`, `a(b)c`) must keep matching itself, which is
    what every existing caller relies on. Regex is opt-in, and an invalid one
    is an error rather than a silent fallback to a literal search.
    """

    def setUp(self):
        super().setUp()
        discover_tools()
        self.handler = get_registry()["search_content"]["handler"]
        set_output(cast(OutputAdapter, _FakeOutput()))
        self.addCleanup(set_output, None)
        self.root = self._tmpdir / "src"
        self.root.mkdir(parents=True, exist_ok=True)
        self._write("a.py", "def one():\n    pass\n\ndef two():\n    return 2\n")
        self._write("b.py", "    self.value = 1\n    self.other = 2\n")
        self._write("notes.md", "# Notes\n\ndef one() is mentioned here too\n")

    def _write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def _run(self, query, **kw):
        return self.handler({"root": str(self.root), "query": query, **kw})

    # --- literal is still the default ---

    def test_literal_search_is_the_default(self):
        result = self._run("self.value")
        assert result["count"] == 1
        assert result["mode"] == "text"
        assert result["results"][0]["file"] == "b.py"

    def test_metacharacters_are_literal_without_regex(self):
        """`config.yaml` must not read as "config" + any char + "yaml"."""
        self._write("config.yaml", "name: configXyaml\nother: config.yaml\n")
        result = self._run("config.yaml")
        assert result["count"] == 1
        assert "other" in result["results"][0]["line_content"]

    def test_literal_search_ignores_regex_syntax_entirely(self):
        result = self._run("^def ")
        assert result["count"] == 0

    # --- regex ---

    def test_regex_anchors_match_per_line(self):
        """'^def ' is the whole reason regex mode exists.

        Without MULTILINE, '^' would only ever match the start of the file, so
        a pattern written the way it looks in grep would silently match
        nothing -- the most damaging possible failure for a search tool. The
        match on line 3 of a *different* file is the proof: it is neither the
        first line nor the first match.
        """
        result = self._run("^def ", regex=True)
        assert result["mode"] == "regex"
        assert {(r["file"], r["line"]) for r in result["results"]} == {
            ("a.py", 1),
            ("a.py", 4),
            ("notes.md", 3),
        }

    def test_regex_metacharacters(self):
        result = self._run(r"self\.\w+ = ", regex=True)
        assert result["count"] == 2
        assert all(r["file"] == "b.py" for r in result["results"])

    def test_regex_case_insensitivity_by_default(self):
        self._write("c.py", "TOKEN = 1\n")
        assert self._run("token", regex=True)["count"] == 1
        assert self._run("token", regex=True, case_sensitive=True)["count"] == 0

    def test_regex_respects_include_patterns(self):
        result = self._run("^def ", regex=True, include_patterns=["*.py"])
        assert {(r["file"], r["line"]) for r in result["results"]} == {
            ("a.py", 1),
            ("a.py", 4),
        }
        result = self._run("^def ", regex=True, include_patterns=["*.md"])
        assert result["count"] == 1

    def test_invalid_regex_is_an_error_not_a_fallback(self):
        result = self._run("def (", regex=True)
        assert "error" in result
        assert "invalid regex" in result["error"]
        assert "regex" in result["error"]

    def test_multiline_requires_regex(self):
        result = self._run("foo", multiline=True)
        assert "error" in result
        assert "requires 'regex'" in result["error"]

    # --- multiline ---

    def test_multiline_matches_across_newlines(self):
        self._write("d.py", "alpha\nbeta\ngamma\n")
        assert self._run("beta\\ngamma", regex=True, multiline=True)["count"] == 1
        # Without multiline the pattern cannot match at all, since no single
        # line contains it.
        assert self._run("beta\\ngamma", regex=True)["count"] == 0

    def test_multiline_reports_the_first_line_of_the_match(self):
        self._write("d.py", "alpha\nbeta\ngamma\n")
        result = self._run("beta\\ngamma", regex=True, multiline=True)
        assert result["results"][0]["line"] == 2

    def test_multiline_context_spans_the_covered_lines(self):
        self._write("d.py", "one\ntwo\nthree\nfour\nfive\n")
        result = self._run(
            "two\\nthree", regex=True, multiline=True, context_lines=1
        )
        ctx = result["results"][0]["context"]
        assert any(line.startswith("> 2:") for line in ctx)
        assert any(line.startswith("> 3:") for line in ctx)
        assert any(line.startswith("  4:") for line in ctx)

    # --- context ---

    def test_context_lines_work_in_regex_mode(self):
        result = self._run("^def two", regex=True, context_lines=1)
        ctx = result["results"][0]["context"]
        # Only the matching line is marked with '>'; the context is unprefixed.
        assert "> 4: def two():" in ctx
        assert any(line.startswith("  3:") for line in ctx)
        assert any(line.startswith("  5:") for line in ctx)
        assert len(ctx) == 3

    # --- cap is reported, not silent ---

    def test_omitted_matches_are_counted(self):
        for i in range(10):
            self._write(f"many{i}.py", "needle\n")
        result = self._run("needle", max_results=4)
        assert result["count"] == 4
        assert result["omitted"] == 6
        assert len(result["results"]) == 4
        # The rendered output has to say so: "Found 4" alone reads as "there
        # are 4", and the caller would conclude the pattern is rarer than it is.
        assert "6 further match(es) omitted" in result["content"]

    def test_max_results_counts_every_match_not_just_reported(self):
        for i in range(3):
            self._write(f"many{i}.py", "needle\n")
        result = self._run("needle", max_results=0)
        assert result["count"] == 3

    # --- existing parameters still honoured ---

    def test_max_depth_still_applies(self):
        nested = self.root / "deep" / "deeper"
        nested.mkdir(parents=True)
        (nested / "e.py").write_text("def deep():\n    pass\n")
        assert self._run("def deep", regex=True, max_depth=0)["count"] == 0
        assert self._run("def deep", regex=True, max_depth=2)["count"] == 1

    def test_missing_directory_is_an_error(self):
        result = self.handler(
            {"root": str(self._tmpdir / "nope"), "query": "x", "regex": True}
        )
        assert "error" in result

    def test_query_is_required(self):
        result = self.handler({"root": str(self.root)})
        assert "error" in result

    def test_binary_files_are_still_skipped(self):
        self._write("real.py", "needle\n")
        (self.root / "blob.bin").write_bytes(b"\x00needle\x00")
        result = self._run("needle")
        assert result["count"] == 1
        assert result["results"][0]["file"] == "real.py"
