"""Tests for tools/files.py: read_file, patch_file and search_content."""

from typing import cast

from tests.conftest import BaseTest
from agent.tools import discover_tools, get_registry
from agent.output import set_output, OutputAdapter


class _FakeOutput:
    """Minimal OutputAdapter stub that swallows all output."""

    def __init__(self, answer: bool = True):
        self.answer = answer
        self.confirmations: list[str] = []

    def print(self, *args, **kwargs):
        pass

    def err(self, *args, **kwargs):
        pass

    def ok(self, *args, **kwargs):
        pass

    def newline(self, *args, **kwargs):
        pass

    def ask_confirm(self, message, default=False):
        # Auto-confirms: these exercise the handlers, not the boundary policy
        # (see tests/test_tool_confirmations.py).
        self.confirmations.append(message)
        return self.answer


class _HandlerCase(BaseTest):
    """Base for the handler tests: resolve one tool and silence output."""

    tool = ""
    output_cls = _FakeOutput

    def setUp(self):
        super().setUp()
        # The @tool decorator does not return the function, so grab the real
        # handler from the registry.
        discover_tools()
        self.handler = get_registry()[self.tool]["handler"]
        self.output = self.output_cls()
        set_output(cast(OutputAdapter, self.output))
        self.addCleanup(set_output, None)


class TestReadFile(_HandlerCase):
    tool = "read_file"

    def test_a_whole_file_read_carries_no_truncation_metadata(self):
        path = self._write_file("full.txt", "line1\nline2\nline3\n")
        result = self.handler({"path": str(path)})
        assert result["content"] == "line1\nline2\nline3\n"
        assert "truncated" not in result
        for bad in (str(self._tmpdir / "nope.txt"), ""):
            assert "error" in self.handler({"path": bad}), bad

    def test_max_lines_reads_a_head_and_numbers_keep_the_real_positions(self):
        path = self._write_file("n.txt", "a\nb\nc\nd\ne\n")
        result = self.handler({"path": str(path), "max_lines": 2})
        assert (result["content"], result["truncated"]) == ("a\nb\n", True)
        assert (result["total_lines"], result["shown_lines"]) == (5, 2)
        result = self.handler({"path": str(path), "show_line_numbers": True})
        assert result["content"] == "1: a\n2: b\n3: c\n4: d\n5: e"


class TestReadFileOffset(_HandlerCase):
    tool = "read_file"

    def _lines(self, name="numbered.txt", n=20):
        path = self._tmpdir / name
        path.write_text("".join(f"line{i}\n" for i in range(1, n + 1)))
        return path

    def test_offset_and_max_lines_compose_as_a_window(self):
        """`offset` is where to start, `max_lines` is how many — not a range."""
        path = self._lines()
        result = self.handler({"path": str(path), "offset": 5, "max_lines": 3})
        assert result["content"] == "line5\nline6\nline7\n"
        assert (result["start_line"], result["end_line"]) == (5, 7)
        assert (result["truncated"], result["total_lines"], result["shown_lines"]) == (
            True, 20, 3
        )
        # Offset alone reads to the end; past the end is an empty read.
        result = self.handler({"path": str(path), "offset": 18})
        assert result["content"] == "line18\nline19\nline20\n"
        assert result["end_line"] == 20
        result = self.handler({"path": str(path), "offset": 50})
        assert (result["content"], result["start_line"]) == ("", 50)

    def test_numbers_reflect_the_file_not_the_window(self):
        """1..n within a window would be useless as an anchor for an edit."""
        path = self._lines(n=200)
        result = self.handler(
            {"path": str(path), "offset": 98, "max_lines": 3, "show_line_numbers": True}
        )
        assert result["content"] == " 98: line98\n 99: line99\n100: line100"

    def test_numbers_may_arrive_as_strings_and_bad_ones_are_ignored(self):
        path = self._lines()
        result = self.handler({"path": str(path), "offset": "5", "max_lines": "2"})
        assert result["content"] == "line5\nline6\n"
        # A negative offset is ignored, not read as `tail -n -N`.
        result = self.handler({"path": str(path), "offset": -5, "max_lines": 2})
        assert result["content"] == "line1\nline2\n"
        for key in ("offset", "max_lines"):
            result = self.handler({"path": str(path), "offset": 1, key: "not-a-number"})
            assert result["content"].startswith("line1\n"), key


class TestPatchFile(_HandlerCase):
    tool = "patch_file"

    def _file(self, text, name="f.py"):
        return self._write_file(name, text)

    def test_replaces_a_block_and_forgives_trailing_whitespace(self):
        path = self._file("def f():\n    return 1\n")
        result = self.handler({
            "path": str(path),
            "old_string": "def f():\n    return 1",
            "new_string": "def f():\n    return 2",
        })
        assert (result["success"], result["changed"]) == (True, True)
        assert path.read_text() == "def f():\n    return 2\n"
        # Trailing whitespace in the stored file is not part of the match.
        path = self._file("a\nb   \nc  \nd   \n", "ws.py")
        result = self.handler(
            {"path": str(path), "old_string": "b\nc\nd", "new_string": "x\ny\nz"}
        )
        assert result["success"] is True
        assert path.read_text() == "a\nx\ny\nz\n"

    def test_indentation_is_not_forgiven(self):
        """Leading whitespace is meaning: forgiving it moves a block in or out
        of a function body."""
        path = self._file("    beta\n")
        result = self.handler(
            {"path": str(path), "old_string": "beta", "new_string": "BETA"}
        )
        assert "error" in result
        assert path.read_text() == "    beta\n"

    def test_an_ambiguous_match_names_the_lines_a_window_disambiguates(self):
        path = self._file("x\ndup\ny\ndup\n")
        result = self.handler(
            {"path": str(path), "old_string": "dup", "new_string": "DUP"}
        )
        assert "2, 4" in result["error"]
        assert path.read_text() == "x\ndup\ny\ndup\n"

        # The window is a search range, inclusive at both ends.
        result = self.handler({
            "path": str(path), "old_string": "dup", "new_string": "DUP",
            "start_line": 1, "end_line": 3,
        })
        assert result["success"] is True
        assert path.read_text() == "x\nDUP\ny\ndup\n"

        # A window that excludes the match names where the text really is.
        path = self._file("x\ndup\ny\ndup\n", "win.py")
        result = self.handler({
            "path": str(path), "old_string": "dup", "new_string": "DUP",
            "start_line": 3, "end_line": 3,
        })
        assert "within lines 3-3" in result["error"]
        assert "line(s) 2, 4" in result["error"]

    def test_a_missing_match_names_line_mode_and_a_no_op_is_not_rewritten(self):
        path = self._file("alpha\n")
        result = self.handler(
            {"path": str(path), "old_string": "nope", "new_string": "x"}
        )
        assert "start_line" in result["error"]
        # Deleting is `new_string: ""`, so a missing old_string is different.
        for args in ({"old_string": "", "new_string": ""},
                     {"old_string": "alpha"}):
            assert "error" in self.handler({"path": str(path), **args})
        result = self.handler(
            {"path": str(path), "old_string": "alpha", "new_string": "alpha"}
        )
        assert (result["success"], result["changed"]) == (True, False)
        assert path.read_text() == "alpha\n"

    def test_line_mode_replaces_a_range_without_matching_text(self):
        """The characters do not have to be right, only the line numbers."""
        path = self._file("a\nb\nc\nd\ne\n")
        result = self.handler(
            {"path": str(path), "start_line": 2, "end_line": 3, "new_string": "X\nY"}
        )
        assert (result["start_line"], result["end_line"]) == (2, 3)
        assert (result["removed_lines"], result["added_lines"]) == (2, 2)
        assert path.read_text() == "a\nX\nY\nd\ne\n"

        path = self._file("a\n        b\nc\n", "indent.py")
        self.handler({"path": str(path), "start_line": 2, "new_string": "\tb"})
        assert path.read_text() == "a\n\tb\nc\n"

    def test_an_empty_new_string_deletes_the_lines(self):
        path = self._file("a\nb\nc\nd\n")
        result = self.handler(
            {"path": str(path), "start_line": 2, "end_line": 3, "new_string": ""}
        )
        assert result["success"] is True
        assert path.read_text() == "a\nd\n"
        # Deleting every line leaves an empty file, not a stray newline.
        path = self._file("a\nb\n", "all.py")
        self.handler({"path": str(path), "start_line": 1, "end_line": 2, "new_string": ""})
        assert path.read_text() == ""

    def test_line_ranges_are_validated_and_a_stale_end_is_clamped(self):
        path = self._file("a\nb\nc\n")
        # A start past the end is a stale read, not an append (line 4 is).
        result = self.handler({"path": str(path), "start_line": 50, "new_string": "x"})
        assert "3 lines" in result["error"]
        result = self.handler({"path": str(path), "start_line": 2, "end_line": 99,
                               "new_string": "B"})
        assert result["success"] is True
        assert path.read_text() == "a\nB\n"
        # A reversed range, or one that does not parse, is refused: a failed
        # parse must not mean "line 0" and rewrite the whole file.
        for args in ({"start_line": 3, "end_line": 2}, {"start_line": "not-a-number"}):
            path = self._file("a\nb\nc\n", "again.py")
            assert "error" in self.handler({"path": str(path), "new_string": "x", **args})
            assert path.read_text() == "a\nb\nc\n"

    def test_numbers_as_strings_insert_at_the_end_and_a_missing_newline_holds(self):
        path = self._file("a\nb\nc\n")
        self.handler({"path": str(path), "start_line": "2", "end_line": "2",
                      "new_string": "B"})
        assert path.read_text() == "a\nB\nc\n"
        # start_line = total + 1 appends.
        path = self._file("a\nb\n", "tail.py")
        self.handler({"path": str(path), "start_line": 3, "new_string": "c\n"})
        assert path.read_text() == "a\nb\nc\n"
        # Gaining a trailing newline would be a whole-file diff.
        path = self._file("a\nb", "nonl.py")
        self.handler({"path": str(path), "start_line": 2, "new_string": "B\nC"})
        assert path.read_text() == "a\nB\nC"


class TestSearchContent(_HandlerCase):
    tool = "search_content"

    def setUp(self):
        super().setUp()
        self.root = self._tmpdir / "src"
        self.root.mkdir(parents=True, exist_ok=True)
        self._write("a.py", "def one():\n    pass\n\ndef two():\n    return 2\n")
        self._write("b.py", "    self.value = 1\n    self.other = 2\n")
        self._write("notes.md", "# Notes\n\ndef one() is mentioned here too\n")

    def _write(self, name, text):
        path = self.root / name
        path.write_text(text)
        return path

    def _run(self, query, **kw):
        return self.handler({"root": str(self.root), "query": query, **kw})

    def test_a_literal_search_is_the_default(self):
        """Metacharacters stay literal, so `config.yaml` matches itself."""
        result = self._run("self.value")
        assert result["mode"] == "text"
        assert [(r["file"], r["line"]) for r in result["results"]] == [("b.py", 1)]
        self._write("config.yaml", "name: configXyaml\nother: config.yaml\n")
        result = self._run("config.yaml")
        assert result["count"] == 1 and "other" in result["results"][0]["line_content"]
        # And a regex-shaped query finds nothing literal.
        assert self._run("^def ")["count"] == 0

    def test_a_regex_anchors_per_line_and_honours_case(self):
        """`^def ` is the whole reason regex mode exists."""
        result = self._run("^def ", regex=True)
        assert result["mode"] == "regex"
        assert {(r["file"], r["line"]) for r in result["results"]} == {
            ("a.py", 1), ("a.py", 4), ("notes.md", 3),
        }
        result = self._run(r"self\.\w+ = ", regex=True)
        assert result["count"] == 2 and all(r["file"] == "b.py" for r in result["results"])
        self._write("c.py", "TOKEN = 1\n")
        assert self._run("token", regex=True)["count"] == 1
        assert self._run("token", regex=True, case_sensitive=True)["count"] == 0

    def test_patterns_select_files_and_an_invalid_regex_is_an_error(self):
        result = self._run("^def ", regex=True, include_patterns=["*.md"])
        assert [r["file"] for r in result["results"]] == ["notes.md"]
        # Never a silent fallback to a literal search: that is a typo worth
        # reporting, not a query to reinterpret.
        assert "invalid regex" in self._run("def (", regex=True)["error"]

    def test_multiline_requires_regex_and_spans_lines(self):
        assert "requires 'regex'" in self._run("foo", multiline=True)["error"]
        self._write("d.py", "alpha\nbeta\ngamma\n")
        assert self._run("beta\\ngamma", regex=True)["count"] == 0
        result = self._run("beta\\ngamma", regex=True, multiline=True)
        assert result["results"][0]["line"] == 2  # the first line of the match

    def test_context_lines_and_a_cap_that_is_reported_not_silent(self):
        result = self._run("^def two", regex=True, context_lines=1)
        assert result["results"][0]["context"] == [
            "  3: ", "> 4: def two():", "  5:     return 2"
        ]
        for i in range(10):
            self._write(f"many{i}.py", "needle\n")
        result = self._run("needle", max_results=4)
        assert (result["count"], result["omitted"]) == (4, 6)
        # "Found 4" alone reads as "there are 4".
        assert "6 further match(es) omitted" in result["content"]
        assert self._run("needle", max_results=0)["count"] == 10

    def test_depth_binary_files_and_missing_arguments(self):
        nested = self.root / "deep" / "deeper"
        nested.mkdir(parents=True)
        (nested / "e.py").write_text("def deep():\n    pass\n")
        assert self._run("def deep", regex=True, max_depth=0)["count"] == 0
        assert self._run("def deep", regex=True, max_depth=2)["count"] == 1

        self._write("real.py", "needle\n")
        (self.root / "blob.bin").write_bytes(b"\x00needle\x00")
        assert [r["file"] for r in self._run("needle")["results"]] == ["real.py"]

        assert "error" in self.handler({"root": str(self._tmpdir / "nope"), "query": "x"})
        assert "error" in self.handler({"root": str(self.root)})


class _CapturingOutput(_FakeOutput):
    """Records every printed line instead of discarding it."""

    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def print(self, *args, **kwargs):
        self.lines.append(" ".join(str(a) for a in args))


class TestPatchFileDiffOutput(_HandlerCase):
    """`patch_file` prints a diff, not the whole block it replaced."""

    tool = "patch_file"
    output_cls = _CapturingOutput

    @property
    def lines(self) -> list[str]:
        return cast(_CapturingOutput, self.output).lines

    def _printed(self, tag: str) -> list[str]:
        # The number sits in its own [weak] tag, so the marker is not at the
        # start of the line: find it anywhere.
        return [line for line in self.lines if f"[{tag}]" in line]

    def test_only_the_changed_lines_are_printed_on_their_own_side(self):
        path = self._write_file("f.py", "".join(f"line{i}\n" for i in range(1, 11)))
        self.handler({"path": str(path), "old_string": "line5", "new_string": "LINE5"})
        assert self._printed("patch-remove") == [
            "[weak]5|[/weak] [patch-remove]- line5[/patch-remove]"
        ]
        assert self._printed("patch-add") == [
            "[weak]5|[/weak] [patch-add]+ LINE5[/patch-add]"
        ]
        # No line of the untouched surrounding block is reprinted.
        printed = "".join(self.lines)
        assert "line4" not in printed and "line6" not in printed

        path = self._write_file("f.py", "a\nb\nc\n")
        self.lines.clear()
        self.handler({"path": str(path), "start_line": 2, "new_string": ""})
        assert self._printed("patch-add") == []
        assert self._printed("patch-remove") == [
            "[weak]2|[/weak] [patch-remove]- b[/patch-remove]"
        ]
