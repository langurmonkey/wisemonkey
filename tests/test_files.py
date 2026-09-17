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
