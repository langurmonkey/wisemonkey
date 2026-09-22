"""Tests for @-file reference expansion (agent/at_files.py)."""

from pathlib import Path

from tests.conftest import BaseTest

from agent.at_files import (
    _is_filelike,
    expand_at_references,
    find_at_references,
)


class TestFindAtReferences(BaseTest):
    def test_path_with_separator(self):
        assert find_at_references("see @agent/agent.py") == ["agent/agent.py"]

    def test_backtick_quoted(self):
        assert find_at_references("see @`README.md`") == ["README.md"]

    def test_home_path(self):
        refs = find_at_references("look at @~/.bashrc")
        assert refs == ["~/.bashrc"]

    def test_ignores_handles(self):
        # @handle without extension or separator and not existing on disk.
        assert find_at_references("ping @langur about a@b.com") == []

    def test_bare_existing_dir(self):
        refs = find_at_references("what is in @tools")
        assert refs == ["tools"]

    def test_bare_nonexistent_word(self):
        assert find_at_references("hello @world") == []


class TestExpandAtReferences(BaseTest):
    def _tmp_file(self, name: str, content: str) -> Path:
        p = self.tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("line1\nline2\n")
        return p

    def setUp(self):
        super().setUp()
        import tempfile

        self.tmp_path = Path(tempfile.mkdtemp())

    def test_small_file_inlined(self):
        self._tmp_file("small.py", "")
        out = expand_at_references(f"check @{self.tmp_path}/small.py", 8000)
        assert "Attached context" in out
        assert "line1" in out
        assert "```py" in out

    def test_original_text_preserved(self):
        self._tmp_file("a.md", "")
        out = expand_at_references(f"read @{self.tmp_path}/a.md", 8000)
        assert out.startswith(f"read @{self.tmp_path}/a.md")

    def test_large_file_direct_stub(self):
        p = self._tmp_file("big.py", "")
        p.write_text("y" * 20000)
        out = expand_at_references(f"see @{p}", 100)
        assert "too large to inline" in out
        assert "use `read_file`" in out

    def test_directory_listing(self):
        self._tmp_file("d/one.py", "")
        self._tmp_file("d/two.py", "")
        out = expand_at_references(f"what is in @{self.tmp_path}/d", 8000)
        assert "(directory)" in out
        assert "- one.py" in out

    def test_nonexistent_untouched(self):
        text = "hello @nonexistent_file_xyz.py"
        assert expand_at_references(text, 8000) == text

    def test_dedup(self):
        self._tmp_file("dup.py", "")
        ref = f"@{self.tmp_path}/dup.py"
        out = expand_at_references(f"{ref} and {ref}", 8000)
        assert out.count("**Attached context") == 1

    def test_email_not_expanded_even_with_existing_prefix(self):
        # a@b.com: 'b.com' has an extension-like suffix but doesn't exist.
        text = "mail me at a@b.com"
        assert expand_at_references(text, 8000) == text

    def test_email_not_matched(self):
        text = "mail me at a@b.com"
        assert expand_at_references(text, 8000) == text


class TestIsFilelike(BaseTest):
    def test_extension(self):
        assert _is_filelike("script.py")
        assert _is_filelike("notes.md")

    def test_separator(self):
        assert _is_filelike("agent/core.py")
        assert _is_filelike("a/b")

    def test_tilde(self):
        assert _is_filelike("~/.bashrc")

    def test_bare_existing(self, tmp_path=None):
        # 'tests' exists relative to the project root where tests run.
        assert _is_filelike("tests")

    def test_bare_nonexistent(self):
        assert not _is_filelike("totally_not_here_qq")