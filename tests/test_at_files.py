"""Tests for @-file reference expansion (agent/at_files.py)."""

from tests.conftest import BaseTest

from agent.at_files import _is_filelike, expand_at_references, find_at_references


class TestFindAtReferences(BaseTest):
    def test_paths_are_found_and_handles_are_not(self):
        assert find_at_references("see @agent/agent.py") == ["agent/agent.py"]
        assert find_at_references("see @`README.md`") == ["README.md"]
        assert find_at_references("look at @~/.bashrc") == ["~/.bashrc"]
        # A bare word counts only when it exists on disk.
        assert find_at_references("what is in @tools") == ["tools"]
        assert find_at_references("hello @world") == []
        assert find_at_references("ping @langur about a@b.com") == []


class TestExpandAtReferences(BaseTest):
    def _file(self, name: str, content: str = "line1\nline2\n"):
        path = self._tmpdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_small_files_are_inlined_and_the_text_kept(self):
        path = self._file("small.py")
        text = f"check @{path}"
        out = expand_at_references(text, 8000)
        assert out.startswith(text)  # the original reference is preserved
        assert "**Attached context" in out
        assert "line1" in out and "```py" in out

    def test_a_large_file_gets_a_stub_pointing_at_read_file(self):
        path = self._file("big.py", "y" * 20000)
        out = expand_at_references(f"see @{path}", 100)
        assert "too large to inline" in out
        assert "use `read_file`" in out

    def test_directories_list_their_entries(self):
        self._file("d/one.py")
        self._file("d/two.py")
        out = expand_at_references(f"what is in @{self._tmpdir}/d", 8000)
        assert "(directory)" in out
        assert "- one.py" in out

    def test_a_repeated_reference_is_inlined_once_and_missing_paths_are_untouched(self):
        path = self._file("dup.py")
        reference = f"@{path}"
        out = expand_at_references(f"{reference} and {reference}", 8000)
        assert out.count("**Attached context") == 1

        for text in ("hello @nonexistent_file_xyz.py", "mail me at a@b.com"):
            assert expand_at_references(text, 8000) == text


class TestIsFilelike(BaseTest):
    def test_extension_separator_tilde_and_existing_bare_words(self):
        assert _is_filelike("script.py")
        assert _is_filelike("notes.md")
        assert _is_filelike("agent/core.py")
        assert _is_filelike("a/b")
        assert _is_filelike("~/.bashrc")
        assert _is_filelike("tests")  # exists relative to the project root
        assert not _is_filelike("totally_not_here_qq")
