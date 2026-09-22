"""Tests for the shared smart path completer (agent/completion.py)."""

from tests.conftest import BaseTest

from agent.completion import SmartPathCompleter, complete_path, _last_token
from prompt_toolkit.document import Document


class TestLastToken(BaseTest):
    def test_simple(self):
        assert _last_token("hello agent/agent.py") == ("agent/agent.py", 6)

    def test_trailing_space_gives_empty_token(self):
        token, offset = _last_token("hello ")
        assert token == ""
        assert offset == 6

    def test_empty(self):
        assert _last_token("") == ("", 0)

    def test_single_token(self):
        assert _last_token("ag") == ("ag", 0)


class TestCompletePath(BaseTest):
    def test_home_dir_expansion(self):
        completions, common = complete_path("look at ~/Pro")
        assert any(c.startswith("~/Projects") for c in completions)
        assert common.startswith("~/Pro")

    def test_tilde_only(self):
        completions, _ = complete_path("~/")
        assert len(completions) > 0
        assert all(c.startswith("~/") for c in completions)

    def test_relative_path(self):
        completions, common = complete_path("agent/ag")
        assert "agent/agent.py" in completions
        assert common.startswith("agent/ag")

    def test_relative_dir_listing(self):
        completions, _ = complete_path("agent/")
        assert len(completions) > 0

    def test_bare_word(self):
        completions, _ = complete_path("look at ag")
        assert "agent/" in completions

    def test_short_bare_word_ignored(self):
        # Single-char bare words are not completed (too noisy).
        completions, _ = complete_path("a")
        assert completions == []

    def test_no_match(self):
        completions, common = complete_path("hello world th")
        assert completions == []
        assert common == ""

    def test_dirs_get_trailing_separator(self):
        completions, _ = complete_path("ag")
        assert any(c.endswith("/") for c in completions)

    def test_tilde_prefix_preserved(self):
        # Completions for ~/ paths must keep the ~, not the expanded home.
        completions, _ = complete_path("~/Pro")
        assert all(c.startswith("~") for c in completions)


class TestSmartPathCompleter(BaseTest):
    def _complete(self, text):
        pc = SmartPathCompleter()
        doc = Document(text=text, cursor_position=len(text))
        return list(pc.get_completions(doc, None))

    def test_mid_sentence_path(self):
        comps = self._complete("look at agent/ag")
        assert any(c.text == "agent/agent.py" for c in comps)

    def test_embed_slash_command_with_path(self):
        comps = self._complete("/embed ~/Doc")
        assert any(c.text.startswith("~/Documents") for c in comps)

    def test_start_position_replaces_token(self):
        comps = self._complete("read agent/agent.p")
        assert comps
        # start_position is negative, equal to the typed token length
        assert comps[0].start_position == -len("agent/agent.p")

    def test_display_is_basename(self):
        comps = self._complete("agent/ag")
        assert comps[0].display_text == "agent.py"


class TestBacktickQuoted(BaseTest):
    def test_split(self):
        from agent.completion import _split_backticks
        assert _split_backticks("`agent/agent.py`") == ("agent/agent.py", "`", "`")
        assert _split_backticks("`agent/ag") == ("agent/ag", "`", "")
        assert _split_backticks("agent/ag") == ("agent/ag", "", "")

    def test_complete_quoted_full(self):
        completions, common = complete_path("in `agent/agent.p`")
        assert completions == ["`agent/agent.py`"]
        assert common == "`agent/agent.py`"

    def test_complete_quoted_open(self):
        completions, _ = complete_path("in `agent/ag")
        assert completions == ["`agent/agent.py"]

    def test_complete_quoted_home(self):
        completions, _ = complete_path("in `~/Pro`")
        assert all(c.startswith("`~/Pro") and c.endswith("`") for c in completions)

    def test_completer_quoted_mid_sentence(self):
        pc = SmartPathCompleter()
        doc = Document(text="in `agent/ag", cursor_position=len("in `agent/ag"))
        comps = list(pc.get_completions(doc, None))
        assert any(c.text == "`agent/agent.py" for c in comps)
        assert comps[0].start_position == -len("`agent/ag")
