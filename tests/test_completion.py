"""Tests for the shared smart path completer (agent/completion.py)."""

from prompt_toolkit.document import Document

from agent.completion import SmartPathCompleter, complete_path, _last_token
from tests.conftest import BaseTest


class TestLastToken(BaseTest):
    def test_splits_the_word_under_the_cursor_from_its_offset(self):
        assert _last_token("hello agent/agent.py") == ("agent/agent.py", 6)
        assert _last_token("hello ") == ("", 6)
        assert _last_token("") == ("", 0)
        assert _last_token("ag") == ("ag", 0)


class TestCompletePath(BaseTest):
    def test_relative_home_and_directory_completions(self):
        completions, common = complete_path("agent/ag")
        assert "agent/agent.py" in completions
        assert common.startswith("agent/ag")
        # Directories get a trailing separator so completion can continue.
        assert any(c.endswith("/") for c in complete_path("ag")[0])
        # A `~/` token keeps the tilde, never the expanded home.
        completions, common = complete_path("look at ~/Pro")
        assert completions and all(c.startswith("~/") for c in completions)
        assert common.startswith("~/Pro")
        assert complete_path("agent/")[0]

    def test_no_match_and_a_too_short_bare_word(self):
        assert complete_path("hello world th") == ([], "")
        # A single-character bare word is not completed: too noisy.
        assert complete_path("a")[0] == []

    def test_backtick_quoting_is_preserved(self):
        assert complete_path("in `agent/agent.p`")[0] == ["`agent/agent.py`"]
        assert complete_path("in `agent/ag")[0] == ["`agent/agent.py"]


class TestSmartPathCompleter(BaseTest):
    def _complete(self, text):
        pc = SmartPathCompleter()
        return list(pc.get_completions(
            Document(text=text, cursor_position=len(text)), None))

    def test_the_typed_token_is_replaced_and_the_basename_displayed(self):
        comps = self._complete("read agent/agent.p")
        assert comps
        assert comps[0].text == "agent/agent.py"
        assert comps[0].display_text == "agent.py"
        assert comps[0].start_position == -len("agent/agent.p")

    def test_a_path_mid_sentence_and_inside_a_quoted_span(self):
        assert any(c.text == "agent/agent.py"
                   for c in self._complete("look at agent/ag"))
        comps = self._complete("in `agent/ag")
        assert any(c.text == "`agent/agent.py" for c in comps)
        assert comps[0].start_position == -len("`agent/ag")
