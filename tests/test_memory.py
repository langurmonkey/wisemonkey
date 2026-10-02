"""Tests for agent/memory.py — Memory singleton, ChatMemory persistence, trimming."""

import json

from tests.conftest import BaseTest
from agent.memory import Memory, ChatMemory
from agent.tokens import count_tokens


class TestMemoryInit(BaseTest):
    """Test Memory initialization and session directory creation."""

    def test_creates_session_dir(self):
        session_dir = self._tmpdir / "sessions" / "test"
        m = Memory(session_dir=session_dir)
        assert session_dir.exists()
        assert session_dir.is_dir()

    def test_creates_metadata_file(self):
        session_dir = self._tmpdir / "sessions" / "meta"
        m = Memory(session_dir=session_dir)
        metadata_file = session_dir / ".session-metadata"
        assert metadata_file.exists()

    def test_metadata_contains_created_and_accessed(self):
        session_dir = self._tmpdir / "sessions" / "meta2"
        m = Memory(session_dir=session_dir)
        content = (session_dir / ".session-metadata").read_text()
        assert "created:" in content
        assert "accessed:" in content


class TestMemoryUserProfile(BaseTest):
    """Test user profile get/set."""

    def setUp(self):
        super().setUp()
        session_dir = self._tmpdir / "sessions" / "profile"
        self.m = Memory(session_dir=session_dir)

    def test_get_default_profile(self):
        assert self.m.get_user_profile() == {}

    def test_set_profile(self):
        self.m.set_user_profile({"name": "Alice"})
        assert self.m.get_user_profile()["name"] == "Alice"

    def test_set_profile_merges(self):
        self.m.set_user_profile({"name": "Alice"})
        self.m.set_user_profile({"role": "dev"})
        profile = self.m.get_user_profile()
        assert profile["name"] == "Alice"
        assert profile["role"] == "dev"

    def test_profile_persists(self):
        self.m.set_user_profile({"name": "Bob"})
        self.m.save()
        data = json.loads((self.m._user_profile_path).read_text())
        assert data["name"] == "Bob"


class TestMemoryNotes(BaseTest):
    """Test note adding and retrieval."""

    def setUp(self):
        super().setUp()
        session_dir = self._tmpdir / "sessions" / "notes"
        self.m = Memory(session_dir=session_dir)

    def test_get_default_notes(self):
        assert self.m.get_notes() == []

    def test_add_note(self):
        note = self.m.add_note("Remember this", category="reminder")
        assert note["content"] == "Remember this"
        assert note["category"] == "reminder"
        assert note["id"] == 1

    def test_add_multiple_notes(self):
        self.m.add_note("First")
        self.m.add_note("Second")
        notes = self.m.get_notes()
        assert len(notes) == 2
        assert notes[0]["id"] == 1
        assert notes[1]["id"] == 2

    def test_notes_persist(self):
        self.m.add_note("Persistent note")
        self.m.save()
        data = json.loads((self.m._notes_path).read_text())
        assert len(data) == 1
        assert data[0]["content"] == "Persistent note"


class TestMemoryFormatted(BaseTest):
    """Test get_formatted output."""

    def setUp(self):
        super().setUp()
        session_dir = self._tmpdir / "sessions" / "fmt"
        self.m = Memory(session_dir=session_dir)

    def test_empty_returns_none(self):
        assert self.m.get_user_profile_formatted() is None

    def test_user_profile_formatted(self):
        self.m.set_user_profile({"name": "Alice"})
        result = self.m.get_user_profile_formatted(notes=False)
        assert "## User Profile" in result
        assert "name: Alice" in result

    def test_notes_formatted(self):
        self.m.add_note("Test note", category="test")
        result = self.m.get_user_profile_formatted(user_profile=False)
        assert "## Persistent Notes" in result
        assert "[test] Test note" in result


class TestChatMemory(BaseTest):
    """Test ChatMemory add, trim, clear, persistence."""

    def setUp(self):
        super().setUp()
        self.session_dir = self._tmpdir / "sessions" / "chat"
        self.cm = ChatMemory(self.session_dir, max_tokens=200)

    def test_add_exchange(self):
        self.cm.add_exchange(None, "user", "Hello")
        assert len(self.cm._exchanges) == 1
        assert self.cm._exchanges[0]["role"] == "user"
        assert self.cm._exchanges[0]["content"] == "Hello"

    def test_add_multiple_exchanges(self):
        self.cm.add_exchange(None, "user", "Hi")
        self.cm.add_exchange(None, "assistant", "Hello!")
        assert len(self.cm._exchanges) == 2

    def test_total_tokens_tracks_content(self):
        self.cm.add_exchange(None, "user", "abc def ghi")
        self.cm.add_exchange(None, "assistant", "one two three four")
        # Exact token counts (tiktoken), so just check they are positive and
        # grow with content.
        assert self.cm.total_tokens > 0
        before = self.cm.total_tokens
        self.cm.add_exchange(None, "user", "more text here")
        assert self.cm.total_tokens > before

    def test_trim_removes_oldest(self):
        # Directly test _trim() instead of add_exchange (which triggers
        # /history-compact via the command registry and needs a real core).
        for i in range(20):
            self.cm._exchanges.append({"role": "user", "content": "x" * 50})
            self.cm.total_tokens += 12
        self.cm._trim()
        assert self.cm.total_tokens <= 200

    def test_clear_n_exchanges(self):
        for i in range(5):
            self.cm.add_exchange(None, "user", f"msg {i}")
        cleared = self.cm._clear(2)
        assert cleared == 2
        assert len(self.cm._exchanges) == 3

    def test_clear_all_with_zero(self):
        for i in range(3):
            self.cm.add_exchange(None, "user", f"msg {i}")
        cleared = self.cm._clear(0)
        assert cleared == 3
        assert len(self.cm._exchanges) == 0

    def test_clear_more_than_exists(self):
        self.cm.add_exchange(None, "user", "only one")
        cleared = self.cm._clear(10)
        assert cleared == 1

    def test_persistence_round_trip(self):
        self.cm.add_exchange(None, "user", "persist me")
        self.cm.save()

        cm2 = ChatMemory(self.session_dir, max_tokens=200)
        assert len(cm2._exchanges) == 1
        assert cm2._exchanges[0]["content"] == "persist me"

    def test_get_formatted(self):
        self.cm.add_exchange(None, "user", "Hello")
        self.cm.add_exchange(None, "assistant", "Hi there")
        result = self.cm.get_formatted(2, timestamps=False, width=0)
        assert "## User:" in result
        assert "## Assistant:" in result
        assert "Hello" in result
        assert "Hi there" in result

    def test_get_formatted_empty(self):
        assert self.cm.get_formatted(0, timestamps=False, width=0) is None

    def test_get_formatted_with_width(self):
        long_content = "a" * 200
        self.cm.add_exchange(None, "user", long_content)
        result = self.cm.get_formatted(1, timestamps=False, width=50)
        # Should be truncated
        assert len(result) < len(long_content) + 100

    def test_tool_call_exchange(self):
        self.cm.add_exchange(
            None, "tool_call", "", name="read_file", arguments='{"path": "x.py"}'
        )
        result = self.cm.get_formatted(0, timestamps=False, width=0)
        assert "## Tool Call (read_file):" in result
        assert "x.py" in result

    def test_tool_result_exchange_rendered(self):
        self.cm.add_exchange(None, "tool_result", "result body", name="grep")
        result = self.cm.get_formatted(0, timestamps=False, width=0)
        assert "## Tool Result (grep):" in result
        assert "result body" in result

    def test_tool_result_truncated_by_default(self):
        cm = ChatMemory(self.session_dir / "big", max_tokens=25000)
        long_result = "z" * 5000
        cm.add_exchange(None, "tool_result", long_result, name="big")
        # Content is truncated at storage time to the config limit (200).
        stored = cm._exchanges[-1]["content"]
        assert len(stored) == 200
        assert stored == "z" * 200

    def test_tool_call_counts_extra_tokens(self):
        self.cm.add_exchange(None, "user", "hi")
        before = self.cm.total_tokens
        self.cm.add_exchange(None, "tool_call", "", name="x", arguments="abcd")
        # role/utc aren't counted, but the extra string fields are
        assert self.cm.total_tokens > before

    def test_large_tool_result_counted_as_truncated(self):
        """A huge tool result must not inflate total_tokens beyond the limit.

        total_tokens tracks what is actually injected into the prompt, where
        tool results are truncated to chat_history_tool_result_max_chars.
        Otherwise a single big read_file would trigger premature compaction.
        """
        cm = ChatMemory(self.session_dir / "big", max_tokens=25000)
        cm.add_exchange(None, "tool_result", "z" * 100000, name="read_file")
        # Truncated to 200 chars -> well under 25000 tokens.
        assert cm.total_tokens < 500

    def test_adjacent_tool_call_and_result_compact_into_one_block(self):
        self.cm.add_exchange(
            None,
            "tool_call",
            "",
            name="read_file",
            arguments='{"path":"a.py"}',
            tool_call_id="call-1",
        )
        self.cm.add_exchange(
            None,
            "tool_result",
            "file contents",
            name="read_file",
            tool_call_id="call-1",
        )
        rendered = self.cm.get_formatted(0, timestamps=False, width=0)
        assert "## Tool: read_file" in rendered
        assert 'Args: {"path":"a.py"}' in rendered
        assert "Result: file contents" in rendered
        assert "## Tool Call" not in rendered
        assert "## Tool Result" not in rendered

    def test_unmatched_tool_entries_remain_separate(self):
        self.cm.add_exchange(
            None,
            "tool_call",
            "",
            name="read_file",
            arguments="{}",
            tool_call_id="call-1",
        )
        self.cm.add_exchange(
            None,
            "tool_result",
            "other result",
            name="read_file",
            tool_call_id="call-2",
        )
        rendered = self.cm.get_formatted(0, timestamps=False, width=0)
        assert "## Tool Call (read_file):" in rendered
        assert "## Tool Result (read_file):" in rendered
        assert "## Tool: read_file" not in rendered

    def test_token_accounting_matches_exact_rendered_prompt_history(self):
        self.cm.add_exchange(None, "user", "hello")
        self.cm.add_exchange(
            None,
            "tool_call",
            "",
            name="read_file",
            arguments='{"path":"a.py"}',
            tool_call_id="call-1",
        )
        self.cm.add_exchange(
            None,
            "tool_result",
            "contents",
            name="read_file",
            tool_call_id="call-1",
        )
        self.cm.add_exchange(None, "assistant", "done")
        formatted = self.cm.get_formatted(0, timestamps=False, width=0)
        assert self.cm.total_tokens == count_tokens(formatted or "")

    def test_clear_recounts_rendered_history_tokens(self):
        for content in ("old exchange", "new exchange"):
            self.cm.add_exchange(None, "user", content)
        self.cm._clear(1)
        formatted = self.cm.get_formatted(0, timestamps=False, width=0)
        assert self.cm.total_tokens == count_tokens(formatted or "")

    def test_get_unformatted(self):
        self.cm.add_exchange(None, "user", "raw")
        raw = self.cm.get_unformatted()
        assert isinstance(raw, list)
        assert raw[0]["content"] == "raw"


class TestChatMemoryWindow(BaseTest):
    """Test the turn-based rolling window (window_turns)."""

    def setUp(self):
        super().setUp()
        self.session_dir = self._tmpdir / "sessions" / "window"

    def _fill(self, cm, n):
        for i in range(n):
            cm.add_exchange(None, "user", f"msg {i}")
            cm.add_exchange(None, "assistant", f"reply {i}")

    def test_window_zero_keeps_all(self):
        cm = ChatMemory(self.session_dir, max_tokens=10**9, window_turns=0)
        self._fill(cm, 4)
        assert len(cm._exchanges) == 8

    def test_window_keeps_last_n_exchanges(self):
        cm = ChatMemory(self.session_dir, max_tokens=10**9, window_turns=2)
        self._fill(cm, 4)
        # 2 exchanges x 2 entries each.
        assert len(cm._exchanges) == 4
        assert cm._exchanges[0]["content"] == "msg 2"
        assert cm._exchanges[-1]["content"] == "reply 3"

    def test_window_includes_tool_entries(self):
        cm = ChatMemory(self.session_dir, max_tokens=10**9, window_turns=1)
        cm.add_exchange(None, "user", "msg 0")
        cm.add_exchange(None, "assistant", "reply 0")
        cm.add_exchange(None, "tool_call", "call", name="t", arguments="{}")
        cm.add_exchange(None, "tool_result", "res", name="t")
        cm.add_exchange(None, "user", "msg 1")
        cm.add_exchange(None, "assistant", "reply 1")
        # Only the last exchange (msg 1 + reply 1) survives.
        assert len(cm._exchanges) == 2
        assert cm._exchanges[0]["content"] == "msg 1"

    def test_load_trims_oversized_history(self):
        cm = ChatMemory(self.session_dir, max_tokens=10**9, window_turns=0)
        self._fill(cm, 4)
        cm.save()
        # Reopen with a smaller window: startup reconciliation trims.
        cm2 = ChatMemory(self.session_dir, max_tokens=10**9, window_turns=2)
        assert len(cm2._exchanges) == 4
        assert cm2._exchanges[0]["content"] == "msg 2"


class TestChatMemoryDropLast(BaseTest):
    """`drop_last` removes whole exchanges from the tail of the history.

    The unit under test is the real ChatMemory, not a stand-in: the subtlety
    is which entries count as an exchange, and only the real implementation
    knows that.
    """

    def setUp(self):
        super().setUp()
        self.session_dir = self._tmpdir / "sessions" / "droplast"

    def _fill(self) -> ChatMemory:
        cm = ChatMemory(self.session_dir, max_tokens=10**9)
        for role, content in [
            ("user", "q1"),
            ("assistant", "a1"),
            ("user", "q2"),
            ("assistant", "a2"),
            ("tool_call", "{}"),
            ("tool_result", "42"),
            ("user", "q3"),
            ("assistant", "a3"),
        ]:
            cm._exchanges.append({"role": role, "content": content})
        cm._recount_tokens()
        return cm

    def _roles(self, cm: ChatMemory) -> list[str]:
        return [e["role"] for e in cm._exchanges]

    def test_drops_last_exchange_only(self):
        cm = self._fill()
        removed = cm.drop_last(1)
        assert removed == 2
        assert self._roles(cm) == [
            "user", "assistant", "user", "assistant",
            "tool_call", "tool_result",
        ]

    def test_keeps_tool_calls_with_their_exchange(self):
        cm = self._fill()
        cm.drop_last(2)
        assert self._roles(cm) == ["user", "assistant"]

    def test_n_zero_drops_everything(self):
        cm = self._fill()
        removed = cm.drop_last(0)
        assert removed == 8
        assert cm._exchanges == []

    def test_n_beyond_history_clears_it(self):
        cm = self._fill()
        assert cm.drop_last(99) == 8
        assert cm._exchanges == []

    def test_empty_history(self):
        cm = ChatMemory(self.session_dir, max_tokens=10**9)
        assert cm.drop_last(1) == 0

    def test_recounts_and_persists(self):
        cm = self._fill()
        before = cm.total_tokens
        cm.drop_last(1)
        assert cm.total_tokens < before
        # Reloaded from disk, so the drop survived the session.
        reopened = ChatMemory(self.session_dir, max_tokens=10**9)
        assert len(reopened._exchanges) == 6

    def test_leading_orphan_is_not_mistaken_for_an_exchange(self):
        """A stray tool result before any user message is not a turn start."""
        cm = ChatMemory(self.session_dir, max_tokens=10**9)
        cm._exchanges = [
            {"role": "tool_result", "content": "leftover"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        assert cm.drop_last(1) == 2
        assert [e["role"] for e in cm._exchanges] == [
            "tool_result", "user", "assistant",
        ]

    def test_dropping_the_only_exchange_clears_the_orphan_too(self):
        """Nothing coherent is left, so nothing is kept -- orphan included."""
        cm = ChatMemory(self.session_dir, max_tokens=10**9)
        cm._exchanges = [
            {"role": "tool_result", "content": "leftover"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        assert cm.drop_last(1) == 3
        assert cm._exchanges == []

    def test_last_user_prompt(self):
        cm = self._fill()
        assert cm.last_user_prompt() == "q3"

    def test_last_user_prompt_ignores_injections_after_the_prompt(self):
        cm = self._fill()
        cm._exchanges.append({"role": "user", "content": "steered mid-turn"})
        assert cm.last_user_prompt() == "steered mid-turn"

    def test_last_user_prompt_on_empty(self):
        assert ChatMemory(
            self.session_dir, max_tokens=10**9).last_user_prompt() == ""
