"""Tests for agent/memory.py — Memory singleton, ChatMemory persistence, trimming."""

import datetime
import json
from pathlib import Path

from tests.conftest import BaseTest
from agent.memory import (
    ChatMemory,
    Memory,
    is_valid_session_name,
    list_sessions,
    session_info,
)
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


class _SessionDirMixin(BaseTest):
    """Point `agent.memory.SESSIONS_DIR` at a temp dir and bind by name.

    Sessions are addressed by name in the product (`wmk <name>`, `/resume
    <name>`); `session_dir` is only a test/injection override and does not set
    the name. Patching the root keeps these tests on the same path as the real
    code, including `switch_session`'s default `SESSIONS_DIR / name`.
    """

    def setUp(self):
        super().setUp()
        import agent.memory as memory_mod

        self.sessions_dir = self._tmpdir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        memory_mod.SESSIONS_DIR = self.sessions_dir
        self.addCleanup(setattr, memory_mod, "SESSIONS_DIR", memory_mod.SESSIONS_DIR)

    def _fresh(self, name: str) -> Memory:
        """A fresh process on *name*.

        `Memory` is a singleton whose ``__init__`` returns early once it has
        been initialized, so a second call in the same test cannot rebind it.
        Reset it explicitly, the way a new `wmk <name>` process would.
        """
        from tests.conftest import reset_singletons

        reset_singletons()
        return Memory(session=name)

    def _write(self, name: str, messages: int = 0, accessed=None) -> Path:
        d = self.sessions_dir / name
        d.mkdir(parents=True, exist_ok=True)
        when = accessed or datetime.datetime.now()
        (d / ".session-metadata").write_text(
            f"created: {when.isoformat()}\naccessed: {when.isoformat()}\n"
        )
        (d / "chat_history.json").write_text(
            json.dumps(
                {"exchanges": [{"role": "user", "content": str(i)} for i in range(messages)]}
            )
        )
        return d


class TestSessionSwitching(_SessionDirMixin):
    """`Memory` is a singleton, so switching a session must rebind it in place.

    Everything downstream (Core, the footer status line, the prompt message)
    holds the same object, so an in-place rebind is the only way a switch can
    be seen by all of them at once.
    """

    def test_switch_reports_the_previous_name(self):
        m = self._fresh("first")
        assert m.session == "first"
        assert m.switch_session("second") == "first"
        assert m.session == "second"

    def test_switch_loads_the_other_sessions_state(self):
        other = self._fresh("second")
        other.add_note("a note from the second session")
        other.set_user_profile({"name": "Langur"})
        # Notes are buffered; a different process only sees them once saved,
        # which is exactly the state a previously used session is left in.
        other.save()

        m = self._fresh("first")
        m.add_note("a note from the first session")
        m.set_user_profile({"name": "Someone Else"})

        m.switch_session("second")
        assert m.session == "second"
        assert [n["content"] for n in m.get_notes()] == ["a note from the second session"]
        assert m.get_user_profile() == {"name": "Langur"}

    def test_the_singleton_is_rebound_not_replaced(self):
        """Everything holding a Memory reference must see the switch."""
        m = self._fresh("first")
        holder = {"memory": m}
        m.switch_session("second")
        assert holder["memory"] is m
        assert holder["memory"].session == "second"

    def test_switch_moves_the_paths_to_the_new_directory(self):
        m = self._fresh("first")
        m.switch_session("second")
        assert m.session_dir == self.sessions_dir / "second"
        assert m._user_profile_path == m.session_dir / "user_profile.json"
        assert m._notes_path == m.session_dir / "notes.json"
        assert m._chat_history._chat_path == m.session_dir / "chat_history.json"

    def test_switch_loads_the_chat_history(self):
        first = self._fresh("first")
        first._chat_history.set_exchanges([{"role": "user", "content": "hello there"}])
        first._chat_history.save()

        m = self._fresh("second")
        assert m.get_chat_history_unformatted() == []
        m.switch_session("first")
        assert m.get_chat_history_unformatted() == [
            {"role": "user", "content": "hello there"}
        ]

    def test_switch_saves_the_previous_session_first(self):
        m = self._fresh("first")
        m.add_note("unsaved when the switch happens")
        first_dir = m.session_dir
        m.switch_session("second")

        # The note must be on disk in the *old* directory, not lost.
        saved = json.loads((first_dir / "notes.json").read_text())
        assert [n["content"] for n in saved] == ["unsaved when the switch happens"]

    def test_switching_to_the_same_session_is_a_noop(self):
        m = self._fresh("first")
        m.add_note("kept")
        assert m.switch_session("first") == "first"
        assert [n["content"] for n in m.get_notes()] == ["kept"]

    def test_switch_keeps_the_token_budget(self):
        m = Memory(session="budget", max_chat_history=1234)
        m.switch_session("other")
        assert m.get_chat_stats()[1] == 1234

    def test_switch_keeps_the_turn_window(self):
        m = Memory(session="win", max_chat_history=10**9, window_turns=3)
        m.switch_session("other")
        assert m._chat_history.window_turns == 3

    def test_switch_drops_the_vector_store(self):
        """The store is scoped to a session dir and must not be reused."""
        m = self._fresh("first")
        m.vectorstore = object()
        m.switch_session("second")
        assert m.vectorstore is None

    def test_switch_creates_a_missing_session(self):
        """`wmk <new>` creates sessions, so `/resume <new>` should too."""
        m = self._fresh("first")
        target = self.sessions_dir / "brand-new"
        m.switch_session("brand-new")
        assert target.is_dir()
        assert (target / ".session-metadata").exists()

    def test_switch_updates_the_accessed_timestamp(self):
        m = self._fresh("first")
        first = (self.sessions_dir / "first" / ".session-metadata").read_text()
        m.switch_session("second")
        second = (self.sessions_dir / "second" / ".session-metadata").read_text()
        assert "created:" in first
        assert "accessed:" in second

    def test_switch_rejects_a_traversing_name(self):
        m = self._fresh("first")
        for bad in ("../escape", "a/b", "..", ".", "", "x\\y"):
            try:
                m.switch_session(bad)
            except ValueError:
                continue
            self.fail(f"{bad!r} should not be accepted as a session name")
        assert m.session == "first"

    def test_invalid_names(self):
        for good in ("default", "wisemonkey", "gaiasky", "my-session", "a b", "unicode"):
            assert is_valid_session_name(good), good
        for bad in ("", ".", "..", "../x", "a/b", "a\\b", "x\0y"):
            assert not is_valid_session_name(bad), bad


class TestSessionListing(BaseTest):
    """`/sessions` must list without opening any session's memory."""

    def setUp(self):
        super().setUp()
        import agent.memory as memory_mod

        self.sessions_dir = self._tmpdir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._patched = memory_mod.SESSIONS_DIR
        memory_mod.SESSIONS_DIR = self.sessions_dir
        self.addCleanup(lambda: setattr(memory_mod, "SESSIONS_DIR", self._patched))

    def _make(self, name: str, messages: int = 0, accessed=None):
        d = self.sessions_dir / name
        d.mkdir(parents=True, exist_ok=True)
        when = accessed or datetime.datetime.now()
        (d / ".session-metadata").write_text(
            f"created: {when.isoformat()}\naccessed: {when.isoformat()}\n"
        )
        (d / "chat_history.json").write_text(
            json.dumps({"exchanges": [{"role": "user", "content": str(i)} for i in range(messages)]})
        )
        return d

    def test_lists_every_session_newest_first(self):
        now = datetime.datetime.now()
        self._make("old", accessed=now - datetime.timedelta(days=2))
        self._make("newest", accessed=now)
        self._make("middle", accessed=now - datetime.timedelta(hours=1))
        names = [s["name"] for s in list_sessions()]
        assert names == ["newest", "middle", "old"]

    def test_reports_the_message_count(self):
        self._make("a", messages=3)
        self._make("b", messages=7)
        counts = {s["name"]: s["messages"] for s in list_sessions()}
        assert counts == {"a": 3, "b": 7}

    def test_a_hand_made_session_still_lists(self):
        """No metadata file: fall back to the directory mtime, do not crash."""
        d = self.sessions_dir / "handmade"
        d.mkdir()
        infos = list_sessions()
        assert [s["name"] for s in infos] == ["handmade"]
        assert infos[0]["messages"] == 0
        assert infos[0]["accessed"] is not None

    def test_a_corrupt_chat_history_does_not_break_the_listing(self):
        self._make("broken")
        (self.sessions_dir / "broken" / "chat_history.json").write_text("{not json")
        assert {s["name"]: s["messages"] for s in list_sessions()} == {"broken": 0}

    def test_loose_files_are_not_sessions(self):
        self._make("real")
        (self.sessions_dir / "stray.txt").write_text("x")
        assert [s["name"] for s in list_sessions()] == ["real"]

    def test_empty_sessions_dir(self):
        assert list_sessions() == []

    def test_session_info_reads_only_the_directory(self):
        """No Memory is constructed, so no profile/history is loaded."""
        d = self._make("x", messages=2)
        info = session_info(d)
        assert info["name"] == "x"
        assert info["dir"] == d
        assert info["messages"] == 2
        assert isinstance(info["accessed"], datetime.datetime)


class TestConversationalRecap(BaseTest):
    """The startup recap is for the human; the system prompt is for the model.

    The two want opposite things from the same stored history: a recap needs
    the conversation, the prompt needs the machinery. `roles` is what lets one
    renderer serve both.
    """

    ROLES = ("user", "assistant", "summary")

    def _cm(self, *entries, max_tokens=10**6):
        cm = ChatMemory(self._tmpdir / self._testMethodName, max_tokens=max_tokens)
        for role, content, extra in entries:
            cm.add_exchange(None, role, content, **extra)
        return cm

    def _recap(self, cm, n=2, **kw):
        opts = {
            "timestamps": False,
            "width": 320,
            "roles": self.ROLES,
            "assistant_width": 400,
            "mark_incomplete": True,
        }
        opts.update(kw)
        return cm.get_formatted(n, **opts)

    # --- roles filtering ---

    def test_tool_entries_are_dropped(self):
        cm = self._cm(
            ("user", "what changed?", {}),
            ("tool_call", "", {"name": "run_command", "arguments": "{}"}),
            ("tool_result", "tick 1 tick 2", {"name": "run_command"}),
            ("assistant", "Only the answer.", {}),
            ("user", "next question", {}),
            ("tool_result", "more noise", {"name": "grep"}),
        )
        out = self._recap(cm)
        assert "run_command" not in out
        assert "tick 1" not in out
        assert "next question" in out
        assert "Only the answer." in out

    def test_the_user_question_survives_a_tool_heavy_tail(self):
        """The reported bug: the recap showed tool output and no question."""
        cm = self._cm(
            ("user", "fix the failing test", {}),
            ("tool_call", "", {"name": "run_command", "arguments": "{}"}),
            ("tool_result", "AssertionError", {"name": "run_command"}),
            ("assistant", "Fixed.", {}),
            ("user", "run it again", {}),
            ("tool_result", "all 594 tests pass", {"name": "run_command"}),
        )
        out = self._recap(cm, 2)
        assert "run it again" in out
        assert "all 594 tests pass" not in out

    def test_summaries_are_kept(self):
        cm = self._cm(
            ("user", "q", {}),
            ("summary", "earlier: we discussed rivers", {}),
            ("assistant", "a", {}),
        )
        assert "earlier: we discussed rivers" in self._recap(cm)

    def test_the_prompt_path_still_keeps_tool_entries(self):
        """The default is no filter: the model must see its own tool results."""
        cm = self._cm(
            ("user", "q", {}),
            ("tool_call", "", {"name": "read_file", "arguments": "{}"}),
            ("tool_result", "file body", {"name": "read_file"}),
            ("assistant", "a", {}),
        )
        out = cm.get_formatted(0, timestamps=False, width=0)
        assert "read_file" in out
        assert "file body" in out

    # --- slicing happens after filtering ---

    def test_num_exchanges_counts_conversational_turns(self):
        """Two entries means two entries of *conversation*, not of history."""
        cm = self._cm(
            ("user", "q1", {}),
            ("assistant", "a1", {}),
            ("user", "q2", {}),
            ("tool_result", "noise", {"name": "x"}),
            ("assistant", "a2", {}),
            ("user", "q3", {}),
        )
        out = self._recap(cm, 2)
        assert "q3" in out and "a2" in out
        assert "q1" not in out and "a1" not in out
        assert "noise" not in out

    def test_conversational_entries_is_the_filtered_slice(self):
        cm = self._cm(
            ("user", "q1", {}),
            ("tool_result", "n", {"name": "x"}),
            ("user", "q2", {}),
            ("assistant", "a2", {}),
        )
        entries = cm.conversational_entries(2)
        assert [e["content"] for e in entries] == ["q2", "a2"]
        assert len(cm.conversational_entries(0)) == 3

    def test_the_title_count_matches_what_is_shown(self):
        cm = self._cm(("user", "q", {}), ("assistant", "a", {}))
        assert cm.conversational_entries(2) == cm.conversational_entries(0)

    # --- assistant_width ---

    def test_assistant_entries_get_more_room(self):
        cm = self._cm(
            ("user", "qword " * 200, {}), ("assistant", "aword " * 200, {})
        )
        out = self._recap(cm, 2, width=320)
        body = out.split("## ")
        user_line = next(b for b in body if b.startswith("User:"))
        assistant_line = next(b for b in body if b.startswith("Assistant:"))
        # The assistant gets 400 and the question 320, so the answer may be
        # longer -- but both are bounded, and by different amounts. A shared
        # width would make them equal.
        assert len(assistant_line) > len(user_line)
        # 400 plus the heading, the "[...]" marker and the "(continued)"
        # flag the recap adds to a truncated tail.
        assert len(assistant_line) <= 440

    def test_no_width_means_no_truncation(self):
        cm = self._cm(("user", "q" * 500, {}))
        out = self._recap(cm, 1, width=0, assistant_width=0)
        assert "q" * 500 in out

    # --- mark_incomplete ---

    def test_a_truncated_answer_is_marked_continued(self):
        cm = self._cm(("assistant", "a" * 900, {}))
        assert "(continued)" in self._recap(cm, 1)

    def test_a_session_that_ended_mid_sentence_is_marked(self):
        cm = self._cm(
            ("user", "q", {}), ("assistant", "The river was dry when we", {})
        )
        assert "(continued)" in self._recap(cm, 2)

    def test_a_complete_answer_is_not_marked(self):
        cm = self._cm(("user", "q", {}), ("assistant", "All done.", {}))
        assert "(continued)" not in self._recap(cm, 2)

    def test_a_complete_answer_ending_in_a_quote_is_not_marked(self):
        cm = self._cm(("user", "q", {}), ("assistant", 'He said "no."', {}))
        assert "(continued)" not in self._recap(cm, 2)

    def test_only_the_last_assistant_is_marked(self):
        cm = self._cm(
            ("user", "q1", {}),
            ("assistant", "The first answer is complete.", {}),
            ("user", "q2", {}),
            ("assistant", "The second answer is complete.", {}),
        )
        assert self._recap(cm, 4).count("(continued)") == 0

    def test_a_trailing_user_turn_does_not_hide_the_marker(self):
        """The last entry is the question, but the answer is still the tail.

        Marking by `len(history) - 1` would skip the answer entirely here,
        which is exactly the turn you most want flagged.
        """
        cm = self._cm(
            ("user", "q1", {}),
            ("assistant", "Here is what I found so far and it cuts off", {}),
            ("user", "what about the rest?", {}),
        )
        out = self._recap(cm, 3)
        assert "(continued)" in out
        assert "what about the rest?" in out

    def test_the_marker_is_off_by_default(self):
        cm = self._cm(("assistant", "The river was dry when we", {}))
        out = cm.get_formatted(1, timestamps=False, width=0)
        assert "(continued)" not in out

    def test_marker_costs_nothing_on_an_empty_history(self):
        cm = self._cm(("user", "q", {}))
        assert self._recap(cm, 5).strip() != ""
