"""Tests for agent/memory.py — memory, chat history, windows, sessions."""

import datetime
import json
from unittest import mock

from agent.memory import (
    CONVERSATIONAL_ROLES,
    ChatMemory,
    Memory,
    is_valid_session_name,
    list_sessions,
    session_info,
)
from agent.tokens import count_tokens
from tests.conftest import BaseTest


class TestMemoryFiles(BaseTest):
    """Profile, notes and the session metadata file."""

    def _memory(self, name="m"):
        return Memory(session_dir=self._tmpdir / "sessions" / name)

    def test_init_creates_the_session_dir_and_metadata(self):
        session_dir = self._tmpdir / "sessions" / "test"
        Memory(session_dir=session_dir)
        content = (session_dir / ".session-metadata").read_text()
        assert "created:" in content and "accessed:" in content

    def test_profile_and_notes_merge_and_persist(self):
        m = self._memory()
        m.set_user_profile({"name": "Alice"})
        m.set_user_profile({"role": "dev"})
        note = m.add_note("Remember this", category="reminder")
        assert note["id"] == 1 and note["category"] == "reminder"
        m.save()
        assert json.loads(m._user_profile_path.read_text()) == {
            "name": "Alice",
            "role": "dev",
        }
        assert [n["content"] for n in json.loads(m._notes_path.read_text())] == [
            "Remember this"
        ]

    def test_formatted_sections(self):
        m = self._memory()
        assert m.get_user_profile_formatted() is None
        m.set_user_profile({"name": "Alice"})
        m.add_note("Test note", category="test")
        out = m.get_user_profile_formatted()
        assert "## User Profile" in out and "name: Alice" in out
        assert "## Persistent Notes" in out and "[test] Test note" in out
        # Either section can be asked for alone.
        assert "## Persistent Notes" not in m.get_user_profile_formatted(notes=False)


class TestChatMemory(BaseTest):
    def setUp(self):
        super().setUp()
        self.session_dir = self._tmpdir / "sessions" / "chat"
        self.cm = ChatMemory(self.session_dir, max_tokens=200)

    def test_add_exchange_persists_a_round_trip(self):
        self.cm.add_exchange(None, "user", "persist me")
        self.cm.save()
        cm2 = ChatMemory(self.session_dir, max_tokens=200)
        assert cm2.get_unformatted()[0]["content"] == "persist me"

    def test_tokens_grow_with_content_and_trim_back_down(self):
        self.cm.add_exchange(None, "user", "abc def ghi")
        before = self.cm.total_tokens
        assert before > 0
        self.cm.add_exchange(None, "assistant", "one two three four")
        assert self.cm.total_tokens > before
        # _trim() directly: add_exchange() would compact via the command registry.
        for _ in range(20):
            self.cm._exchanges.append({"role": "user", "content": "x" * 50})
            self.cm.total_tokens += 12
        self.cm._trim()
        assert self.cm.total_tokens <= 200

    def test_clear_removes_n_or_everything(self):
        for i in range(5):
            self.cm.add_exchange(None, "user", f"msg {i}")
        assert self.cm._clear(2) == 2
        assert len(self.cm._exchanges) == 3
        assert self.cm._clear(0) == 3
        assert self.cm._exchanges == []

    def test_rendering_headings_roles_and_width(self):
        self.cm.add_exchange(None, "user", "Hello")
        self.cm.add_exchange(None, "assistant", "Hi there")
        out = self.cm.get_formatted(2, timestamps=False, width=0)
        assert "## User:" in out and "## Assistant:" in out
        assert "Hello" in out and "Hi there" in out
        assert ChatMemory(self.session_dir / "empty").get_formatted(0) is None
        # A width truncates the body it is given.
        long = ChatMemory(self.session_dir / "wide", max_tokens=10**6)
        long.add_exchange(None, "user", "a" * 200)
        assert len(long.get_formatted(1, timestamps=False, width=50)) < 300

    def test_tool_entries_render_and_compact_when_matched(self):
        self.cm.add_exchange(
            None, "tool_call", "", name="read_file",
            arguments='{"path":"a.py"}', tool_call_id="c1",
        )
        self.cm.add_exchange(
            None, "tool_result", "file contents", name="read_file", tool_call_id="c1"
        )
        rendered = self.cm.get_formatted(0, timestamps=False, width=0)
        assert "## Tool: read_file" in rendered
        assert 'Args: {"path":"a.py"}' in rendered
        assert "Result: file contents" in rendered
        assert "## Tool Call" not in rendered and "## Tool Result" not in rendered

    def test_an_unmatched_tool_result_stays_separate(self):
        self.cm.add_exchange(
            None, "tool_call", "", name="read_file",
            arguments="{}", tool_call_id="c1",
        )
        self.cm.add_exchange(
            None, "tool_result", "other", name="read_file", tool_call_id="c2"
        )
        rendered = self.cm.get_formatted(0, timestamps=False, width=0)
        assert "## Tool Call (read_file):" in rendered
        assert "## Tool Result (read_file):" in rendered
        assert "## Tool: read_file" not in rendered

    def test_a_huge_tool_result_is_truncated_at_storage(self):
        cm = ChatMemory(self.session_dir / "big", max_tokens=25000)
        cm.add_exchange(None, "tool_result", "z" * 100000, name="read_file")
        assert cm._exchanges[-1]["content"] == "z" * 200
        assert cm.total_tokens < 500

    def test_token_accounting_matches_the_rendered_history(self):
        self.cm.add_exchange(None, "user", "hello")
        self.cm.add_exchange(
            None, "tool_call", "", name="read_file",
            arguments='{"path":"a.py"}', tool_call_id="c1",
        )
        self.cm.add_exchange(
            None, "tool_result", "contents", name="read_file", tool_call_id="c1"
        )
        self.cm.add_exchange(None, "assistant", "done")

        def rendered():
            return self.cm.get_formatted(0, timestamps=False, width=0) or ""

        assert self.cm.total_tokens == count_tokens(rendered())
        self.cm._clear(1)
        assert self.cm.total_tokens == count_tokens(rendered())


class TestChatMemoryWindow(BaseTest):
    """The turn-based rolling window."""

    def setUp(self):
        super().setUp()
        self.session_dir = self._tmpdir / "sessions" / "window"

    def _fill(self, cm, n):
        for i in range(n):
            cm.add_exchange(None, "user", f"msg {i}")
            cm.add_exchange(None, "assistant", f"reply {i}")

    def _archive(self, cm):
        path = cm._chat_path.parent / "chat_archive.jsonl"
        return [json.loads(l)["content"] for l in path.read_text().splitlines()]

    def test_zero_keeps_everything_and_n_keeps_the_last_turns(self):
        cm = ChatMemory(self.session_dir / "z", max_tokens=10**9, window_turns=0)
        self._fill(cm, 4)
        assert len(cm._exchanges) == 8
        cm = ChatMemory(self.session_dir / "n", max_tokens=10**9, window_turns=2)
        self._fill(cm, 4)
        assert [e["content"] for e in cm._exchanges] == [
            "msg 2", "reply 2", "msg 3", "reply 3",
        ]

    def test_a_tool_heavy_turn_counts_as_one_exchange(self):
        cm = ChatMemory(self.session_dir / "t", max_tokens=10**9, window_turns=1)
        for role, content in (
            ("user", "msg 0"), ("assistant", "reply 0"), ("tool_call", "call"),
            ("tool_result", "res"), ("user", "msg 1"), ("assistant", "reply 1"),
        ):
            cm.add_exchange(None, role, content, name="t", arguments="{}")
        assert [e["content"] for e in cm._exchanges] == ["msg 1", "reply 1"]

    def test_loading_reconciles_a_lowered_window(self):
        cm = ChatMemory(self.session_dir / "load", max_tokens=10**9, window_turns=0)
        self._fill(cm, 4)
        cm.save()
        cm2 = ChatMemory(self.session_dir / "load", max_tokens=10**9, window_turns=2)
        assert cm2._exchanges[0]["content"] == "msg 2"

    def test_dropped_entries_are_archived_not_destroyed(self):
        cm = ChatMemory(self.session_dir / "arch", max_tokens=10**9, window_turns=2)
        self._fill(cm, 4)
        assert self._archive(cm) == ["msg 0", "reply 0", "msg 1", "reply 1"]
        # The token-cap backstop archives too, and the archive is never reloaded.
        cm2 = ChatMemory(self.session_dir / "cap", max_tokens=10**9)
        self._fill(cm2, 3)
        cm2.max_tokens = 1
        cm2._trim()
        assert cm2._exchanges == []
        assert self._archive(cm2) == [
            "msg 0", "reply 0", "msg 1", "reply 1", "msg 2", "reply 2",
        ]


class TestChatMemoryDropLast(BaseTest):
    """`drop_last` removes whole exchanges from the tail."""

    ENTRIES = (
        ("user", "q1"), ("assistant", "a1"), ("user", "q2"), ("assistant", "a2"),
        ("tool_call", "{}"), ("tool_result", "42"), ("user", "q3"),
        ("assistant", "a3"),
    )

    def setUp(self):
        super().setUp()
        self.session_dir = self._tmpdir / "sessions" / "droplast"

    def _fill(self, entries=ENTRIES):
        self._n = getattr(self, "_n", 0) + 1
        cm = ChatMemory(self.session_dir / str(self._n), max_tokens=10**9)
        self._last_dir = self.session_dir / str(self._n)
        cm._exchanges = [{"role": r, "content": c} for r, c in entries]
        cm._recount_tokens()
        return cm

    def test_a_tool_turn_is_dropped_with_its_exchange(self):
        cm = self._fill()
        assert cm.drop_last(1) == 2
        assert [e["role"] for e in cm._exchanges] == [
            "user", "assistant", "user", "assistant", "tool_call", "tool_result",
        ]
        cm = self._fill()
        assert cm.drop_last(2) == 6
        assert [e["role"] for e in cm._exchanges] == ["user", "assistant"]

    def test_everything_or_more_clears_the_history(self):
        for n in (0, 99):
            cm = self._fill()
            assert cm.drop_last(n) == 8
            assert cm._exchanges == []
        assert ChatMemory(self.session_dir / "empty").drop_last(1) == 0

    def test_the_drop_is_recounted_and_persisted(self):
        cm = self._fill()
        before = cm.total_tokens
        cm.drop_last(1)
        cm.save()
        assert cm.total_tokens < before
        assert len(ChatMemory(self._last_dir)._exchanges) == 6

    def test_a_leading_orphan_is_not_a_turn_start(self):
        orphan = (
            ("tool_result", "leftover"), ("user", "q1"), ("assistant", "a1"),
            ("user", "q2"), ("assistant", "a2"),
        )
        cm = self._fill(orphan)
        assert cm.drop_last(1) == 2
        assert [e["role"] for e in cm._exchanges] == [
            "tool_result", "user", "assistant",
        ]
        # Dropping the only exchange leaves nothing coherent, orphan included.
        cm = self._fill((("tool_result", "leftover"), ("user", "q"), ("assistant", "a")))
        assert cm.drop_last(1) == 3
        assert cm._exchanges == []

    def test_last_user_prompt(self):
        assert self._fill().last_user_prompt() == "q3"
        cm = self._fill()
        cm._exchanges.append({"role": "user", "content": "steered mid-turn"})
        assert cm.last_user_prompt() == "steered mid-turn"
        assert ChatMemory(self.session_dir / "none").last_user_prompt() == ""


class _SessionsDir(BaseTest):
    """Point `agent.memory.SESSIONS_DIR` at a temp dir for the test's lifetime."""

    def setUp(self):
        super().setUp()
        import agent.memory as memory_mod

        self.sessions_dir = self._tmpdir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._patched = memory_mod.SESSIONS_DIR
        memory_mod.SESSIONS_DIR = self.sessions_dir
        self.addCleanup(setattr, memory_mod, "SESSIONS_DIR", self._patched)

    def _fresh(self, name: str) -> Memory:
        """A fresh process on *name*."""
        from tests.conftest import reset_singletons

        reset_singletons()
        return Memory(session=name)

    def _make(self, name: str, messages: int = 0, accessed=None):
        d = self.sessions_dir / name
        d.mkdir(parents=True, exist_ok=True)
        when = accessed or datetime.datetime.now()
        (d / ".session-metadata").write_text(
            f"created: {when.isoformat()}\naccessed: {when.isoformat()}\n"
        )
        (d / "chat_history.json").write_text(
            json.dumps({"exchanges": [
                {"role": "user", "content": str(i)} for i in range(messages)
            ]})
        )
        return d


class TestSessionSwitching(_SessionsDir):
    """`Memory` is a singleton, so switching must rebind it in place."""

    def test_the_singleton_is_rebound_with_its_paths(self):
        m = self._fresh("first")
        holder = {"memory": m}
        assert m.switch_session("second") == "first"
        assert holder["memory"] is m and m.session == "second"
        assert m.session_dir == self.sessions_dir / "second"
        assert m._user_profile_path == m.session_dir / "user_profile.json"
        assert m._notes_path == m.session_dir / "notes.json"
        assert m._chat_history._chat_path == m.session_dir / "chat_history.json"

    def test_switch_loads_the_other_sessions_state(self):
        other = self._fresh("second")
        other.add_note("a note from the second session")
        other.set_user_profile({"name": "Langur"})
        other._chat_history.set_exchanges([{"role": "user", "content": "hello there"}])
        other.save()

        m = self._fresh("first")
        m.add_note("a note from the first session")
        m.set_user_profile({"name": "Someone Else"})
        assert m.get_chat_history_unformatted() == []

        m.switch_session("second")
        assert [n["content"] for n in m.get_notes()] == ["a note from the second session"]
        assert m.get_user_profile() == {"name": "Langur"}
        assert m.get_chat_history_unformatted() == [
            {"role": "user", "content": "hello there"}
        ]

    def test_switch_saves_the_previous_session_first(self):
        m = self._fresh("first")
        m.add_note("unsaved when the switch happens")
        first_dir = m.session_dir
        m.switch_session("second")
        saved = json.loads((first_dir / "notes.json").read_text())
        assert [n["content"] for n in saved] == ["unsaved when the switch happens"]

    def test_switch_keeps_the_budget_and_window_and_drops_the_vector_store(self):
        m = Memory(session="budget", max_chat_history=1234, window_turns=3)
        m.vectorstore = object()
        m.switch_session("other")
        assert m.get_chat_stats()[1] == 1234
        assert m._chat_history.window_turns == 3
        assert m.vectorstore is None

    def test_switch_creates_a_missing_session_and_rejects_a_traversing_name(self):
        m = self._fresh("first")
        target = self.sessions_dir / "brand-new"
        m.switch_session("brand-new")
        assert target.is_dir() and (target / ".session-metadata").exists()
        for bad in ("../escape", "a/b", "..", ".", "", "x\\y"):
            with self.assertRaises(ValueError, msg=bad):
                m.switch_session(bad)
        assert m.session == "brand-new"

    def test_switching_to_the_same_session_is_a_noop(self):
        m = self._fresh("first")
        m.add_note("kept")
        assert m.switch_session("first") == "first"
        assert [n["content"] for n in m.get_notes()] == ["kept"]

    def test_session_name_validation(self):
        for good in ("default", "gaiasky", "my-session", "a b"):
            assert is_valid_session_name(good), good
        for bad in ("", ".", "..", "../x", "a/b", "a\\b", "x\0y"):
            assert not is_valid_session_name(bad), bad


class TestSessionListing(_SessionsDir):
    """`/sessions` must list without opening any session's memory."""

    def test_lists_newest_first_with_message_counts(self):
        now = datetime.datetime.now()
        self._make("old", accessed=now - datetime.timedelta(days=2))
        self._make("newest", messages=7, accessed=now)
        self._make("middle", messages=3, accessed=now - datetime.timedelta(hours=1))
        infos = list_sessions()
        assert [s["name"] for s in infos] == ["newest", "middle", "old"]
        assert {s["name"]: s["messages"] for s in infos} == {
            "newest": 7, "middle": 3, "old": 0,
        }

    def test_a_hand_made_session_still_lists_and_debris_is_ignored(self):
        """No metadata file: fall back to the directory mtime, do not crash."""
        (self.sessions_dir / "handmade").mkdir()
        (self.sessions_dir / "stray.txt").write_text("x")
        infos = list_sessions()
        assert [s["name"] for s in infos] == ["handmade"]
        assert infos[0]["messages"] == 0
        assert infos[0]["accessed"] is not None

    def test_a_corrupt_chat_history_does_not_break_the_listing(self):
        self._make("broken")
        (self.sessions_dir / "broken" / "chat_history.json").write_text("{not json")
        assert {s["name"]: s["messages"] for s in list_sessions()} == {"broken": 0}

    def test_session_info_reads_only_the_directory(self):
        d = self._make("x", messages=2)
        info = session_info(d)
        assert (info["name"], info["dir"], info["messages"]) == ("x", d, 2)
        assert isinstance(info["accessed"], datetime.datetime)
        assert list_sessions() != [] and list_sessions() is not None


# A two-turn history whose second turn ran tools: the shape that made
# `/history 3` print two tool calls and an answer with no question.
_TOOL_HEAVY_TURNS = (
    ("user", "prompt 1", {}),
    ("assistant", "assistant 1", {}),
    ("user", "prompt 2", {}),
    ("tool_call", "", {"name": "read_file", "arguments": "{}"}),
    ("tool_result", "file body", {"name": "read_file"}),
    ("tool_call", "", {"name": "run_command", "arguments": "{}"}),
    ("tool_result", "a b c", {"name": "run_command"}),
    ("assistant", "assistant 2", {}),
)

_RECAP = {
    "timestamps": False,
    "width": 320,
    "roles": CONVERSATIONAL_ROLES,
    "assistant_width": 400,
    "mark_incomplete": True,
}


class _HistoryBase(BaseTest):
    """A `ChatMemory` builder on a per-test, per-call directory."""

    def _cm(self, *entries, max_tokens=10**6):
        self._n = getattr(self, "_n", 0) + 1
        cm = ChatMemory(
            self._tmpdir / f"{self._testMethodName}-{self._n}", max_tokens=max_tokens
        )
        for role, content, extra in entries:
            cm.add_exchange(None, role, content, **extra)
        return cm

    def _recap(self, cm, n, **kw):
        return cm.get_formatted(n, **{**_RECAP, **kw})


class TestConversationalRecap(_HistoryBase):
    """The startup recap is for the human; the system prompt is for the model."""

    def test_tool_machinery_is_filtered_out(self):
        cm = self._cm(
            ("user", "fix the failing test", {}),
            ("tool_call", "", {"name": "run_command", "arguments": "{}"}),
            ("tool_result", "AssertionError", {"name": "run_command"}),
            ("assistant", "Fixed.", {}),
            ("user", "run it again", {}),
            ("tool_result", "all 594 tests pass", {"name": "run_command"}),
        )
        out = self._recap(cm, 2)
        assert "run it again" in out and "Fixed." in out
        assert "run_command" not in out and "594" not in out

    def test_the_prompt_path_keeps_tool_entries_and_summaries(self):
        cm = self._cm(
            ("summary", "earlier: we discussed rivers", {}),
            ("user", "q", {}),
            ("tool_result", "file body", {"name": "read_file"}),
            ("assistant", "a", {}),
        )
        prompt = cm.get_formatted(0, timestamps=False, width=0)
        assert "read_file" in prompt and "file body" in prompt
        assert "earlier: we discussed rivers" in self._recap(cm, 0)

    def test_slicing_counts_conversational_turns(self):
        cm = self._cm(
            ("user", "q1", {}), ("assistant", "a1", {}), ("user", "q2", {}),
            ("tool_result", "noise", {"name": "x"}), ("assistant", "a2", {}),
            ("user", "q3", {}),
        )
        out = self._recap(cm, 2)
        assert "q3" in out and "a2" in out
        assert "q1" not in out and "a1" not in out and "noise" not in out
        assert [e["content"] for e in cm.conversational_entries(2)] == ["q2", "a2", "q3"]
        assert len(cm.conversational_entries(0)) == 5

    def test_assistant_entries_get_more_room_but_stay_bounded(self):
        cm = self._cm(
            ("user", "qword " * 200, {}), ("assistant", "aword " * 200, {})
        )
        body = self._recap(cm, 2).split("## ")
        user_line = next(b for b in body if b.startswith("User:"))
        assistant_line = next(b for b in body if b.startswith("Assistant:"))
        assert len(assistant_line) > len(user_line)
        # 400 plus heading, "[...]" and "(continued)".
        assert len(assistant_line) <= 440
        # Zero width means no truncation at all.
        long = self._cm(("user", "q" * 500, {}))
        assert "q" * 500 in self._recap(long, 1, width=0, assistant_width=0)

    def test_a_fragment_is_marked_continued(self):
        for cm in (
            self._cm(("user", "q", {}), ("assistant", "a" * 900, {})),
            self._cm(("user", "q", {}), ("assistant", "The river was dry when we", {})),
            self._cm(
                ("user", "q", {}),
                ("assistant", "Looked finished but was not.", {"incomplete": True}),
            ),
            self._cm(
                ("user", "q1", {}),
                ("assistant", "Here is what I found so far and it cuts off", {}),
                ("user", "what about the rest?", {}),
            ),
        ):
            assert "(continued)" in self._recap(cm, 3)

    def test_a_whole_answer_is_not_marked_and_the_flag_stays_out_of_the_prompt(self):
        for answer in ("All done.", 'He said "no."'):
            cm = self._cm(("user", "q", {}), ("assistant", answer, {}))
            assert "(continued)" not in self._recap(cm, 2)
        cm = self._cm(
            ("user", "q", {}), ("assistant", "A fragment, cut mid-", {"incomplete": True})
        )
        assert "(continued)" not in cm.get_formatted(1, timestamps=False, width=0)
        # Off by default, too.
        plain = self._cm(("assistant", "The river was dry when we", {}))
        assert "(continued)" not in plain.get_formatted(1, timestamps=False, width=0)


class TestExchangeSlicing(_HistoryBase):
    """`num_exchanges` counts exchanges on both ends of the history."""

    def test_a_count_is_a_count_of_exchanges(self):
        cm = self._cm(*_TOOL_HEAVY_TURNS)
        out = cm.get_formatted(1, timestamps=False, width=0)
        assert "prompt 2" in out and "assistant 2" in out
        assert "file body" in out  # the tool turn is not split
        assert "prompt 1" not in out and "assistant 1" not in out
        everything = cm.get_formatted(0, timestamps=False, width=0)
        for n in (2, 3, 99):
            assert cm.get_formatted(n, timestamps=False, width=0) == everything

    def test_the_two_cutoffs_agree(self):
        cm = self._cm(*_TOOL_HEAVY_TURNS)
        assert cm._oldest_cutoff(1) == cm._newest_cutoff(1) == 2
        assert cm._oldest_cutoff(0) == len(cm._exchanges) == 8
        assert cm._newest_cutoff(0) == cm._newest_cutoff(9) == 0

    def test_a_history_without_a_question_still_renders(self):
        cm = self._cm(("assistant", "an opening line", {}))
        assert "an opening line" in cm.get_formatted(1, timestamps=False, width=0)


class TestConversationalPreamble(_HistoryBase):
    """A summary in front of the first question is not a turn, but is shown."""

    ENTRIES = (("summary", "earlier: we discussed rivers", {}), *_TOOL_HEAVY_TURNS)

    def test_the_preamble_travels_with_a_whole_history_and_not_with_a_slice(self):
        cm = self._cm(*self.ENTRIES)
        for n in (0, 2):
            assert "earlier: we discussed rivers" in self._recap(cm, n)
        out = self._recap(cm, 1)
        assert "earlier: we discussed rivers" not in out and "prompt 2" in out

    def test_the_preamble_is_neither_a_turn_nor_lost_from_the_front(self):
        cm = self._cm(*self.ENTRIES)
        assert cm.conversational_count(0) == 2 and cm.conversational_count(1) == 1
        assert [e["content"] for e in cm.conversational_entries(0)] == [
            "earlier: we discussed rivers",
            "prompt 1", "assistant 1", "prompt 2", "assistant 2",
        ]

    def test_a_preamble_only_history_is_still_shown(self):
        cm = self._cm(("summary", "a briefing with no question", {}))
        assert "a briefing with no question" in self._recap(cm, 0)
        assert cm.conversational_count(0) == 0


class TestHistoryCommand(_HistoryBase):
    """`/history N` prints N turns: prompt plus final answer, no machinery."""

    def setUp(self):
        super().setUp()
        from types import SimpleNamespace

        import agent.memory as memory_mod

        sessions = self._tmpdir / "sessions"
        sessions.mkdir()
        self._patched = memory_mod.SESSIONS_DIR
        memory_mod.SESSIONS_DIR = sessions
        self.addCleanup(setattr, memory_mod, "SESSIONS_DIR", self._patched)

        self.memory = Memory(session="history")
        for role, content, extra in _TOOL_HEAVY_TURNS:
            self.memory.add_chat_exchange(None, role, content, **extra)
        self.core = SimpleNamespace(memory=self.memory)

    def _run(self, *params):
        from agent.commands import registry

        command, parsed = registry.lookup(["/history", *params])
        assert command is not None
        return command.handler(self.core, parsed)

    def test_the_count_is_turns(self):
        ok, _, _, markdown = self._run("1")
        assert ok
        assert "prompt 2" in markdown and "assistant 2" in markdown
        assert "prompt 1" not in markdown
        _, _, _, all_turns = self._run("3")
        for text in ("prompt 1", "assistant 1", "prompt 2", "assistant 2"):
            assert text in all_turns

    def test_tool_machinery_stays_out_of_the_panel(self):
        for params in (("2",), ()):
            _, _, _, markdown = self._run(*params)
            for noise in ("read_file", "run_command", "file body", "Tool"):
                assert noise not in markdown
        assert "prompt 1" in self._run()[3]

    def test_the_stats_are_reported(self):
        ok, msg, _, _ = self._run("1")
        assert ok and "Memory status:" in msg

    def test_a_non_integer_is_rejected(self):
        ok, msg, _, _ = self._run("two")
        assert not ok and "must be integer" in msg


class TestCorruptFilesAreNotOverwritten(BaseTest):
    """A file that fails to parse is quarantined, never silently overwritten."""

    def _session(self, name):
        d = self._tmpdir / "sessions" / name
        d.mkdir(parents=True)
        return d

    def test_corrupt_chat_history_is_quarantined_across_a_save(self):
        session_dir = self._session("corrupt_chat")
        bad = '{"exchanges": [{"role": "user", "con'
        (session_dir / "chat_history.json").write_text(bad)

        cm = ChatMemory(session_dir)
        assert cm.get_unformatted() == []
        kept = list(session_dir.glob("chat_history.json.corrupt-*"))
        assert len(kept) == 1 and kept[0].read_text() == bad
        # The live path is gone, so a save cannot clobber the evidence.
        assert not (session_dir / "chat_history.json").exists()
        cm.save()
        assert json.loads((session_dir / "chat_history.json").read_text()) == {
            "exchanges": []
        }
        assert len(list(session_dir.glob("chat_history.json.corrupt-*"))) == 1

    def test_corrupt_profile_and_notes_are_quarantined(self):
        session_dir = self._session("corrupt_memory")
        (session_dir / "user_profile.json").write_text("{broken")
        (session_dir / "notes.json").write_text("[broken")
        m = Memory(session_dir=session_dir)
        assert m.get_notes() == []
        assert len(list(session_dir.glob("user_profile.json.corrupt-*"))) == 1
        assert len(list(session_dir.glob("notes.json.corrupt-*"))) == 1

    def test_valid_files_round_trip_and_leave_no_temp(self):
        session_dir = self._session("roundtrip")
        m = Memory(session_dir=session_dir)
        m.set_user_profile({"name": "Langur"})
        m.add_note("remember this")
        m.save()
        m2 = Memory(session_dir=session_dir)
        assert m2.get_user_profile()["name"] == "Langur"
        assert m2.get_notes()[0]["content"] == "remember this"
        assert list(session_dir.glob("*.tmp")) == []
        assert list(session_dir.glob("*.corrupt-*")) == []

    def test_a_failed_write_leaves_the_previous_file_intact(self):
        session_dir = self._session("atomic")
        cm = ChatMemory(session_dir)
        cm.add_exchange(None, "user", "first")
        cm.save()
        original = (session_dir / "chat_history.json").read_text()
        # add_exchange() saves itself; make that write fail mid-replace.
        with mock.patch("agent.memory.os.replace", side_effect=OSError("disk")):
            with self.assertRaises(OSError):
                cm.add_exchange(None, "user", "second")
        assert (session_dir / "chat_history.json").read_text() == original
