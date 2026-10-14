"""Tests for agent/hooks.py — external tool hooks (Claude Code / Codex protocol).

Two layers are covered:

* the hook runner itself (config parsing, matchers, payloads, every verdict
  shape and exit code, the fail-open/fail-closed split), exercised against
  small throwaway shell scripts;
* the `Core._tool_calls` seam, so a deny really does stop a tool call and a
  post hook really does run without being able to veto the result.
"""

import json
import stat
import unittest

from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from tests.conftest import BaseTest
from agent.hooks import (
    HookGroup,
    HookHandler,
    build_payload,
    load_hook_groups,
    matcher_matches,
    run_handler,
    run_post_hooks,
    run_pre_hooks,
)


# ── helpers ──────────────────────────────────────────────────────────────────


class _Config:
    """Config stand-in with dot-notation `get`, matching agent.config.Config."""

    def __init__(self, values=None):
        self.values = values or {}

    def get(self, key, default=None):
        node: Any = self.values
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node


def _make_executable(path: Path, body: str) -> Path:
    """Write a shell script and mark it executable."""
    path.write_text(body.strip() + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class _HookScripts(BaseTest):
    """Base class offering a temp dir to drop throwaway hook scripts into."""

    def _script(self, name: str, body: str) -> str:
        return str(_make_executable(self._tmpdir / name, body))

    def _handler(self, body: str, **kw) -> HookHandler:
        return HookHandler(command=self._script("hook", body), **kw)


# ── config parsing ───────────────────────────────────────────────────────────


class TestConfigParsing(BaseTest):
    def test_parses_a_pre_tool_use_group(self):
        cfg = _Config({"agent": {"hooks": {"pre_tool_use": [
            {"matcher": "run_command", "hooks": [
                {"type": "command", "command": "/bin/dcg", "timeout": 3},
            ]},
        ]}}})
        groups = load_hook_groups(cfg)
        assert len(groups) == 1
        group = groups[0]
        assert group.event == "pre_tool_use"
        assert group.matcher == "run_command"
        assert group.handlers[0].command == "/bin/dcg"
        assert group.handlers[0].timeout == 3
        # Defaults: always runs (honor_unsafe False), library fail policy.
        assert group.handlers[0].honor_unsafe is False
        assert group.handlers[0].fail_open is None

    def test_parses_both_events(self):
        cfg = _Config({"agent": {"hooks": {
            "pre_tool_use": [{"hooks": [{"command": "/a"}]}],
            "post_tool_use": [{"hooks": [{"command": "/b"}]}],
        }}})
        events = [g.event for g in load_hook_groups(cfg)]
        assert events == ["pre_tool_use", "post_tool_use"]

    def test_no_hooks_block_yields_nothing(self):
        assert load_hook_groups(_Config({})) == []

    def test_type_defaults_to_command(self):
        cfg = _Config({"agent": {"hooks": {"pre_tool_use": [
            {"hooks": [{"command": "/a"}]},
        ]}}})
        assert len(load_hook_groups(cfg)[0].handlers) == 1

    def test_a_non_command_type_is_skipped(self):
        cfg = _Config({"agent": {"hooks": {"pre_tool_use": [
            {"hooks": [{"type": "prompt", "command": "/a"}]},
        ]}}})
        assert load_hook_groups(cfg) == []

    def test_a_handler_without_a_command_is_skipped(self):
        cfg = _Config({"agent": {"hooks": {"pre_tool_use": [
            {"hooks": [{"type": "command"}]},
        ]}}})
        assert load_hook_groups(cfg) == []

    def test_a_group_with_no_surviving_handler_is_dropped(self):
        cfg = _Config({"agent": {"hooks": {"pre_tool_use": [
            {"matcher": "x", "hooks": [{"type": "prompt"}]},
        ]}}})
        assert load_hook_groups(cfg) == []

    def test_a_malformed_event_is_ignored_not_fatal(self):
        cfg = _Config({"agent": {"hooks": {"pre_tool_use": "not a list"}}})
        assert load_hook_groups(cfg) == []

    def test_a_malformed_entry_is_skipped(self):
        cfg = _Config({"agent": {"hooks": {"pre_tool_use": [
            "nope",
            {"hooks": [{"command": "/a"}]},
        ]}}})
        groups = load_hook_groups(cfg)
        assert len(groups) == 1

    def test_honor_unsafe_and_fail_open_are_read(self):
        cfg = _Config({"agent": {"hooks": {"pre_tool_use": [
            {"hooks": [{"command": "/a", "honor_unsafe": True, "fail_open": True}]},
        ]}}})
        handler = load_hook_groups(cfg)[0].handlers[0]
        assert handler.honor_unsafe is True
        assert handler.fail_open is True

    def test_a_bad_timeout_falls_back_to_the_default(self):
        cfg = _Config({"agent": {"hooks": {"pre_tool_use": [
            {"hooks": [{"command": "/a", "timeout": -4}]},
            {"hooks": [{"command": "/b", "timeout": "soon"}]},
        ]}}})
        handlers = load_hook_groups(cfg)[0].handlers
        assert all(h.timeout > 0 for h in handlers)


# ── matchers ─────────────────────────────────────────────────────────────────


class TestMatchers(unittest.TestCase):
    def test_empty_matches_everything(self):
        assert matcher_matches("", "run_command") is True

    def test_star_matches_everything(self):
        assert matcher_matches("*", "anything") is True

    def test_regex_selects(self):
        assert matcher_matches("run_command|spawn", "spawn") is True
        assert matcher_matches("run_command|spawn", "read_file") is False

    def test_a_broken_regex_matches_everything(self):
        """A typo in a guard's matcher must not silently disable the guard."""
        assert matcher_matches("[", "run_command") is True


# ── payloads ─────────────────────────────────────────────────────────────────


class TestPayloads(unittest.TestCase):
    def test_a_shell_tool_is_presented_as_bash(self):
        payload = build_payload("pre_tool_use", "run_command",
                                {"command": "git reset --hard"})
        assert payload["hook_event_name"] == "PreToolUse"
        assert payload["tool_name"] == "Bash"
        assert payload["tool_input"] == {"command": "git reset --hard"}

    def test_spawn_is_also_bash(self):
        payload = build_payload("pre_tool_use", "spawn", {"command": "make"})
        assert payload["tool_name"] == "Bash"
        assert payload["tool_input"]["command"] == "make"

    def test_another_tool_keeps_its_own_name(self):
        payload = build_payload("pre_tool_use", "read_file", {"path": "x.py"})
        assert payload["tool_name"] == "read_file"
        assert payload["tool_input"] == {"path": "x.py"}

    def test_post_payload_carries_the_response(self):
        payload = build_payload("post_tool_use", "read_file", {"path": "x"}, "body")
        assert payload["hook_event_name"] == "PostToolUse"
        assert payload["tool_response"] == "body"

    def test_a_structured_response_is_serialised(self):
        payload = build_payload("post_tool_use", "read_file", {}, {"ok": True})
        assert json.loads(payload["tool_response"]) == {"ok": True}


# ── running one hook ─────────────────────────────────────────────────────────


class TestRunHandler(_HookScripts):
    def _run(self, handler, args=None):
        return run_handler(handler, build_payload("pre_tool_use", "run_command",
                                                  args or {"command": "x"}))

    def test_silent_stdout_is_a_no_opinion_allow(self):
        v = self._run(self._handler("cat >/dev/null; exit 0"))
        assert v.decision == "allow"

    def test_a_deny_verdict(self):
        v = self._run(self._handler(
            'cat >/dev/null; echo \'{"hookSpecificOutput":'
            '{"permissionDecision":"deny","permissionDecisionReason":"no"}}\''
        ))
        assert v.decision == "deny"
        assert v.reason == "no"

    def test_an_ask_verdict(self):
        v = self._run(self._handler(
            'cat >/dev/null; echo \'{"hookSpecificOutput":'
            '{"permissionDecision":"ask","permissionDecisionReason":"review"}}\''
        ))
        assert v.decision == "ask"
        assert v.reason == "review"

    def test_a_flat_top_level_decision_is_accepted(self):
        v = self._run(self._handler(
            'cat >/dev/null; echo \'{"permissionDecision":"deny",'
            '"permissionDecisionReason":"flat"}\''
        ))
        assert v.decision == "deny"
        assert v.reason == "flat"

    def test_valid_json_without_a_decision_is_an_allow(self):
        v = self._run(self._handler(
            'cat >/dev/null; echo \'{"hookSpecificOutput":{"context":"fyi"}}\''
        ))
        assert v.decision == "allow"

    def test_exit_2_blocks_with_stderr_as_the_reason(self):
        v = self._run(self._handler("cat >/dev/null; echo 'because' >&2; exit 2"))
        assert v.decision == "deny"
        assert v.reason == "because"

    def test_a_missing_binary_fails_open(self):
        """A broken install must not refuse every tool call (exit 127)."""
        v = self._run(HookHandler(command="/nonexistent/dcg-does-not-exist"))
        assert v.decision == "allow"

    def test_unparseable_output_fails_closed(self):
        v = self._run(self._handler("cat >/dev/null; echo 'not json'"))
        assert v.decision == "deny"

    def test_unparseable_output_fails_open_when_asked(self):
        v = self._run(self._handler("cat >/dev/null; echo 'not json'", fail_open=True))
        assert v.decision == "allow"

    def test_a_timeout_fails_closed(self):
        v = self._run(self._handler("sleep 2", timeout=0.3))
        assert v.decision == "deny"

    def test_a_timeout_fails_open_when_asked(self):
        v = self._run(self._handler("sleep 2", timeout=0.3, fail_open=True))
        assert v.decision == "allow"

    def test_the_hook_receives_the_payload_on_stdin(self):
        seen = self._tmpdir / "seen.json"
        self._run(self._handler(f"cat > {seen}; exit 0"))
        payload = json.loads(seen.read_text())
        assert payload["hook_event_name"] == "PreToolUse"
        assert payload["tool_name"] == "Bash"


# ── orchestration ────────────────────────────────────────────────────────────


class TestRunPreHooks(_HookScripts):
    def _deny_group(self, **kw):
        return HookGroup(
            event="pre_tool_use",
            matcher="run_command",
            handlers=[self._handler(
                'cat >/dev/null; echo \'{"hookSpecificOutput":'
                '{"permissionDecision":"deny","permissionDecisionReason":"no"}}\'', **kw
            )],
        )

    def test_a_matching_deny_blocks(self):
        v = run_pre_hooks([self._deny_group()], "run_command", {"command": "x"})
        assert v.decision == "deny"

    def test_a_non_matching_tool_is_untouched(self):
        v = run_pre_hooks([self._deny_group()], "read_file", {"path": "x"})
        assert v.decision == "allow"

    def test_a_deny_is_not_cancelled_by_a_later_allow(self):
        allow = HookGroup(event="pre_tool_use", matcher="",
                          handlers=[HookHandler(command="/bin/true")])
        v = run_pre_hooks([self._deny_group(), allow], "run_command", {"command": "x"})
        assert v.decision == "deny"

    def test_a_later_deny_still_blocks_after_an_allow(self):
        allow = HookGroup(event="pre_tool_use", matcher="",
                          handlers=[HookHandler(command="/bin/true")])
        v = run_pre_hooks([allow, self._deny_group()], "run_command", {"command": "x"})
        assert v.decision == "deny"

    def test_an_ask_is_returned_immediately(self):
        ask = HookGroup(event="pre_tool_use", matcher="run_command",
                        handlers=[self._handler(
                            'cat >/dev/null; echo \'{"hookSpecificOutput":'
                            '{"permissionDecision":"ask",'
                            '"permissionDecisionReason":"review"}}\'')])
        v = run_pre_hooks([ask], "run_command", {"command": "x"})
        assert v.decision == "ask"
        assert v.reason == "review"

    def test_a_post_group_is_not_run_as_a_pre_hook(self):
        post = HookGroup(event="post_tool_use", matcher="",
                         handlers=[HookHandler(command="/bin/false")])
        assert run_pre_hooks([post], "run_command", {}).decision == "allow"

    def test_honor_unsafe_skips_the_hook_under_unsafe(self):
        group = self._deny_group(honor_unsafe=True)
        v = run_pre_hooks([group], "run_command", {"command": "x"}, unsafe=True)
        assert v.decision == "allow"
        # ... and still blocks when unsafe is off.
        assert run_pre_hooks([group], "run_command", {"command": "x"}).decision == "deny"

    def test_a_hook_without_honor_unsafe_runs_under_unsafe(self):
        v = run_pre_hooks([self._deny_group()], "run_command", {"command": "x"},
                          unsafe=True)
        assert v.decision == "deny"


class TestRunPostHooks(_HookScripts):
    def test_a_post_hook_runs_and_sees_the_result(self):
        seen = self._tmpdir / "post.json"
        group = HookGroup(event="post_tool_use", matcher="read_file",
                          handlers=[HookHandler(command=self._script(
                              "post", f"cat > {seen}; exit 0"))])
        run_post_hooks([group], "read_file", {"path": "p"}, "file body")
        payload = json.loads(seen.read_text())
        assert payload["tool_response"] == "file body"

    def test_a_post_deny_is_ignored(self):
        """Post hooks are informational: nothing they say can veto a result."""
        group = HookGroup(event="post_tool_use", matcher="",
                          handlers=[self._handler(
                              'cat >/dev/null; echo \'{"hookSpecificOutput":'
                              '{"permissionDecision":"deny"}}\''
                          )])
        # No exception and no return value: the denial simply has nowhere to go.
        assert run_post_hooks([group], "read_file", {}, "body") is None

    def test_a_post_group_is_not_run_as_a_pre_hook(self):
        group = HookGroup(event="pre_tool_use", matcher="",
                          handlers=[HookHandler(command="/bin/false")])
        assert run_post_hooks([group], "read_file", {}, "body") is None


# ── the Core._tool_calls seam ────────────────────────────────────────────────


class _Memory:
    """Records chat exchanges without touching the filesystem."""

    def __init__(self):
        self.exchanges = []

    def add_chat_exchange(self, core, role, content, **extra):
        self.exchanges.append({"role": role, "content": content, **extra})


def _make_core(config=None):
    from agent.core import Core

    core = cast(Any, Core.__new__(Core))
    core.config = config or _Config({})
    core.memory = _Memory()
    core.messages = []
    core.thinking_buffer = ""
    core._pending_image = None
    core._hooks = None
    return core


def _tool_call(name="run_command", args='{"command": "ls"}', call_id="c1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": args}}


class TestToolCallsSeam(_HookScripts):
    def _core_with(self, hooks: dict, **agent_extra):
        config = _Config({"agent": {"hooks": hooks, **agent_extra}})
        return _make_core(config)

    def _run(self, core, calls, **kw):
        return core._tool_calls(calls, **kw)

    def test_a_deny_blocks_the_tool_and_feeds_a_tool_error(self):
        core = self._core_with({"pre_tool_use": [
            {"matcher": "run_command", "hooks": [{"command": self._script(
                "deny", 'cat >/dev/null; echo \'{"hookSpecificOutput":'
                        '{"permissionDecision":"deny",'
                        '"permissionDecisionReason":"destroys work"}}\'')}]},
        ]})
        with patch("agent.core.execute_tool") as execute:
            core._tool_calls([_tool_call()])
        # The tool never ran.
        execute.assert_not_called()
        # The model got a tool message carrying the hook's reason.
        tool_msgs = [m for m in core.messages if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        payload = json.loads(tool_msgs[0]["content"])
        assert payload["blocked_by_hook"] is True
        assert "destroys work" in payload["error"]
        # ... and it was recorded like any other tool result.
        assert any(e["role"] == "tool_result" for e in core.memory.exchanges)

    def test_a_deny_reaches_the_tool_result_callback(self):
        core = self._core_with({"pre_tool_use": [
            {"matcher": "", "hooks": [{"command": self._script(
                "deny", 'cat >/dev/null; echo \'{"hookSpecificOutput":'
                        '{"permissionDecision":"deny"}}\'')}]},
        ]})
        seen = []
        with patch("agent.core.execute_tool") as execute:
            core._tool_calls([_tool_call()],
                             tool_result_callback=lambda *a: seen.append(a))
        execute.assert_not_called()
        assert len(seen) == 1
        assert seen[0][3] is True  # is_error

    def test_a_silent_hook_lets_the_tool_run(self):
        core = self._core_with({"pre_tool_use": [
            {"matcher": "", "hooks": [{"command": self._script("ok", "exit 0")}]},
        ]})
        with patch("agent.core.execute_tool", return_value="ran") as execute:
            core._tool_calls([_tool_call()])
        execute.assert_called_once()

    def test_no_hooks_configured_still_runs_the_tool(self):
        core = _make_core()
        with patch("agent.core.execute_tool", return_value="ran") as execute:
            core._tool_calls([_tool_call()])
        execute.assert_called_once()

    def test_a_post_hook_runs_after_the_tool(self):
        seen = self._tmpdir / "post.json"
        core = self._core_with({"post_tool_use": [
            {"matcher": "run_command", "hooks": [{"command": self._script(
                "post", f"cat > {seen}; exit 0")}]},
        ]})
        with patch("agent.core.execute_tool", return_value="tool output"):
            core._tool_calls([_tool_call()])
        payload = json.loads(seen.read_text())
        assert payload["tool_response"] == "tool output"

    def test_a_post_hook_that_fails_does_not_break_the_turn(self):
        core = self._core_with({"post_tool_use": [
            {"matcher": "", "hooks": [{"command": "/nonexistent/hook"}]},
        ]})
        with patch("agent.core.execute_tool", return_value="tool output"):
            core._tool_calls([_tool_call()])
        # The tool result is still there and the turn was not aborted.
        assert any(e["role"] == "tool_result" for e in core.memory.exchanges)

    def test_an_ask_the_user_declines_becomes_a_cancelled_tool_error(self):
        core = self._core_with({"pre_tool_use": [
            {"matcher": "", "hooks": [{"command": self._script(
                "ask", 'cat >/dev/null; echo \'{"hookSpecificOutput":'
                       '{"permissionDecision":"ask","permissionDecisionReason":"review"}}\''
            )}]},
        ]})
        with patch.object(type(core), "_hook_ask", return_value=False), \
                patch("agent.core.execute_tool") as execute:
            core._tool_calls([_tool_call()])
        execute.assert_not_called()
        payload = json.loads(
            [m for m in core.messages if m["role"] == "tool"][0]["content"]
        )
        assert payload["user_cancelled"] is True

    def test_an_ask_the_user_approves_lets_the_tool_run(self):
        core = self._core_with({"pre_tool_use": [
            {"matcher": "", "hooks": [{"command": self._script(
                "ask", 'cat >/dev/null; echo \'{"hookSpecificOutput":'
                       '{"permissionDecision":"ask"}}\'')}]},
        ]})
        with patch.object(type(core), "_hook_ask", return_value=True), \
                patch("agent.core.execute_tool", return_value="ran") as execute:
            core._tool_calls([_tool_call()])
        execute.assert_called_once()

    def test_a_hook_that_honors_unsafe_is_skipped_under_unsafe(self):
        hooks = {"pre_tool_use": [
            {"matcher": "", "hooks": [{"command": self._script(
                "deny", 'cat >/dev/null; echo \'{"hookSpecificOutput":'
                        '{"permissionDecision":"deny"}}\''), "honor_unsafe": True}]},
        ]}
        core = self._core_with(hooks, unsafe=True)
        with patch("agent.core.execute_tool", return_value="ran") as execute:
            core._tool_calls([_tool_call()])
        execute.assert_called_once()

    def test_a_default_hook_still_blocks_under_unsafe(self):
        hooks = {"pre_tool_use": [
            {"matcher": "", "hooks": [{"command": self._script(
                "deny", 'cat >/dev/null; echo \'{"hookSpecificOutput":'
                        '{"permissionDecision":"deny"}}\'')}]},
        ]}
        core = self._core_with(hooks, unsafe=True)
        with patch("agent.core.execute_tool") as execute:
            core._tool_calls([_tool_call()])
        execute.assert_not_called()

    def test_the_hook_set_is_parsed_once_and_cached(self):
        core = self._core_with({"pre_tool_use": [
            {"matcher": "", "hooks": [{"command": self._script("ok", "exit 0")}]},
        ]})
        first = core._hook_groups()
        assert first and core._hook_groups() is first

    def test_clearing_the_cache_reparses_the_config(self):
        """`reload_config` drops the cache, so an edited block takes effect."""
        core = self._core_with({"pre_tool_use": [
            {"matcher": "", "hooks": [{"command": self._script("ok", "exit 0")}]},
        ]})
        assert core._hook_groups() != []
        core.config = _Config({"agent": {"hooks": {}}})
        core._hooks = None
        assert core._hook_groups() == []


# ── the real dcg binary, when one is available ───────────────────────────────
#
# `dcg` (destructive_command_guard) is the motivating case for the whole
# feature, and its Claude-compatible arm is exactly what an installed binary
# answers: a `Bash` payload with no `turn_id` returns the structured
# `hookSpecificOutput` denial. These tests drive the genuine binary, so the
# wire shape is pinned against it rather than against a hand-written stand-in.
#
# They are skipped unless a binary can be found, because dcg is an external
# Rust project (a slow `cargo build --release`) and CI must stay hermetic. Point
# `DCG_BIN` at one, or let the search find the usual checkout/`PATH` locations.


def _find_dcg() -> str:
    """Path to a real dcg binary, or "" when there is none to test against."""
    import os
    import shutil

    explicit = os.environ.get("DCG_BIN")
    if explicit and Path(explicit).is_file():
        return explicit
    checkout = Path.home() / "Projects" / "destructive_command_guard"
    for candidate in (
        checkout / "target" / "release" / "dcg",
        Path(shutil.which("dcg") or ""),
    ):
        if candidate and candidate.is_file():
            return str(candidate)
    return ""


_DCG = _find_dcg()


@unittest.skipUnless(_DCG, "no dcg binary found (set DCG_BIN to run these)")
class TestRealDcg(_HookScripts):
    """The real `dcg` binary, through wisemonkey's hook runner."""

    def _groups(self):
        return [HookGroup(
            event="pre_tool_use",
            matcher="run_command|spawn",
            handlers=[HookHandler(command=_DCG, timeout=15)],
        )]

    def test_a_destructive_command_is_denied(self):
        verdict = run_pre_hooks(self._groups(), "run_command",
                                {"command": "git reset --hard HEAD~5"})
        assert verdict.decision == "deny"
        # dcg's own reason text, which the model receives verbatim.
        assert "dcg" in verdict.reason
        assert "reset --hard" in verdict.reason

    def test_a_safe_command_is_silent(self):
        verdict = run_pre_hooks(self._groups(), "run_command", {"command": "ls -la"})
        assert verdict.decision == "allow"

    def test_a_non_shell_tool_never_reaches_dcg(self):
        """The matcher is the only thing standing between dcg and every tool."""
        verdict = run_pre_hooks(self._groups(), "read_file", {"path": "x"})
        assert verdict.decision == "allow"

    def test_the_seam_blocks_the_tool_call(self):
        core = _make_core(_Config({"agent": {"hooks": {"pre_tool_use": [
            {"matcher": "run_command|spawn",
             "hooks": [{"command": _DCG, "timeout": 15}]},
        ]}}}))
        with patch("agent.core.execute_tool") as execute:
            core._tool_calls([_tool_call(
                args='{"command": "git reset --hard HEAD~5"}')])
        execute.assert_not_called()
        payload = json.loads(
            [m for m in core.messages if m["role"] == "tool"][0]["content"]
        )
        assert payload["blocked_by_hook"] is True
        assert "dcg" in payload["error"]

    def test_the_seam_lets_a_safe_command_through(self):
        core = _make_core(_Config({"agent": {"hooks": {"pre_tool_use": [
            {"matcher": "run_command|spawn",
             "hooks": [{"command": _DCG, "timeout": 15}]},
        ]}}}))
        with patch("agent.core.execute_tool", return_value="ran") as execute:
            core._tool_calls([_tool_call(args='{"command": "ls -la"}')])
        execute.assert_called_once()


if __name__ == "__main__":
    unittest.main()
