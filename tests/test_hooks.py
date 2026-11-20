"""Tests for agent/hooks.py: external tool hooks (Claude Code / Codex protocol)."""

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
    hook_label,
    load_hook_groups,
    matcher_matches,
    run_handler,
    run_post_hooks,
    run_pre_hooks,
)
from agent.output import HOOK_BLOCK_KIND, HOOK_DECLINED_KIND, hook_notice


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
    path.write_text(body.strip() + "\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


DENY = (
    'cat >/dev/null; echo \'{"hookSpecificOutput":'
    '{"permissionDecision":"deny","permissionDecisionReason":"no"}}\''
)
ASK = (
    'cat >/dev/null; echo \'{"hookSpecificOutput":'
    '{"permissionDecision":"ask","permissionDecisionReason":"review"}}\''
)


class _HookScripts(BaseTest):
    """A temp dir to drop throwaway hook scripts into."""

    def _script(self, name: str, body: str) -> str:
        return str(_make_executable(self._tmpdir / name, body))

    def _handler(self, body: str, **kw) -> HookHandler:
        return HookHandler(command=self._script("hook", body), **kw)


class TestConfigParsing(BaseTest):
    def test_parses_groups_handlers_and_their_defaults(self):
        cfg = _Config({"agent": {"hooks": {
            "pre_tool_use": [{"matcher": "run_command", "hooks": [
                {"type": "command", "command": "/bin/dcg", "timeout": 3,
                 "honor_unsafe": True, "fail_open": True},
            ]}],
            "post_tool_use": [{"hooks": [{"command": "/b"}]}],
        }}})
        groups = load_hook_groups(cfg)
        assert [g.event for g in groups] == ["pre_tool_use", "post_tool_use"]
        assert (groups[0].matcher, groups[0].handlers[0].command) == (
            "run_command", "/bin/dcg"
        )
        handler = groups[0].handlers[0]
        assert (handler.timeout, handler.honor_unsafe, handler.fail_open) == (3, True, True)
        # Defaults are library policy: nothing is opted out of the guard.
        default = groups[1].handlers[0]
        assert default.honor_unsafe is False and default.fail_open is None
        assert default.timeout > 0

    def test_junk_and_a_bad_timeout_are_skipped_rather_than_fatal(self):
        """A typo in the config must not take the agent down."""
        for hooks in (
            {},
            {"pre_tool_use": "not a list"},
            {"pre_tool_use": [{"hooks": [{"type": "prompt", "command": "/a"}]}]},
            {"pre_tool_use": [{"hooks": [{"type": "command"}]}]},
            {"pre_tool_use": ["nope"]},
        ):
            assert load_hook_groups(_Config({"agent": {"hooks": hooks}})) == [], hooks
        cfg = _Config({"agent": {"hooks": {"pre_tool_use": [
            {"hooks": [{"command": "/a", "timeout": -4}]},
            {"hooks": [{"command": "/b", "timeout": "soon"}]},
        ]}}})
        assert all(h.timeout > 0 for h in load_hook_groups(cfg)[0].handlers)


class TestMatchers(unittest.TestCase):
    def test_matcher_semantics(self):
        for pattern, tool, expected in (
            ("", "run_command", True),
            ("*", "anything", True),
            ("run_command|spawn", "spawn", True),
            ("run_command|spawn", "read_file", False),
            # A typo must not silently disable the guard.
            ("[", "run_command", True),
        ):
            assert matcher_matches(pattern, tool) is expected, pattern


class TestHookLabel(unittest.TestCase):
    """The UI frames a hook's output, so it has to be nameable."""

    def test_label_resolution(self):
        for command, expected in (
            ("/usr/local/bin/dcg", "dcg"),
            ("python3 /opt/guard.py", "guard"),
            ("/usr/bin/env bash /opt/check.sh", "check"),
            ("dcg --strict", "dcg"),
            ("", "hook"),
        ):
            assert hook_label(command) == expected, command
        # An explicit name always wins over the derived one.
        assert HookHandler(command="/bin/dcg", name="dcg (strict)").label == "dcg (strict)"


def _render(renderable) -> str:
    from io import StringIO

    from rich.console import Console

    from agent.console import monkee_theme

    buffer = StringIO()
    Console(file=buffer, width=70, no_color=True, theme=monkee_theme).print(renderable)
    return buffer.getvalue()


class TestHookNotice(unittest.TestCase):
    """The frame that stops a guard's output reading as the model speaking."""

    def test_it_names_the_hook_keeps_the_reason_and_tells_the_two_apart(self):
        reason = "[red]not markup[/red] destroys uncommitted work"
        text = _render(hook_notice(HOOK_BLOCK_KIND, "dcg", "run_command", reason))
        assert "dcg" in text and "run_command" in text
        assert "destroys uncommitted work" in text
        # The point of the frame, and a guard's brackets left alone.
        assert "not the model's" in text
        assert "[red]not markup[/red]" in text

        declined = _render(hook_notice(HOOK_DECLINED_KIND, "dcg", "run_command", "r"))
        assert "Declined" in declined and "you declined" in declined


class TestPayloads(unittest.TestCase):
    def test_shell_tools_are_presented_as_bash_and_results_ride_along(self):
        for tool in ("run_command", "spawn"):
            payload = build_payload("pre_tool_use", tool, {"command": "git reset --hard"})
            assert payload["hook_event_name"] == "PreToolUse"
            assert payload["tool_name"] == "Bash", tool
            assert payload["tool_input"] == {"command": "git reset --hard"}
        payload = build_payload("pre_tool_use", "read_file", {"path": "x.py"})
        assert payload["tool_name"] == "read_file"
        payload = build_payload("post_tool_use", "read_file", {"path": "x"}, {"ok": True})
        assert payload["hook_event_name"] == "PostToolUse"
        assert json.loads(payload["tool_response"]) == {"ok": True}


class TestRunHandler(_HookScripts):
    def _run(self, handler, args=None):
        return run_handler(
            handler,
            build_payload("pre_tool_use", "run_command", args or {"command": "x"}),
        )

    def test_verdicts(self):
        for body, decision, reason in (
            ("cat >/dev/null; exit 0", "allow", ""),
            (DENY, "deny", "no"),
            (ASK, "ask", "review"),
            # A flat top-level decision is accepted too.
            ('cat >/dev/null; echo \'{"permissionDecision":"deny",'
             '"permissionDecisionReason":"flat"}\'', "deny", "flat"),
            # Valid JSON with no decision is no opinion.
            ('cat >/dev/null; echo \'{"hookSpecificOutput":{"context":"fyi"}}\'',
             "allow", ""),
            # Exit 2 blocks, with stderr as the reason.
            ("cat >/dev/null; echo 'because' >&2; exit 2", "deny", "because"),
        ):
            verdict = self._run(self._handler(body))
            assert (verdict.decision, verdict.reason) == (decision, reason), body

    def test_failure_policy(self):
        # A broken install must not refuse every tool call (exit 127).
        assert self._run(
            HookHandler(command="/nonexistent/dcg-does-not-exist")
        ).decision == "allow"
        # Garbage or a hang has not cleared the call, so it fails closed.
        for body, kw in (("cat >/dev/null; echo 'not json'", {}),
                         ("sleep 2", {"timeout": 0.3})):
            assert self._run(self._handler(body, **kw)).decision == "deny", body
            assert self._run(
                self._handler(body, fail_open=True, **kw)
            ).decision == "allow", body

    def test_the_hook_receives_the_payload_on_stdin(self):
        seen = self._tmpdir / "seen.json"
        self._run(self._handler(f"cat > {seen}; exit 0"))
        payload = json.loads(seen.read_text())
        assert payload["hook_event_name"] == "PreToolUse"
        assert payload["tool_name"] == "Bash"


class TestRunPreHooks(_HookScripts):
    def _deny_group(self, **kw):
        return HookGroup(event="pre_tool_use", matcher="run_command",
                         handlers=[self._handler(DENY, **kw)])

    def test_deny_and_ask_verdicts(self):
        assert run_pre_hooks([self._deny_group()], "run_command", {}).decision == "deny"
        assert run_pre_hooks([self._deny_group()], "read_file", {}).decision == "allow"
        # A deny survives a later allow, in either order.
        allow = HookGroup(event="pre_tool_use", matcher="",
                          handlers=[HookHandler(command="/bin/true")])
        for groups in ([self._deny_group(), allow], [allow, self._deny_group()]):
            assert run_pre_hooks(groups, "run_command", {}).decision == "deny"
        ask = HookGroup(event="pre_tool_use", matcher="run_command",
                        handlers=[self._handler(ASK)])
        verdict = run_pre_hooks([ask], "run_command", {})
        assert (verdict.decision, verdict.reason) == ("ask", "review")

    def test_only_pre_groups_run_as_pre_hooks(self):
        post = HookGroup(event="post_tool_use", matcher="",
                         handlers=[HookHandler(command="/bin/false")])
        assert run_pre_hooks([post], "run_command", {}).decision == "allow"

    def test_unsafe_skips_only_the_hooks_that_opted_out(self):
        honoured = self._deny_group(honor_unsafe=True)
        assert run_pre_hooks([honoured], "run_command", {}, unsafe=True).decision == "allow"
        assert run_pre_hooks([honoured], "run_command", {}).decision == "deny"
        default = self._deny_group()
        assert run_pre_hooks([default], "run_command", {}, unsafe=True).decision == "deny"


class TestRunPostHooks(_HookScripts):
    def test_a_post_hook_sees_the_result_and_can_never_veto(self):
        seen = self._tmpdir / "post.json"
        group = HookGroup(event="post_tool_use", matcher="read_file",
                          handlers=[HookHandler(command=self._script(
                              "post", f"cat > {seen}; exit 0"))])
        run_post_hooks([group], "read_file", {"path": "p"}, "file body")
        assert json.loads(seen.read_text())["tool_response"] == "file body"

        deny = HookGroup(event="post_tool_use", matcher="",
                         handlers=[self._handler(DENY)])
        pre = HookGroup(event="pre_tool_use", matcher="",
                        handlers=[HookHandler(command="/bin/false")])
        # Post hooks are informational: the denial simply has nowhere to go.
        assert run_post_hooks([deny, pre], "read_file", {}, "body") is None


class _Memory:
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


def _tool_message(core) -> dict:
    return json.loads([m for m in core.messages if m["role"] == "tool"][0]["content"])


class TestToolCallsSeam(_HookScripts):
    def _core_with(self, hooks: dict, **agent_extra):
        return _make_core(_Config({"agent": {"hooks": hooks, **agent_extra}}))

    def _deny_core(self, reason="destroys work"):
        body = DENY.replace('"no"', f'"{reason}"')
        return self._core_with({"pre_tool_use": [
            {"matcher": "", "hooks": [{"command": self._script("deny", body)}]},
        ]})

    def _ask_core(self):
        return self._core_with({"pre_tool_use": [
            {"matcher": "", "hooks": [{"command": self._script("ask", ASK)}]},
        ]})

    def test_a_deny_blocks_the_tool_and_is_reported_as_the_guard_speaking(self):
        core = self._deny_core()
        seen = []
        with patch("agent.core.execute_tool") as execute:
            core._tool_calls(
                [_tool_call()], tool_result_callback=lambda *a, **k: seen.append((a, k))
            )
        execute.assert_not_called()
        payload = _tool_message(core)
        assert payload["blocked_by_hook"] is True
        # The model gets instructions; the human gets the guard's own reason.
        assert "destroys work" in payload["error"] and "Do not retry" in payload["error"]
        assert any(e["role"] == "tool_result" for e in core.memory.exchanges)
        args, kwargs = seen[0]
        assert args[3] is True  # is_error
        assert args[2] == "destroys work"
        assert (kwargs["kind"], kwargs["hook"]) == (HOOK_BLOCK_KIND, "deny")

    def test_an_ask_may_be_declined_or_approved(self):
        core = self._ask_core()
        seen = []
        with patch.object(type(core), "_hook_ask", return_value=False), \
                patch("agent.core.execute_tool") as execute:
            core._tool_calls([_tool_call()],
                             tool_result_callback=lambda *a, **k: seen.append(k))
        execute.assert_not_called()
        assert _tool_message(core)["user_cancelled"] is True
        assert (seen[0]["kind"], seen[0]["hook"]) == (HOOK_DECLINED_KIND, "ask")

        core = self._ask_core()
        with patch.object(type(core), "_hook_ask", return_value=True), \
                patch("agent.core.execute_tool", return_value="ran") as execute:
            core._tool_calls([_tool_call()])
        execute.assert_called_once()

    def test_a_silent_hook_and_no_hooks_both_let_the_tool_run(self):
        silent = self._core_with({"pre_tool_use": [
            {"matcher": "", "hooks": [{"command": self._script("ok", "exit 0")}]},
        ]})
        for core in (silent, _make_core()):
            seen = []
            with patch("agent.core.execute_tool", return_value="ran") as execute:
                core._tool_calls([_tool_call()],
                                 tool_result_callback=lambda *a, **k: seen.append(k))
            execute.assert_called_once()
            # `kind`/`hook` are empty for a tool that ran: the UI's default.
            assert seen[0] == {}

    def test_a_post_hook_runs_after_the_tool_and_cannot_break_the_turn(self):
        seen = self._tmpdir / "post.json"
        core = self._core_with({"post_tool_use": [
            {"matcher": "run_command",
             "hooks": [{"command": self._script("post", f"cat > {seen}; exit 0")},
                       {"command": "/nonexistent/hook"}]},
        ]})
        with patch("agent.core.execute_tool", return_value="tool output"):
            core._tool_calls([_tool_call()])
        assert json.loads(seen.read_text())["tool_response"] == "tool output"
        assert any(e["role"] == "tool_result" for e in core.memory.exchanges)

    def test_unsafe_skips_only_the_hooks_that_opted_out(self):
        def deny(honor_unsafe):
            return {"pre_tool_use": [{"matcher": "", "hooks": [{
                "command": self._script("deny", DENY), "honor_unsafe": honor_unsafe,
            }]}]}

        for honour_unsafe, ran in ((True, True), (False, False)):
            core = self._core_with(deny(honour_unsafe), unsafe=True)
            with patch("agent.core.execute_tool", return_value="ran") as execute:
                core._tool_calls([_tool_call()])
            assert execute.called is ran, honour_unsafe

    def test_the_hook_set_is_parsed_once_and_reloaded_on_demand(self):
        core = self._core_with({"pre_tool_use": [
            {"matcher": "", "hooks": [{"command": self._script("ok", "exit 0")}]},
        ]})
        first = core._hook_groups()
        assert first and core._hook_groups() is first
        # `reload_config` drops the cache, so an edited block takes effect.
        core.config = _Config({"agent": {"hooks": {}}})
        core._hooks = None
        assert core._hook_groups() == []


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
        return [HookGroup(event="pre_tool_use", matcher="run_command|spawn",
                          handlers=[HookHandler(command=_DCG, timeout=15)])]

    def test_a_destructive_command_is_denied_and_a_safe_one_allowed(self):
        verdict = run_pre_hooks(self._groups(), "run_command",
                                {"command": "git reset --hard HEAD~5"})
        assert verdict.decision == "deny"
        # dcg's own reason text, which the model receives verbatim.
        assert "dcg" in verdict.reason and "reset --hard" in verdict.reason
        assert run_pre_hooks(self._groups(), "run_command",
                             {"command": "ls -la"}).decision == "allow"
        # The matcher is what keeps dcg off every other tool.
        assert run_pre_hooks(self._groups(), "read_file",
                             {"path": "x"}).decision == "allow"

    def test_the_seam_blocks_the_tool_call(self):
        core = _make_core(_Config({"agent": {"hooks": {"pre_tool_use": [
            {"matcher": "run_command|spawn",
             "hooks": [{"command": _DCG, "timeout": 15}]},
        ]}}}))
        with patch("agent.core.execute_tool") as execute:
            core._tool_calls([_tool_call(args='{"command": "git reset --hard HEAD~5"}')])
        execute.assert_not_called()
        payload = _tool_message(core)
        assert payload["blocked_by_hook"] is True and "dcg" in payload["error"]


if __name__ == "__main__":
    unittest.main()
