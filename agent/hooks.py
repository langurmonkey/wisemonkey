"""External tool hooks, Claude Code / Codex compatible.

Wisemonkey can shell out to guard programs around every tool call. The
motivating case is ``dcg`` (destructive_command_guard), which is *already* a
Claude Code ``PreToolUse`` hook: a binary that reads JSON on stdin and emits a
verdict on stdout. Pointing ``agent.hooks.pre_tool_use`` at it needs no
dcg-specific code, and neither does any other hook written for Claude or Codex.

Config shape (``$XDG_CONFIG_HOME/wisemonkey/config.yaml``)::

    agent:
      hooks:
        pre_tool_use:
          - matcher: "run_command|spawn"   # regex over wisemonkey tool names
            hooks:
              - type: command
                command: /usr/local/bin/dcg
                timeout: 5        # seconds
                honor_unsafe: false   # true = skipped when agent.unsafe is set
        post_tool_use: []

The protocol is byte-compatible with Claude Code, which is what makes the
installed ``dcg`` work unchanged:

* Input on **stdin**: ``{"hook_event_name", "tool_name", "tool_input"}``.
  Shell tools are presented under the Claude name (``Bash``) with the command
  in ``tool_input.command``, because that is the name every protocol-aware
  hook matches on -- a stock ``dcg`` ignores anything else.
* Verdict on **stdout**, on exit 0: ``hookSpecificOutput.permissionDecision``
  (``allow`` / ``deny`` / ``ask``) plus ``permissionDecisionReason``. A flat
  top-level ``permissionDecision`` is accepted too. Empty stdout means "no
  opinion" and the tool proceeds.
* **Exit codes**: 0 is success (parse stdout); 2 blocks, with stderr as the
  reason; any other non-zero is a non-blocking hook error (Claude's own
  contract). A shell's ``command not found`` exit 127 therefore fails open,
  so a broken install cannot block every tool call.

Two policies are deliberate and pinned here rather than scattered:

* ``PRE_TOOL_USE_FAIL_CLOSED`` -- a hook that *hangs* (past ``timeout``) or
  emits unparseable stdout fails closed by default, because a guard that
  cannot answer has not cleared the call. A hook that cannot be *started*
  fails open (exit 127 is a non-blocking error above): a broken install is a
  configuration problem, not a reason to refuse all work. Per-handler
  ``fail_open: true`` overrides the timeout/garbage direction.
* ``honor_unsafe`` -- ``agent.unsafe`` silences wisemonkey's own confirmations
  but does **not** disable hooks. Only a handler that sets ``honor_unsafe:
  true`` is skipped under unsafe, so a security hook is never switched off by
  a flag nobody asked it to respect.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field

from agent.console import warn

# Event names, as they appear in the config.
PRE_TOOL_USE = "pre_tool_use"
POST_TOOL_USE = "post_tool_use"

# The wire value of `hook_event_name` (Claude/Codex spelling).
_WIRE_EVENT = {PRE_TOOL_USE: "PreToolUse", POST_TOOL_USE: "PostToolUse"}

# Tools that run a shell command, and the Claude tool name they are presented
# under. Every protocol-aware hook (dcg included) matches on `Bash` and reads
# `tool_input.command`; a hook would silently ignore `run_command`.
_SHELL_TOOL_NAMES = {"run_command", "spawn"}
_CLAUDE_SHELL_TOOL = "Bash"

DEFAULT_TIMEOUT = 5.0

# A hook that times out or answers garbage has not cleared the call. A hook
# that cannot start at all is handled separately (it fails open).
PRE_TOOL_USE_FAIL_CLOSED = True

# Interpreters a command may start with. They are skipped when naming a hook,
# so `python /opt/guard.py` is "guard", not "python".
_INTERPRETERS = {
    "python", "python3", "pypy", "sh", "bash", "zsh", "dash", "fish",
    "node", "bun", "deno", "ruby", "perl", "uv", "uvx", "npx", "env",
}

_SCRIPT_EXTENSIONS = ("py", "sh", "bash", "js", "ts", "rb", "pl")


def _basename(token: str) -> str:
    """`/usr/local/bin/dcg` -> `dcg`, `/opt/guard.py` -> `guard`."""
    base = os.path.basename(token)
    stem, dot, ext = base.rpartition(".")
    if dot and ext in _SCRIPT_EXTENSIONS:
        return stem
    return base


def hook_label(command: str) -> str:
    """A short human name for a hook: ``dcg``, not its full path.

    Display only -- it names the speaker on the frame that reports a block --
    so it errs towards *something* readable rather than failing: a command it
    cannot make sense of comes back as its first token.
    """
    tokens = (command or "").split()
    while tokens and _basename(tokens[0]) in _INTERPRETERS:
        tokens = tokens[1:]
    return (_basename(tokens[0]) if tokens else "") or "hook"


@dataclass
class HookHandler:
    """One command hook inside a matcher group."""

    command: str
    timeout: float = DEFAULT_TIMEOUT
    # True = skipped while `agent.unsafe` is set. False (the default) = always runs.
    honor_unsafe: bool = False
    # Display name; defaults to the command's basename (see hook_label).
    name: str = ""

    @property
    def label(self) -> str:
        """How this hook is named when it reports a block."""
        return self.name or hook_label(self.command)
    # None = use PRE_TOOL_USE_FAIL_CLOSED; True/False override the
    # timeout/unparseable-output direction for this handler.
    fail_open: bool | None = None


@dataclass
class HookGroup:
    """A matcher plus the hooks it selects."""

    event: str
    matcher: str = ""
    handlers: list[HookHandler] = field(default_factory=list)


@dataclass
class HookVerdict:
    """A normalised hook decision.

    ``decision`` is ``"allow"`` (proceed), ``"deny"`` (block outright) or
    ``"ask"`` (the hook wants a human to review the call).
    """

    decision: str = "allow"
    reason: str = ""
    # The hook that produced this verdict (its display label, e.g. "dcg").
    # Set by run_handler; empty on the allow a caller constructs itself.
    source: str = ""


# ── config parsing ───────────────────────────────────────────────────────────


def _as_bool(value, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _as_timeout(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return DEFAULT_TIMEOUT
    return float(value) if value > 0 else DEFAULT_TIMEOUT


def load_hook_groups(config) -> list[HookGroup]:
    """Parse ``agent.hooks.{pre,post}_tool_use`` from *config*.

    Malformed entries are reported and skipped: a broken hook definition must
    not stop the agent from starting, and the warning names what was ignored.
    """
    groups: list[HookGroup] = []
    for event in (PRE_TOOL_USE, POST_TOOL_USE):
        raw = config.get(f"agent.hooks.{event}", [])
        groups.extend(_parse_groups(event, raw))
    return groups


def _parse_groups(event: str, raw) -> list[HookGroup]:
    if not raw:
        return []
    if not isinstance(raw, list):
        warn(f"agent.hooks.{event} must be a list of matcher groups; ignoring it")
        return []

    groups: list[HookGroup] = []
    for entry in raw:
        if not isinstance(entry, dict):
            warn(f"agent.hooks.{event}: each entry must be a mapping; skipping {entry!r}")
            continue
        raw_hooks = entry.get("hooks", [])
        if not isinstance(raw_hooks, list):
            warn(f"agent.hooks.{event}: 'hooks' must be a list; skipping the group")
            continue
        handlers = []
        for raw_handler in raw_hooks:
            handler = _parse_handler(event, raw_handler)
            if handler is not None:
                handlers.append(handler)
        if not handlers:
            continue
        groups.append(
            HookGroup(
                event=event,
                matcher=str(entry.get("matcher", "") or ""),
                handlers=handlers,
            )
        )
    return groups


def _parse_handler(event: str, entry) -> HookHandler | None:
    if not isinstance(entry, dict):
        warn(f"agent.hooks.{event}: each hook must be a mapping; skipping {entry!r}")
        return None
    kind = entry.get("type", "command")
    if kind != "command":
        # Only command hooks exist today; naming the unsupported type is more
        # useful than failing silently.
        warn(f"agent.hooks.{event}: unsupported hook type {kind!r}; only 'command' is supported")
        return None
    command = entry.get("command")
    if not isinstance(command, str) or not command.strip():
        warn(f"agent.hooks.{event}: hook has no 'command'; skipping it")
        return None
    fail_open = entry.get("fail_open")
    name = entry.get("name")
    return HookHandler(
        command=command,
        timeout=_as_timeout(entry.get("timeout")),
        honor_unsafe=_as_bool(entry.get("honor_unsafe")),
        name=name.strip() if isinstance(name, str) else "",
        fail_open=fail_open if isinstance(fail_open, bool) else None,
    )


# ── matchers and payloads ────────────────────────────────────────────────────


def matcher_matches(matcher: str, tool_name: str) -> bool:
    """Whether *matcher* selects *tool_name*.

    An empty matcher (or ``*``) matches every tool. Otherwise it is a Python
    regular expression over the *wisemonkey* tool name, so ``run_command|spawn``
    selects both shell tools. A matcher that does not compile matches
    everything rather than nothing: a typo in a security config must not
    silently disable the guard.
    """
    text = (matcher or "").strip()
    if not text or text == "*":
        return True
    try:
        return re.search(matcher, tool_name) is not None
    except re.error as e:
        warn(f"agent.hooks: matcher {matcher!r} is not a valid regex ({e}); matching every tool")
        return True


def build_payload(event: str, tool_name: str, args, result=None) -> dict:
    """Build the Claude/Codex hook payload for one tool call."""
    if tool_name in _SHELL_TOOL_NAMES:
        wire_name = _CLAUDE_SHELL_TOOL
        tool_input = {"command": (args or {}).get("command", "") if isinstance(args, dict) else ""}
    else:
        wire_name = tool_name
        tool_input = args if isinstance(args, dict) else {"value": args}

    payload = {
        "hook_event_name": _WIRE_EVENT[event],
        "tool_name": wire_name,
        "tool_input": tool_input,
    }
    if event == POST_TOOL_USE:
        # Fire-and-forget result, for logging/audit hooks. Anything not
        # already a string is serialised so the payload is always JSON-safe.
        payload["tool_response"] = (
            result if isinstance(result, str) else json.dumps(result, default=str)
        )
    return payload


# ── running a hook ───────────────────────────────────────────────────────────


def _fail_closed(handler: HookHandler) -> bool:
    if handler.fail_open is None:
        return PRE_TOOL_USE_FAIL_CLOSED
    return not handler.fail_open


def _failure(handler: HookHandler, reason: str) -> HookVerdict:
    """A hook we could not get an answer from."""
    if _fail_closed(handler):
        warn(f"Hook failed ({reason}); blocking the tool call")
        return HookVerdict("deny", f"Hook failed: {reason}")
    warn(f"Hook failed ({reason}); allowing the tool call")
    return HookVerdict("allow")


def _parse_json_verdict(text: str) -> HookVerdict | None:
    """Parse a hook's stdout. ``None`` means it was not JSON at all."""
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None

    spec = data.get("hookSpecificOutput")
    if not isinstance(spec, dict):
        # A flat top-level decision (Copilot's shape) is accepted too.
        spec = data
    decision = spec.get("permissionDecision", data.get("decision"))
    reason = spec.get("permissionDecisionReason") or data.get("reason") or ""
    if isinstance(decision, str):
        lowered = decision.strip().lower()
        if lowered in ("deny", "block"):
            return HookVerdict("deny", reason or "blocked by hook")
        if lowered == "ask":
            return HookVerdict("ask", reason or "review requested by hook")
    # Valid JSON carrying no blocking opinion: a warning, or an explicit allow.
    return HookVerdict("allow")


def run_handler(handler: HookHandler, payload: dict) -> HookVerdict:
    """Run one hook command and normalise its answer.

    Never raises: a hook is an external process, and its failure modes are
    answers too (see the module docstring for the fail-open/fail-closed split).

    The verdict is stamped with the handler's ``label``, because the UI names
    the hook that spoke when it reports a block -- a frame attributing the
    message to "a hook" is exactly the confusion this is here to avoid.
    """
    verdict = _run_handler(handler, payload)
    verdict.source = verdict.source or handler.label
    return verdict


def _run_handler(handler: HookHandler, payload: dict) -> HookVerdict:
    body = json.dumps(payload).encode("utf-8")
    try:
        proc = subprocess.run(
            handler.command,
            shell=True,
            input=body,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=handler.timeout,
        )
    except subprocess.TimeoutExpired:
        return _failure(handler, f"timed out after {handler.timeout:g}s")
    except OSError as e:  # pragma: no cover - shell=True rarely raises here
        return _failure(handler, f"could not be started: {e}")

    out = proc.stdout.decode("utf-8", "replace").strip()
    err = proc.stderr.decode("utf-8", "replace").strip()

    if proc.returncode == 0:
        if not out:
            return HookVerdict("allow")  # silent = no opinion
        verdict = _parse_json_verdict(out)
        if verdict is None:
            return _failure(handler, f"unparseable output: {out[:200]!r}")
        return verdict

    if proc.returncode == 2:
        # Claude Code's blocking exit: the reason is on stderr.
        return HookVerdict("deny", err or "blocked by hook (exit 2)")

    # Any other non-zero exit is a *non-blocking* hook error per the Claude
    # contract, and this is where a shell's "command not found" (127) lands --
    # a broken install warns rather than refusing every tool call.
    warn(f"Hook exited {proc.returncode}: {err or out or 'no output'}")
    return HookVerdict("allow")


# ── orchestration, called from Core._tool_calls ──────────────────────────────


def run_pre_hooks(
    groups: list[HookGroup], tool_name: str, args, unsafe: bool = False
) -> HookVerdict:
    """Run every matching pre hook; the first blocking verdict wins.

    Groups run in config order and a deny is not cancellable by a later hook
    that allows: only a deny proves the call unsafe, so the narrowest guard
    decides. An ``ask`` is returned immediately too -- the caller prompts the
    user, and there is nothing a later hook could add to that.
    """
    for group in groups:
        if group.event != PRE_TOOL_USE or not matcher_matches(group.matcher, tool_name):
            continue
        for handler in group.handlers:
            if unsafe and handler.honor_unsafe:
                continue
            verdict = run_handler(handler, build_payload(PRE_TOOL_USE, tool_name, args))
            if verdict.decision in ("deny", "ask"):
                return verdict
    return HookVerdict("allow")


def run_post_hooks(
    groups: list[HookGroup], tool_name: str, args, result, unsafe: bool = False
) -> None:
    """Run every matching post hook, ignoring what they answer.

    Fire-and-forget by design: a logging or audit hook gets to see the result
    but cannot veto it, and anything it raises is swallowed so it cannot break
    a turn that has already done its work.
    """
    for group in groups:
        if group.event != POST_TOOL_USE or not matcher_matches(group.matcher, tool_name):
            continue
        for handler in group.handlers:
            if unsafe and handler.honor_unsafe:
                continue
            try:
                run_handler(handler, build_payload(POST_TOOL_USE, tool_name, args, result))
            except Exception as e:  # pragma: no cover - run_handler swallows its own
                warn(f"Post-tool hook failed: {e}")
