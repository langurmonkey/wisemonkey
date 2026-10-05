"""Tests for the per-turn UI stage state in agent/agent.py.

Focus: the prompt stage fires once per LLM round (`Core.run_turn` loops back
to the model after every tool batch), and the spinner + completion line must
only appear for the *first* round of a turn.
"""

import unittest
from typing import Any, Callable, cast
from unittest import mock

from agent.agent import Agent


class _FakeFooterOutput:
    """Minimal OutputAdapter stand-in recording what the UI asked for."""

    def __init__(self, footer_active: bool = True) -> None:
        self._footer_active = footer_active
        self.spinner_labels: list[str] = []
        self.refreshes = 0

    # `Agent._spinner` reaches for `_console` on the output adapter; the fake
    # just needs the attribute to exist (and to be replaceable).
    _console: Any = None

    def footer_active(self) -> bool:
        return self._footer_active

    def footer_update(self, status_line: str, input_line: str = "") -> None:
        self.refreshes += 1

    def footer_spinner(self, text: str) -> None:
        """Record the label the agent handed to the footer's own animation."""
        self.spinner_labels.append(text)

    def steer_line(self) -> str:
        return ""

    def steer_pending(self) -> str | None:
        return None


class _SpyAgent(Agent):
    """An Agent with the network, config and console stubbed out."""

    def __init__(self, footer_active: bool = True) -> None:  # noqa: D107
        self.spinner_prompt = None
        self.spinner_thinking = None
        self._footer_spinner: str | None = None
        self._footer_spinner_mine = False
        self._prompt_reported = False
        self._prompt_rounds = 0
        self._turn_in_progress = True
        self._fake_output = _FakeFooterOutput(footer_active)
        self.output = self._fake_output

    def _footer_refresh(self) -> None:
        """The redraw needs the real footer plumbing; not under test here."""
        self._fake_output.refreshes += 1

    def _spinner_started(self) -> list[str]:
        """Labels the agent asked the footer to animate ("" means stop)."""
        return [t for t in self._fake_output.spinner_labels if t]

    def _spinner_stops(self) -> int:
        """How many times the agent told the footer to stop animating."""
        return sum(1 for t in self._fake_output.spinner_labels if not t)


class TestPromptStage(unittest.TestCase):
    """`Processing prompt...` belongs to the user's prompt only."""

    def _agent(self, footer_active: bool = True) -> _SpyAgent:
        return _SpyAgent(footer_active)

    def test_first_round_shows_the_spinner(self):
        agent = self._agent()
        agent._prompt_start()
        self.assertEqual(agent._spinner_started(), ["⏳ Processing prompt..."])

    def test_later_rounds_do_not_repeat_the_spinner(self):
        """A tool round re-enters the LLM; no second "Processing prompt"."""
        agent = self._agent()
        agent._prompt_start()
        with mock.patch("agent.agent.ok"):
            agent._prompt_stop()
        agent._prompt_start()  # round 2, after a tool batch
        self.assertEqual(agent._spinner_started(), ["⏳ Processing prompt..."])
        self.assertEqual(agent._prompt_rounds, 2)

    def test_many_tool_rounds_still_show_one_spinner(self):
        agent = self._agent()
        for _ in range(5):
            agent._prompt_start()
            agent._prompt_stop()
        self.assertEqual(agent._spinner_started(), ["⏳ Processing prompt..."])

    def test_completion_line_printed_once(self):
        agent = self._agent()
        with mock.patch("agent.agent.ok") as printed:
            for _ in range(3):
                agent._prompt_start()
                agent._prompt_stop()
        self.assertEqual(printed.call_count, 1)

    def test_spinner_cleared_on_stop(self):
        agent = self._agent()
        agent._prompt_start()
        self.assertIsNotNone(agent._footer_spinner)
        with mock.patch("agent.agent.ok"):
            agent._prompt_stop()
        self.assertIsNone(agent._footer_spinner)

    def test_spinner_is_handed_to_the_footer_not_a_live_display(self):
        """While the footer is armed, the label goes to `footer_spinner`.

        A Rich `Live` display would wrap stdout in a FileProxy and turn the
        footer's cursor addressing into literal text on screen.
        """
        agent = self._agent()
        agent._prompt_start()
        self.assertEqual(
            agent._fake_output.spinner_labels, ["⏳ Processing prompt..."]
        )
        agent._prompt_stop()
        self.assertEqual(agent._spinner_stops(), 1)

    def test_no_footer_falls_back_to_a_rich_status(self):
        """With no sticky footer there is no FileProxy hazard, so keep Rich."""
        agent = self._agent(footer_active=False)

        console = mock.MagicMock()
        console.status.return_value = "handle"
        agent.output._console = console
        self.assertEqual(agent._spinner("⏳ Processing prompt..."), "handle")
        console.status.assert_called_once_with("⏳ Processing prompt...")
        self.assertEqual(agent._fake_output.spinner_labels, [])

    def test_second_round_does_not_clear_a_foreign_spinner(self):
        """Round 2 has no spinner of its own, so it must not stop one."""
        agent = self._agent()
        agent._prompt_start()
        with mock.patch("agent.agent.ok"):
            agent._prompt_stop()
        # Something else (a tool) owns the footer spinner now.
        agent._footer_spinner = "🔧 Tool running"
        stops_before = agent._spinner_stops()
        agent._prompt_start()
        with mock.patch("agent.agent.ok"):
            agent._prompt_stop()
        self.assertEqual(agent._footer_spinner, "🔧 Tool running")
        self.assertEqual(agent._spinner_stops(), stops_before)



class _RecordingOutput:
    """OutputAdapter stand-in capturing what the turn printed."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.oks: list[str] = []
        self.rules: list[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def ok(self, msg: str) -> None:
        self.oks.append(msg)

    def rule(self, title: str = "", style: str = "", **kwargs: object) -> None:
        self.rules.append(title)

    def footer_active(self) -> bool:
        return False

    def footer_update(self, status_line: str, input_line: str = "") -> None:
        pass

    def steer_line(self) -> str:
        return ""

    def steer_pending(self) -> str | None:
        return None


class _ToolResultAgent(Agent):
    """An Agent with just enough state for `tool_result_callback`."""

    def __init__(self, threshold: object = None) -> None:
        self.output = _RecordingOutput()

        class _Config:
            def get(self, _key: str, default: object = None) -> object:
                return threshold if threshold is not None else default

        class _Core:
            config = _Config()

        self.core = _Core()


class _SteerOutput:
    """OutputAdapter stand-in with a real queue for the injection seam."""

    def __init__(self, lines: list[str]) -> None:
        self.queue = list(lines)
        self.printed: list[str] = []

    def steer_take_all(self) -> list[str]:
        lines, self.queue = self.queue, []
        return lines

    def steer_pending(self) -> str | None:
        return self.queue[0] if self.queue else None

    def newline(self) -> None:
        pass

    def print(self, text: str, end: str = "\n", indent: int = 0) -> None:
        self.printed.append(text)


class _SteerAgent(Agent):
    """Just enough Agent state for `_steer_inject`."""

    def __init__(self, lines: list[str]) -> None:
        self.out = _SteerOutput(lines)
        self.output: Any = self.out
        self._turn_in_progress = True
        self.refreshes = 0

    def _footer_refresh(self) -> None:
        self.refreshes += 1


class TestSteerInjection(unittest.TestCase):
    """A delivered line must not leave `↳ queued:` on the footer row.

    `_announce_steered` echoes the line on stdout (in blue), but the footer
    keeps showing the queued text until something redraws that row. Without a
    redraw here, the user sees "queued: <text>" next to a line that has
    obviously already been delivered.
    """

    def test_delivery_redraws_the_footer(self):
        agent = _SteerAgent(["do this instead"])
        lines = agent._steer_inject()
        self.assertEqual(lines, ["do this instead"])
        self.assertEqual(agent.refreshes, 1)
        self.assertEqual(len(agent.out.printed), 1)

    def test_the_queued_row_is_stale_after_delivery(self):
        agent = _SteerAgent(["a", "b"])
        agent._steer_inject()
        self.assertIsNone(agent.out.steer_pending())

    def test_an_empty_queue_still_refreshes(self):
        """Cheap and keeps the seam free of special cases."""
        agent = _SteerAgent([])
        self.assertEqual(agent._steer_inject(), [])
        self.assertEqual(agent.refreshes, 1)


class TestToolResultLine(unittest.TestCase):
    """A successful fast tool is silent; slow and failed ones are not.

    The activation line ("Activating tool: read_file") already says the tool
    ran. A "finished in 0.0s" line under every call turns a fifteen-tool turn
    into fifteen lines of noise, so the duration is only worth printing when
    the tool was actually slow enough to make you wait.
    """

    def _run(self, *, threshold=1.0, duration=0.1, is_error=False):
        agent = _ToolResultAgent(threshold)
        agent.tool_result_callback("1", "read_file", "ok", is_error, duration)
        return agent

    def test_fast_success_prints_nothing(self):
        agent = self._run(duration=0.05)
        self.assertEqual(agent.output.oks, [])
        self.assertEqual(agent.output.errors, [])

    def test_slow_success_prints_the_duration(self):
        agent = self._run(duration=2.4)
        self.assertEqual(len(agent.output.oks), 1)
        self.assertIn("read_file", agent.output.oks[0])
        self.assertIn("2.4s", agent.output.oks[0])

    def test_threshold_is_inclusive(self):
        """Exactly at the threshold still counts as slow."""
        self.assertEqual(self._run(duration=1.0).output.oks,
                         ["Tool read_file finished in 1.0s"])

    def test_error_is_always_reported(self):
        agent = self._run(is_error=True, duration=0.01)
        self.assertEqual(agent.output.oks, [])
        self.assertEqual(len(agent.output.errors), 1)
        self.assertIn("failed", agent.output.errors[0])

    def test_error_line_carries_the_duration_too(self):
        agent = self._run(is_error=True, duration=3.0)
        self.assertIn("3.0s", agent.output.errors[0])

    def test_zero_threshold_always_prints(self):
        self.assertEqual(len(self._run(threshold=0, duration=0.0).output.oks), 1)

    def test_bad_threshold_falls_back_to_the_default(self):
        agent = _ToolResultAgent("not a number")
        agent._tool_slow_threshold()
        self.assertEqual(agent._tool_slow_threshold(), 1.0)

    def test_negative_threshold_is_clamped(self):
        self.assertEqual(self._run(threshold=-5)._tool_slow_threshold(), 0.0)


class _StatsMemory:
    """Memory stand-in whose chat stats the test can set."""

    session = "test"

    def __init__(self, used: int, max_tokens: int) -> None:
        self.stats = (used, max_tokens, used * 100.0 / max_tokens)

    def get_chat_stats(self) -> tuple[int, int, float]:
        return self.stats


class _PricingConfig:
    """Config stand-in; `pricing` is what `model.pricing` returns."""

    def __init__(self, pricing: dict[str, float] | None = None) -> None:
        self.pricing = pricing

    def get(self, _key: str, default: object = None) -> object:
        return self.pricing if self.pricing is not None else default


class _LiveTokenAgent(Agent):
    """An Agent stubbed for the live token counter and the status line."""

    def __init__(
        self,
        used: int = 1000,
        max_tokens: int = 200000,
        pricing: dict[str, float] | None = None,
    ) -> None:
        self.output = _RecordingOutput()
        self.remote = None
        self._md_stream = None
        self._turn_in_progress = False
        self._turn_start_mem = used
        self._live_chars = 0
        self._last_footer_tick = 0.0

        class _Core:
            config = _PricingConfig(pricing)
            memory = _StatsMemory(used, max_tokens)

        self.core = _Core()


class TestLiveTokenCounter(unittest.TestCase):
    """The token count should move *while* the model streams.

    A cumulative figure updated at turn end looks identical whether the turn
    was trivial or enormous. The delta -- what this turn added -- is the
    number that climbs, so it is shown live in the footer (estimated from
    streamed characters) and exactly in the status line at turn end.
    """

    def _agent(self, used: int = 1000, pricing: dict[str, float] | None = None):
        return _LiveTokenAgent(used=used, pricing=pricing)

    def test_statusline_shows_the_turn_delta(self):
        agent = self._agent(used=1000)
        cast(_StatsMemory, agent.core.memory).stats = (4200, 200000, 2.1)
        agent._statusline(3200, 0, 1.0)
        rule = cast(_RecordingOutput, agent.output).rules[-1]
        self.assertIn("Δ+3200", rule)

    def test_statusline_omits_the_delta_when_nothing_was_added(self):
        agent = self._agent(used=1000)
        cast(_StatsMemory, agent.core.memory).stats = (1000, 200000, 0.5)
        agent._statusline(0, 0, 1.0)
        self.assertNotIn("Δ", cast(_RecordingOutput, agent.output).rules[-1])

    def test_statusline_shows_provider_prompt_tokens(self):
        agent = self._agent()
        agent._statusline(500, 0, 1.0, prompt_tokens=12000)
        self.assertIn("p:12.0k", agent.output.rules[-1])

    def test_statusline_omits_prompt_tokens_when_unreported(self):
        """A backend that reports no usage must not show a fake number."""
        agent = self._agent()
        agent._statusline(500, 0, 1.0, prompt_tokens=0)
        self.assertNotIn("p:", agent.output.rules[-1])

    def test_cost_is_shown_only_when_pricing_is_configured(self):
        agent = _LiveTokenAgent(pricing={"input": 3.0, "output": 15.0})
        # 12k prompt at $3/Mtok + 500 completion at $15/Mtok = $0.0435
        agent._statusline(500, 0, 1.0, prompt_tokens=12000)
        output = cast(_RecordingOutput, agent.output)
        self.assertIn("$0.0435", output.rules[-1])

    def test_no_pricing_means_no_cost(self):
        """A made-up price would be worse than no price."""
        agent = self._agent()
        agent._statusline(500, 0, 1.0, prompt_tokens=12000)
        self.assertNotIn("$", agent.output.rules[-1])

    def test_live_tick_accumulates_streamed_characters(self):
        agent = self._agent()
        agent._turn_in_progress = True
        agent.content_callback("hello world ")
        agent.content_callback("more text")
        self.assertEqual(agent._live_chars, len("hello world more text"))

    def test_live_tick_throttles_footer_refreshes(self):
        """Per-chunk redraws would fight the spinner thread for the lock."""
        agent = self._agent()
        agent._turn_in_progress = True
        agent.output.footer_active = lambda: True
        agent.output.footer_update = lambda *a, **k: setattr(
            agent.output, "n", getattr(agent.output, "n", 0) + 1
        )
        agent._last_footer_tick = __import__("time").time() - 1.0
        for _ in range(50):
            agent.content_callback("x")
        # 50 chunks inside the 0.25 s window must produce at most one redraw.
        self.assertLessEqual(getattr(agent.output, "n", 0), 1)

    def test_live_tick_is_silent_outside_a_turn(self):
        agent = self._agent()
        agent._turn_in_progress = False
        agent.output.footer_active = lambda: True
        agent.output.footer_update = lambda *a, **k: setattr(
            agent.output, "n", getattr(agent.output, "n", 0) + 1
        )
        agent.content_callback("hello")
        self.assertEqual(getattr(agent.output, "n", 0), 0)

    def test_fmt_k(self):
        self.assertEqual(Agent._fmt_k(823), "823")
        self.assertEqual(Agent._fmt_k(12400), "12.4k")


if __name__ == "__main__":
    unittest.main()


class TestTurnHeaderOrdering(unittest.TestCase):
    """The footer must be armed *before* anything is printed for the turn.

    ``Footer.start()`` blanks the bottom ``FOOTER_LINES`` rows and sets the
    scroll region. If output is printed first, a screen that is already
    full puts it on one of those rows and ``start()`` erases it -- which is
    how the "Wisemonkey <model>" line intermittently disappeared. The turn
    header is gone now (the model lives in the footer), but the invariant
    is not: the blank line that separates the turn from the prompt above it
    is still printed after ``footer_start()``, and it would vanish the same
    way.

    These tests pin the *ordering* in the source, since the failure only
    shows up on a full screen and cannot be reproduced in a StringIO.
    """

    @staticmethod
    def _call_order(func: Callable[..., Any]) -> list[str]:
        """Attribute names of every call in *func*, in source order."""
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        calls.sort(key=lambda n: (n.lineno, n.col_offset))
        return [
            n.func.attr
            for n in calls
            if isinstance(n.func, ast.Attribute)
        ]

    def test_local_turn_arms_footer_before_printing_anything(self):
        order = self._call_order(Agent.run_interactive)
        self.assertLess(
            order.index("footer_start"),
            len(order) - 1 - order[::-1].index("newline"),
        )

    def test_remote_turn_arms_footer_before_printing_anything(self):
        order = self._call_order(Agent._run_interactive_remote)
        self.assertLess(
            order.index("footer_start"),
            len(order) - 1 - order[::-1].index("newline"),
        )

    def test_footer_stops_after_the_turn(self):
        """The footer is disarmed in the finally block, after the header."""
        for func in (Agent.run_interactive, Agent._run_interactive_remote):
            order = self._call_order(func)
            self.assertIn("footer_stop", order)
            self.assertLess(
                order.index("footer_start"), order.index("footer_stop")
            )


if __name__ == "__main__":
    unittest.main()
