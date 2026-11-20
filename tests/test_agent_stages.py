"""Tests for the per-turn UI stage state in agent/agent.py."""

import time
import unittest
from collections import deque
from types import SimpleNamespace
from typing import Any, Callable, cast
from unittest import mock

from agent.agent import Agent
from agent.footer import _CURSOR, _RESET, visible_len
from agent.ipc import Event, Message, ReasoningPayload, StageKind
from agent.palette import PALETTE
from agent.utils import term_width


class _FakeFooterOutput:
    """Minimal OutputAdapter stand-in recording what the UI asked for."""

    def __init__(self, footer_active: bool = True) -> None:
        self._footer_active = footer_active
        self.spinner_labels: list[str] = []
        self.refreshes = 0
        self.thinking: list[bool] = []
        self._console: Any = None  # `Agent._spinner` reaches for it

    def footer_active(self) -> bool:
        return self._footer_active

    def footer_update(self, status_line: str, input_line: str = "") -> None:
        self.refreshes += 1

    def footer_spinner(self, text: str) -> None:
        self.spinner_labels.append(text)

    def footer_thinking(self, active: bool) -> None:
        self.thinking.append(active)

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
        self._reasoning_active = False
        self._reasoning_mid_line = False
        self._fake_output = _FakeFooterOutput(footer_active)
        self.output = self._fake_output

    def _footer_refresh(self) -> None:
        """The redraw needs the real footer plumbing; not under test here."""
        self._fake_output.refreshes += 1

    def _spinner_started(self) -> list[str]:
        return [t for t in self._fake_output.spinner_labels if t]

    def _spinner_stops(self) -> int:
        return sum(1 for t in self._fake_output.spinner_labels if not t)


class TestPromptStage(unittest.TestCase):
    """`Processing prompt...` belongs to the user's prompt only."""

    def test_one_spinner_across_many_rounds_and_never_a_completion_line(self):
        """A tool round re-enters the LLM; no second "Processing prompt"."""
        agent = _SpyAgent()
        with mock.patch("agent.agent.print") as printed:
            for _ in range(5):
                agent._prompt_start()
                agent._prompt_stop()
        self.assertEqual(agent._spinner_started(), ["⏳ Processing prompt..."])
        self.assertEqual(agent._spinner_stops(), 1)
        self.assertEqual(agent._prompt_rounds, 5)
        self.assertIsNone(agent._footer_spinner)
        # "✓ Prompt processed" was removed: on a full screen it overwrites the
        # last line of the user's own prompt.
        printed.assert_not_called()

    def test_the_footer_owns_the_spinner_and_a_foreign_one_is_left_alone(self):
        """With no sticky footer there is no FileProxy hazard, so keep Rich."""
        agent = _SpyAgent(footer_active=False)
        console = mock.MagicMock()
        console.status.return_value = "handle"
        agent.output._console = console
        self.assertEqual(agent._spinner("⏳ Processing prompt..."), "handle")
        console.status.assert_called_once_with("⏳ Processing prompt...")
        self.assertEqual(agent._fake_output.spinner_labels, [])

        # Round 2 has no spinner of its own, so it must not stop one a tool owns.
        agent = _SpyAgent()
        agent._prompt_start()
        agent._prompt_stop()
        agent._footer_spinner = "🔧 Tool running"
        stops = agent._spinner_stops()
        agent._prompt_start()
        agent._prompt_stop()
        self.assertEqual(agent._footer_spinner, "🔧 Tool running")
        self.assertEqual(agent._spinner_stops(), stops)


class TestReasoningFooterIndicator(unittest.TestCase):
    """Thinking status is shown on the footer's key-hints row, not in chat."""

    def test_reasoning_drives_the_indicator_and_closes_its_line(self):
        agent = _SpyAgent()
        with (
            mock.patch("agent.agent.info") as info,
            mock.patch("agent.agent.print"),
            mock.patch("agent.agent.newline") as newline,
        ):
            agent._reasoning_start(visible=True)
            self.assertTrue(agent._reasoning_active)
            event = Message.event(
                Event.REASONING,
                ReasoningPayload(stage=StageKind.PROCESS, text="thinking", visible=True),
            )
            agent._handle_event(event)
            self.assertTrue(agent._reasoning_mid_line)
            agent._reasoning_stop()
        # The block streamed mid-line, so it must be closed with one newline.
        newline.assert_called_once()
        self.assertFalse(agent._reasoning_active)
        self.assertFalse(agent._reasoning_mid_line)
        self.assertEqual(agent._fake_output.thinking, [True, False])
        info.assert_not_called()

        # A block already at a line boundary, or an invisible one, adds none.
        for text, visible in (("thinking\n", True), ("thinking", False)):
            agent = _SpyAgent()
            agent._reasoning_start(visible=visible)
            with mock.patch("agent.agent.print"), mock.patch(
                "agent.agent.newline"
            ) as newline:
                agent._handle_event(Message.event(
                    Event.REASONING,
                    ReasoningPayload(stage=StageKind.PROCESS, text=text, visible=visible),
                ))
                agent._reasoning_stop()
            newline.assert_not_called()


class _RecordingOutput:
    """OutputAdapter stand-in capturing what the turn printed."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.oks: list[str] = []
        self.rules: list[str] = []
        # Terminal width stand-in; the status line picks a terse variant to fit.
        self._console = SimpleNamespace(width=200)

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

    def steer_active(self) -> bool:
        return False


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
    """Just enough Agent state for `_steer_inject` and `_steer_requeue`."""

    def __init__(self, lines: list[str]) -> None:
        self.out = _SteerOutput(lines)
        self.output: Any = self.out
        self._turn_in_progress = True
        self.refreshes = 0

    def _footer_refresh(self) -> None:
        self.refreshes += 1


class TestSteeringPlumbing(unittest.TestCase):
    """Delivery redraws the row; leftovers reach the prompt, not the model."""

    def test_delivery_redraws_the_footer(self):
        agent = _SteerAgent(["do this instead"])
        self.assertEqual(agent._steer_inject(), ["do this instead"])
        self.assertEqual(agent.refreshes, 1)
        self.assertEqual(len(agent.out.printed), 1)
        # The row is stale after delivery, or it would read "queued:" forever.
        self.assertIsNone(agent.out.steer_pending())
        # An empty queue still refreshes: cheap, and keeps the seam simple.
        agent = _SteerAgent([])
        self.assertEqual(agent._steer_inject(), [])
        self.assertEqual(agent.refreshes, 1)

    def test_a_queued_line_becomes_the_prompt_text_and_is_drained(self):
        self.assertEqual(_SteerAgent([])._steer_requeue(), "")
        # One queue entry per Enter, so the author's own breaks are the join.
        agent = _SteerAgent(["first thought", "second thought"])
        self.assertEqual(agent._steer_requeue(), "first thought\nsecond thought")
        # Otherwise the line would be pasted again on the next loop pass.
        self.assertIsNone(agent.out.steer_pending())

    def test_the_carry_over_is_announced_as_pasted_not_steered(self):
        """The note is markup-rendered text: an unknown tag would be dropped."""
        import io

        from rich.console import Console

        agent = _SteerAgent(["one", "two"])
        agent._steer_requeue()
        buffer = io.StringIO()
        Console(file=buffer, width=200, force_terminal=False).print(
            agent.out.printed[0]
        )
        rendered = buffer.getvalue()
        self.assertIn("2 queued lines", rendered)
        self.assertIn("Ctrl+U", rendered)  # how to discard it
        self.assertIn("pasted into the prompt", rendered)
        self.assertNotIn("steered", rendered)


class _InputRowOutput:
    """OutputAdapter stand-in for the steering input row."""

    def __init__(
        self,
        line: str = "",
        queued: str | None = None,
        armed: bool = True,
        caret: int | None = None,
    ) -> None:
        self.line = line
        self.queued = queued
        self.armed = armed
        # Default to the end of the line, which is where typing leaves it.
        self.caret = len(line) if caret is None else caret

    def steer_line(self) -> str:
        return self.line

    def steer_caret(self) -> int:
        return self.caret

    def steer_pending(self) -> str | None:
        return self.queued

    def steer_active(self) -> bool:
        return self.armed


class _InputRowAgent(Agent):
    """Just enough Agent state for `_footer_input`."""

    def __init__(self, **kwargs: Any) -> None:
        self.output: Any = _InputRowOutput(**kwargs)
        self._turn_in_progress = True


class TestSteeringInputRow(unittest.TestCase):
    """The steering row is an input field, even while it is empty."""

    _OVER = PALETTE.footer_sgr()["cursor_over"]

    def _row(self, **kwargs: Any) -> str:
        return _InputRowAgent(**kwargs)._footer_input()

    def _caret_at(self, row: str) -> int:
        return row.index(self._OVER) if self._OVER in row else row.index(_CURSOR)

    def test_the_idle_row_is_an_input_field_and_blank_outside_a_turn(self):
        row = self._row()
        self.assertIn("❯", row)
        self.assertIn(_CURSOR, row)
        agent = _InputRowAgent()
        agent._turn_in_progress = False
        self.assertEqual(agent._footer_input(), "")

    def test_the_caret_covers_the_character_it_points_at(self):
        """A caret cell of its own shifted the whole line on every keypress."""
        line = "use pandas"
        row = self._row(line=line, caret=4)  # on the 'p'
        self.assertIn(f"{self._OVER}p{_RESET}", row)
        self.assertIn("use ", row)
        self.assertIn("andas", row)
        # Moving within the line changes no width at all.
        widths = {visible_len(self._row(line=line, caret=i)) for i in range(len(line))}
        self.assertEqual(widths, {visible_len(self._row(line=line, caret=0))})
        # The block is drawn *in* the text, not always after it: " ❯ use".
        row = self._row(line=line, caret=3)
        self.assertEqual(visible_len(row[: self._caret_at(row)]), 6)
        self.assertGreater(row.index("pandas"), self._caret_at(row))
        # Past the last character there is nothing to paint over, so the cell
        # takes a column of its own -- the one `steer_window` reserves.
        self.assertEqual(
            visible_len(self._row(line=line, caret=len(line))),
            visible_len(self._row(line=line, caret=0)) + 1,
        )

    def test_queued_versus_typed_and_a_paused_reader(self):
        row = self._row(queued="wait, use polars")
        self.assertIn("↳", row)
        self.assertIn("queued: wait, use polars", row)
        row = self._row(line="typing now", queued="older line")
        self.assertIn("typing now", row)
        self.assertNotIn("older line", row)

        # A paused reader (an ask_*, a subprocess) must not invite typing.
        self.assertEqual(self._row(armed=False), "")
        row = self._row(line="half typed", armed=False)
        self.assertIn("half typed", row)
        self.assertNotIn(_CURSOR, row)
        self.assertNotIn(self._OVER, row)

    def test_a_long_line_scrolls_to_keep_the_caret_visible(self):
        row = self._row(line="x" * 500)
        self.assertIn("…", row)
        self.assertLessEqual(visible_len(row), term_width())
        self.assertIn(_CURSOR, row)  # the caret is past the end of the window
        # A caret in the middle cuts both ends, keeping what is being edited.
        row = self._row(line="abc" + "-" * 500 + "xyz", caret=503)
        self.assertEqual(row.count("…"), 2)
        self.assertIn(self._OVER, row)
        self.assertLessEqual(visible_len(row), term_width())
        self.assertNotIn("abc", row)


class TestToolResultLine(unittest.TestCase):
    """A successful fast tool is silent; slow and failed ones are not."""

    def _run(self, *, threshold=1.0, duration=0.1, is_error=False):
        agent = _ToolResultAgent(threshold)
        agent.tool_result_callback("1", "read_file", "ok", is_error, duration)
        return agent

    def test_fast_success_is_silent_and_slow_success_is_not(self):
        agent = self._run(duration=0.05)
        self.assertEqual((agent.output.oks, agent.output.errors), ([], []))
        agent = self._run(duration=2.4)
        self.assertEqual(agent.output.oks, ["Tool read_file finished in 2.4s"])
        # The threshold is inclusive; zero prints everything.
        self.assertEqual(
            self._run(duration=1.0).output.oks, ["Tool read_file finished in 1.0s"]
        )
        self.assertEqual(len(self._run(threshold=0, duration=0.0).output.oks), 1)

    def test_an_error_is_always_reported_and_bad_thresholds_fall_back(self):
        agent = self._run(is_error=True, duration=0.01)
        self.assertEqual(agent.output.oks, [])
        self.assertEqual(agent.output.errors, ["Tool read_file failed (0.0s): ok"])
        self.assertEqual(_ToolResultAgent("not a number")._tool_slow_threshold(), 1.0)
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
        self._reasoning_active = False

        class _Core:
            config = _PricingConfig(pricing)
            memory = _StatsMemory(used, max_tokens)

        self.core = _Core()


class TestLiveTokenCounter(unittest.TestCase):
    """The token count should move *while* the model streams."""

    def _agent(self, used=1000, pricing=None, max_tokens=200000):
        return _LiveTokenAgent(used=used, max_tokens=max_tokens, pricing=pricing)

    @staticmethod
    def _seed_rate(agent, samples: list[tuple[float, int]]) -> None:
        """Plant ``(age_seconds, chars)`` samples in the rate window."""
        now = time.monotonic()
        agent._rate_samples = deque((now - age, chars) for age, chars in samples)

    def test_statusline_shows_and_omits_the_turn_delta(self):
        agent = self._agent(used=1000)
        cast(_StatsMemory, agent.core.memory).stats = (4200, 200000, 2.1)
        agent._statusline(3200, 0, 1.0)
        self.assertIn("Δ+3200", agent.output.rules[-1])
        cast(_StatsMemory, agent.core.memory).stats = (1000, 200000, 0.5)
        agent._statusline(0, 0, 1.0)
        self.assertNotIn("Δ", agent.output.rules[-1])

    def test_prompt_tokens_and_cost_are_omitted_when_unknown(self):
        agent = self._agent(pricing={"input": 3.0, "output": 15.0})
        # 12k prompt at $3/Mtok + 500 completion at $15/Mtok = $0.0435
        agent._statusline(500, 0, 1.0, prompt_tokens=12000)
        self.assertIn("p:12.0k", agent.output.rules[-1])
        self.assertIn("$0.0435", agent.output.rules[-1])
        # A backend that reports nothing must not show a made-up number.
        agent = self._agent()
        agent._statusline(500, 0, 1.0, prompt_tokens=0)
        self.assertNotIn("p:", agent.output.rules[-1])
        self.assertNotIn("$", agent.output.rules[-1])

    def test_footer_shows_the_current_rate_after_the_delta(self):
        agent = self._agent(used=1000)
        agent._turn_in_progress = True
        cast(_StatsMemory, agent.core.memory).stats = (4200, 200000, 2.1)
        # 2000 chars (= 500 tokens) spread over the last 5 s -> ~100 tok/s.
        self._seed_rate(agent, [(5.0, 1000), (2.5, 1000)])
        line = agent._footer_status()
        self.assertIn("~", line)
        self.assertIn("100", line.split("Δ+")[1])
        self.assertIn("tok/s", line)
        self.assertLess(line.index("Δ+"), line.index("tok/s"))

    def test_footer_omits_the_rate_without_samples_or_outside_a_turn(self):
        agent = self._agent()
        agent._turn_in_progress = True
        self.assertNotIn("tok/s", agent._footer_status())
        # The rate is per-turn state; it must not linger after the answer.
        agent._turn_in_progress = False
        self._seed_rate(agent, [(4.0, 4000)])
        self.assertNotIn("tok/s", agent._footer_status())

    def test_the_rate_ignores_a_single_burst_prunes_and_decays(self):
        agent = self._agent()
        # One chunk has a span of one round trip, not a decode rate.
        self._seed_rate(agent, [(0.05, 4000)])
        self.assertIsNone(agent._live_rate())
        # The span runs to *now*, so silence empties the window.
        self._seed_rate(agent, [(6.0, 4000), (5.5, 4000)])
        self.assertIsNone(agent._live_rate())
        now = time.monotonic()
        window = deque([(now - 30.0, 100), (now - 1.0, 400)])
        Agent._rate_prune(window, now)
        self.assertEqual([c for _, c in window], [400])

    def test_content_callback_feeds_the_rate_window_and_the_delta(self):
        agent = self._agent()
        agent._turn_in_progress = True
        with mock.patch("agent.agent.print"):
            agent.content_callback("hello")
            agent.content_callback(" world")
        self.assertEqual(sum(c for _, c in agent._rate_samples), len("hello world"))
        self.assertEqual(agent._live_chars, len("hello world"))

    def test_the_live_tick_is_throttled_and_silent_outside_a_turn(self):
        """Per-chunk redraws would fight the spinner thread for the lock."""
        agent = self._agent()
        agent.output.footer_active = lambda: True
        agent.output.footer_update = lambda *a, **k: setattr(
            agent.output, "n", getattr(agent.output, "n", 0) + 1
        )
        with mock.patch("agent.agent.print"):
            for _ in range(50):
                agent.content_callback("x")
            self.assertEqual(getattr(agent.output, "n", 0), 0)  # no turn running

            agent._turn_in_progress = True
            agent._last_footer_tick = time.time() - 1.0
            for _ in range(50):
                agent.content_callback("x")
        self.assertLessEqual(getattr(agent.output, "n", 0), 1)

    def test_statusline_shows_the_decode_rate_only_when_meaningful(self):
        agent = self._agent()
        agent._statusline(1000, 0, 2.0)
        self.assertIn("500 tok/s", agent.output.rules[-1])
        # One chunk is one round trip; no output is no rate at all.
        for tokens, gen_time in ((1000, 0.01), (0, 2.0), (1000, 0.0)):
            agent = self._agent()
            agent._statusline(tokens, 0, gen_time)
            self.assertNotIn("tok/s", agent.output.rules[-1]), (tokens, gen_time)

    def test_the_title_is_shed_to_fit_never_truncated_by_rich(self):
        agent = self._agent(used=13524, max_tokens=120000)
        for width in range(22, 141):
            agent.output._console.width = width
            agent._statusline(690, 0, 4.9, prompt_tokens=40400)
            title = agent.output.rules[-1]
            # A Rule would cut it to "…" and leave a number with no meaning.
            self.assertLessEqual(len(title), width - 2, f"overflow at {width}: {title!r}")
            self.assertIn("690 tokens", title)

        # With room, everything is there.
        agent.output._console.width = 140
        agent._statusline(690, 0, 4.9, prompt_tokens=40400)
        title = agent.output.rules[-1]
        for segment in ("tks (11.27%)", "141 tok/s", "p:40.4k"):
            self.assertIn(segment, title)
        self.assertEqual(Agent._fmt_k(823), "823")
        self.assertEqual(Agent._fmt_k(12400), "12.4k")

        # Cheapest first: the percentage, then the unit, then the memory figure.
        for width, gone, kept in (
            (80, "%", ("Mem: 13524/120000", "tks")),
            (72, "tks", ("Mem: 13524/120000",)),
            (60, "Mem:", ("4.9s", "690 tokens", "141 tok/s")),
        ):
            agent.output._console.width = width
            agent._statusline(690, 0, 4.9, prompt_tokens=40400)
            title = agent.output.rules[-1]
            self.assertNotIn(gone, title, width)
            for segment in kept:
                self.assertIn(segment, title, width)

        # No console at all falls back to 80 columns.
        del agent.output._console
        agent._statusline(690, 0, 4.9, prompt_tokens=40400)
        self.assertLessEqual(len(agent.output.rules[-1]), 78)


class TestTurnHeaderOrdering(unittest.TestCase):
    """The footer must be armed *before* anything is printed for the turn."""

    @staticmethod
    def _call_order(func: Callable[..., Any]) -> list[str]:
        """Attribute names of every call in *func*, in source order."""
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        calls.sort(key=lambda n: (n.lineno, n.col_offset))
        return [n.func.attr for n in calls if isinstance(n.func, ast.Attribute)]

    def test_both_loops_arm_the_footer_before_printing_and_stop_it_after(self):
        for func in (Agent.run_interactive, Agent._run_interactive_remote):
            order = self._call_order(func)
            # `start()` blanks the reserved rows, so anything printed first
            # vanishes whenever the screen was already full.
            self.assertLess(
                order.index("footer_start"),
                len(order) - 1 - order[::-1].index("newline"),
            )
            self.assertLess(order.index("footer_start"), order.index("footer_stop"))


if __name__ == "__main__":
    unittest.main()
