"""Tests for how RichOutputAdapter's prompts interact with the terminal.

Three properties, each a real bug that was fixed once:

1. A prompt reading through ``readline`` (the builtin ``input()``) must have
   ``readline`` imported, or arrow keys are echoed as ``^[[C``.
2. It must not run with the kitty disambiguate flag pushed, or a Left arrow
   arrives as ``CSI 1;129D`` and is typed in as escape characters.
3. It must hand readline the prompt, or readline believes the buffer starts at
   column 1 and a Backspace erases the question. (That last one is only
   observable over a real pty -- see ``tests/test_readline_prompt.py``.)

``ask_choice`` is prompt_toolkit: it renders and reads the line itself, so it
needs none of the three and keeps Shift+Enter.
"""

import contextlib
import unittest
from unittest import mock

from agent import keys
from agent.output import RichOutputAdapter


def _recorder(seen: list[bool]):
    """A stand-in prompt class that records the flag state when asked."""
    return mock.Mock(ask=mock.Mock(side_effect=lambda *a, **k: seen.append(
        keys.protocol_pushed()) or "answer"))


# The prompt classes the adapter actually calls. They are the readline-aware
# subclasses defined in agent.output, not Rich's originals.
PROMPTS = {
    "ask_string": "_ReadlinePrompt",
    "ask_float": "_ReadlineFloatPrompt",
    "ask_confirm": "_ReadlineConfirm",
}


class PromptKeyEncodingTestCase(unittest.TestCase):
    def setUp(self):
        self.seen: list[bool] = []
        patches = [
            mock.patch.object(keys, "_protocol_pushed", False),
            mock.patch.object(keys, "_legacy_encoding_depth", 0),
            # The pop/push pair is exercised in test_keys; here we only care
            # that the prompt runs while the flag is down.
            mock.patch.object(keys, "disable_kitty_keyboard", return_value=None),
            mock.patch.object(keys, "enable_kitty_keyboard", return_value=True),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        # Steer and footer are irrelevant to the key protocol; skip __init__ so
        # no Config singleton or real terminal is needed.
        self.ui = RichOutputAdapter.__new__(RichOutputAdapter)
        self.ui._steer = None
        self.ui._footer = mock.Mock(active=False)
        self.ui._footer_status_cache = ("", "")

    def _call(self, attr, *args):
        target = f"agent.output.{PROMPTS[attr]}"
        with mock.patch(target, _recorder(self.seen)):
            getattr(self.ui, attr)(*args)


class TestReadlinePromptsEnsureALineEditor(PromptKeyEncodingTestCase):
    """Each readline-backed prompt must load readline before delegating.

    Without it ``input()`` is a bare canonical read and arrow keys are echoed
    as ``^[[C`` rather than moving the cursor -- the actual bug.
    """

    def _call_ensuring(self, attr, *args):
        with mock.patch(
            f"agent.output.{PROMPTS[attr]}", _recorder(self.seen)
        ), mock.patch("agent.output.ensure_readline", return_value=True) as ensure:
            getattr(self.ui, attr)(*args)
        return ensure

    def test_ask_string_loads_readline(self):
        self._call_ensuring("ask_string", "URL").assert_called_once()

    def test_ask_float_loads_readline(self):
        self._call_ensuring("ask_float", "t", 0.5).assert_called_once()

    def test_ask_confirm_loads_readline(self):
        self._call_ensuring("ask_confirm", "ok?", False).assert_called_once()

    def test_ask_choice_does_not_need_it(self):
        # prompt_toolkit brings its own line editor; importing readline would
        # only risk the two competing over the tty.
        with mock.patch("agent.output.choice", return_value="generic"), mock.patch(
            "agent.output.ensure_readline"
        ) as ensure:
            self.ui.ask_choice("Provider:", [("generic", "Generic")])
        ensure.assert_not_called()


class TestReadlinePromptsDropTheFlag(PromptKeyEncodingTestCase):
    """rich.prompt -> builtin input() -> readline: legacy encodings only."""

    def test_ask_string_runs_with_the_flag_popped(self):
        self._call("ask_string", "URL")
        self.assertEqual(self.seen, [False])

    def test_ask_float_runs_with_the_flag_popped(self):
        self._call("ask_float", "temp", 0.5)
        self.assertEqual(self.seen, [False])

    def test_ask_confirm_runs_with_the_flag_popped(self):
        self._call("ask_confirm", "ok?", False)
        self.assertEqual(self.seen, [False])

    def test_the_context_manager_is_released_afterwards(self):
        self._call("ask_string", "URL")
        self.assertEqual(keys._legacy_encoding_depth, 0)


class TestReadlinePromptsKnowTheirPrompt(unittest.TestCase):
    """The prompt must be handed to ``input()``, not printed before it.

    Rich prints the question and calls ``input()`` with nothing, so readline
    thinks the line buffer starts at column 1: the first redraw rewrites from
    there and erases the question.
    """

    def setUp(self):
        self.ui = RichOutputAdapter.__new__(RichOutputAdapter)
        self.ui._steer = None
        self.ui._footer = mock.Mock(active=False)
        self.ui._footer_status_cache = ("", "")
        patches = [
            mock.patch.object(keys, "_protocol_pushed", False),
            mock.patch.object(keys, "_legacy_encoding_depth", 0),
            mock.patch("agent.output.ensure_readline", return_value=True),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _call(self, attr, *args):
        # The answer must be valid for the prompt type, or Rich's retry loop
        # never terminates ("Please enter Y or N", forever).
        answer = {"ask_string": "answer", "ask_float": "0.7", "ask_confirm": "y"}[
            attr
        ]
        with mock.patch("agent.output.readline_prompt", return_value="P: ") as rp, \
                mock.patch("builtins.input", return_value=answer) as inp:
            result = getattr(self.ui, attr)(*args)
        return result, rp, inp

    def test_ask_string_passes_the_prompt_to_input(self):
        value, rp, inp = self._call("ask_string", "URL")
        rp.assert_called_once()
        self.assertEqual(inp.call_args.args, ("P: ",))
        self.assertEqual(value, "answer")

    def test_ask_float_passes_the_prompt_to_input(self):
        value, _, inp = self._call("ask_float", "temp", 0.5)
        self.assertEqual(inp.call_args.args, ("P: ",))
        self.assertEqual(value, 0.7)

    def test_ask_confirm_passes_the_prompt_to_input(self):
        value, _, inp = self._call("ask_confirm", "ok?", False)
        self.assertEqual(inp.call_args.args, ("P: ",))
        self.assertIs(value, True)

    def test_a_password_prompt_still_goes_through_getpass(self):
        # getpass does its own terminal handling; readline must not touch it.
        import io

        from rich.console import Console
        from rich.text import Text

        from agent.output import _ReadlinePrompt

        # A real Console: Rich's Console.input dispatches to getpass on
        # password=True, and a Mock would return a Mock and spin the retry loop.
        console = Console(file=io.StringIO())
        with mock.patch("getpass.getpass", return_value="s3cret") as gp, \
                mock.patch("builtins.input") as inp:
            value = _ReadlinePrompt.get_input(console, Text("Password"), True)
        gp.assert_called_once()
        inp.assert_not_called()
        self.assertEqual(value, "s3cret")


class TestRunSubprocessDropsTheFlag(PromptKeyEncodingTestCase):
    def test_a_child_process_gets_legacy_encodings(self):
        # $EDITOR never asked for the protocol; under it, arrow keys arrive
        # re-encoded and are read as text.
        popped: list[bool] = []

        real = keys.legacy_key_encoding

        @contextlib.contextmanager
        def spy():
            with real():
                popped.append(keys.protocol_pushed())
                yield

        with mock.patch("subprocess.run") as run, mock.patch(
            "agent.output.legacy_key_encoding", spy
        ):
            self.ui.run_subprocess(["vi"])

        self.assertEqual(popped, [False])
        self.assertTrue(run.called)


class TestAskChoiceKeepsTheFlag(PromptKeyEncodingTestCase):
    def test_prompt_toolkit_parses_the_sequences_itself(self):
        seen: list[bool] = []
        keys._protocol_pushed = True  # as if the REPL prompt had pushed it

        def fake_choice(**kwargs):
            seen.append(keys.protocol_pushed())
            return "generic"

        with mock.patch("agent.output.choice", fake_choice):
            self.ui.ask_choice("Provider:", [("generic", "Generic")])
        self.assertEqual(seen, [True])


if __name__ == "__main__":
    unittest.main()