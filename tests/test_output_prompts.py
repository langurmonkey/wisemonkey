"""Tests for how RichOutputAdapter's prompts interact with the terminal."""

import unittest
from unittest import mock

from agent import keys
from agent.output import RichOutputAdapter

# The prompt classes the adapter calls: the readline-aware subclasses in
# agent.output, not Rich's originals.
PROMPTS = {
    "ask_string": "_ReadlinePrompt",
    "ask_float": "_ReadlineFloatPrompt",
    "ask_confirm": "_ReadlineConfirm",
}
ARGS = {
    "ask_string": ("URL",),
    "ask_float": ("temp", 0.5),
    "ask_confirm": ("ok?", False),
}
ANSWERS = {"ask_string": "answer", "ask_float": "0.7", "ask_confirm": "y"}


class _Adapter(unittest.TestCase):
    def setUp(self):
        patches = [
            mock.patch.object(keys, "_protocol_pushed", False),
            mock.patch.object(keys, "_legacy_encoding_depth", 0),
            mock.patch("agent.output.ensure_readline", return_value=True),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        # Skip __init__: no Config singleton, no real terminal.
        self.ui = RichOutputAdapter.__new__(RichOutputAdapter)
        self.ui._steer = None
        self.ui._footer = mock.Mock(active=False)
        self.ui._footer_status_cache = ("", "")


class TestReadlineBackedPrompts(_Adapter):
    """rich.prompt -> builtin input() -> readline: legacy encodings only."""

    def test_each_prompt_loads_readline_and_runs_with_the_flag_popped(self):
        for attr, args in ARGS.items():
            seen: list[bool] = []
            stub = mock.Mock(ask=mock.Mock(
                side_effect=lambda *a, **k: seen.append(keys.protocol_pushed())))
            with mock.patch(f"agent.output.{PROMPTS[attr]}", stub), \
                    mock.patch("agent.output.ensure_readline") as ensure:
                getattr(self.ui, attr)(*args)
            ensure.assert_called_once()
            self.assertEqual(seen, [False], attr)
        # The context manager is released afterwards.
        self.assertEqual(keys._legacy_encoding_depth, 0)

    def test_ask_choice_uses_prompt_toolkit_and_keeps_the_flag(self):
        seen: list[bool] = []
        keys._protocol_pushed = True

        def fake_choice(**kwargs):
            seen.append(keys.protocol_pushed())
            return "generic"

        with mock.patch("agent.output.choice", fake_choice), \
                mock.patch("agent.output.ensure_readline") as ensure:
            self.ui.ask_choice("Provider:", [("generic", "Generic")])
        ensure.assert_not_called()
        self.assertEqual(seen, [True])

    def test_the_prompt_reaches_input_rather_than_being_printed_first(self):
        for attr, args in ARGS.items():
            with mock.patch("agent.output.readline_prompt", return_value="P: ") as rp, \
                    mock.patch("builtins.input", return_value=ANSWERS[attr]) as inp:
                getattr(self.ui, attr)(*args)
            rp.assert_called_once()
            self.assertEqual(inp.call_args.args, ("P: ",), attr)

    def test_a_password_prompt_goes_through_getpass(self):
        import io

        from rich.console import Console
        from rich.text import Text

        from agent.output import _ReadlinePrompt

        console = Console(file=io.StringIO())
        with mock.patch("getpass.getpass", return_value="s3cret") as gp, \
                mock.patch("builtins.input") as inp:
            value = _ReadlinePrompt.get_input(console, Text("Password"), True)
        gp.assert_called_once()
        inp.assert_not_called()
        self.assertEqual(value, "s3cret")


class TestRunSubprocess(_Adapter):
    def test_a_child_process_gets_legacy_encodings(self):
        # $EDITOR never asked for the protocol.
        popped: list[bool] = []
        real = keys.legacy_key_encoding

        def spy():
            with real():
                popped.append(keys.protocol_pushed())
                yield

        import contextlib
        with mock.patch("subprocess.run") as run, \
                mock.patch("agent.output.legacy_key_encoding",
                           contextlib.contextmanager(spy)):
            self.ui.run_subprocess(["vi"])
        self.assertEqual(popped, [False])
        self.assertTrue(run.called)
