"""Tests for the /config-edit reload behavior."""

import unittest
from unittest.mock import Mock, patch

from agent.commands import registry


class TestConfigEditCommand(unittest.TestCase):
    def _run(self, editor, reload_result=(True, None)):
        core = Mock()
        core.reload_config.return_value = reload_result
        with patch("agent.config.edit_base_config_visual", return_value=editor), \
                patch("pubsub.pub.sendMessage") as send_message:
            command = registry.lookup(["/config-edit"])[0]
            assert command is not None
            result = command.handler(core, [])
        return result, core, send_message

    def test_success_reloads_and_refreshes_the_prompt(self) -> None:
        (ok, msg, content, markdown), core, send = self._run(
            Mock(returncode=0, stderr="")
        )
        assert ok is True and content is None and markdown is None
        assert msg == "Configuration edited and reloaded successfully"
        core.reload_config.assert_called_once_with()
        send.assert_called_once_with("prompt-update")

    def test_editor_or_reload_failure_is_reported_and_not_announced(self) -> None:
        cases = [
            {"editor": Mock(returncode=1, stderr="editor failed")},
            {"editor": Mock(returncode=0, stderr=""),
             "reload_result": (False, "invalid model config")},
        ]
        for kwargs in cases:
            (ok, msg, _c, _m), core, send = self._run(**kwargs)
            assert ok is False, kwargs
            assert "editor failed" in msg or "invalid model config" in msg
            send.assert_not_called()
        # A failing editor never reloads at all.
        _result, core, _send = self._run(Mock(returncode=1, stderr="x"))
        core.reload_config.assert_not_called()
