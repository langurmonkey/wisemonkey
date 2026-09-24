"""Tests for the /config-edit reload behavior."""

import unittest
from unittest.mock import Mock, patch

from agent.commands import registry


class TestConfigEditCommand(unittest.TestCase):
    def test_success_reloads_runtime_config(self) -> None:
        core = Mock()
        core.reload_config.return_value = (True, None)
        editor_result = Mock(returncode=0, stderr="")

        with patch(
            "agent.config.edit_base_config_visual", return_value=editor_result
        ), patch("pubsub.pub.sendMessage") as send_message:
            command = registry.lookup(["/config-edit"])[0]
            assert command is not None
            ok, message, content, markdown = command.handler(core, [])

        self.assertTrue(ok)
        self.assertEqual(message, "Configuration edited and reloaded successfully")
        self.assertIsNone(content)
        self.assertIsNone(markdown)
        core.reload_config.assert_called_once_with()
        send_message.assert_called_once_with("prompt-update")

    def test_editor_failure_does_not_reload(self) -> None:
        core = Mock()
        editor_result = Mock(returncode=1, stderr="editor failed")

        with patch(
            "agent.config.edit_base_config_visual", return_value=editor_result
        ):
            command = registry.lookup(["/config-edit"])[0]
            assert command is not None
            ok, message, _content, _markdown = command.handler(core, [])

        self.assertFalse(ok)
        self.assertEqual(message, "editor failed")
        core.reload_config.assert_not_called()

    def test_reload_failure_is_reported(self) -> None:
        core = Mock()
        core.reload_config.return_value = (False, "invalid model config")
        editor_result = Mock(returncode=0, stderr="")

        with patch(
            "agent.config.edit_base_config_visual", return_value=editor_result
        ), patch("pubsub.pub.sendMessage") as send_message:
            command = registry.lookup(["/config-edit"])[0]
            assert command is not None
            ok, message, _content, _markdown = command.handler(core, [])

        self.assertFalse(ok)
        self.assertEqual(message, "Configuration reload failed: invalid model config")
        send_message.assert_not_called()
