"""Tests for slash-command categories and the grouped `/help` listing."""

import unittest

from agent.commands import CATEGORY_ORDER, PINNED_FIRST, registry


class TestCategories(unittest.TestCase):
    def test_every_command_declares_a_listed_category(self) -> None:
        """A command in no category falls to the tail; a typo does the same.

        Pinning it here means a new command with a mistyped category is a
        test failure rather than a stray heading at the bottom of /help.
        """
        unknown = [
            f"{cmd.name} -> {cmd.category!r}"
            for cmd in registry.list_commands()
            if cmd.category not in CATEGORY_ORDER
        ]
        self.assertEqual(unknown, [])

    def test_groups_follow_the_declared_order(self) -> None:
        groups = [name for name, _ in registry.list_by_category()]
        self.assertEqual(groups, [c for c in CATEGORY_ORDER if c in groups])

    def test_commands_are_alphabetical_inside_a_group(self) -> None:
        for _name, commands in registry.list_by_category():
            names = [c.name for c in commands if c.name not in PINNED_FIRST]
            self.assertEqual(names, sorted(names))

    def test_pinned_commands_lead_their_group(self) -> None:
        """`/help` is how you find the rest, so it is not just another row."""
        general = dict(registry.list_by_category())["General"]
        self.assertEqual(general[0].name, "/help")

    def test_each_command_appears_exactly_once(self) -> None:
        """Aliases are registered as separate entries; they must dedupe."""
        listed = [c.name for _, cmds in registry.list_by_category() for c in cmds]
        self.assertEqual(len(listed), len(set(listed)))
        self.assertEqual(len(listed), len(registry.list_commands()))

    def test_help_prints_one_heading_per_group(self) -> None:
        text = registry.get_commands_str()
        for name, commands in registry.list_by_category():
            self.assertEqual(text.count(f"{name}[/accent-bold]\n"), 1)
            for command in commands:
                # /help renders a dashed name in its spaced form, which the
                # lookup accepts too (`/config-edit` == `/config edit`).
                self.assertIn(command.name.replace("-", " "), text)


if __name__ == "__main__":
    unittest.main()
