"""Tests for slash-command categories and the grouped `/help` listing."""

import unittest

from agent.commands import CATEGORY_ORDER, PINNED_FIRST, registry


class TestCategories(unittest.TestCase):
    def test_every_command_is_listed_once_in_a_known_category(self) -> None:
        """A typo in `category` would file the command silently at the tail."""
        for _name, commands in registry.list_by_category():
            names = [c.name for c in commands]
            assert len(names) == len(set(names)), names
            # Pinned commands lead their group; the rest are alphabetical.
            body = [n for n in names if n not in PINNED_FIRST]
            assert body == sorted(body), names
        unknown = [
            f"{cmd.name} -> {cmd.category!r}"
            for cmd in registry.list_commands()
            if cmd.category not in CATEGORY_ORDER
        ]
        assert unknown == []

        groups = [name for name, _ in registry.list_by_category()]
        assert groups == [c for c in CATEGORY_ORDER if c in groups]
        listed = [c.name for _, cmds in registry.list_by_category() for c in cmds]
        assert len(listed) == len(registry.list_commands())

    def test_help_prints_one_heading_per_group_and_every_command(self) -> None:
        text = registry.get_commands_str()
        assert dict(registry.list_by_category())["General"][0].name == "/help"
        for name, commands in registry.list_by_category():
            assert text.count(f"{name}[/accent-bold]\n") == 1
            for command in commands:
                # Rendered in the spaced form, which the lookup also accepts.
                assert command.name.replace("-", " ") in text


if __name__ == "__main__":
    unittest.main()
