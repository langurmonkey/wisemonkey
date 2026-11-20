"""Tests for agent/palette.py — the shared colour source."""

import dataclasses
from unittest import TestCase

from rich.theme import Theme

from agent.palette import PALETTE, Palette, bg, fg

# Tags referenced from the frontends, listed explicitly so removing one from
# the theme fails here rather than rendering silently unstyled.
STYLE_TAGS = {
    "accent", "accent-bold", "agent", "cmd", "code", "err", "error",
    "info", "kbd", "link", "list-desc", "list-item", "ok", "output-frame",
    "patch-add", "patch-remove", "path", "prompt", "server", "session-name",
    "rule", "status", "thinking", "time", "title", "tool", "user", "warn", "weak",
}


class TestSgr(TestCase):
    def test_24_bit_sequences(self):
        assert fg("#4c9aff") == "\x1b[38;2;76;154;255m"
        assert fg("4C9AFF") == fg("#4c9aff")  # case and '#' are both optional
        assert bg("#21262d") == "\x1b[48;2;33;38;45m"


class TestRichTheme(TestCase):
    def test_every_referenced_tag_resolves_and_parses(self):
        theme = PALETTE.rich_theme_dict()
        assert STYLE_TAGS <= set(theme), STYLE_TAGS - set(theme)
        # A malformed spec raises at render time, not at definition.
        for name, spec in theme.items():
            assert Theme({name: spec}).styles[name] is not None

    def test_colours_are_distinct_and_semantic(self):
        p, d = PALETTE, PALETTE.rich_theme_dict()
        # One identity hue stepped by weight, not three near-identical blues.
        assert p.user not in (p.accent, p.tool) and p.accent != p.tool
        # Orange means warning and nothing else.
        assert p.warn not in (p.agent, p.accent, p.user, p.tool)
        # One green and one red, shared with the status styles.
        assert d["patch-add"] == d["ok"]
        assert d["patch-remove"] == d["error"].removeprefix("bold ")
        # Foreground only: a background draws a block on a light terminal.
        assert " on " not in d["path"]

    def test_thinking_is_dimmer_than_weak(self):
        def lum(colour: str) -> float:
            r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        p = PALETTE
        assert p.thinking not in (p.weak, p.faint)
        assert lum(p.thinking) < lum(p.weak) < lum(p.text)


class TestFrontendDicts(TestCase):
    def test_prompt_toolkit_style_parses_and_shares_the_accent(self):
        from prompt_toolkit.styles import Style

        d = PALETTE.prompt_toolkit_dict()
        assert set(d) == {
            "prompt", "frame.border", "bottom-toolbar",
            "kbd", "model", "weak", "unsafe-warn",
        }
        Style.from_dict(d)
        assert d["prompt"] == PALETTE.blue
        assert PALETTE.rich_theme_dict()["user"] == PALETTE.blue

    def test_footer_slots_and_the_caret_cell(self):
        sgr = PALETTE.footer_sgr()
        assert set(sgr) == {
            "dim", "reset", "accent", "label", "text", "key", "bar_bg",
            "cursor", "cursor_over",
        }
        r, g, b = (int(PALETTE.text[i:i + 2], 16) for i in (1, 3, 5))
        # The empty-cell caret can only be a background; the one over a
        # character knocks the glyph out with reverse video.
        assert sgr["cursor"] == f"\x1b[48;2;{r};{g};{b}m"
        assert sgr["cursor_over"] == f"\x1b[38;2;{r};{g};{b}m\x1b[7m"


class TestPaletteShape(TestCase):
    def test_fields_are_frozen_and_roles_are_derived(self):
        p = Palette()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            setattr(p, "blue", "#ffffff")
        # Roles are properties, so they cannot drift from the fields.
        assert "agent" not in p.__dataclass_fields__
        assert p.agent == p.blue_dim
