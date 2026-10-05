"""Tests for agent/palette.py — the shared colour source.

The point of the module is that the four frontends cannot disagree, so
most of these tests are about *agreement*: that every style tag the
codebase actually uses resolves, and that each frontend's output is
derived from the same fields. A palette that merely looks right in one
frontend is not the invariant being protected.
"""

import dataclasses
import pathlib
import re
from unittest import TestCase

from rich.theme import Theme

from agent.palette import PALETTE, Palette, bg, fg

# Style tags referenced anywhere in the source. Kept as an explicit list
# rather than grepped, so that removing a tag from the theme is a test
# failure rather than a silent unstyled line.
STYLE_TAGS = {
    "accent", "accent-bold", "agent", "cmd", "code", "err", "error",
    "info", "kbd", "link", "list-desc", "list-item", "ok", "output-frame",
    "patch-add", "patch-remove", "path", "prompt", "server", "session-name",
    "status", "thinking", "time", "title", "tool", "user", "warn", "weak",
}


class TestSgr(TestCase):
    def test_fg_and_bg_produce_24_bit_sequences(self):
        assert fg("#4c9aff") == "\x1b[38;2;76;154;255m"
        assert bg("#21262d") == "\x1b[48;2;33;38;45m"

    def test_hash_prefix_is_optional(self):
        assert fg("4c9aff") == fg("#4c9aff")

    def test_uppercase_hex_works(self):
        assert fg("#4C9AFF") == fg("#4c9aff")


class TestRichTheme(TestCase):
    def test_every_referenced_tag_is_defined(self):
        """`[server]` was referenced by four call sites and never defined.

        Rich silently drops an unknown tag, so this failure mode is
        invisible at runtime — the only defence is to check the names.
        """
        theme = PALETTE.rich_theme_dict()
        assert STYLE_TAGS <= set(theme), STYLE_TAGS - set(theme)

    def test_theme_is_accepted_by_rich(self):
        Theme(PALETTE.rich_theme_dict())

    def test_every_style_parses(self):
        """A malformed spec raises at render time, not at definition."""
        for name, spec in PALETTE.rich_theme_dict().items():
            theme = Theme({name: spec})
            assert theme.styles[name] is not None

    def test_no_tag_is_definitely_the_same_colour(self):
        """`user` and `accent` were both `deep_sky_blue3` for no reason."""
        p = Palette()
        assert p.user != p.accent
        assert p.accent != p.tool

    def test_warning_is_the_only_use_of_orange(self):
        """A warning used to be the same colour as the agent's own name."""
        p = Palette()
        assert p.warn not in (p.agent, p.accent, p.user, p.tool)

    def test_patch_colours_match_the_status_colours(self):
        """`green` vs `chartreuse4`: two greens doing one job."""
        d = PALETTE.rich_theme_dict()
        assert d["patch-add"] == d["ok"]
        assert d["patch-remove"] == d["error"].removeprefix("bold ")

    def test_path_has_no_background(self):
        """`#999999 on #252525` drew a dark block on a light terminal."""
        assert " on " not in PALETTE.rich_theme_dict()["path"]

    def test_thinking_is_dimmer_than_the_body(self):
        """Reasoning text must recede below both the body and the annotations.

        `weak` is an annotation the *agent* wrote; the model's own thinking is
        not part of the answer at all, so it sits on a step of its own. The
        test pins the *ordering*, not the hex: retuning `think` is allowed,
        making it brighter than `weak` is not.
        """
        p = PALETTE
        lum = lambda c: (0.2126 * int(c[1:3], 16)
                         + 0.7152 * int(c[3:5], 16)
                         + 0.0722 * int(c[5:7], 16))
        assert p.thinking != p.weak
        assert p.thinking != p.faint
        assert lum(p.thinking) < lum(p.weak) < lum(p.text)
        theme = p.rich_theme_dict()
        assert theme["thinking"] == p.thinking
        assert theme["thinking"] != theme["weak"]

    def test_reasoning_sites_use_the_thinking_tag(self):
        """Pins the wiring: a `[weak]`/`[dim]` left behind in a frontend would
        pass every palette test above (the TUI rendered reasoning with `[dim]`,
        the REPL with `[weak]` -- two different greys for one thing)."""
        for src_path, needle in (
            ("agent/agent.py", 'print(f"[thinking]{escape(payload.text)}[/]"'),
            ("agent/tui.py", 'f"[thinking]{buf}[/thinking]"'),
            ("agent/tui.py", 'f"[thinking]{self._reasoning_buffer}[/thinking]"'),
        ):
            src = pathlib.Path(src_path).read_text(encoding="utf-8")
            assert needle in src, f"{src_path}: reasoning not using [thinking]"

    def test_kbd_is_the_only_yellow(self):
        p = PALETTE
        assert p.kbd not in (p.blue, p.blue_mid, p.blue_dim)


class TestPromptToolkitStyle(TestCase):
    def test_values_are_hex_and_parse(self):
        from prompt_toolkit.styles import Style

        d = PALETTE.prompt_toolkit_dict()
        assert set(d) == {
            "prompt", "frame.border", "bottom-toolbar",
            "kbd", "model", "weak", "unsafe-warn",
        }
        Style.from_dict(d)

    def test_prompt_shares_the_rich_palette(self):
        """The prompt bar is drawn by a different library; it must not
        hard-code its own copy of the accent."""
        assert PALETTE.prompt_toolkit_dict()["prompt"] == PALETTE.blue
        assert PALETTE.rich_theme_dict()["user"] == PALETTE.blue


class TestFooterSgr(TestCase):
    def test_all_seven_footer_slots_are_present(self):
        """footer.py reads these by name; a rename there breaks the footer."""
        assert set(PALETTE.footer_sgr()) == {
            "dim", "reset", "accent", "label", "text", "key", "bar_bg",
        }

    def test_footer_colours_come_from_the_palette(self):
        """The SGR carries 24-bit RGB components, not the hex text, so the
        check is on the numbers: a `\\x1b[36m`-style approximation would
        not match."""
        sgr = PALETTE.footer_sgr()
        r, g, b = (int(PALETTE.blue[i:i + 2], 16) for i in (1, 3, 5))
        assert sgr["accent"] == f"\x1b[38;2;{r};{g};{b}m"
        assert sgr["key"].endswith(f"38;2;{int(PALETTE.kbd[1:3], 16)};"
                                   f"{int(PALETTE.kbd[3:5], 16)};"
                                   f"{int(PALETTE.kbd[5:7], 16)}m")

    def test_reset_is_a_real_reset(self):
        assert PALETTE.footer_sgr()["reset"] == "\x1b[0m"

    def test_footer_module_uses_the_generated_sequences(self):
        """Pins the wiring, not just the palette: a hand-written
        `\\x1b[36m` left behind would pass every test above."""
        src = pathlib.Path("agent/footer.py").read_text(encoding="utf-8")
        assert "_SGR = PALETTE.footer_sgr()" in src
        # No hard-coded colour escapes left outside the generated block.
        body = src.split("_SGR = PALETTE.footer_sgr()")[1]
        assert "\\x1b[3" not in body
        assert "\\x1b[1;38;5;" not in body


class TestTextualTheme(TestCase):
    def test_theme_builds_and_has_the_wisemonkey_tokens(self):
        theme = PALETTE.textual_theme()
        assert theme.name == "wisemonkey"
        colors = theme.to_color_system().generate()
        # These are the two tokens wm.tcss used, which resolved to
        # Textual's purple palette before.
        assert colors["primary"] == "#4C9AFF"
        assert colors["accent"] == "#2F81F7"

    def test_textual_tokens_are_derived_from_our_primary(self):
        """`$primary-muted` is a darkened form of *our* primary now. When it
        came from Textual's built-in theme it was purple, which is the whole
        reason the TUI did not match the REPL."""
        colors = PALETTE.textual_theme().to_color_system().generate()
        primary = colors["primary"].lower()
        assert primary == PALETTE.blue
        muted = colors["primary-muted"].lower()
        assert muted != "#3c3c3c"  # Textual's default muted purple
        # A muted token is the same hue at lower luminance.
        pr, pg, pb = (int(primary[i:i + 2], 16) for i in (1, 3, 5))
        mr, mg, mb = (int(muted[i:i + 2], 16) for i in (1, 3, 5))
        assert mr <= pr and mg <= pg and mb <= pb
        assert (mr, mg, mb) != (pr, pg, pb)

    def test_tui_registers_the_theme_before_css_is_parsed(self):
        src = pathlib.Path("agent/tui.py").read_text(encoding="utf-8")
        assert "self.register_theme(PALETTE.textual_theme())" in src
        assert 'self.theme = "wisemonkey"' in src


class TestPaletteIsFrozen(TestCase):
    def test_fields_cannot_be_mutated_at_runtime(self):
        """The palette is shared module state; a half-recoloured agent is
        worse than an un-recolourable one."""
        p = Palette()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            p.__dict__  # noqa: B018 - read only
            setattr(p, "blue", "#ffffff")

    def test_roles_are_derived_not_stored(self):
        """`agent`, `user`, `tool` etc. are properties, so they cannot be
        set independently and drift apart."""
        p = Palette()
        assert "agent" not in p.__dataclass_fields__
        assert p.agent == p.blue_dim


class TestNoScatteredCopies(TestCase):
    def test_console_theme_is_generated(self):
        src = pathlib.Path("agent/console.py").read_text(encoding="utf-8")
        assert "PALETTE.rich_theme_dict()" in src
        # The old inline dict of colour names is gone.
        assert "deep_sky_blue3" not in src

    def test_no_stray_rich_colour_names_left_in_the_source(self):
        """A leftover `steel_blue3` somewhere would be a fourth palette.

        This is what found the real strays: `tui.py` styled tool activation
        with `steel_blue3` inline, `output.py` with `bold deep_sky_blue3`, and
        four spots in `commands.py` with `grey30/39/50`, none of which the
        theme controlled.
        """
        root = pathlib.Path("agent")
        offenders = []
        for path in root.rglob("*.py"):
            if path.name == "palette.py":
                continue
            for i, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if re.search(r"\b(steel_blue\d|deep_sky_blue\d|chartreuse\d|"
                             r"indian_red|light_pink\d|dark_olive_green\d|"
                             r"orange_red\d|orange\d|grey\d+|gray\d+)\b", line):
                    offenders.append(f"{path}:{i}: {line.strip()}")
        assert not offenders, "hard-coded Rich colour names remain:\n" + "\n".join(offenders)