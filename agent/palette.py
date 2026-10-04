"""The single source of truth for every colour Wisemonkey draws.

There used to be four independent palettes — a Rich `Theme`, a
prompt_toolkit `Style`, hand-written SGR constants in the footer, and
Textual's built-in design tokens — with no shared values and no shared
intent. That is why the TUI looked purple while the REPL looked blue, and
why `path` rendered as a dark block on a light terminal: the copy had
drifted.

This module fixes that structurally rather than by taste alone:

* **One identity hue.** Everything that is not a semantic signal is a
  lightness step of the same blue. Three near-identical blues used to
  carry `user`, `accent` and `tool`, so the model's name and your own
  prompt looked equivalent for no reason.
* **Hue means something.** Orange is a warning and nothing else, red is an
  error and a removed line, green is success and an added line. `agent`
  used to be the same orange as `warn`, which meant a warning read as
  though the agent itself were complaining.
* **One role, one colour, in every frontend.** `rich_theme_dict()`,
  `prompt_toolkit_dict()`, `footer_sgr()` and `textual_theme()` are
  generated from the same fields, so the four frontends cannot disagree.

Colours are truecolor hex. Rich and Textual both downsample them to the
terminal's actual palette when it is not a truecolor terminal, and the
footer's SGR uses 24-bit form for the same reason — so a hex value here
means the same thing in all three.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def fg(hex_color: str) -> str:
    """24-bit SGR sequence setting the foreground to *hex_color*."""
    r, g, b = _rgb(hex_color)
    return f"\x1b[38;2;{r};{g};{b}m"


def bg(hex_color: str) -> str:
    """24-bit SGR sequence setting the background to *hex_color*."""
    r, g, b = _rgb(hex_color)
    return f"\x1b[48;2;{r};{g};{b}m"


@dataclass(frozen=True)
class Palette:
    """Every colour the agent draws, as truecolor hex.

    The blue scale is deliberately ordered by *weight*, not by hue: `blue`
    is the one saturated accent, `blue_mid` is repeated chrome, and
    `blue_dim` is for text that should recede. Distinguishing `user` from
    `accent` from `tool` by lightness rather than by three near-identical
    hues is what makes the palette read as one system.
    """

    # --- the identity scale -------------------------------------------
    blue: str = "#4c9aff"  # the one saturated accent: user input, primary
    blue_mid: str = "#2f81f7"  # repeated chrome: model name, links, rules
    blue_dim: str = "#2b6cb0"  # tool names, secondary chrome

    # --- semantic signals ---------------------------------------------
    # Saturated orange is reserved for warnings. Nothing else may use it.
    warn: str = "#f0883e"
    ok: str = "#3fb950"
    error: str = "#f85149"

    # --- neutrals ------------------------------------------------------
    text: str = "#c9d1d9"  # default body text, if a style ever needs one
    weak: str = "#8b949e"  # de-emphasised annotation
    faint: str = "#6e7681"  # separators, the dimmest readable grey
    kbd: str = "#ffd787"  # keycaps: the only yellow in the system
    bar_bg: str = "#21262d"  # background of the hint bar and status chips

    # --- roles ---------------------------------------------------------
    @property
    def title(self) -> str:
        return self.blue

    @property
    def accent(self) -> str:
        return self.blue_mid

    @property
    def agent(self) -> str:
        """Wisemonkey's own name in output.

        Chrome, not a speaker. It is dimmed so that a real warning in
        orange next to it is unmistakable — the collision that previously
        made `warn` read as `agent`.
        """
        return self.blue_dim

    @property
    def user(self) -> str:
        return self.blue

    @property
    def tool(self) -> str:
        return self.blue_dim

    @property
    def cmd(self) -> str:
        """A slash command echoed back. Text the user typed, not a signal."""
        return self.blue_dim

    @property
    def path(self) -> str:
        """Foreground only, no background block.

        The old value was `#999999 on #252525`, which drew a dark rectangle
        around every path — and around every `$ command` in a light
        terminal.
        """
        return self.weak

    # --- consumers -----------------------------------------------------

    def rich_theme_dict(self) -> dict[str, str]:
        """The Rich `Theme` used by `agent/console.py`.

        Names are the style tags used throughout the codebase (`[agent]`,
        `[weak]`, `[patch-add]`, ...), so this is a lookup table rather
        than an arbitrary palette dump.
        """
        return {
            # Global
            "title": f"bold {self.title}",
            "accent": self.accent,
            "accent-bold": f"bold {self.accent}",
            "output-frame": self.weak,
            "time": f"{self.faint} i",
            # Turns
            "agent": self.agent,
            "user": self.user,
            # Features
            "tool": self.tool,
            "status": f"{self.text} on {self.bar_bg}",
            "path": self.path,
            "cmd": self.cmd,
            # Inline code: no second hue. A tinted block rather than pink,
            # so code stops competing with the error/warn signals.
            "code": f"{self.text} on {self.bar_bg}",
            "prompt": f"{self.ok} bold",
            "weak": self.weak,
            "kbd": f"{self.kbd} bold on {self.bar_bg}",
            "link": f"{self.accent} underline",
            # Patching: the same green/red as ok/error, so a diff and a
            # status message do not disagree about what "added" means.
            "patch-add": self.ok,
            "patch-remove": self.error,
            "list-item": self.blue_mid,
            "list-desc": self.weak,
            # Logging
            "ok": self.ok,
            "info": f"dim {self.blue_mid}",
            "warn": self.warn,
            "error": f"bold {self.error}",
            "err": f"bold {self.error}",
            # Daemon/remote mode. Used by agent.py, tui.py and server.py;
            # it was referenced but never defined, and Rich silently drops
            # an unknown tag, so those lines rendered unstyled.
            "server": self.blue_dim,
            "session-name": self.accent,
        }

    def prompt_toolkit_dict(self) -> dict[str, str]:
        """The prompt_toolkit `Style` for the REPL prompt.

        prompt_toolkit needs literal hex, not Rich style names, so these
        cannot be derived from `rich_theme_dict()`. Deriving both from the
        same fields is what stops the prompt drifting from the output
        above it.
        """
        return {
            "prompt": self.blue,
            "frame.border": self.blue,
            "bottom-toolbar": f"{self.text} bg:{self.bar_bg} noreverse",
            "kbd": f"{self.kbd} bold",
            "model": self.blue_dim,
            "weak": self.faint,
            "unsafe-warn": f"bold bg:{self.error} {self.text}",
        }

    def footer_sgr(self) -> dict[str, str]:
        """Raw SGR sequences for the sticky footer.

        The footer writes to the terminal directly and cannot go through
        Rich, so it needs escape sequences. Generating them here rather
        than hand-writing `\\x1b[36m` is the point: the footer can no longer
        disagree with the theme.
        """
        return {
            "dim": fg(self.faint),
            "reset": "\x1b[0m",
            "accent": fg(self.blue),
            "label": fg(self.weak),
            "text": fg(self.text),
            "key": f"\x1b[1m{fg(self.kbd)}",
            "bar_bg": bg(self.bar_bg),
        }

    def textual_theme(self) -> Any:
        """A Textual `Theme` matching the same palette.

        The TUI previously used Textual's built-in `$primary-muted` and
        `$accent-muted`, which is where the purple came from — the TUI was
        literally running a different colour scheme. Overriding the theme
        makes `$primary` etc. resolve to these values instead.
        """
        from textual.theme import Theme

        return Theme(
            name="wisemonkey",
            primary=self.blue,
            secondary=self.blue_dim,
            accent=self.blue_mid,
            warning=self.warn,
            error=self.error,
            success=self.ok,
            foreground=self.text,
            background="#0d1117",
            surface="#161b22",
            panel="#21262d",
            boost=self.blue_mid,
            dark=True,
            variables={
                "footer-key-foreground": self.kbd,
                "footer-background": self.bar_bg,
                "block-cursor-foreground": self.blue,
            },
        )


#: The palette everything uses. A module-level instance rather than a
#: constructor call at each site, so importing it is enough.
PALETTE = Palette()