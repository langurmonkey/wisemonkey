from rich.console import Console
from rich.theme import Theme
from rich.traceback import install

# Replace default error tracebacks with better version
install()

# Theme
langur_theme = Theme({
    "title": "bold deep_sky_blue3",

    # Global
    "accent": "deep_sky_blue3",
    "accent-bold": "bold deep_sky_blue3",
    "output-frame": "gray39",

    # Turns
    "agent": "medium_orchid",
    "user": "gold1",

    # Features
    "tool": "steel_blue3",
    "status": "white on grey15",
    "weak": "grey39",
    "kbd": "light_goldenrod1 bold on grey15", 

    "list-item": "cyan",
    "list-desc": "grey39",

    # Logging
    "ok": "chartreuse4",
    "info": "dim cyan",
    "warn": "orange_red1",
    "warning": "orange_red1",
    "error": "bold red",
    "err": "bold red"
})

# Create consoles
console = Console(theme=langur_theme)
err_console = Console(theme=langur_theme, stderr=True)

def print(msg:str=None):
    console.print(msg)

def err(msg:str):
    err_console.print(f"[err]⨯[/] {msg}")

def ok(msg:str):
    console.print(f"[ok]✓[/] {msg}")

def info(msg:str):
    console.print(f"[info]⇨[/] {msg}")

def warn(msg:str):
    console.print(f"[warn]⚠[/] {msg}")
