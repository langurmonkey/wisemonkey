"""Wisemonkey package."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("wisemonkey")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

from agent.agent import Agent

__all__ = ["Agent"]
