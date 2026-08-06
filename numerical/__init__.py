"""Numerical methods package."""

from . import chebyshev, core, newton
from .core import *  # noqa: F401,F403

__all__ = ("core", "newton", "chebyshev")
