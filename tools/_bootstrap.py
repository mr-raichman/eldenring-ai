"""
_bootstrap.py - shared setup for the tools scripts.

Import this first in every tool. It puts the repo root on sys.path (so the
`eldenring_ai` package imports resolve when a tool is run as a loose script,
not as a module) and sets the Wayland/X display defaults the game needs.

    import _bootstrap  # noqa: F401
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("DISPLAY", ":0")
os.environ.setdefault("WAYLAND_DISPLAY", "wayland-1")
