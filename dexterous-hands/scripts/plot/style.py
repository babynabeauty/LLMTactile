"""Shared matplotlib styling for plot scripts.

Registers the bundled Hanken Grotesk fonts (under ``scripts/plot/fonts/``) and sets it as the
default sans-serif so every figure produced by these scripts has consistent typography without
depending on a system-installed font.

The directory holds the variable font plus static SemiBold (600) and Bold (700) instances cut from
it with ``fontTools.varLib.instancer``. matplotlib registers a variable font at its default instance
only, so without those statics every ``fontweight`` above 400 silently renders at 400.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_FAMILY = "Hanken Grotesk"
_APPLIED = False


def apply() -> None:
    """Register the bundled font files and set matplotlib defaults. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return

    for ttf in FONT_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(ttf))

    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [FONT_FAMILY] + matplotlib.rcParams["font.sans-serif"]
    _APPLIED = True


# Apply on import so callers only need to import this module.
apply()
