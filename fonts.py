"""Font discovery with fallbacks so rendering never fails for lack of a font."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

_FONT_DIRS = [
    Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
    Path.home() / "AppData/Local/Microsoft/Windows/Fonts",
    Path("/usr/share/fonts/truetype/dejavu"),
    Path("/Library/Fonts"),
    Path("/System/Library/Fonts/Supplemental"),
]

# role -> candidate file names, best first
_CANDIDATES = {
    "title": ["palab.ttf", "georgiab.ttf", "timesbd.ttf", "DejaVuSerif-Bold.ttf"],
    "type": ["palab.ttf", "georgiab.ttf", "timesbd.ttf", "DejaVuSerif-Bold.ttf"],
    "rules": ["pala.ttf", "georgia.ttf", "times.ttf", "DejaVuSerif.ttf"],
    "italic": ["palai.ttf", "georgiai.ttf", "timesi.ttf", "DejaVuSerif-Italic.ttf"],
    "bold": ["palab.ttf", "georgiab.ttf", "timesbd.ttf", "DejaVuSerif-Bold.ttf"],
    "symbol": ["arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"],
    "info": ["arial.ttf", "DejaVuSans.ttf", "Arial.ttf"],
    "info_bold": ["arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"],
    "glyph": ["seguisym.ttf", "DejaVuSans.ttf"],
}


@lru_cache(maxsize=None)
def _path_for(role: str) -> str | None:
    for fname in _CANDIDATES[role]:
        for d in _FONT_DIRS:
            p = d / fname
            if p.exists():
                return str(p)
    return None


def font(role: str, size: float) -> ImageFont.FreeTypeFont:
    # Round before the cache lookup so fractional sizes share one font object per pixel size.
    return _font(role, max(1, int(round(size))))


@lru_cache(maxsize=512)
def _font(role: str, size: int) -> ImageFont.FreeTypeFont:
    path = _path_for(role)
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)
