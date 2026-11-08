"""Frame palettes and color-identity rules."""

from __future__ import annotations

import re
from dataclasses import dataclass

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class Palette:
    frame: RGB   # outer frame
    bar: RGB     # title / type bars
    box: RGB     # rules text box
    edge: RGB    # outlines


PALETTES = {
    "W": Palette((222, 214, 186), (249, 246, 234), (251, 249, 241), (120, 110, 85)),
    "U": Palette((38, 105, 168), (196, 219, 238), (228, 238, 246), (20, 55, 95)),
    "B": Palette((52, 47, 45), (196, 189, 184), (228, 224, 221), (20, 18, 17)),
    "R": Palette((196, 68, 42), (241, 203, 183), (248, 232, 222), (105, 30, 15)),
    "G": Palette((36, 117, 68), (198, 223, 196), (230, 241, 228), (15, 60, 30)),
    "gold": Palette((201, 165, 74), (246, 230, 175), (251, 243, 215), (110, 85, 25)),
    "artifact": Palette((146, 156, 165), (216, 221, 226), (236, 239, 241), (70, 78, 85)),
    "colorless": Palette((186, 186, 186), (230, 230, 230), (243, 243, 243), (95, 95, 95)),
    "land": Palette((156, 127, 92), (226, 211, 186), (241, 233, 219), (85, 65, 40)),
}

# Mana symbol fills.
MANA_FILL = {
    "W": (248, 244, 214),
    "U": (172, 207, 234),
    "B": (180, 171, 166),
    "R": (233, 152, 120),
    "G": (158, 196, 145),
    "C": (205, 200, 196),
    "generic": (205, 200, 196),
}

RARITY_COLORS = {
    "common": (25, 25, 25),
    "uncommon": (150, 170, 180),
    "rare": (205, 170, 70),
    "mythic": (225, 100, 35),
    "special": (130, 80, 160),
    "bonus": (130, 80, 160),
}

WUBRG = "WUBRG"


def colors_from_cost(cost: str | None) -> list[str]:
    found = set(re.findall(r"[WUBRG]", "".join(re.findall(r"\{([^}]*)\}", cost or ""))))
    return [c for c in WUBRG if c in found]


def is_all_hybrid(cost: str | None) -> bool:
    syms = re.findall(r"\{([^}]*)\}", cost or "")
    colored = [s for s in syms if re.search(r"[WUBRG]", s)]
    return bool(colored) and all("/" in s and not s.endswith("/P") for s in colored)


def frame_style(type_line: str, colors: list[str], mana_cost: str | None,
                color_identity: list[str] | None = None) -> tuple[Palette, Palette | None]:
    """Return (primary palette, secondary palette for a two-tone gradient or None)."""
    type_line = type_line or ""
    colors = [c for c in WUBRG if c in (colors or [])]
    if "Land" in type_line and not colors:
        base = PALETTES["land"]
        ident = [c for c in WUBRG if c in (color_identity or [])]
        if len(ident) == 1:
            p = PALETTES[ident[0]]
            return Palette(base.frame, p.bar, p.box, base.edge), None
        if len(ident) == 2:
            a, b = PALETTES[ident[0]], PALETTES[ident[1]]
            return (Palette(base.frame, a.bar, a.box, base.edge),
                    Palette(base.frame, b.bar, b.box, base.edge))
        return base, None
    if not colors:
        if "Artifact" in type_line or "Vehicle" in type_line:
            return PALETTES["artifact"], None
        return PALETTES["colorless"], None
    if len(colors) == 1:
        return PALETTES[colors[0]], None
    if len(colors) == 2 and is_all_hybrid(mana_cost):
        return PALETTES[colors[0]], PALETTES[colors[1]]
    return PALETTES["gold"], None
