"""Mana / tap / misc symbols drawn from scratch with Pillow."""

from __future__ import annotations

import math
import re
from functools import lru_cache

from PIL import Image, ImageDraw

from . import svgbadge
from .colors import MANA_FILL
from .fonts import font

SYMBOL_RE = re.compile(r"\{([^}]+)\}")
SS = 4  # supersampling factor for anti-aliasing
INK = (20, 20, 20)
COLOR_LETTERS = set("WUBRGC")


def parse_cost(cost: str | None) -> list[str]:
    return SYMBOL_RE.findall(cost or "")


def _fill(sym: str):
    return MANA_FILL.get(sym, MANA_FILL["generic"])


def _text_centered(d: ImageDraw.ImageDraw, cx, cy, text, size, role="symbol", fill=INK, max_w=None):
    f = font(role, size)
    if max_w:
        while size > 4 and d.textlength(text, font=f) > max_w:
            size *= 0.9
            f = font(role, size)
    d.text((cx, cy), text, font=f, fill=fill, anchor="mm")


def _phyrexian_mark(d, cx, cy, r, color=INK):
    w = max(2, int(r * 0.16))
    d.ellipse([cx - r * 0.42, cy - r * 0.32, cx + r * 0.42, cy + r * 0.32], outline=color, width=w)
    d.line([cx, cy - r * 0.62, cx, cy + r * 0.62], fill=color, width=w)


def _tap(d, cx, cy, r, color=INK):
    rr = r * 0.52
    cy += r * 0.08
    w = max(2, int(r * 0.2))
    d.arc([cx - rr, cy - rr, cx + rr, cy + rr], start=165, end=365, fill=color, width=w)
    ex = cx + rr * math.cos(math.radians(5))
    ey = cy + rr * math.sin(math.radians(5))
    h = r * 0.5
    d.polygon([(ex - h * 0.6, ey - h * 0.1), (ex + h * 0.6, ey - h * 0.1), (ex, ey + h * 0.7)], fill=color)


def _badge_box(D: float, cx: float, cy: float, scale: float = 1.0):
    """Box for a badge's 512-unit viewBox so its r=248 circle has diameter D*scale at (cx, cy)."""
    k = D * scale / (2 * 248)
    return (cx - 256 * k, cy - 256 * k, 512 * k, 512 * k)


def _draw_badge(img: Image.Image, d: ImageDraw.ImageDraw, s: str, D: int) -> bool:
    """Draw `{s}` from the SVG badges (assets/). Returns False when no badge covers it."""
    parts = s.split("/")
    size = img.width
    box = [0, 0, D, D]

    def glyph(sym, cx, cy, scale=1.0):
        img.alpha_composite(svgbadge.glyph_layer(sym, size, _badge_box(D, cx, cy, scale)))

    if s in svgbadge.BADGES:
        d.ellipse(box, fill=svgbadge.circle_color(s))
        glyph(s, D / 2, D / 2)
    elif len(parts) == 2 and parts[1] == "P" and parts[0] in svgbadge.BADGES:
        d.ellipse(box, fill=svgbadge.circle_color(parts[0]))  # Phyrexian: color + phi
        glyph("P", D / 2, D / 2)
    elif len(parts) >= 2 and parts[1] in svgbadge.BADGES and (
            parts[0] in svgbadge.BADGES or parts[0].isdigit()):
        # Hybrid: split diagonally, one small glyph (or the generic number) per half.
        a, b = parts[0], parts[1]
        d.pieslice(box, 135, 315, fill=svgbadge.circle_color(a) or _fill("generic"))
        d.pieslice(box, 315, 135 + 360, fill=svgbadge.circle_color(b))
        d.line([D * 0.15, D * 0.85, D * 0.85, D * 0.15], fill=(40, 40, 40), width=max(1, SS))
        if len(parts) == 3 and parts[2] == "P":  # hybrid Phyrexian: both colors, one phi
            glyph("P", D / 2, D / 2, 0.8)
            return True
        if a.isdigit():
            _text_centered(d, D * 0.33, D * 0.35, a, D * 0.42)
        else:
            glyph(a, D * 0.33, D * 0.33, 0.5)
        glyph(b, D * 0.67, D * 0.67, 0.5)
    else:
        return False
    return True


@lru_cache(maxsize=1024)
def symbol_image(sym: str, diameter: int, badge: bool = False) -> Image.Image:
    """Return an RGBA image of the symbol `{sym}` with the given diameter.
    badge=True uses the SVG badges in assets/ where one exists."""
    diameter = max(6, int(diameter))
    D = diameter * SS
    img = Image.new("RGBA", (D + SS * 2, D + SS * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # drop shadow
    off = int(D * 0.05)
    d.ellipse([off, off + int(D * 0.04), D + off, D + off], fill=(0, 0, 0, 170))
    box = [0, 0, D, D]
    cx = cy = D / 2
    r = D / 2
    s = sym.upper()
    parts = s.split("/")

    if badge and _draw_badge(img, d, s, D):
        pass
    elif s == "T":
        d.ellipse(box, fill=_fill("generic"))
        _tap(d, cx, cy, r)
    elif s == "Q":
        d.ellipse(box, fill=(30, 30, 30))
        _tap(d, cx, cy, r, color=(240, 240, 240))
    elif s == "S":
        d.ellipse(box, fill=_fill("generic"))
        _text_centered(d, cx, cy, "❄", D * 0.7, role="glyph")
    elif s == "E":
        d.ellipse(box, fill=(40, 40, 40))
        _text_centered(d, cx, cy, "⚡", D * 0.6, role="glyph", fill=(240, 240, 240))
    elif len(parts) == 2 and parts[1] == "P":
        d.ellipse(box, fill=_fill(parts[0]))
        _phyrexian_mark(d, cx, cy, r)
    elif len(parts) >= 2:
        # Hybrid: split diagonally (top-left / bottom-right).
        a, b = parts[0], parts[1]
        d.pieslice(box, 135, 315, fill=_fill(a if a in COLOR_LETTERS else "generic"))
        d.pieslice(box, 315, 135 + 360, fill=_fill(b if b in COLOR_LETTERS else "generic"))
        d.line([D * 0.15, D * 0.85, D * 0.85, D * 0.15], fill=(60, 60, 60), width=max(1, SS))
        _text_centered(d, D * 0.33, D * 0.35, a, D * 0.42)
        if len(parts) == 3 and parts[2] == "P":
            _phyrexian_mark(d, D * 0.67, D * 0.65, r * 0.45)
        else:
            _text_centered(d, D * 0.67, D * 0.65, b, D * 0.42)
    elif s in COLOR_LETTERS:
        d.ellipse(box, fill=_fill(s))
        _text_centered(d, cx, cy + D * 0.02, s, D * 0.66)
    else:
        d.ellipse(box, fill=_fill("generic"))
        text = {"1/2": "½", "INFINITY": "∞", "HALF": "½"}.get(s, sym)
        size = D * (0.72 if len(text) == 1 else 0.6 if len(text) == 2 else 0.4)
        _text_centered(d, cx, cy + D * 0.02, text, size, max_w=D * 0.82)

    return img.resize((diameter + 2, diameter + 2), Image.LANCZOS)


def paste_symbol(canvas: Image.Image, sym: str, x: float, y: float, diameter: int,
                 badge: bool = False):
    """Paste a symbol with its top-left at (x, y)."""
    im = symbol_image(sym, int(diameter), badge)
    canvas.alpha_composite(im, (int(round(x)), int(round(y))))


def draw_cost(canvas: Image.Image, cost: str | None, right: float, cy: float,
              diameter: int, gap: int = 3) -> float:
    """Draw a mana cost right-aligned at `right`, vertically centered at `cy`.
    Returns the x where the cost starts (its left edge)."""
    syms = parse_cost(cost)
    if not syms:
        return right
    total = len(syms) * diameter + (len(syms) - 1) * gap
    x = right - total
    left = x
    for s in syms:
        paste_symbol(canvas, s, x, cy - diameter / 2, diameter, badge=True)
        x += diameter + gap
    return left
