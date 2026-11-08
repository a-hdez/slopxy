"""Mana badges from the SVGs in assets/: a colored circle plus a glyph path.

Only the subset those files use is supported (circle, one <g transform="translate() scale()">,
paths of absolute M/L/C/Z with nonzero or evenodd fill), rasterized with numpy scanlines."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

ASSET_DIR = Path(__file__).parent / "assets"
VIEW = 512.0  # every badge uses viewBox 0 0 512 512
BEZIER_STEPS = 12

# Mana symbol -> badge file stem
BADGES = {"W": "star", "U": "droplet", "B": "skull", "R": "fireball", "G": "tree",
          "C": "diamond", "P": "phi"}

_NUM = re.compile(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?")


@dataclass
class Glyph:
    edges: np.ndarray   # (n, 4) x0, y0, x1, y1 in viewBox units
    fill: tuple
    evenodd: bool


@dataclass
class Badge:
    circle: tuple | None  # (cx, cy, r, rgb) in viewBox units
    glyphs: list[Glyph]


def _rgb(value: str) -> tuple:
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _transform(attr: str | None):
    tx = ty = 0.0
    s = 1.0
    if attr:
        m = re.search(r"translate\(\s*([-\d.]+)[\s,]+([-\d.]+)\s*\)", attr)
        if m:
            tx, ty = float(m.group(1)), float(m.group(2))
        m = re.search(r"scale\(\s*([-\d.]+)", attr)
        if m:
            s = float(m.group(1))
    return tx, ty, s


def _path_edges(d: str, tx: float, ty: float, s: float) -> np.ndarray:
    tokens = re.findall(r"[MLCZ]|" + _NUM.pattern, d)
    edges, start, cur = [], None, None
    i, cmd = 0, None
    while i < len(tokens):
        if tokens[i] in "MLCZ":
            cmd = tokens[i]
            i += 1
            if cmd == "Z":
                if cur and start and cur != start:
                    edges.append((*cur, *start))
                cur = start
                continue
        nums = lambda k: [float(t) for t in tokens[i:i + k]]
        if cmd == "M":
            cur = start = tuple(nums(2))
            i += 2
            cmd = "L"  # extra coordinate pairs after M are line-tos
        elif cmd == "L":
            p = tuple(nums(2))
            edges.append((*cur, *p))
            cur = p
            i += 2
        elif cmd == "C":
            x1, y1, x2, y2, x3, y3 = nums(6)
            t = np.linspace(0, 1, BEZIER_STEPS + 1)[1:]
            mt = 1 - t
            xs = mt**3 * cur[0] + 3 * mt**2 * t * x1 + 3 * mt * t**2 * x2 + t**3 * x3
            ys = mt**3 * cur[1] + 3 * mt**2 * t * y1 + 3 * mt * t**2 * y2 + t**3 * y3
            prev = cur
            for p in zip(xs, ys):
                edges.append((*prev, *p))
                prev = p
            cur = (x3, y3)
            i += 6
        else:
            raise ValueError(f"unsupported path data near {tokens[i]!r}")
    a = np.array(edges, dtype=float).reshape(-1, 4)
    a[:, [0, 2]] = a[:, [0, 2]] * s + tx
    a[:, [1, 3]] = a[:, [1, 3]] * s + ty
    return a


@lru_cache(maxsize=None)
def load(stem: str) -> Badge:
    root = ET.parse(ASSET_DIR / f"{stem}_badge.svg").getroot()
    circle, glyphs = None, []

    def walk(el, tf):
        nonlocal circle
        tag = el.tag.rsplit("}", 1)[-1]
        if tag == "g":
            t = _transform(el.get("transform"))
            tf = (tf[0] + t[0] * tf[2], tf[1] + t[1] * tf[2], tf[2] * t[2])
        elif tag == "circle":
            circle = (float(el.get("cx")), float(el.get("cy")), float(el.get("r")),
                      _rgb(el.get("fill", "#000000")))
        elif tag == "path":
            glyphs.append(Glyph(_path_edges(el.get("d"), *tf), _rgb(el.get("fill", "#000000")),
                                el.get("fill-rule") == "evenodd"))
        for child in el:
            walk(child, tf)

    walk(root, (0.0, 0.0, 1.0))
    return Badge(circle, glyphs)


def _mask(g: Glyph, size: int, scale: float, ox: float, oy: float) -> np.ndarray:
    """Coverage mask (size x size, 0/255) of a glyph scaled by `scale` and offset by (ox, oy)
    pixels, sampled at pixel centers; supersample by passing a large size."""
    e = g.edges * scale + np.array([ox, oy, ox, oy])
    x0, y0, x1, y1 = e.T
    keep = y0 != y1
    x0, y0, x1, y1 = x0[keep], y0[keep], x1[keep], y1[keep]
    direction = np.where(y1 > y0, 1, -1)
    out = np.zeros((size, size), dtype=np.uint8)
    px = np.arange(size) + 0.5
    for row in range(size):
        y = row + 0.5
        hit = ((y0 <= y) & (y1 > y)) | ((y1 <= y) & (y0 > y))
        if not hit.any():
            continue
        xs = x0[hit] + (y - y0[hit]) * (x1[hit] - x0[hit]) / (y1[hit] - y0[hit])
        order = np.argsort(xs)
        xs = xs[order]
        idx = np.searchsorted(xs, px)
        if g.evenodd:
            inside = idx % 2 == 1
        else:
            wind = np.concatenate([[0], np.cumsum(direction[hit][order])])
            inside = wind[idx] != 0
        out[row, inside] = 255
    return out


def circle_color(sym: str) -> tuple | None:
    stem = BADGES.get(sym)
    return load(stem).circle[3] if stem and load(stem).circle else None


def glyph_layer(sym: str, size: int, box: tuple[float, float, float, float]) -> Image.Image:
    """RGBA (size x size) with the badge glyph for `sym` fitted into box=(x, y, w, h), where
    the box stands for the badge's full 512x512 viewBox."""
    badge = load(BADGES[sym])
    x, y, w, _ = box
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for g in badge.glyphs:
        m = Image.fromarray(_mask(g, size, w / VIEW, x, y), "L")
        layer.paste(Image.new("RGBA", (size, size), g.fill + (255,)), (0, 0), m)
    return layer
