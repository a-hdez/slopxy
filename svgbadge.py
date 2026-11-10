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


_T = np.linspace(0, 1, BEZIER_STEPS + 1)[1:]  # curve parameters of each flattened segment end


def _path_edges(d: str, tx: float, ty: float, s: float) -> np.ndarray:
    tokens = re.findall(r"[MLCZ]|" + _NUM.pattern, d)
    # pieces: a line edge (x0, y0, x1, y1), or the index of a cubic curve in `curves`,
    # which are all flattened together afterwards.
    pieces, curves, start, cur = [], [], None, None
    i, cmd = 0, None
    while i < len(tokens):
        if tokens[i] in "MLCZ":
            cmd = tokens[i]
            i += 1
            if cmd == "Z":
                if cur and start and cur != start:
                    pieces.append((*cur, *start))
                cur = start
                continue
        nums = lambda k: [float(t) for t in tokens[i:i + k]]
        if cmd == "M":
            cur = start = tuple(nums(2))
            i += 2
            cmd = "L"  # extra coordinate pairs after M are line-tos
        elif cmd == "L":
            p = tuple(nums(2))
            pieces.append((*cur, *p))
            cur = p
            i += 2
        elif cmd == "C":
            x1, y1, x2, y2, x3, y3 = nums(6)
            pieces.append(len(curves))
            curves.append((*cur, x1, y1, x2, y2, x3, y3))
            cur = (x3, y3)
            i += 6
        else:
            raise ValueError(f"unsupported path data near {tokens[i]!r}")
    if curves:
        cx, cy, x1, y1, x2, y2, x3, y3 = (c[:, None] for c in np.array(curves, dtype=float).T)
        t, mt = _T, 1 - _T
        xs = mt**3 * cx + 3 * mt**2 * t * x1 + 3 * mt * t**2 * x2 + t**3 * x3
        ys = mt**3 * cy + 3 * mt**2 * t * y1 + 3 * mt * t**2 * y2 + t**3 * y3
        # Segment k of curve n runs from point k-1 (the curve's start for k = 0) to point k.
        seg = np.stack([np.hstack([cx, xs[:, :-1]]), np.hstack([cy, ys[:, :-1]]), xs, ys],
                       axis=-1)  # (curves, steps, 4)
    chunks = [seg[p] if isinstance(p, int) else np.array([p], dtype=float) for p in pieces]
    a = np.concatenate(chunks) if chunks else np.zeros((0, 4))
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
    pixels, sampled at pixel centers; supersample by passing a large size.

    Each edge crossing a row's center line at x adds its direction (or 1, for evenodd) to
    every pixel whose center lies right of x; a running sum along the row then gives each
    pixel's winding number (or crossing count). Only the rows each edge spans are visited."""
    e = g.edges * scale + np.array([ox, oy, ox, oy])
    x0, y0, x1, y1 = e.T
    keep = y0 != y1
    x0, y0, x1, y1 = x0[keep], y0[keep], x1[keep], y1[keep]
    direction = np.where(y1 > y0, 1, -1)
    # Rows whose center y = row + 0.5 satisfies min(y0, y1) <= y < max(y0, y1). Take one
    # extra row on each side, then keep the pairs that pass the exact test below.
    lo = np.clip(np.ceil(np.minimum(y0, y1) - 0.5) - 1, 0, size).astype(np.int64)
    hi = np.clip(np.ceil(np.maximum(y0, y1) - 0.5) + 1, 0, size).astype(np.int64)
    counts = hi - lo
    cols = np.repeat(np.arange(len(x0)), counts)
    rows = np.repeat(lo - np.cumsum(counts) + counts, counts) + np.arange(counts.sum())
    y = rows + 0.5
    a0, b0, a1, b1 = x0[cols], y0[cols], x1[cols], y1[cols]
    hit = ((b0 <= y) & (b1 > y)) | ((b1 <= y) & (b0 > y))
    rows, cols, y = rows[hit], cols[hit], y[hit]
    a0, b0, a1, b1 = a0[hit], b0[hit], a1[hit], b1[hit]
    xs = a0 + (y - b0) * (a1 - a0) / (b1 - b0)
    # Pixel j (center j + 0.5) is right of the crossing when j + 0.5 > xs.
    first = np.clip(np.floor(xs - 0.5).astype(np.int64) + 1, 0, size)
    weights = np.ones(len(rows)) if g.evenodd else direction[cols]
    delta = np.bincount(rows * (size + 1) + first, weights=weights,
                        minlength=size * (size + 1)).reshape(size, size + 1)
    acc = np.cumsum(delta, axis=1)[:, :size]
    inside = (np.rint(acc) % 2 == 1) if g.evenodd else (np.rint(acc) != 0)
    return np.where(inside, 255, 0).astype(np.uint8)


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
        layer.paste(g.fill + (255,), (0, 0, size, size), m)
    return layer
