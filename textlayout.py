"""Rich rules-text layout: word wrap with inline symbols, italic reminder/flavor
text, bold planeswalker costs, and auto-shrink to fit a box."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from PIL import Image, ImageDraw

from .fonts import font
from .symbols import paste_symbol

TOKEN_RE = re.compile(r"\{[^}]+\}|\s+|[^\s{]+|\{")
LOYALTY_RE = re.compile(r"^([+−\-]?(?:\d+|X)):")
STYLE_ROLE = {"rules": "rules", "italic": "italic", "bold": "bold"}
TEXT_COLOR = (15, 15, 15)

SYM_SCALE = 0.80      # symbol diameter relative to font size
LINE_SCALE = 1.16     # line height relative to font size
PARA_GAP = 0.38       # extra space between rules paragraphs
FLAVOR_GAP = 0.75     # space around the flavor divider


@dataclass
class Paragraph:
    units: list            # list[list[(kind, text, style)]]
    gap: float = PARA_GAP  # gap before paragraph, in font-size units
    rule_before: bool = False


def _style_segments(text: str, base: str) -> list[tuple[str, str]]:
    """Split text into (segment, style) by parenthesis depth (reminder text -> italic)."""
    if base != "rules":
        return [(text, base)]
    segs, buf, depth, cur = [], [], 0, "rules"
    for ch in text:
        if ch == "(":
            if depth == 0 and buf:
                segs.append(("".join(buf), cur)); buf = []
            depth += 1
            cur = "italic"
            buf.append(ch)
        elif ch == ")" and depth:
            buf.append(ch)
            depth -= 1
            if depth == 0:
                segs.append(("".join(buf), cur)); buf = []
                cur = "rules"
        else:
            buf.append(ch)
    if buf:
        segs.append(("".join(buf), cur))
    return segs


def _units_from(segments: list[tuple[str, str]]) -> list[list[tuple[str, str, str]]]:
    units, cur = [], []
    for text, style in segments:
        for m in TOKEN_RE.finditer(text):
            tok = m.group(0)
            if tok.isspace():
                if cur:
                    units.append(cur); cur = []
            elif tok.startswith("{") and tok.endswith("}") and len(tok) > 2:
                cur.append(("sym", tok[1:-1], style))
            else:
                cur.append(("text", tok, style))
    if cur:
        units.append(cur)
    return units


def build_paragraphs(oracle: str | None, flavor: str | None = None,
                     extra_rules: list[str] | None = None) -> list[Paragraph]:
    paras: list[Paragraph] = []
    lines = (oracle or "").split("\n") if oracle else []
    lines += extra_rules or []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        segs = []
        m = LOYALTY_RE.match(line)
        if m:
            segs.append((m.group(0), "bold"))
            line = line[m.end():]
        segs += _style_segments(line, "rules")
        paras.append(Paragraph(_units_from(segs), gap=0 if not paras else PARA_GAP))
    if flavor:
        for j, line in enumerate(flavor.split("\n")):
            if not line.strip():
                continue
            first = j == 0
            paras.append(Paragraph(
                _units_from([(line, "italic")]),
                gap=(FLAVOR_GAP * 2 if paras else 0) if first else 0.05,
                rule_before=first and bool(paras),
            ))
    return paras


def _unit_widths(unit, size: float) -> list[float]:
    """Advance width of each piece of a unit (word) at the given font size."""
    widths = []
    for kind, text, style in unit:
        if kind == "sym":
            widths.append(size * SYM_SCALE + size * 0.08)
        else:
            widths.append(font(STYLE_ROLE[style], size).getlength(text))
    return widths


@dataclass
class TextLayout:
    size: float
    height: float
    ops: list = field(default_factory=list)  # draw ops relative to box origin
    fits: bool = True


def layout(paras: list[Paragraph], width: float, size: float) -> TextLayout:
    line_h = size * LINE_SCALE
    space = font("rules", size).getlength(" ")
    ops, y = [], 0.0
    for p in paras:
        y += p.gap * size
        if p.rule_before:
            ops.append(("rule", y - p.gap * size / 2))
        x = 0.0
        line_started = False
        for unit in p.units:
            widths = _unit_widths(unit, size)  # measured once, used for wrapping and placement
            uw = sum(widths)
            if line_started and x + space + uw > width:
                y += line_h
                x = 0.0
                line_started = False
            if line_started:
                x += space
            for pc, w in zip(unit, widths):
                ops.append(("text" if pc[0] == "text" else "sym", x, y, pc[1], pc[2]))
                x += w
            line_started = True
        y += line_h
    return TextLayout(size=size, height=y, ops=ops)


def fit(paras: list[Paragraph], width: float, height: float,
        max_size: float = 34, min_size: float = 14) -> TextLayout:
    """Largest size from max_size down in 1pt steps (ending at min_size) whose layout fits.
    Text height grows with size, so instead of trying every size this gallops down from
    the top (1, 2, 4, 8... steps) and then bisects: about 2 log2(steps) layouts at most,
    and 2-3 when the text nearly fits at max_size. If nothing fits, the min_size layout is
    returned with fits=False."""
    first = layout(paras, width, max_size)  # before listing sizes: bad sizes fail here
    if first.height <= height or max_size <= min_size:
        first.fits = first.height <= height
        return first
    sizes = [max_size]
    while sizes[-1] > min_size:
        sizes.append(max(min_size, sizes[-1] - 1))
    # sizes[0] doesn't fit; find the first index that does.
    tried = {0: first}

    def fits_at(i: int) -> bool:
        if i not in tried:
            tried[i] = layout(paras, width, sizes[i])
        return tried[i].height <= height

    last = len(sizes) - 1
    bad, step = 0, 1                # sizes[bad] doesn't fit
    while True:
        probe = min(bad + step, last)
        if fits_at(probe):
            good = probe            # answer is in (bad, good]
            break
        if probe == last:           # nothing fits; return the min_size layout
            tried[last].fits = False
            return tried[last]
        bad, step = probe, step * 2
    while good - bad > 1:
        mid = (bad + good) // 2
        if fits_at(mid):
            good = mid
        else:
            bad = mid
    tried[good].fits = True
    return tried[good]


def render(canvas: Image.Image, lay: TextLayout, x0: float, y0: float,
           width: float, height: float, valign: str = "center"):
    draw = ImageDraw.Draw(canvas)
    size = lay.size
    if valign == "center" and lay.height < height:
        y0 += (height - lay.height) / 2
    ascent = size * 0.86
    dia = size * SYM_SCALE
    for op in lay.ops:
        if op[0] == "rule":
            yy = y0 + op[1]
            draw.line([x0 + width * 0.08, yy, x0 + width * 0.92, yy], fill=(90, 90, 90), width=2)
            continue
        kind, x, y, text, style = op
        baseline = y0 + y + ascent
        if kind == "text":
            draw.text((x0 + x, baseline), text, font=font(STYLE_ROLE[style], size),
                      fill=TEXT_COLOR, anchor="ls")
        else:
            top = baseline - size * 0.34 - dia / 2
            paste_symbol(canvas, text, x0 + x + size * 0.04, top, round(dia), badge=True)
