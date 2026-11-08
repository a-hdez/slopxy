"""Printable sheets: lay card images out on pages at true size and save a PDF."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from PIL import Image, ImageDraw

from .fonts import font
from .render import DPI, H, W

PAPER_SIZES = {  # width x height in inches, portrait
    "letter": (8.5, 11.0),
    "a4": (210 / 25.4, 297 / 25.4),
    "legal": (8.5, 14.0),
    "a3": (297 / 25.4, 420 / 25.4),
    "tabloid": (11.0, 17.0),
}
CUSTOM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*(mm|cm|in)?\s*$", re.I)
LINE_COLOR = (187, 187, 187)
MARK_COLOR = (120, 120, 120)


def parse_paper(spec: str) -> tuple[float, float]:
    """'letter', 'a4', ... or a custom size like '210x297mm' / '8.5x11' (inches)."""
    key = spec.strip().lower()
    if key in PAPER_SIZES:
        return PAPER_SIZES[key]
    m = CUSTOM_RE.match(key)
    if not m:
        raise ValueError(f"Unknown paper size '{spec}'. Use one of {', '.join(PAPER_SIZES)} "
                         "or a custom size like 210x297mm / 8.5x11in.")
    w, h, unit = float(m.group(1)), float(m.group(2)), (m.group(3) or "in").lower()
    per_inch = {"mm": 25.4, "cm": 2.54, "in": 1.0}[unit]
    return w / per_inch, h / per_inch


@dataclass
class SheetOptions:
    paper: str = "letter"
    orientation: str = "auto"   # portrait | landscape | auto
    margin_in: float = 0.2
    gap_mm: float = 1.0
    cut_lines: str = "lines"    # lines | marks | none
    card_scale: float = 100.0   # percent
    dpi: int = DPI
    page_numbers: bool = True


@dataclass
class Layout:
    page_w: int
    page_h: int
    card_w: int
    card_h: int
    cols: int
    rows: int
    x0: int
    y0: int
    gap: int
    orientation: str

    @property
    def per_page(self) -> int:
        return self.cols * self.rows

    def slot(self, i: int) -> tuple[int, int]:
        r, c = divmod(i, self.cols)
        return self.x0 + c * (self.card_w + self.gap), self.y0 + r * (self.card_h + self.gap)


def _grid(page_w, page_h, margin, card_w, card_h, gap):
    cols = max(0, math.floor((page_w - 2 * margin + gap) / (card_w + gap)))
    rows = max(0, math.floor((page_h - 2 * margin + gap) / (card_h + gap)))
    return cols, rows


def compute_layout(opts: SheetOptions) -> Layout:
    dpi = opts.dpi
    pw_in, ph_in = parse_paper(opts.paper)
    scale = opts.card_scale / 100.0
    card_w = round(W * dpi / DPI * scale)
    card_h = round(H * dpi / DPI * scale)
    margin = round(opts.margin_in * dpi)
    gap = round(opts.gap_mm / 25.4 * dpi)

    candidates = []
    for orient in ("portrait", "landscape"):
        if opts.orientation not in ("auto", orient):
            continue
        pw, ph = (pw_in, ph_in) if orient == "portrait" else (ph_in, pw_in)
        pw, ph = round(pw * dpi), round(ph * dpi)
        cols, rows = _grid(pw, ph, margin, card_w, card_h, gap)
        candidates.append((cols * rows, orient, pw, ph, cols, rows))
    best = max(candidates, key=lambda c: c[0])  # ties keep portrait (listed first)
    n, orient, pw, ph, cols, rows = best
    if n == 0:
        raise ValueError(f"No card fits on {opts.paper} paper with a {opts.margin_in}in margin.")
    grid_w = cols * card_w + (cols - 1) * gap
    grid_h = rows * card_h + (rows - 1) * gap
    return Layout(pw, ph, card_w, card_h, cols, rows,
                  (pw - grid_w) // 2, (ph - grid_h) // 2, gap, orient)


def _crop_marks(d: ImageDraw.ImageDraw, lay: Layout, width: int, n_cards: int):
    """Marks outside the occupied part of the grid, aligned with every card edge."""
    cols = min(n_cards, lay.cols)
    rows = math.ceil(n_cards / lay.cols)
    gx0, gy0 = lay.x0, lay.y0
    gx1 = lay.x0 + cols * lay.card_w + (cols - 1) * lay.gap
    gy1 = lay.y0 + rows * lay.card_h + (rows - 1) * lay.gap
    length = max(4, min(gx0, gy0) - 6)
    xs = []
    for c in range(cols):
        x = lay.x0 + c * (lay.card_w + lay.gap)
        xs += [x, x + lay.card_w - 1]
    ys = []
    for r in range(rows):
        y = lay.y0 + r * (lay.card_h + lay.gap)
        ys += [y, y + lay.card_h - 1]
    for x in xs:
        d.line([x, gy0 - 4 - length, x, gy0 - 4], fill=MARK_COLOR, width=width)
        d.line([x, gy1 + 4, x, gy1 + 4 + length], fill=MARK_COLOR, width=width)
    for y in ys:
        d.line([gx0 - 4 - length, y, gx0 - 4, y], fill=MARK_COLOR, width=width)
        d.line([gx1 + 4, y, gx1 + 4 + length, y], fill=MARK_COLOR, width=width)


def build_pages(cards: list[Image.Image], opts: SheetOptions) -> tuple[list[Image.Image], Layout]:
    lay = compute_layout(opts)
    n_pages = math.ceil(len(cards) / lay.per_page) if cards else 0
    line_w = max(1, round(opts.dpi / 300))
    pages = []
    resized: dict[int, Image.Image] = {}
    for p in range(n_pages):
        page = Image.new("RGB", (lay.page_w, lay.page_h), (255, 255, 255))
        d = ImageDraw.Draw(page)
        batch = cards[p * lay.per_page:(p + 1) * lay.per_page]
        for i, card in enumerate(batch):
            img = resized.get(id(card))
            if img is None:
                img = card.convert("RGBA")
                if img.size != (lay.card_w, lay.card_h):
                    img = img.resize((lay.card_w, lay.card_h), Image.LANCZOS)
                resized[id(card)] = img
            x, y = lay.slot(i)
            page.paste(img, (x, y), img)  # alpha onto white: transparent areas print white
            if opts.cut_lines == "lines":
                d.rectangle([x - line_w, y - line_w, x + lay.card_w - 1 + line_w,
                             y + lay.card_h - 1 + line_w], outline=LINE_COLOR, width=line_w)
        if opts.cut_lines == "marks":
            _crop_marks(d, lay, line_w, len(batch))
        if opts.page_numbers and n_pages > 1:
            bottom = lay.page_h - (lay.y0 + lay.rows * lay.card_h + (lay.rows - 1) * lay.gap)
            if bottom >= opts.dpi * 0.15:
                f = font("info", max(10, round(opts.dpi * 0.08)))
                d.text((lay.page_w / 2, lay.page_h - bottom / 2), f"{p + 1} / {n_pages}",
                       font=f, fill=MARK_COLOR, anchor="mm")
        pages.append(page)
    return pages, lay


def save_pdf(pages: list[Image.Image], path, dpi: int = DPI) -> None:
    pages[0].save(path, "PDF", save_all=True, append_images=pages[1:], resolution=dpi)
