"""Printable sheets: lay card images out on pages at true size and save a PDF."""

from __future__ import annotations

import codecs
import io
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

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


def _opaque_slots(lay: Layout, opts: SheetOptions) -> bool:
    """Whether each card slot is plain white when its card is pasted: slots don't overlap
    and no earlier card's cut line reaches into them. Then pasting a card flattened onto
    white gives the same pixels as pasting it through its alpha."""
    line_w = max(1, round(opts.dpi / 300))
    return lay.gap >= (line_w if opts.cut_lines == "lines" else 0)


def prepare_card(card: Image.Image, lay: Layout, opts: SheetOptions) -> Image.Image:
    """A card as it goes on the page, resized to the layout's card size. Where the slots
    are plain white (_opaque_slots) it is flattened onto white, so a sheet only needs to
    keep an RGB copy; otherwise it stays RGBA and is pasted through its alpha."""
    img = card if card.mode == "RGBA" else card.convert("RGBA")
    if img.size != (lay.card_w, lay.card_h):
        img = img.resize((lay.card_w, lay.card_h), Image.LANCZOS)
    if not _opaque_slots(lay, opts):
        return img if img is not card else img.copy()
    flat = Image.new("RGB", img.size, (255, 255, 255))
    flat.paste(img, (0, 0), img)
    return flat


def page_count(n_cards: int, lay: Layout) -> int:
    return math.ceil(n_cards / lay.per_page) if n_cards else 0


def iter_pages(cards: list[Image.Image], opts: SheetOptions, lay: Layout | None = None):
    """Yield the sheet pages one at a time, so only one page bitmap needs to be in memory.
    Cards from prepare_card() are pasted as they are; others are prepared once per distinct
    image."""
    lay = lay or compute_layout(opts)
    n_pages = page_count(len(cards), lay)
    line_w = max(1, round(opts.dpi / 300))
    size = (lay.card_w, lay.card_h)
    prepared: dict[int, Image.Image] = {}
    for p in range(n_pages):
        page = Image.new("RGB", (lay.page_w, lay.page_h), (255, 255, 255))
        d = ImageDraw.Draw(page)
        batch = cards[p * lay.per_page:(p + 1) * lay.per_page]
        for i, card in enumerate(batch):
            img = card
            if not (card.size == size and (card.mode == "RGB" or (
                    card.mode == "RGBA" and not _opaque_slots(lay, opts)))):
                img = prepared.get(id(card))
                if img is None:
                    img = prepared[id(card)] = prepare_card(card, lay, opts)
            x, y = lay.slot(i)
            # alpha onto white: transparent areas print white
            page.paste(img, (x, y), img if img.mode == "RGBA" else None)
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
        yield page


def build_pages(cards: list[Image.Image], opts: SheetOptions) -> tuple[list[Image.Image], Layout]:
    """All pages as a list (holds every page in memory; save_pdf(iter_pages(...)) doesn't)."""
    lay = compute_layout(opts)
    return list(iter_pages(cards, opts, lay)), lay


def _pdf_text(s: str) -> bytes:
    raw = codecs.BOM_UTF16_BE + s.encode("utf-16-be")
    return b"(" + raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)") + b")"


def _write_pdf(pages, fh, dpi: float, title: str | None) -> int:
    """Stream pages into a PDF: each page is JPEG-encoded (DCTDecode, like Pillow's PDF
    writer) and written as soon as it arrives, so only one page is in memory, and nothing is
    read back. Catalog first, page tree and info at the end. Returns the page count."""
    offsets: dict[int, int] = {}
    pos = 0

    def out(data: bytes):
        nonlocal pos
        fh.write(data)
        pos += len(data)

    def obj(num: int, body: bytes, stream: bytes | None = None):
        offsets[num] = pos
        out(b"%d 0 obj\n" % num + body)
        if stream is not None:
            out(b"stream\n" + stream + b"\nendstream")
        out(b"\nendobj\n")

    out(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    kids, num = [], 3
    for page in pages:
        page = page if page.mode in ("RGB", "L") else page.convert("RGB")
        buf = io.BytesIO()
        page.save(buf, "JPEG")
        jpeg = buf.getvalue()
        space = b"/DeviceRGB" if page.mode == "RGB" else b"/DeviceGray"
        image, content, page_obj = num, num + 1, num + 2
        obj(image, b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace %s "
                   b"/BitsPerComponent 8 /Filter /DCTDecode /Length %d >>\n"
                   % (page.width, page.height, space, len(jpeg)), jpeg)
        w_pt, h_pt = page.width * 72.0 / dpi, page.height * 72.0 / dpi
        draw = b"q %f 0 0 %f 0 0 cm /image Do Q\n" % (w_pt, h_pt)
        obj(content, b"<< /Length %d >>\n" % len(draw), draw)
        obj(page_obj, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %s %s] "
                      b"/Resources << /ProcSet [/PDF /ImageC] /XObject << /image %d 0 R >> >> "
                      b"/Contents %d 0 R >>" % (str(w_pt).encode(), str(h_pt).encode(),
                                                image, content))
        kids.append(page_obj)
        num += 3
    if not kids:
        raise ValueError("no pages to save")
    obj(2, b"<< /Type /Pages /Count %d /Kids [%s] >>"
           % (len(kids), b" ".join(b"%d 0 R" % k for k in kids)))
    stamp = time.strftime("D:%Y%m%d%H%M%SZ", time.gmtime()).encode()
    info = b"<< /CreationDate (%s) /ModDate (%s)" % (stamp, stamp)
    if title:
        info += b" /Title " + _pdf_text(title)
    obj(num, info + b" >>")
    xref = pos
    out(b"xref\n0 %d\n0000000000 65535 f \n" % (num + 1))
    for n in range(1, num + 1):
        out(b"%010d 00000 n \n" % offsets[n])
    out(b"trailer\n<< /Size %d /Root 1 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (num + 1, num, xref))
    return len(kids)


def save_pdf(pages, path, dpi: int = DPI) -> int:
    """Write pages (any iterable, e.g. iter_pages(...)) to a PDF, one page in memory at a
    time; returns the page count. `path` may also be a writable file object. A path is
    written to a temporary file next to it and renamed into place once complete, so a failed
    or interrupted run never leaves a truncated PDF or destroys an existing one."""
    if hasattr(path, "write"):
        name = getattr(path, "name", None)
        return _write_pdf(pages, path, dpi, Path(name).stem if isinstance(name, str) else None)
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    try:
        with open(tmp, "wb") as fh:
            n = _write_pdf(pages, fh, dpi, path.stem)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return n
