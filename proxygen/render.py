"""Proxy card renderer: draws the whole card frame and text from scratch."""

from __future__ import annotations

import json
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFilter, PngImagePlugin

from . import textlayout
from .colors import Palette, colors_from_cost, frame_style
from .fonts import font
from .symbols import draw_cost

W, H = 750, 1050          # 2.5" x 3.5" at 300 DPI
DPI = 300
BORDER = 30
CORNER = 36
TEXT_INK = (15, 15, 15)
FOOTER_INK = (235, 235, 235)

DFC_LAYOUTS = {"transform", "modal_dfc", "reversible_card", "double_faced_token", "battle"}
SIDE_PANEL_LAYOUTS = {"adventure", "omen"}
STACKED_LAYOUTS = {"split", "flip", "aftermath"}


@dataclass
class Face:
    name: str
    mana_cost: str
    type_line: str
    oracle_text: str
    flavor_text: str | None
    power: str | None
    toughness: str | None
    loyalty: str | None
    defense: str | None
    colors: list
    color_identity: list
    artist: str


@dataclass
class ArtWindow:
    rect: tuple       # (x0, y0, x1, y1) where art goes under the card
    face: Face        # face the art illustrates (drives the prompt)
    key: str = ""     # distinguishes several windows on one image


@dataclass
class Rendered:
    suffix: str       # filename suffix, e.g. "_front"
    image: Image.Image
    windows: list     # list[ArtWindow]


def _face(src: dict, card: dict, printing: dict, idx: int | None) -> Face:
    flavor = printing.get("flavor_text")
    artist = printing.get("artist", "")
    if idx is not None:
        ff = printing.get("face_flavor") or []
        fa = printing.get("face_artist") or []
        flavor = ff[idx] if idx < len(ff) else None
        artist = (fa[idx] if idx < len(fa) else None) or artist
    colors = src.get("colors")
    if colors is None:
        colors = colors_from_cost(src.get("mana_cost")) if idx is not None else card.get("colors", [])
    return Face(
        name=src.get("name", ""),
        mana_cost=src.get("mana_cost", ""),
        type_line=src.get("type_line", ""),
        oracle_text=src.get("oracle_text", ""),
        flavor_text=flavor,
        power=src.get("power"),
        toughness=src.get("toughness"),
        loyalty=src.get("loyalty"),
        defense=src.get("defense"),
        colors=colors,
        color_identity=card.get("color_identity", []),
        artist=artist or "",
    )


# ---------------------------------------------------------------- primitives

def _gradient(size, c1, c2) -> Image.Image:
    w, h = size
    # linear_gradient is black->white top->bottom; rotated 90° CCW it runs left->right
    g = Image.linear_gradient("L").rotate(90, expand=True).resize((w, h))
    return Image.composite(Image.new("RGB", (w, h), c2), Image.new("RGB", (w, h), c1), g)


def _shape(img: Image.Image, rect, radius, fill, fill2=None, outline=None, width=0):
    """Rounded rectangle; with fill2, a left->right two-tone gradient."""
    x0, y0, x1, y1 = (int(round(v)) for v in rect)
    if fill2 is None:
        ImageDraw.Draw(img).rounded_rectangle([x0, y0, x1, y1], radius, fill=fill,
                                              outline=outline, width=width)
        return
    w, h = x1 - x0 + 1, y1 - y0 + 1
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius, fill=255)
    img.paste(_gradient((w, h), fill, fill2), (x0, y0), mask)
    if outline:
        ImageDraw.Draw(img).rounded_rectangle([x0, y0, x1, y1], radius, outline=outline, width=width)


def _fit_font(draw, text, role, size, max_w, min_size=14):
    f = font(role, size)
    while size > min_size and draw.textlength(text, font=f) > max_w:
        size -= 1
        f = font(role, size)
    return f


def _ctext(d: ImageDraw.ImageDraw, xy, text, font, anchor="lm", **kw):
    """Draw text vertically centered on its cap height (looks centered in bars,
    unlike Pillow's 'm' anchor which centers ascender-to-descender)."""
    cap = -font.getbbox("H", anchor="ls")[1]
    d.text((xy[0], xy[1] + cap / 2), text, font=font, anchor=anchor[0] + "s", **kw)


def _new_canvas() -> Image.Image:
    img = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    ImageDraw.Draw(img).rounded_rectangle([0, 0, W - 1, H - 1], CORNER, fill=(12, 12, 12))
    return img


def _palettes(face: Face):
    return frame_style(face.type_line, face.colors, face.mana_cost, face.color_identity)


def _pick(p: Palette, p2: Palette | None, attr: str):
    return getattr(p, attr), (getattr(p2, attr) if p2 else None)


# ---------------------------------------------------------------- components

def _title_bar(img, rect, face: Face, pal, pal2, cost_d=38, name_size=40, opacity=1.0):
    """opacity < 1 makes only the bar's fill see-through (outline, name and cost stay solid),
    so art layered under a full-art card shows through it."""
    a, b = _pick(pal, pal2, "bar")
    if opacity >= 1:
        _shape(img, rect, 12, a, b, outline=pal.edge, width=3)
    else:
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        _shape(layer, rect, 12, a, b)
        layer.putalpha(layer.getchannel("A").point(lambda v: round(v * opacity)))
        img.alpha_composite(layer)
        ImageDraw.Draw(img).rounded_rectangle([int(round(v)) for v in rect], 12,
                                              outline=pal.edge, width=3)
    x0, y0, x1, y1 = rect
    cy = (y0 + y1) / 2
    left = draw_cost(img, face.mana_cost, x1 - 14, cy, cost_d)
    d = ImageDraw.Draw(img)
    f = _fit_font(d, face.name, "title", name_size, left - (x0 + 18) - 10)
    if opacity < 1:  # soft light glow so the name reads over dark art
        # Blur only the shape and fill it with solid glow color: blurring an RGBA layer
        # would mix its transparent black into the halo and leave dark fringes.
        shape = Image.new("L", img.size, 0)
        _ctext(ImageDraw.Draw(shape), (x0 + 18, cy), face.name, font=f, fill=255,
               anchor="lm", stroke_width=5, stroke_fill=255)
        shape = shape.filter(ImageFilter.GaussianBlur(5))
        glow = Image.new("RGBA", img.size, GLOW_COLOR + (0,))
        glow.putalpha(shape.point(lambda v: round(min(255, v * 2) * GLOW_OPACITY)))
        img.alpha_composite(glow)
    _ctext(d, (x0 + 18, cy), face.name, font=f, fill=TEXT_INK, anchor="lm")


def _type_bar(img, rect, face: Face, printing, pal, pal2, size=33):
    a, b = _pick(pal, pal2, "bar")
    _shape(img, rect, 12, a, b, outline=pal.edge, width=3)
    x0, y0, x1, y1 = rect
    cy = (y0 + y1) / 2
    d = ImageDraw.Draw(img)
    f = _fit_font(d, face.type_line, "type", size, x1 - x0 - 36)
    _ctext(d, (x0 + 18, cy), face.type_line, font=f, fill=TEXT_INK, anchor="lm")


def _art_box(img, rect, pal):
    """Outline the art window and punch its interior to fully transparent,
    so artwork can be layered underneath the card image."""
    x0, y0, x1, y1 = (int(round(v)) for v in rect)
    ImageDraw.Draw(img).rectangle([x0, y0, x1, y1], outline=pal.edge, width=3)
    img.paste((0, 0, 0, 0), (x0 + 3, y0 + 3, x1 - 2, y1 - 2))


def _text_box(img, rect, face: Face, pal, pal2, reserve_bottom=0, max_size=34, pad=18,
              draw_box=True, valign="center"):
    if draw_box:
        a, b = _pick(pal, pal2, "box")
        _shape(img, rect, 4, a, b, outline=pal.edge, width=3)
    x0, y0, x1, y1 = rect
    paras = textlayout.build_paragraphs(face.oracle_text, face.flavor_text)
    if not paras:
        return
    w = x1 - x0 - 2 * pad
    h = y1 - y0 - 2 * pad + 4 - reserve_bottom
    lay = textlayout.fit(paras, w, h, max_size=max_size, min_size=12)
    textlayout.render(img, lay, x0 + pad, y0 + pad - 2, w, h, valign)


def _stat_box(img, face: Face, pal, rect=(566, 928, 708, 986), size=40):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = rect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if face.power is not None and face.toughness is not None:
        _shape(img, rect, 12, pal.bar, outline=pal.edge, width=3)
        txt = f"{face.power}/{face.toughness}"
        f = _fit_font(d, txt, "title", size, x1 - x0 - 16)
        _ctext(d, (cx, cy), txt, font=f, fill=TEXT_INK, anchor="mm")
    elif face.loyalty is not None:
        h = y1 - y0
        pts = [(x0 + 30, y0 - 8), (x1 - 30, y0 - 8), (x1 - 30, y0 + h * 0.6),
               (cx, y1 + 4), (x0 + 30, y0 + h * 0.6)]
        d.polygon(pts, fill=(28, 28, 28), outline=(210, 210, 210), width=3)
        _ctext(d, (cx, cy - 6), str(face.loyalty), font=font("title", size), fill=(250, 250, 250), anchor="mm")
    elif face.defense is not None:
        pts = [(cx, cy - 44), (x1 - 14, cy - 6), (cx, cy + 32), (x0 + 14, cy - 6)]
        d.polygon(pts, fill=(150, 40, 30), outline=(240, 220, 200), width=3)
        _ctext(d, (cx, cy - 6), str(face.defense), font=font("title", size), fill=(250, 250, 250), anchor="mm")


def _has_stat(face: Face) -> bool:
    return (face.power is not None and face.toughness is not None) or \
        face.loyalty is not None or face.defense is not None


def _footer(img, note: str = ""):
    d = ImageDraw.Draw(img)
    f = font("info", 17)
    fb = font("info_bold", 17)
    y1, y2 = 1001, 1024
    _ctext(d, (W - 44, y1), "PROXY — NOT FOR SALE", font=fb, fill=FOOTER_INK, anchor="rm")
    if note:
        _ctext(d, (W - 44, y2), note, font=f, fill=FOOTER_INK, anchor="rm")


def _finish(img: Image.Image) -> Image.Image:
    # Kept as RGBA: the art window is transparent.
    return img


# ---------------------------------------------------------------- layouts

FRAME = (BORDER, BORDER, W - BORDER, 982)
TITLE = (44, 44, 706, 104)
ART = (58, 110, 692, 568)
FULL_ART_BAR_OPACITY = 0.25  # full-art title bar fill; the art shows through the rest
GLOW_COLOR = (255, 255, 255)  # halo behind the full-art title text
GLOW_OPACITY = 0.65           # the halo's peak opacity
TYPE = (44, 574, 706, 632)
TEXT = (58, 638, 692, 950)


def render_standard(face: Face, printing: dict, note: str = "") -> Image.Image:
    img = _new_canvas()
    pal, pal2 = _palettes(face)
    a, b = _pick(pal, pal2, "frame")
    _shape(img, FRAME, 16, a, b)
    _title_bar(img, TITLE, face, pal, pal2)
    _art_box(img, ART, pal)
    _type_bar(img, TYPE, face, printing, pal, pal2)
    stat = _has_stat(face)
    _text_box(img, TEXT, face, pal, pal2, reserve_bottom=24 if stat else 0)
    if stat:
        _stat_box(img, face, pal)
    _footer(img, note)
    return _finish(img), [ArtWindow(ART, face)]


def render_side_panel(main: Face, side: Face, printing: dict) -> Image.Image:
    """Adventure-style: main face text on the right, secondary spell on a left panel."""
    img = _new_canvas()
    pal, pal2 = _palettes(main)
    a, b = _pick(pal, pal2, "frame")
    _shape(img, FRAME, 16, a, b)
    _title_bar(img, TITLE, main, pal, pal2)
    _art_box(img, ART, pal)
    _type_bar(img, TYPE, main, printing, pal, pal2)

    bx, by = _pick(pal, pal2, "box")
    _shape(img, TEXT, 4, bx, by, outline=pal.edge, width=3)
    x0, y0, x1, y1 = TEXT
    split = x0 + int((x1 - x0) * 0.48)

    # Left panel: the adventure/omen spell.
    spal, spal2 = _palettes(side)
    panel = (x0 + 6, y0 + 6, split - 4, y1 - 6)
    _shape(img, panel, 6, spal.box, spal2.box if spal2 else None, outline=spal.edge, width=2)
    px0, py0, px1, py1 = panel
    hdr = (px0, py0, px1, py0 + 40)
    _shape(img, hdr, 6, spal.bar, spal2.bar if spal2 else None, outline=spal.edge, width=2)
    d = ImageDraw.Draw(img)
    left = draw_cost(img, side.mana_cost, px1 - 8, (hdr[1] + hdr[3]) / 2, 24, gap=2)
    f = _fit_font(d, side.name, "title", 24, left - px0 - 18, min_size=12)
    _ctext(d, (px0 + 8, (hdr[1] + hdr[3]) / 2), side.name, font=f, fill=TEXT_INK, anchor="lm")
    tf = _fit_font(d, side.type_line, "type", 20, px1 - px0 - 16, min_size=11)
    _ctext(d, (px0 + 8, hdr[3] + 16), side.type_line, font=tf, fill=TEXT_INK, anchor="lm")
    d.line([px0 + 6, hdr[3] + 32, px1 - 6, hdr[3] + 32], fill=spal.edge, width=1)
    side_text = Face(**{**side.__dict__, "flavor_text": None})
    _text_box(img, (px0, hdr[3] + 30, px1, py1), side_text, spal, None,
              max_size=26, pad=10, draw_box=False, valign="top")

    # Right: the main face.
    stat = _has_stat(main)
    _text_box(img, (split, y0, x1, y1), main, pal, pal2, reserve_bottom=24 if stat else 0,
              max_size=30, pad=14, draw_box=False)
    if stat:
        _stat_box(img, main, pal)
    _footer(img)
    return _finish(img), [ArtWindow(ART, main)]


def render_stacked(faces: list[Face], printing: dict) -> Image.Image:
    """Split / flip / aftermath: two half-cards stacked top and bottom."""
    img = _new_canvas()
    fx0, fy0, fx1, fy1 = FRAME
    half = (fy1 - fy0) / len(faces)
    windows = []
    for i, face in enumerate(faces):
        top = fy0 + i * half
        bot = top + half
        pal, pal2 = _palettes(face)
        a, b = _pick(pal, pal2, "frame")
        _shape(img, (fx0, top, fx1, bot), 16, a, b)
        title = (44, top + 12, 706, top + 62)
        art = (58, title[3] + 6, 692, title[3] + 6 + 140)
        typ = (44, art[3] + 6, 706, art[3] + 54)
        txt = (58, typ[3] + 6, 692, bot - 12)
        _title_bar(img, title, face, pal, pal2, cost_d=34, name_size=36)
        _art_box(img, art, pal)
        windows.append(ArtWindow(art, face, f"_{i + 1}"))
        _type_bar(img, typ, face, printing, pal, pal2, size=30)
        stat = _has_stat(face)
        _text_box(img, txt, face, pal, pal2, reserve_bottom=30 if stat else 0, max_size=30, pad=14)
        if stat:
            _stat_box(img, face, pal, rect=(600, txt[3] - 44, 700, txt[3] - 4), size=30)
    # divider between the halves
    ImageDraw.Draw(img).line([fx0, fy0 + half, fx1, fy0 + half], fill=(12, 12, 12), width=6)
    _footer(img)
    return _finish(img), windows


def render_full_art(face: Face, note: str = "") -> Image.Image:
    """Full-art: the whole area inside the black border is a transparent art
    window, with only the title bar (name + mana cost) drawn over it."""
    img = _new_canvas()
    x0, y0, x1, y1 = FRAME
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([x0, y0, x1, y1], 16, fill=255)
    img.paste((0, 0, 0, 0), (0, 0), mask)
    pal, pal2 = _palettes(face)
    _title_bar(img, TITLE, face, pal, pal2, opacity=FULL_ART_BAR_OPACITY)
    _footer(img, note)
    return _finish(img), [ArtWindow(FRAME, face)]


def _dfc_note(raw_faces, i, labels) -> str:
    if len(raw_faces) != 2:
        return ""
    other = raw_faces[1 - i]["name"]
    return f"{labels[i].title()} face • {'back' if i == 0 else 'front'}: {other}"


def render_card(card: dict, printing: dict, full_art: bool = False) -> list[Rendered]:
    """Render a card into one or more images, each with its art window(s)."""
    layout = card.get("layout", "normal")
    raw_faces = card.get("faces") or []
    labels = ["front", "back"] + [f"face{i}" for i in range(3, len(raw_faces) + 1)]
    if full_art:
        if raw_faces and layout in DFC_LAYOUTS:
            return [Rendered(f"_{labels[i]}", *render_full_art(_face(rf, card, printing, i),
                                                               _dfc_note(raw_faces, i, labels)))
                    for i, rf in enumerate(raw_faces)]
        if raw_faces and layout in SIDE_PANEL_LAYOUTS | {"flip"}:
            face = _face(raw_faces[0], card, printing, 0)
            face.colors = card.get("colors", face.colors)
            return [Rendered("", *render_full_art(face))]
        return [Rendered("", *render_full_art(_face(card, card, printing, None)))]
    if raw_faces and layout in DFC_LAYOUTS:
        return [Rendered(f"_{labels[i]}", *render_standard(_face(rf, card, printing, i), printing,
                                                            _dfc_note(raw_faces, i, labels)))
                for i, rf in enumerate(raw_faces)]
    if raw_faces and len(raw_faces) == 2 and layout in SIDE_PANEL_LAYOUTS:
        main = _face(raw_faces[0], card, printing, 0)
        main.power = main.power if main.power is not None else card.get("power")
        main.toughness = main.toughness if main.toughness is not None else card.get("toughness")
        side = _face(raw_faces[1], card, printing, 1)
        return [Rendered("", *render_side_panel(main, side, printing))]
    if raw_faces and layout != "meld":
        faces = [_face(rf, card, printing, i) for i, rf in enumerate(raw_faces)]
        if layout == "flip":
            # flip cards share the card's colors
            for f in faces:
                f.colors = card.get("colors", f.colors)
        return [Rendered("", *render_stacked(faces, printing))]
    return [Rendered("", *render_standard(_face(card, card, printing, None), printing))]


META_KEY = "proxygen"  # PNG iTXt chunk holding the JSON needed to reproduce the image


def save(img: Image.Image, path, meta: dict | None = None) -> None:
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "proxygen")
    if meta is not None:
        info.add_itxt(META_KEY, json.dumps(meta, ensure_ascii=False))
    img.save(path, dpi=(DPI, DPI), pnginfo=info)


def read_meta(path) -> dict | None:
    """The proxygen metadata embedded in an output PNG, or None."""
    with Image.open(path) as im:
        raw = getattr(im, "text", {}).get(META_KEY)
    return json.loads(raw) if raw else None
