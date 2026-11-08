"""Fitting art images to art windows and layering them under the card frame."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

FIT_MODES = ("crop", "fit")


def fit_art(art: Image.Image, size: tuple[int, int], mode: str = "crop") -> Image.Image:
    """Size art to a window. crop: fill the window, trimming the overflow (centered).
    fit: show the whole image, filling the leftover space with a blurred, darkened copy."""
    art = art.convert("RGBA")
    cover = ImageOps.fit(art, size, Image.LANCZOS)
    if mode != "fit":
        return cover
    bg = cover.filter(ImageFilter.GaussianBlur(max(size) / 30))
    bg = Image.blend(bg, Image.new("RGBA", size, (0, 0, 0, 255)), 0.35)
    inner = ImageOps.contain(art, size, Image.LANCZOS)
    bg.alpha_composite(inner, ((size[0] - inner.width) // 2, (size[1] - inner.height) // 2))
    return bg


def compose(card: Image.Image, layers: list[tuple[Image.Image, tuple, str]]) -> Image.Image:
    """Put each art image under the card at its window rect: [(art, rect, fit_mode), ...]."""
    base = Image.new("RGBA", card.size, (0, 0, 0, 0))
    for art, rect, mode in layers:
        x0, y0, x1, y1 = (int(round(v)) for v in rect)
        base.paste(fit_art(art, (x1 - x0 + 1, y1 - y0 + 1), mode), (x0, y0))
    base.alpha_composite(card)
    return base


def load_image_art(path: str | Path) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img
