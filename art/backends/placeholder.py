"""Placeholder backend: draws the prompt instead of generating art, and writes a canned scene
description. Needs nothing, costs nothing; for trying layouts, runs and sheets, and for tests.
Deterministic: the same spec (seed included) always gives the same image."""

from __future__ import annotations

import hashlib
import random
import textwrap

from PIL import Image, ImageDraw

from . import SD_SIZES, ImageCaps
from ...fonts import font


def image_backend(settings: dict) -> "PlaceholderImageBackend":
    return PlaceholderImageBackend(settings)


def text_backend(settings: dict) -> "PlaceholderTextBackend":
    return PlaceholderTextBackend()


class PlaceholderImageBackend:
    name = "placeholder"
    model = "placeholder"
    style = None
    flavor = True
    paid = False

    def __init__(self, settings: dict):
        self.caps = ImageCaps(sizes=list(SD_SIZES), seed=True, negative=True)
        self.concurrency = int(settings.get("concurrency", 4))

    def spec_fields(self) -> dict:
        return {}

    def generate(self, spec: dict) -> Image.Image:
        w, h = spec["width"], spec["height"]
        rng = random.Random(f"{spec.get('seed')}|{spec['positive']}")
        c1 = tuple(rng.randrange(40, 200) for _ in range(3))
        c2 = tuple(rng.randrange(40, 200) for _ in range(3))
        grad = Image.linear_gradient("L").resize((w, h))
        img = Image.composite(Image.new("RGB", (w, h), c2), Image.new("RGB", (w, h), c1), grad)
        d = ImageDraw.Draw(img)
        size = max(14, h // 28)
        f = font("info", size)
        lines = textwrap.wrap(spec["positive"], width=max(20, w // (size // 2 + 1)))[:16]
        y = h // 12
        for line in [f"seed {spec.get('seed')}"] + lines:
            d.text((w // 16, y), line, font=f, fill=(250, 250, 250))
            y += int(size * 1.3)
        return img


class PlaceholderTextBackend:
    name = "placeholder"
    model = "placeholder"
    concurrency = 4
    paid = False

    def complete(self, prompt: str, seed: int) -> str:
        digest = hashlib.sha1(f"{seed}|{prompt}".encode()).hexdigest()[:8]
        return f"A placeholder scene description ({digest}) for the card described in the prompt."
