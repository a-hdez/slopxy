"""ArtGenerator: turns card faces into generation specs and images, using an image backend and
(for scene descriptions) a text backend. Nothing here knows which backend is in use."""

from __future__ import annotations

import math
import random

from PIL import Image

from .backends import ArtError, ImageBackend, TextBackend
from .prompts import STYLE, basic_land_type, build_prompt, clean_description, describe_prompt, \
    land_hints

# SDXL/Flux-friendly resolutions (~1 megapixel)
BUCKETS = [(1024, 1024), (1152, 896), (1216, 832), (1344, 768), (1536, 640),
           (896, 1152), (832, 1216), (768, 1344), (640, 1536)]


def bucket_for(w: float, h: float) -> tuple[int, int]:
    target = math.log(w / h)
    return min(BUCKETS, key=lambda b: abs(math.log(b[0] / b[1]) - target))


def new_seed() -> int:
    return random.randrange(2**32)


class ArtGenerator:
    """Generates art with one image backend. Nothing is cached: every call to plan() picks a
    seed (random unless fixed) and the resulting spec is stored in the output PNG instead."""

    def __init__(self, image: ImageBackend, text: TextBackend | None = None,
                 seed: int | None = None, describe: bool = True, log=print):
        self.image = image
        self.text = text
        self.model = image.model
        self.checkpoint = image.checkpoint
        self.style = image.style or STYLE
        self.flavor = image.flavor
        self.seed = seed
        self.describe = describe and text is not None
        self.log = log

    def _seed(self) -> int:
        return self.seed if self.seed is not None else new_seed()

    def prompt_for(self, face, override: str | None = None, label: str = "",
                   hint: str | None = None) -> tuple[str, str, dict]:
        """(positive, negative, info): user override, else an LLM scene description,
        else the template prompt. info records where the prompt came from."""
        info = {"prompt_source": "custom" if override else "template"}
        if not override and self.describe:
            if hint is None and basic_land_type(face):
                hint = land_hints(face, 1)[0]
            try:
                override = self.describe_face(face, label, hint)
                info = {"prompt_source": "description", "description": override}
                if hint:
                    info["direction"] = hint
            except ArtError as e:
                self.log(f"  scene description failed ({e}); using the template prompt. "
                         "Pass --no-describe to skip this step.")
                self.describe = False
        positive, negative = build_prompt(face, override, self.style, self.flavor)
        return positive, negative, info

    def describe_face(self, face, label: str = "", hint: str | None = None) -> str:
        """Scene description written by the language model (random seed unless --seed)."""
        if self.text is None:
            raise ArtError("no text backend for scene descriptions")
        prompt = describe_prompt(face, hint)
        seed = self._seed()
        self.log(f"  describing scene for {label or face.name}...")
        text = clean_description(self.text.complete(prompt, seed))
        if not text:
            raise ArtError("the language model returned an empty description")
        return text

    def plan(self, face, rect, override: str | None = None, label: str = "",
             hint: str | None = None) -> dict:
        """Everything needed to generate (and later reproduce) art for one window."""
        positive, negative, info = self.prompt_for(face, override, label, hint)
        w, h = bucket_for(rect[2] - rect[0], rect[3] - rect[1])
        return {"source": "generated", "model": self.model, "checkpoint": self.checkpoint,
                "positive": positive, "negative": negative, "seed": self._seed(),
                "width": w, "height": h, **info}

    def restyle(self, spec: dict, face, label: str = "", hint: str | None = None) -> dict:
        """A copy of spec with a fresh scene description (for --redescribe)."""
        description = self.describe_face(face, label, hint)
        positive, negative = build_prompt(face, description, self.style, self.flavor)
        return {**spec, "positive": positive, "negative": negative,
                "prompt_source": "description", "description": description}

    def generate(self, spec: dict, label: str = "") -> Image.Image:
        self.log(f"  generating art for {label} ({spec['width']}x{spec['height']}, "
                 f"{spec['model']}, seed {spec['seed']})...")
        return self.image.generate(spec)
