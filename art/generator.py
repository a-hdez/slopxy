"""ArtGenerator: turns card faces into generation specs and images, using an image backend and
(for scene descriptions) a text backend. Nothing here knows which backend is in use."""

from __future__ import annotations

import random
import threading

from PIL import Image

from .backends import ArtDeclined, ArtError, ImageBackend, TextBackend
from .prompts import STYLE, basic_land_type, build_prompt, clean_description, describe_prompt, \
    land_hints


def new_seed() -> int:
    return random.randrange(2**32)


class ArtGenerator:
    """Generates art with one image backend. Nothing is cached: every call to plan() picks a
    seed (random unless fixed) and the resulting spec is stored in the output PNG instead."""

    def __init__(self, image: ImageBackend, text: TextBackend | None = None,
                 seed: int | None = None, describe: bool = True, log=print,
                 profile: str | None = None):
        self.image = image
        self.text = text
        self.model = image.model
        self.style = image.style or STYLE
        self.flavor = image.flavor
        self.seed = seed
        self.describe = describe and text is not None
        self.log = log
        self.profile = profile  # config profile it came from, kept in specs for reproducing
        self._lock = threading.Lock()  # descriptions may run in parallel

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
            except ArtDeclined as e:  # this card only: the next one may well work
                self.log(f"  no scene description for {label or face.name} ({e}); using the "
                         "template prompt")
            except ArtError as e:
                with self._lock:
                    if self.describe:
                        self.log(f"  scene description failed ({e}); using the template "
                                 "prompt. Pass --no-describe to skip this step.")
                    self.describe = False
        positive, negative = build_prompt(face, override, self.style, self.flavor)
        return positive, negative, info

    def describe_face(self, face, label: str = "", hint: str | None = None) -> str:
        """Scene description written by the language model (random seed unless --seed)."""
        if self.text is None:
            raise ArtError("no backend writes scene descriptions here; choose one with "
                           "--describe-backend")
        prompt = describe_prompt(face, hint)
        seed = self._seed()
        self.log(f"  describing scene for {label or face.name}...")
        text = clean_description(self.text.complete(prompt, seed))
        if not text:
            raise ArtDeclined("the language model returned an empty description")
        return text

    def plan(self, face, rect, override: str | None = None, label: str = "",
             hint: str | None = None) -> dict:
        """Everything needed to generate (and later reproduce) art for one window."""
        positive, negative, info = self.prompt_for(face, override, label, hint)
        caps = self.image.caps
        w, h = caps.size_for(rect[2] - rect[0], rect[3] - rect[1])
        spec = {"source": "generated", "backend": self.image.name, "model": self.model,
                **self.image.spec_fields(), "positive": positive}
        if caps.negative:
            spec["negative"] = negative
        spec.update({"seed": self._seed() if caps.seed else None, "width": w, "height": h})
        if self.profile and self.profile != self.image.name:
            spec["profile"] = self.profile
        if info.get("prompt_source") == "description" and self.text is not None:
            info["describer"] = f"{self.text.name}:{self.text.model}"
        return {**spec, **info}

    def restyle(self, spec: dict, face, label: str = "", hint: str | None = None) -> dict:
        """A copy of spec with a fresh scene description (for --redescribe)."""
        description = self.describe_face(face, label, hint)
        positive, negative = build_prompt(face, description, self.style, self.flavor)
        out = {**spec, "positive": positive, "prompt_source": "description",
               "description": description, "describer": f"{self.text.name}:{self.text.model}"}
        if self.image.caps.negative:
            out["negative"] = negative
        return out

    def generate(self, spec: dict, label: str = "") -> Image.Image:
        seed = f", seed {spec['seed']}" if spec.get("seed") is not None else ""
        self.log(f"  generating art for {label} ({spec['width']}x{spec['height']}, "
                 f"{spec.get('backend', 'comfyui')} {spec['model']}{seed})...")
        return self.image.generate(spec)
