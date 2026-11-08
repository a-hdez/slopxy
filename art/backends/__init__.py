"""Generation backends. An image backend turns a prompt into an image; a text backend runs the
language model that writes scene descriptions. ComfyUI provides both today (comfyui.py)."""

from __future__ import annotations

from typing import Protocol

from PIL import Image


class ArtError(RuntimeError):
    """A backend could not be reached or failed to produce a result."""


class ImageBackend(Protocol):
    model: str        # preset name, stored in the PNG metadata
    checkpoint: str   # model file, stored in the PNG metadata
    style: str | None  # style text for every prompt with this model (None: the default)
    flavor: bool      # whether template prompts may quote flavor text

    def generate(self, spec: dict) -> Image.Image:
        """Image for a spec: positive, negative, seed, width, height (see ArtGenerator.plan)."""


class TextBackend(Protocol):
    def complete(self, prompt: str, seed: int) -> str:
        """The language model's answer to prompt, sampled with seed."""
