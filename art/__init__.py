"""AI art: prompts (prompts.py), generation through pluggable backends (generator.py,
backends/), and layering art under the card (compose.py)."""

from .backends import ArtError
from .compose import FIT_MODES, compose, fit_art, load_image_art
from .generator import ArtGenerator, new_seed
from .prompts import load_prompt_file, variation_hints

__all__ = ["ArtError", "ArtGenerator", "FIT_MODES", "compose", "fit_art", "load_image_art",
           "load_prompt_file", "new_seed", "variation_hints"]
