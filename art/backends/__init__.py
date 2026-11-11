"""Generation backends. An image backend turns a prompt into an image; a text backend runs the
language model that writes scene descriptions. Each backend module exposes a factory that
takes a profile (a dict of settings, see art/config.py) and returns a backend.

Image backends: comfyui, a1111 (Automatic1111 / Forge / SD.Next), openai, gemini, placeholder.
Text backends: comfyui, openai, gemini, anthropic, placeholder.
Hosted backends import their vendor SDK only when used (pip install slopxy[openai] etc.)."""

from __future__ import annotations

import importlib
import math
import os
from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image


class ArtError(RuntimeError):
    """A backend could not be reached or failed to produce a result."""


class ArtUnavailable(ArtError):
    """The backend can't be used at all: not reachable, no API key, SDK not installed."""


class ArtDeclined(ArtError):
    """The model declined or gave nothing for this one request (a refusal, a blocked or empty
    answer); other requests to the same backend may still work."""


# SDXL/Flux-friendly resolutions (~1 megapixel), for backends that take any size.
SD_SIZES = [(1024, 1024), (1152, 896), (1216, 832), (1344, 768), (1536, 640),
            (896, 1152), (832, 1216), (768, 1344), (640, 1536)]

# Style text for models that follow plain language and draw quoted words literally (hosted
# models, Krea 2): no flavor quotes, no "card" wording, and the no-text rule in words, since
# they take no negative prompt.
PLAIN_STYLE = ("epic fantasy digital painting, full-bleed illustration filling the entire "
               "image, dramatic lighting, highly detailed. The image contains no text, no "
               "letters, no border and no frame")


@dataclass(frozen=True)
class ImageCaps:
    """What an image backend accepts; the generator plans specs to match."""
    sizes: list = field(default_factory=lambda: list(SD_SIZES))  # [(w, h)] it can produce
    seed: bool = True       # takes a seed, so a spec reproduces the same image
    negative: bool = True   # takes a negative prompt

    def size_for(self, w: float, h: float) -> tuple[int, int]:
        """The supported size closest in aspect ratio to a w x h art window."""
        target = math.log(w / h)
        return min(self.sizes, key=lambda s: abs(math.log(s[0] / s[1]) - target))


class ImageBackend(Protocol):
    name: str            # backend name (registry key), stored in the PNG metadata
    model: str           # model / preset, stored in the PNG metadata
    style: str | None    # style text for every prompt with this model (None: the default)
    flavor: bool         # whether template prompts may quote flavor text
    caps: ImageCaps
    concurrency: int     # requests it can usefully run at once (1 for a local GPU)
    paid: bool           # charges per image (runs ask before making many)

    def spec_fields(self) -> dict:
        """Backend-specific settings to store in a spec so it can be reproduced."""

    def generate(self, spec: dict) -> Image.Image:
        """Image for a spec: positive, negative, seed, width, height (see ArtGenerator.plan)."""


class TextBackend(Protocol):
    name: str
    model: str
    concurrency: int
    paid: bool

    def complete(self, prompt: str, seed: int) -> str:
        """The language model's answer to prompt (sampled with seed where supported)."""


# name -> (module, has image backend, has text backend, what it needs)
BACKENDS = {
    "comfyui": ("comfyui", True, True, "a running ComfyUI (default http://127.0.0.1:8189)"),
    "a1111": ("a1111", True, False, "a running Automatic1111 / Forge / SD.Next WebUI "
                                    "started with --api (default http://127.0.0.1:7860)"),
    "openai": ("openai_api", True, True, "pip install slopxy[openai]; OPENAI_API_KEY "
                                         "(or base_url for an OpenAI-compatible server)"),
    "gemini": ("gemini_api", True, True, "pip install slopxy[gemini]; GEMINI_API_KEY "
                                         "or GOOGLE_API_KEY"),
    "anthropic": ("anthropic_api", False, True, "pip install slopxy[anthropic]; "
                                                "ANTHROPIC_API_KEY"),
    "placeholder": ("placeholder", True, True, "nothing: draws the prompt (for testing)"),
}


def _module(name: str):
    if name not in BACKENDS:
        raise ArtError(f"unknown backend '{name}' (known: {', '.join(BACKENDS)})")
    return importlib.import_module(f"{__name__}.{BACKENDS[name][0]}")


def image_backend(settings: dict) -> ImageBackend:
    """An image backend for a resolved profile (settings["backend"] names it)."""
    name = settings["backend"]
    if not BACKENDS.get(name, (None, False))[1]:
        raise ArtError(f"'{name}' can't generate images; image backends: "
                       + ", ".join(n for n, b in BACKENDS.items() if b[1]))
    return _make(_module(name).image_backend, settings)


def text_backend(settings: dict) -> TextBackend:
    """A text backend for a resolved profile (settings["backend"] names it)."""
    name = settings["backend"]
    if not BACKENDS.get(name, (None, False, False))[2]:
        raise ArtError(f"'{name}' can't write scene descriptions; text backends: "
                       + ", ".join(n for n, b in BACKENDS.items() if b[2]))
    return _make(_module(name).text_backend, settings)


def _make(factory, settings: dict):
    try:
        return factory(settings)
    except (ValueError, TypeError) as e:  # e.g. concurrency = "four"
        raise ArtError(f"profile '{settings.get('profile', settings['backend'])}' has a bad "
                       f"setting: {e}") from None


def require_sdk(module: str, extra: str):
    """Import a vendor SDK, or explain how to install it."""
    try:
        return importlib.import_module(module)
    except ImportError:
        raise ArtUnavailable(f"the {extra} backend needs its SDK: "
                             f"pip install slopxy[{extra}]") from None


def api_key(settings: dict, *default_envs: str) -> str | None:
    """The API key for a profile: from the env var it names (api_key_env), else the vendor's
    usual variables. Keys never live in config files or PNG metadata."""
    envs = [settings["api_key_env"]] if settings.get("api_key_env") else list(default_envs)
    for env in envs:
        if os.environ.get(env):
            return os.environ[env]
    return None
