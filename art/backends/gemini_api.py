"""Google Gemini backend (official `google-genai` SDK: pip install slopxy[gemini]) for images
(Nano Banana models) and scene descriptions, through the Interactions API.

Profile settings:
  model        image model (default gemini-3.1-flash-image; e.g. gemini-nano-banana-2.1,
               gemini-3-pro-image, gemini-3.1-flash-lite-image)
               or text model (default gemini-3.6-flash)
  api_key_env  env var holding the key (default GOOGLE_API_KEY, then GEMINI_API_KEY)
  base_url     another endpoint for the Gemini API (e.g. a proxy)
  image_size   images: "512", "1K" (default), "2K" or "4K"
  thinking_level  text: minimal | low | medium | high (default low for the default model)
  max_tokens   text: output limit (default 4000)
  concurrency  parallel requests (default 4)"""

from __future__ import annotations

import base64
import io

from PIL import Image

from . import PLAIN_STYLE, ArtDeclined, ArtError, ImageCaps, api_key, require_sdk

DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image"
DEFAULT_TEXT_MODEL = "gemini-3.6-flash"
# Aspect ratios every current Nano Banana model accepts, with a ~1 megapixel nominal size each
# (the size only picks the ratio; the model decides the pixels).
RATIOS = {"1:1": (1024, 1024), "2:3": (832, 1248), "3:2": (1248, 832), "3:4": (896, 1184),
          "4:3": (1184, 896), "4:5": (928, 1152), "5:4": (1152, 928), "9:16": (768, 1344),
          "16:9": (1344, 768), "21:9": (1536, 672)}


def _client(settings: dict):
    require_sdk("google.genai", "gemini")
    from google import genai
    from google.genai import errors
    key = api_key(settings, "GOOGLE_API_KEY", "GEMINI_API_KEY")
    if not key:
        raise ArtError("the gemini backend needs an API key: set GEMINI_API_KEY (or "
                       "GOOGLE_API_KEY, or the profile's api_key_env)")
    options = {"base_url": settings["base_url"]} if settings.get("base_url") else None
    return errors, genai.Client(api_key=key, http_options=options)


def image_backend(settings: dict) -> "GeminiImageBackend":
    return GeminiImageBackend(settings)


def text_backend(settings: dict) -> "GeminiTextBackend":
    return GeminiTextBackend(settings)


def _check(interaction, what: str):
    status = getattr(interaction, "status", None)
    if status not in (None, "completed"):
        detail = getattr(interaction, "output_text", None) or ""
        raise ArtDeclined(f"Gemini {what} request ended with status '{status}'"
                       + (f": {detail[:300]}" if detail else ""))


class GeminiImageBackend:
    name = "gemini"
    style = PLAIN_STYLE
    flavor = False
    paid = True

    def __init__(self, settings: dict):
        self.errors, self.client = _client(settings)
        self.model = settings.get("model") or DEFAULT_IMAGE_MODEL
        # Gemini 2.5 Flash Image has no size option.
        self.image_size = None if self.model.startswith("gemini-2.5") else (
            settings.get("image_size") or "1K")
        self.caps = ImageCaps(sizes=list(RATIOS.values()), seed=False, negative=False)
        self.concurrency = int(settings.get("concurrency", 4))

    def spec_fields(self) -> dict:
        return {"image_size": self.image_size} if self.image_size else {}

    def generate(self, spec: dict) -> Image.Image:
        size = (spec["width"], spec["height"])
        ratio = next((r for r, s in RATIOS.items() if s == size), None)
        if ratio is None:  # a spec from elsewhere: nearest ratio
            ratio = min(RATIOS, key=lambda r: abs(RATIOS[r][0] / RATIOS[r][1] - size[0] / size[1]))
        fmt = {"type": "image", "aspect_ratio": ratio}
        if spec.get("image_size", self.image_size):
            fmt["image_size"] = spec.get("image_size", self.image_size)
        try:
            interaction = self.client.interactions.create(
                model=spec.get("model", self.model), input=spec["positive"], response_format=fmt)
        except self.errors.APIError as e:
            raise ArtError(f"Gemini image request failed: {e}") from None
        _check(interaction, "image")
        image = getattr(interaction, "output_image", None)
        if image is None or not image.data:
            text = getattr(interaction, "output_text", None) or ""
            raise ArtDeclined("Gemini returned no image" + (f": {text[:300]}" if text else ""))
        img = Image.open(io.BytesIO(base64.b64decode(image.data)))
        img.load()
        return img


class GeminiTextBackend:
    name = "gemini"
    paid = True

    def __init__(self, settings: dict):
        self.errors, self.client = _client(settings)
        self.model = settings.get("model") or DEFAULT_TEXT_MODEL
        default_level = "low" if self.model == DEFAULT_TEXT_MODEL else None
        self.thinking_level = settings.get("thinking_level", default_level)
        self.max_tokens = int(settings.get("max_tokens", 4000))
        self.concurrency = int(settings.get("concurrency", 4))

    def complete(self, prompt: str, seed: int) -> str:
        config = {"max_output_tokens": self.max_tokens, "seed": seed}
        if self.thinking_level:
            config["thinking_level"] = self.thinking_level
        try:
            interaction = self.client.interactions.create(model=self.model, input=prompt,
                                                          generation_config=config)
        except self.errors.APIError as e:
            raise ArtError(f"Gemini text request failed: {e}") from None
        _check(interaction, "text")
        return interaction.output_text or ""
