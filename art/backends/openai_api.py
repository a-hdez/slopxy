"""OpenAI backend (official `openai` SDK: pip install slopxy[openai]) for images and scene
descriptions. With base_url it also talks to OpenAI-compatible servers (local or hosted).

Profile settings:
  model        image model (default gpt-image-2.6-sunburst; gpt-image-2.6-flare is faster)
               or text model (default gpt-6.1-terra; required with base_url)
  base_url     an OpenAI-compatible server instead of api.openai.com
  api_key_env  env var holding the key (default OPENAI_API_KEY; optional with base_url)
  quality      images: low | medium | high | auto (API default)
  sizes        images: list of "WxH" the model accepts (default 1024x1024, 1536x1024, 1024x1536)
  api          text: "responses" (default for api.openai.com) or "chat" (default with base_url)
  reasoning_effort  text: e.g. "low" (default for the default model; unset otherwise)
  max_tokens   text: output limit including reasoning (default 4000)
  concurrency  parallel requests (default 4; 1 with base_url)
  paid         whether it charges per request (default true; false with base_url)"""

from __future__ import annotations

import base64
import io
import urllib.request

from PIL import Image

from . import PLAIN_STYLE, ArtError, ImageCaps, api_key, require_sdk

DEFAULT_IMAGE_MODEL = "gpt-image-2.6-sunburst"
DEFAULT_TEXT_MODEL = "gpt-6.1-terra"
DEFAULT_SIZES = [(1024, 1024), (1536, 1024), (1024, 1536)]


def _client(settings: dict):
    openai = require_sdk("openai", "openai")
    key = api_key(settings, "OPENAI_API_KEY")
    base_url = settings.get("base_url")
    if not key:
        if not base_url:
            raise ArtError("the openai backend needs an API key: set OPENAI_API_KEY (or the "
                           "profile's api_key_env)")
        key = "not-needed"  # many local OpenAI-compatible servers take any key
    return openai, openai.OpenAI(api_key=key, base_url=base_url,
                                 max_retries=int(settings.get("max_retries", 3)),
                                 timeout=float(settings.get("timeout", 300)))


def _sizes(settings: dict) -> list[tuple[int, int]]:
    if not settings.get("sizes"):
        return list(DEFAULT_SIZES)
    try:
        return [tuple(int(v) for v in s.lower().split("x")) for s in settings["sizes"]]
    except (ValueError, AttributeError):
        raise ArtError("sizes must be a list like [\"1024x1024\", \"1536x1024\"]") from None


def image_backend(settings: dict) -> "OpenAIImageBackend":
    return OpenAIImageBackend(settings)


def text_backend(settings: dict) -> "OpenAITextBackend":
    return OpenAITextBackend(settings)


class OpenAIImageBackend:
    name = "openai"
    style = PLAIN_STYLE
    flavor = False

    def __init__(self, settings: dict):
        self.sdk, self.client = _client(settings)
        local = bool(settings.get("base_url"))
        self.model = settings.get("model") or DEFAULT_IMAGE_MODEL
        self.quality = settings.get("quality")
        self.caps = ImageCaps(sizes=_sizes(settings), seed=False, negative=False)
        self.concurrency = int(settings.get("concurrency", 1 if local else 4))
        self.paid = bool(settings.get("paid", not local))

    def spec_fields(self) -> dict:
        return {"quality": self.quality} if self.quality else {}

    def generate(self, spec: dict) -> Image.Image:
        kwargs = {"model": spec.get("model", self.model), "prompt": spec["positive"],
                  "size": f"{spec['width']}x{spec['height']}", "n": 1}
        if spec.get("quality", self.quality):
            kwargs["quality"] = spec.get("quality", self.quality)
        try:
            resp = self.client.images.generate(**kwargs)
        except self.sdk.APIError as e:
            raise ArtError(f"OpenAI image request failed: {e}") from None
        item = (resp.data or [None])[0]
        if item is not None and getattr(item, "b64_json", None):
            data = base64.b64decode(item.b64_json)
        elif item is not None and getattr(item, "url", None):  # some compatible servers
            with urllib.request.urlopen(item.url, timeout=120) as r:
                data = r.read()
        else:
            raise ArtError("OpenAI returned no image")
        img = Image.open(io.BytesIO(data))
        img.load()
        return img


class OpenAITextBackend:
    name = "openai"

    def __init__(self, settings: dict):
        self.sdk, self.client = _client(settings)
        local = bool(settings.get("base_url"))
        if local and not settings.get("model"):
            raise ArtError("set the model name for an OpenAI-compatible server (profile model)")
        self.model = settings.get("model") or DEFAULT_TEXT_MODEL
        self.api = settings.get("api", "chat" if local else "responses")
        if self.api not in ("responses", "chat"):
            raise ArtError("api must be \"responses\" or \"chat\"")
        default_effort = "low" if self.model == DEFAULT_TEXT_MODEL else None
        self.effort = settings.get("reasoning_effort", default_effort)
        self.max_tokens = int(settings.get("max_tokens", 4000))
        self.concurrency = int(settings.get("concurrency", 1 if local else 4))
        self.paid = bool(settings.get("paid", not local))

    def complete(self, prompt: str, seed: int) -> str:
        try:
            if self.api == "responses":
                kwargs = {"model": self.model, "input": prompt,
                          "max_output_tokens": self.max_tokens}
                if self.effort:
                    kwargs["reasoning"] = {"effort": self.effort}
                return self.client.responses.create(**kwargs).output_text or ""
            kwargs = {"model": self.model, "max_tokens": self.max_tokens, "seed": seed,
                      "messages": [{"role": "user", "content": prompt}]}
            if self.effort:
                kwargs["reasoning_effort"] = self.effort
            resp = self.client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "") if resp.choices else ""
        except self.sdk.APIError as e:
            raise ArtError(f"OpenAI text request failed: {e}") from None
