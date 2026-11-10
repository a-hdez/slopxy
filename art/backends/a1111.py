"""Automatic1111 / Forge / SD.Next backend, through the WebUI API (start it with --api).
Stdlib HTTP only.

Profile settings: url (default http://127.0.0.1:7860), checkpoint (a model title or file
name; default: whatever the WebUI has loaded), steps (25), cfg_scale (6.0), sampler
("DPM++ 2M"), scheduler (e.g. "Karras"; WebUIs that have no separate scheduler ignore it),
model_family ("sdxl" or "flux": with "flux" the style text is plain language and no negative
prompt is sent)."""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request

from PIL import Image

from . import PLAIN_STYLE, SD_SIZES, ArtError, ImageCaps

DEFAULT_URL = "http://127.0.0.1:7860"


def image_backend(settings: dict) -> "A1111Backend":
    return A1111Backend(settings)


class A1111Backend:
    name = "a1111"
    concurrency = 1
    paid = False

    def __init__(self, settings: dict):
        self.url = (settings.get("url") or DEFAULT_URL).rstrip("/")
        self.timeout = float(settings.get("timeout", 600))
        self.steps = int(settings.get("steps", 25))
        self.cfg_scale = float(settings.get("cfg_scale", 6.0))
        self.sampler = settings.get("sampler", "DPM++ 2M")
        self.scheduler = settings.get("scheduler")
        flux = settings.get("model_family", "sdxl") == "flux"
        self.style = PLAIN_STYLE if flux else None
        self.flavor = not flux
        self.caps = ImageCaps(sizes=list(SD_SIZES), seed=True, negative=not flux)
        try:
            models = self._get("/sdapi/v1/sd-models", timeout=10)
            current = self._get("/sdapi/v1/options", timeout=10).get("sd_model_checkpoint")
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise ArtError(f"the WebUI API is not reachable at {self.url} ({e}). Start "
                           "Automatic1111/Forge with --api, or set the profile's url.") from None
        self.checkpoint = self._resolve(settings.get("checkpoint") or settings.get("model")
                                        or current, models)
        self.model = self.checkpoint

    def _resolve(self, wanted: str | None, models: list) -> str:
        if not wanted:
            raise ArtError("the WebUI reports no loaded checkpoint; set the profile's checkpoint")
        key = wanted.lower().replace("\\", "/")
        for m in models:
            names = [m.get("title", ""), m.get("model_name", ""), m.get("filename", "")]
            for n in names:
                n = n.lower().replace("\\", "/")
                if n and (n == key or n.rsplit("/", 1)[-1] == key.rsplit("/", 1)[-1]
                          or n.split(" [", 1)[0] == key.split(" [", 1)[0]):
                    return m.get("title") or wanted
        raise ArtError(f"checkpoint '{wanted}' not found by the WebUI. Available: "
                       + ", ".join(m.get("title", "?") for m in models))

    def _get(self, path: str, timeout: float):
        with urllib.request.urlopen(self.url + path, timeout=timeout) as r:
            return json.loads(r.read())

    def spec_fields(self) -> dict:
        fields = {"checkpoint": self.checkpoint, "steps": self.steps, "cfg_scale": self.cfg_scale,
                  "sampler": self.sampler}
        if self.scheduler:
            fields["scheduler"] = self.scheduler
        return fields

    def generate(self, spec: dict) -> Image.Image:
        body = {"prompt": spec["positive"], "negative_prompt": spec.get("negative", ""),
                "seed": spec["seed"], "width": spec["width"], "height": spec["height"],
                "steps": spec.get("steps", self.steps),
                "cfg_scale": spec.get("cfg_scale", self.cfg_scale),
                "sampler_name": spec.get("sampler", self.sampler), "batch_size": 1, "n_iter": 1,
                "override_settings": {"sd_model_checkpoint": spec.get("checkpoint",
                                                                      self.checkpoint)},
                "override_settings_restore_afterwards": False}
        if spec.get("scheduler", self.scheduler):
            body["scheduler"] = spec.get("scheduler", self.scheduler)
        req = urllib.request.Request(self.url + "/sdapi/v1/txt2img",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                images = json.loads(r.read()).get("images") or []
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            raise ArtError(f"the WebUI rejected the request ({e.code}): {detail}") from None
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise ArtError(f"the WebUI request failed: {e}") from None
        if not images:
            raise ArtError("the WebUI returned no image")
        data = images[0].split(",", 1)[-1]  # some builds prefix a data: URL header
        img = Image.open(io.BytesIO(base64.b64decode(data)))
        img.load()
        return img
