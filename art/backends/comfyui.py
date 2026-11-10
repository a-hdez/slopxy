"""ComfyUI backend: a minimal HTTP API client (stdlib only), the model presets, and the image
and text backends that fill in the workflow templates in workflows/.

Profile settings: url (default http://127.0.0.1:8189), model (a preset name below, default
krea2-turbo), checkpoint (another model file of the preset's family)."""

from __future__ import annotations

import copy
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from PIL import Image

from . import PLAIN_STYLE, SD_SIZES, ArtError, ImageCaps

DEFAULT_URL = "http://127.0.0.1:8189"  # ComfyUI Desktop's default port
WORKFLOW_DIR = Path(__file__).parent / "workflows"

PRESETS = {
    # Krea 2 follows prompts literally, renders quoted text and ignores negatives (cfg 1),
    # so it gets plain-language style, no flavor quotes and no "card" wording.
    "krea2-turbo": {"workflow": "krea2_turbo.json",
                    "checkpoint": "krea2_turbo_int8_convrot.safetensors",
                    "style": PLAIN_STYLE, "flavor": False, "negative": False},
    "sdxl-turbo": {"workflow": "sdxl_turbo.json",
                   "checkpoint": "dreamshaperXL_v21TurboDPMSDE.safetensors"},
    "flux": {"workflow": "flux_dev.json", "checkpoint": "flux1-dev-fp8.safetensors"},
}
DEFAULT_MODEL = "krea2-turbo"

# Model loader node -> the input that names the model file (node "1" in every workflow).
LOADER_INPUT = {"CheckpointLoaderSimple": "ckpt_name", "UNETLoader": "unet_name"}


class ComfyError(ArtError):
    pass


class ComfyClient:
    def __init__(self, url: str = DEFAULT_URL, timeout: float = 600):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    def _get(self, path: str, timeout: float = 30) -> bytes:
        with urllib.request.urlopen(self.url + path, timeout=timeout) as r:
            return r.read()

    def _get_json(self, path: str, timeout: float = 30):
        return json.loads(self._get(path, timeout))

    def check(self) -> dict:
        try:
            return self._get_json("/system_stats", timeout=5)
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise ComfyError(
                f"ComfyUI is not reachable at {self.url} ({e}). "
                "Open ComfyUI Desktop and retry, or pass --comfy-url."
            ) from None

    def model_files(self, node_class: str, input_name: str) -> list[str]:
        """Files a loader node can choose from, e.g. ("UNETLoader", "unet_name")."""
        info = self._get_json(f"/object_info/{node_class}")
        return list(info[node_class]["input"]["required"][input_name][0])

    def run(self, workflow: dict) -> bytes:
        """Queue a workflow (API format), wait for it, return the first output image's bytes."""
        for out in self._execute(workflow).values():
            for img in out.get("images", []):
                q = urllib.parse.urlencode({"filename": img["filename"],
                                            "subfolder": img.get("subfolder", ""),
                                            "type": img.get("type", "output")})
                return self._get(f"/view?{q}", timeout=60)
        raise ComfyError("ComfyUI finished but produced no image.")

    def run_text(self, workflow: dict) -> str:
        """Queue a workflow ending in a text preview node (PreviewAny) and return its text."""
        for out in self._execute(workflow).values():
            for text in out.get("text", []):
                return str(text)
        raise ComfyError("ComfyUI finished but produced no text.")

    def _execute(self, workflow: dict) -> dict:
        """Queue a workflow and wait for it; return its outputs by node id."""
        body = json.dumps({"prompt": workflow, "client_id": self.client_id}).encode()
        req = urllib.request.Request(self.url + "/prompt", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                prompt_id = json.loads(r.read())["prompt_id"]
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                d = json.loads(detail)
                parts = [d.get("error", {}).get("message", "")]
                for node in (d.get("node_errors") or {}).values():
                    for err in node.get("errors", []):
                        parts.append(f"{node.get('class_type')}: {err.get('message')} {err.get('details', '')}")
                detail = "; ".join(p for p in parts if p)
            except ValueError:
                pass
            raise ComfyError(f"ComfyUI rejected the workflow: {detail}") from None

        deadline = time.time() + self.timeout
        while time.time() < deadline:
            hist = self._get_json(f"/history/{prompt_id}")
            entry = hist.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m[1].get("exception_message", "") for m in status.get("messages", [])
                            if m[0] == "execution_error"]
                    raise ComfyError("ComfyUI execution failed: " + ("; ".join(msgs) or "unknown error"))
                if entry.get("outputs") or status.get("completed"):
                    return entry.get("outputs", {})
            time.sleep(1.0)
        raise ComfyError(f"Timed out after {self.timeout:.0f}s waiting for ComfyUI.")


def _load_workflow(name: str) -> dict:
    return json.loads((WORKFLOW_DIR / name).read_text())


_CLIENTS: dict[str, ComfyClient] = {}


def _client(settings: dict) -> ComfyClient:
    """One checked client per ComfyUI address, shared by the image and text backends."""
    url = (settings.get("url") or DEFAULT_URL).rstrip("/")
    if url not in _CLIENTS:
        client = ComfyClient(url, timeout=float(settings.get("timeout", 600)))
        client.check()
        _CLIENTS[url] = client
    return _CLIENTS[url]


def image_backend(settings: dict) -> "ComfyImageBackend":
    return ComfyImageBackend(_client(settings), settings.get("model") or DEFAULT_MODEL,
                             settings.get("checkpoint"))


def text_backend(settings: dict) -> "ComfyTextBackend":
    return ComfyTextBackend(_client(settings))


class ComfyImageBackend:
    """Image generation with one model preset. The workflow's node ids are fixed: "1" loads
    the model, "2"/"3" are the positive/negative prompts, "4" the latent size, "5" the sampler."""

    name = "comfyui"
    concurrency = 1  # one GPU: ComfyUI runs its queue in order anyway
    paid = False

    def __init__(self, client: ComfyClient, model: str = DEFAULT_MODEL,
                 checkpoint: str | None = None):
        if model not in PRESETS:
            raise ComfyError(f"unknown ComfyUI preset '{model}' (presets: {', '.join(PRESETS)})")
        self.client = client
        self.model = model
        preset = PRESETS[model]
        self.style = preset.get("style")
        self.flavor = preset.get("flavor", True)
        self.caps = ImageCaps(sizes=list(SD_SIZES), seed=True,
                              negative=preset.get("negative", True))
        self.template = _load_workflow(preset["workflow"])
        self.loader = self.template["1"]["class_type"]
        self.loader_input = LOADER_INPUT[self.loader]
        self.checkpoint = self._resolve_checkpoint(checkpoint or preset["checkpoint"])

    def _resolve_checkpoint(self, wanted: str) -> str:
        available = self.client.model_files(self.loader, self.loader_input)
        key = wanted.lower().replace("\\", "/")
        for name in available:
            n = name.lower().replace("\\", "/")
            if n == key or n.rsplit("/", 1)[-1] == key.rsplit("/", 1)[-1]:
                return name
        raise ComfyError(f"Model '{wanted}' not found by ComfyUI's {self.loader}. Available: "
                         + ", ".join(available))

    def spec_fields(self) -> dict:
        return {"checkpoint": self.checkpoint}

    def generate(self, spec: dict) -> Image.Image:
        wf = copy.deepcopy(self.template)
        wf["1"]["inputs"][self.loader_input] = spec["checkpoint"]
        wf["2"]["inputs"]["text"] = spec["positive"]
        if wf["3"]["class_type"] == "CLIPTextEncode":
            wf["3"]["inputs"]["text"] = spec.get("negative", "")
        wf["4"]["inputs"]["width"], wf["4"]["inputs"]["height"] = spec["width"], spec["height"]
        wf["5"]["inputs"]["seed"] = spec["seed"]
        img = Image.open(io.BytesIO(self.client.run(wf)))
        img.load()
        return img


class ComfyTextBackend:
    """The language model behind scene descriptions: Qwen3-VL through ComfyUI's Generate Text
    node (workflows/describe.json)."""

    name = "comfyui"
    concurrency = 1
    paid = False

    def __init__(self, client: ComfyClient):
        self.client = client
        self.template = _load_workflow("describe.json")
        self.model = self.template["1"]["inputs"]["clip_name"].rsplit(".", 1)[0]

    def complete(self, prompt: str, seed: int) -> str:
        wf = copy.deepcopy(self.template)
        wf["2"]["inputs"]["prompt"] = prompt
        wf["2"]["inputs"]["sampling_mode.seed"] = seed
        return self.client.run_text(wf)
