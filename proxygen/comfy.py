"""Minimal ComfyUI HTTP API client (stdlib only)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

DEFAULT_URL = "http://127.0.0.1:8189"  # ComfyUI Desktop's default port


class ComfyError(RuntimeError):
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
