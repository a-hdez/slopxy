"""Which backends a run uses: resolves profiles from the config and the command line, makes
each backend once, and limits how many requests go to each at a time."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

from .backends import BACKENDS, ArtError, image_backend, text_backend
from .config import Config
from .generator import ArtGenerator

# Backends whose own server also runs the description model, used for descriptions when no
# description backend is chosen (local, so nothing is charged without being asked for).
COMPANION_TEXT = {"comfyui"}


class ArtSession:
    def __init__(self, args, config: Config, log=print):
        self.args = args
        self.config = config
        self.log = log
        self._gens: dict[str, ArtGenerator] = {}
        self._texts: dict[str, object] = {}
        self._limits: dict[int, threading.Semaphore] = {}
        self._lock = threading.RLock()
        self._noted: set[str] = set()

    # ---------- profiles ----------
    def _image_settings(self, name: str, spec: dict | None = None) -> dict:
        settings = self.config.resolve(name)
        if spec is None:  # the run's own backend: command-line overrides apply
            for key, value in (("model", self.args.art_model),
                               ("checkpoint", self.args.checkpoint),
                               ("url", self.args.art_url)):
                if value:
                    settings[key] = value
        else:  # reproducing: the spec says what made the image
            settings["model"] = spec.get("model") or settings.get("model")
            if spec.get("checkpoint"):
                settings["checkpoint"] = spec["checkpoint"]
            if self.args.art_url:
                settings["url"] = self.args.art_url
        return settings

    def _text_settings(self, image_settings: dict) -> dict | None:
        if self.args.no_describe:
            return None
        name = self.args.describe_backend or self.config.describe
        if name:
            settings = self.config.resolve(name)
            if self.args.describe_model:
                settings["model"] = self.args.describe_model
            return settings
        if image_settings["backend"] in COMPANION_TEXT:
            return {"backend": image_settings["backend"], "url": image_settings.get("url"),
                    "profile": image_settings.get("profile")}
        self.note("describe", "  scene descriptions are off for this backend; using template "
                              "prompts (choose a writer with --describe-backend, e.g. "
                              "anthropic, openai or gemini)")
        return None

    def note(self, key: str, message: str) -> None:
        with self._lock:
            if key not in self._noted:
                self._noted.add(key)
                self.log(message)

    # ---------- generators ----------
    def main(self) -> ArtGenerator:
        """The generator for --art: --art-backend, else the config's [art] image profile."""
        name = self.args.art_backend or self.config.image
        if not name:
            raise ArtError("no image backend chosen. Pass --art-backend NAME (" +
                           ", ".join(n for n, b in BACKENDS.items() if b[1]) +
                           ", or a profile from your config), or set image = \"NAME\" under "
                           "[art] in slopxy.toml. See --list-backends.")
        return self._gen(self._image_settings(name), describe=True)

    def for_spec(self, spec: dict) -> ArtGenerator:
        """The generator that can redo a spec read from a PNG (no backend: made by ComfyUI).
        It only gets a description backend for --redescribe."""
        backend = spec.get("backend", "comfyui")
        profile = spec.get("profile")
        name = profile if profile in self.config.profiles and \
            self.config.profiles[profile].get("backend", profile) == backend else backend
        return self._gen(self._image_settings(name, spec), describe=self.args.redescribe)

    def _gen(self, settings: dict, describe: bool) -> ArtGenerator:
        key = json.dumps([settings, describe], sort_keys=True, default=str)
        with self._lock:
            if key not in self._gens:
                image = image_backend(settings)
                text_settings = self._text_settings(settings) if describe else None
                text = self._text(text_settings) if text_settings else None
                self._gens[key] = ArtGenerator(image, text, seed=self.args.seed,
                                               describe=text is not None, log=self.log,
                                               profile=settings.get("profile"))
            return self._gens[key]

    def _text(self, settings: dict):
        key = json.dumps(settings, sort_keys=True, default=str)
        if key not in self._texts:
            self._texts[key] = text_backend(settings)
        return self._texts[key]

    # ---------- concurrency ----------
    def limit(self, backend) -> threading.Semaphore:
        """At most backend.concurrency requests in flight to one backend."""
        with self._lock:
            if id(backend) not in self._limits:
                self._limits[id(backend)] = threading.Semaphore(max(1, backend.concurrency))
            return self._limits[id(backend)]

    def workers(self, kind: str) -> int:
        backends = ([g.image for g in self._gens.values()] if kind == "image" else
                    [g.text for g in self._gens.values() if g.text is not None])
        return max([b.concurrency for b in backends] or [1])


def ordered(items, fn, workers: int):
    """Yield (item, fn(item)) in input order, running up to `workers` calls at once and
    keeping at most 2*workers results ahead of the consumer. fn must not raise."""
    if workers <= 1:
        for item in items:
            yield item, fn(item)
        return
    it = iter(items)
    ex = ThreadPoolExecutor(max_workers=workers)
    pending = deque((item, ex.submit(fn, item)) for item in islice(it, workers * 2))
    try:
        while pending:
            item, fut = pending.popleft()
            result = fut.result()
            for nxt in islice(it, 1):
                pending.append((nxt, ex.submit(fn, nxt)))
            yield item, result
    finally:
        ex.shutdown(wait=True, cancel_futures=True)


CONFIRM_OVER = 20  # paid requests a run may make without asking


def confirm_paid(images: list, texts: list, yes: bool) -> bool:
    """Ask before a run makes many paid requests. images/texts: the paid backends, one entry
    per request. True to go ahead."""
    total = len(images) + len(texts)
    if total <= CONFIRM_OVER or yes:
        return True

    def describe(backends) -> str:
        counts: dict[str, int] = {}
        for b in backends:
            counts[f"{b.name} {b.model}"] = counts.get(f"{b.name} {b.model}", 0) + 1
        return ", ".join(f"{n} with {k}" for k, n in counts.items())

    parts = []
    if images:
        parts.append(f"{len(images)} paid image(s) ({describe(images)})")
    if texts:
        parts.append(f"{len(texts)} paid scene description(s) ({describe(texts)})")
    msg = "This run will request " + " and ".join(parts) + "."
    if not sys.stdin.isatty():
        print(f"{msg} Pass --yes to allow that.", file=sys.stderr)
        return False
    answer = input(f"{msg} Continue? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def list_backends(config: Config) -> str:
    """Human-readable overview for --list-backends."""
    sdk = {"openai": "openai", "gemini": "google.genai", "anthropic": "anthropic"}
    keys = {"openai": ["OPENAI_API_KEY"], "gemini": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"]}
    lines = ["Backends (use a name with --art-backend / --describe-backend, or in a profile):"]
    for name, (_, img, txt, needs) in BACKENDS.items():
        roles = "/".join(r for r, ok in (("images", img), ("descriptions", txt)) if ok)
        status = []
        if name in sdk:
            installed = importlib.util.find_spec(sdk[name].split(".")[0]) is not None and (
                "." not in sdk[name] or _has_module(sdk[name]))
            status.append("SDK installed" if installed else f"pip install slopxy[{name}]")
            env = next((k for k in keys[name] if os.environ.get(k)), None)
            status.append(f"{env} set" if env else f"no {keys[name][0]}")
        lines.append(f"  {name:12} {roles:21} {needs}" +
                     (f"  [{', '.join(status)}]" if status else ""))
    if config.path:
        lines.append(f"\nConfig: {config.path}")
        lines.append(f"  [art] image = {config.image!r}, describe = {config.describe!r}")
        for name, prof in config.profiles.items():
            shown = {k: v for k, v in prof.items()}
            lines.append(f"  profile {name}: {shown}")
    else:
        lines.append("\nNo config file (looked for --config, ./slopxy.toml and the user config).")
    return "\n".join(lines)


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False
