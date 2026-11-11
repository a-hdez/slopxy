"""Backend profiles: named settings for image and text backends, from a TOML file.

    [art]
    image = "local"          # profile --art uses unless --art-backend says otherwise
    describe = "claude"      # profile that writes scene descriptions

    [profiles.local]
    backend = "comfyui"
    url = "http://127.0.0.1:8189"
    model = "krea2-turbo"

    [profiles.claude]
    backend = "anthropic"

A backend's own name also works as a profile with its defaults (--art-backend openai).
API keys are never stored here: a profile may name the environment variable that holds its
key (api_key_env = "MY_KEY"); otherwise each vendor's usual variable is used.

The file is the first of: --config PATH, ./slopxy.toml, the user config file
(%APPDATA%\\slopxy\\config.toml on Windows, else $XDG_CONFIG_HOME or ~/.config/slopxy/config.toml).
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .backends import BACKENDS, ArtError

_SECRET_NAME = re.compile(r"(^|_)(api_?key|key|token|password|passwd|secret)$", re.I)
_SECRET_VALUE = re.compile(r"^(sk-|sk_|AIza|xai-|gsk_)")  # common API key prefixes
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _secret_settings(prof: dict) -> list[str]:
    """Setting names in a profile that look like they hold a secret."""
    return [k for k, v in prof.items()
            if k.lower() != "api_key_env" and (_SECRET_NAME.search(k) or (
                isinstance(v, str) and _SECRET_VALUE.match(v)))]


@dataclass
class Config:
    path: Path | None = None
    image: str | None = None      # default image profile
    describe: str | None = None   # default text profile
    profiles: dict = field(default_factory=dict)

    def resolve(self, name: str) -> dict:
        """Settings for a profile name, or for a backend name with its defaults."""
        if name in self.profiles:
            settings = dict(self.profiles[name])
            if "backend" not in settings and name not in BACKENDS:
                raise ArtError(f"profile '{name}' needs a backend = \"...\" line (one of "
                               f"{', '.join(BACKENDS)})")
            settings.setdefault("backend", name)
        elif name in BACKENDS:
            settings = {"backend": name}
        else:
            known = sorted(set(self.profiles) | set(BACKENDS))
            where = f" in {self.path}" if self.path else ""
            raise ArtError(f"no profile or backend named '{name}'{where} "
                           f"(known: {', '.join(known)})")
        if settings["backend"] not in BACKENDS:
            raise ArtError(f"profile '{name}' uses unknown backend '{settings['backend']}' "
                           f"(known: {', '.join(BACKENDS)})")
        settings["profile"] = name
        return settings


def user_config_path() -> Path:
    if os.name == "nt" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "slopxy" / "config.toml"
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "slopxy" / "config.toml"


def find_config(explicit: str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise ArtError(f"config file not found: {p}")
        return p
    for p in (Path("slopxy.toml"), user_config_path()):
        if p.exists():
            return p
    return None


def load_config(explicit: str | None = None) -> Config:
    path = find_config(explicit)
    if path is None:
        return Config()
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # Python 3.10: the tomli package (a dependency there)
        try:
            import tomli as tomllib
        except ImportError:
            raise ArtError(f"reading {path} on Python 3.10 needs tomli: "
                           "pip install tomli") from None
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ArtError(f"can't read {path}: {e}") from None
    art = data.get("art", {})
    profiles = data.get("profiles", {})
    if not isinstance(art, dict) or not isinstance(profiles, dict) or not all(
            isinstance(p, dict) for p in profiles.values()):
        raise ArtError(f"{path}: expected an [art] table and [profiles.NAME] tables")
    for name, prof in profiles.items():
        leaked = _secret_settings(prof)
        if leaked:
            raise ArtError(f"{path}: profile '{name}' seems to hold a key in "
                           f"{', '.join(sorted(leaked))}; keep keys out of the file: put the "
                           "key in an environment variable and set api_key_env to its name")
        env = prof.get("api_key_env")
        if env is not None and not (isinstance(env, str) and _ENV_NAME.match(env)
                                    and not _SECRET_VALUE.match(env)):
            raise ArtError(f"{path}: profile '{name}': api_key_env must be the name of an "
                           "environment variable (like MY_OPENAI_KEY), not the key itself")
    return Config(path, art.get("image"), art.get("describe"), profiles)
