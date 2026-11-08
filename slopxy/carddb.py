"""Card database: builds a slim, cached index from a Scryfall bulk JSONL export."""

from __future__ import annotations

import difflib
import json
import os
import pickle
import time
import unicodedata
from pathlib import Path

INDEX_VERSION = 2

# Oracle-level fields kept for the card and for each face.
CARD_FIELDS = (
    "name", "layout", "mana_cost", "type_line", "oracle_text", "power",
    "toughness", "loyalty", "defense", "colors", "color_identity",
    "produced_mana", "hand_modifier", "life_modifier",
)
FACE_FIELDS = (
    "name", "mana_cost", "type_line", "oracle_text", "power", "toughness",
    "loyalty", "defense", "colors",
)
TOKEN_LAYOUTS = {"token", "double_faced_token", "emblem"}
SKIP_LAYOUTS = {"art_series"}
SKIP_SET_TYPES = {"memorabilia"}


def normalize(name: str) -> str:
    """Lookup key: lowercase, ASCII-folded, unified apostrophes and spacing."""
    name = name.replace("’", "'").replace("‘", "'")
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return " ".join(name.lower().split())


def _slim_card(raw: dict) -> dict:
    card = {k: raw[k] for k in CARD_FIELDS if k in raw}
    if "card_faces" in raw:
        card["faces"] = [
            {k: f[k] for k in FACE_FIELDS if k in f} for f in raw["card_faces"]
        ]
    return card


def _slim_printing(raw: dict) -> dict:
    p = {
        "set": raw.get("set", "").upper(),
        "set_name": raw.get("set_name", ""),
        "collector_number": raw.get("collector_number", ""),
        "rarity": raw.get("rarity", ""),
        "artist": raw.get("artist", ""),
        "flavor_text": raw.get("flavor_text"),
        "released_at": raw.get("released_at", ""),
    }
    if "card_faces" in raw:
        p["face_flavor"] = [f.get("flavor_text") for f in raw["card_faces"]]
        p["face_artist"] = [f.get("artist") for f in raw["card_faces"]]
    return p


class CardDB:
    def __init__(self, cards: dict, keys: dict):
        # cards: name -> {"card": {...}, "latest": printing, "sets": {SET: printing}, "token": bool}
        # keys: normalized name/face name -> card name
        self.cards = cards
        self.keys = keys

    # ---------- building / loading ----------
    @classmethod
    def load(cls, data_path: Path, rebuild: bool = False, log=print) -> "CardDB":
        data_path = Path(data_path)
        cache = data_path.parent / ".cache" / "card_index.pkl"
        st = data_path.stat()
        stamp = (INDEX_VERSION, data_path.name, st.st_size, int(st.st_mtime))
        if cache.exists() and not rebuild:
            try:
                with open(cache, "rb") as fh:
                    saved = pickle.load(fh)
                if saved.get("stamp") == stamp:
                    return cls(saved["cards"], saved["keys"])
            except Exception:
                pass
        log(f"Building card index from {data_path.name} (one-time, may take a minute)...")
        t0 = time.time()
        db = cls._build(data_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as fh:
            pickle.dump({"stamp": stamp, "cards": db.cards, "keys": db.keys}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        log(f"Indexed {len(db.cards):,} cards in {time.time() - t0:.1f}s -> {cache}")
        return db

    @classmethod
    def _build(cls, data_path: Path) -> "CardDB":
        cards: dict[str, dict] = {}
        with open(data_path, "r", encoding="utf-8") as fh:
            for line in fh:
                # Cheap prefilter before parsing JSON.
                if '"lang":"en"' not in line or '"digital":false' not in line:
                    continue
                line = line.strip().rstrip(",")
                if not line.startswith("{"):
                    continue
                raw = json.loads(line)
                if "paper" not in raw.get("games", ()) or raw.get("oversized"):
                    continue
                layout = raw.get("layout", "")
                if layout in SKIP_LAYOUTS or raw.get("set_type") in SKIP_SET_TYPES:
                    continue
                is_token = layout in TOKEN_LAYOUTS or raw.get("set_type") == "token"
                name = raw["name"]
                entry = cards.get(name)
                if entry is not None and entry["token"] and not is_token:
                    entry = None  # a real card replaces a same-named token
                elif entry is not None and is_token and not entry["token"]:
                    continue
                printing = _slim_printing(raw)
                if entry is None:
                    entry = cards[name] = {
                        "card": _slim_card(raw), "latest": printing,
                        "sets": {}, "token": is_token,
                    }
                elif printing["released_at"] > entry["latest"]["released_at"]:
                    entry["latest"] = printing
                    entry["card"] = _slim_card(raw)
                prev = entry["sets"].get(printing["set"])
                if prev is None or printing["released_at"] >= prev["released_at"]:
                    entry["sets"][printing["set"]] = printing

        keys: dict[str, str] = {}
        # Face names first so that full names (and real cards) win collisions.
        for name, entry in cards.items():
            for face in entry["card"].get("faces", ()):
                keys.setdefault(normalize(face["name"]), name)
        for name, entry in cards.items():
            if not entry["token"]:
                keys[normalize(name)] = name
        for name, entry in cards.items():
            keys.setdefault(normalize(name), name)
        return cls(cards, keys)

    # ---------- lookup ----------
    def find(self, query: str, set_code: str | None = None):
        """Return (card, printing) or raise LookupError with suggestions."""
        key = normalize(query)
        name = self.keys.get(key)
        if name is None:
            close = difflib.get_close_matches(key, self.keys.keys(), n=5, cutoff=0.75)
            suggestions = []
            for k in close:
                if self.keys[k] not in suggestions:
                    suggestions.append(self.keys[k])
            raise LookupError(suggestions)
        entry = self.cards[name]
        printing = entry["latest"]
        if set_code:
            printing = entry["sets"].get(set_code.upper())
            if printing is None:
                raise LookupError(
                    [], f"no English paper printing in set {set_code.upper()} "
                        f"(available: {', '.join(sorted(entry['sets']))})"
                )
        return entry["card"], printing


def find_data_file(start: Path) -> Path | None:
    candidates = sorted(Path(start).glob("default-cards-*.json*"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(Path(start).glob("*.jsonl"), key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None
