"""Card database: builds a slim, cached index from a Scryfall bulk JSONL export."""

from __future__ import annotations

import difflib
import io
import json
import os
import pickle
import tempfile
import time
import unicodedata
from pathlib import Path

import numpy as np

INDEX_VERSION = 3  # 3: stamp ahead of the data; per-set printings in their own file

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
    def __init__(self, cards: dict, keys: dict, sets_loader=None):
        # cards: name -> {"card": {...}, "latest": printing, "sets": {SET: printing}, "token": bool}
        # keys: normalized name/face name -> card name
        # sets_loader: when the entries come without "sets" (loaded from the cache), a
        # function returning {name: {SET: printing}}, called the first time a set is needed.
        self.cards = cards
        self.keys = keys
        self._sets_loader = sets_loader
        self._chars = None  # per-key character counts for suggest(), built on first use

    # ---------- building / loading ----------
    # The cache is two files, each starting with a stamp (card_sets.pkl: the stamp and its
    # body size), so a stale or partial file is detected without loading the rest:
    # card_index.pkl (cards without their per-set printings, and the lookup keys) is
    # unpickled on every run; card_sets.pkl is read and checked on every run too, but only
    # unpickled for a Name|SET lookup. Holding its bytes means a later lookup never depends
    # on the cache files or the data file staying as they were.
    @classmethod
    def load(cls, data_path: Path, rebuild: bool = False, log=print) -> "CardDB":
        data_path = Path(data_path)
        index, sets_file = _cache_files(data_path)
        stamp = _stamp(data_path)
        if index.exists() and not rebuild:
            try:
                with open(index, "rb") as fh:
                    if pickle.load(fh) == stamp:
                        sets_buf = io.BytesIO(sets_file.read_bytes())
                        sets_stamp, size = pickle.load(sets_buf)
                        complete = size == len(sets_buf.getbuffer()) - sets_buf.tell()
                        if sets_stamp == stamp and complete:
                            saved = pickle.load(fh)
                            cards = saved["cards"]

                            def sets_loader():
                                try:
                                    return _link_sets(cards, pickle.load(sets_buf))
                                except Exception:  # damaged after all: rebuild
                                    return {n: e["sets"] for n, e in
                                            cls._rebuild(data_path, log).cards.items()}

                            return cls(cards, saved["keys"], sets_loader=sets_loader)
            except Exception:
                pass
        return cls._rebuild(data_path, log)

    @classmethod
    def _rebuild(cls, data_path: Path, log) -> "CardDB":
        log(f"Building card index from {data_path.name} (one-time, may take a minute)...")
        t0 = time.time()
        stamp = _stamp(data_path)
        db = cls._build(data_path)
        index, sets_file = _cache_files(data_path)
        index.parent.mkdir(parents=True, exist_ok=True)
        slim = {n: {k: v for k, v in e.items() if k != "sets"} for n, e in db.cards.items()}
        # The latest printing is usually also its set's entry; store it once, in the index,
        # and mark its place in the sets with None (see _link_sets).
        sets = {n: {code: None if p is e["latest"] else p for code, p in e["sets"].items()}
                for n, e in db.cards.items()}
        body = pickle.dumps(sets, protocol=pickle.HIGHEST_PROTOCOL)
        # Sets first: an index is never newer than the sets file next to it. Each file is
        # written to a temporary name and renamed, so a reader never sees a partial file.
        _write_atomic(sets_file, pickle.dumps((stamp, len(body)), protocol=pickle.HIGHEST_PROTOCOL)
                      + body)
        _write_atomic(index, pickle.dumps(stamp, protocol=pickle.HIGHEST_PROTOCOL)
                      + pickle.dumps({"cards": slim, "keys": db.keys},
                                     protocol=pickle.HIGHEST_PROTOCOL))
        log(f"Indexed {len(db.cards):,} cards in {time.time() - t0:.1f}s -> {index}")
        return db

    def _sets_of(self, name: str) -> dict:
        entry = self.cards[name]
        if "sets" not in entry and self._sets_loader is not None:
            loader, self._sets_loader = self._sets_loader, None
            for n, sets in loader().items():
                if n in self.cards:
                    self.cards[n]["sets"] = sets
        return entry.get("sets", {})

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
    def suggest(self, key: str, n: int = 5, cutoff: float = 0.75) -> list[str]:
        """Card names whose keys are close to `key` (difflib), best first.
        Same result as difflib.get_close_matches over every key, but keys that can't reach
        the cutoff are dropped first, all at once with numpy: get_close_matches keeps a key
        only if its quick_ratio, 2 * (characters in common, with multiplicity) / (total
        length), reaches the cutoff."""
        if self._chars is None:
            self._chars = _char_counts(list(self.keys))
        keys, lengths, alphabet, counts = self._chars
        codes = np.frombuffer(key.encode("utf-32-le", "surrogatepass"), dtype=np.uint32)
        pos = np.searchsorted(alphabet, codes)
        known = pos < len(alphabet)
        known[known] = alphabet[pos[known]] == codes[known]
        query = np.bincount(pos[known], minlength=len(alphabet))
        common = np.minimum(counts, query).sum(axis=1)
        total = lengths + len(key)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(total > 0, 2.0 * common / total, 1.0)
        pool = [keys[i] for i in np.flatnonzero(ratio >= cutoff)]
        suggestions = []
        for k in difflib.get_close_matches(key, pool, n=n, cutoff=cutoff):
            if self.keys[k] not in suggestions:
                suggestions.append(self.keys[k])
        return suggestions

    def find(self, query: str, set_code: str | None = None):
        """Return (card, printing) or raise LookupError with suggestions."""
        key = normalize(query)
        name = self.keys.get(key)
        if name is None:
            raise LookupError(self.suggest(key))
        entry = self.cards[name]
        printing = entry["latest"]
        if set_code:
            sets = self._sets_of(name)
            printing = sets.get(set_code.upper())
            if printing is None:
                raise LookupError(
                    [], f"no English paper printing in set {set_code.upper()} "
                        f"(available: {', '.join(sorted(sets))})"
                )
        return entry["card"], printing


def _write_atomic(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".pkl")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _link_sets(cards: dict, sets: dict) -> dict:
    """Put each card's latest printing back where card_sets.pkl marks it with None."""
    for name, by_set in sets.items():
        for code, printing in by_set.items():
            if printing is None:
                by_set[code] = cards[name]["latest"]
    return sets


def _cache_files(data_path: Path) -> tuple[Path, Path]:
    cache = data_path.parent / ".cache"
    return cache / "card_index.pkl", cache / "card_sets.pkl"


def _stamp(data_path: Path) -> tuple:
    st = data_path.stat()
    return (INDEX_VERSION, data_path.name, st.st_size, int(st.st_mtime))


def _char_counts(keys: list[str]):
    """(keys, key lengths, sorted alphabet of code points, keys x alphabet count matrix)."""
    lengths = np.fromiter(map(len, keys), dtype=np.int64, count=len(keys))
    codes = np.frombuffer("".join(keys).encode("utf-32-le", "surrogatepass"),
                          dtype=np.uint32)
    alphabet, idx = np.unique(codes, return_inverse=True)
    owner = np.repeat(np.arange(len(keys)), lengths)
    counts = np.bincount(owner * len(alphabet) + idx, minlength=len(keys) * len(alphabet))
    return keys, lengths, alphabet, counts.reshape(len(keys), len(alphabet)).astype(np.uint16)


def find_data_file(start: Path) -> Path | None:
    candidates = sorted(Path(start).glob("default-cards-*.json*"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(Path(start).glob("*.jsonl"), key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None
