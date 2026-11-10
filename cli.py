"""Command line interface: card names in, proxy PNGs out."""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .art import (FIT_MODES, ArtError, compose, load_image_art, load_prompt_file, new_seed,
                  variation_hints)
from .art.backends import ArtUnavailable
from .art.config import Config, load_config
from .art.session import ArtSession, confirm_paid, list_backends, ordered, say
from .carddb import CardDB, find_data_file
from .render import read_meta, render_card, save
from .sheets import (PAPER_SIZES, SheetOptions, compute_layout, iter_pages, page_count,
                     prepare_card, save_pdf)

SECTION_HEADERS = {"deck", "sideboard", "commander", "companion", "maybeboard", "mainboard"}
QTY_RE = re.compile(r"^\s*(\d+)\s*x?\s+", re.IGNORECASE)
ARENA_RE = re.compile(r"\s+\(([A-Za-z0-9]{2,6})\)(?:\s+\S+)?\s*$")
FULL_ART_RE = re.compile(r"\s*\[\s*full[\s_-]?(?:art)?\s*\]", re.IGNORECASE)
ART_PROMPT_RE = re.compile(r"\s*\[\s*art\s*:\s*(.*?)\s*\]", re.IGNORECASE)
IMAGE_RE = re.compile(r"\s*\[\s*image\s*:\s*(.*?)\s*\]", re.IGNORECASE)
RUNS_RE = re.compile(r"\s*\[\s*runs?\s*:\s*(\d+)\s*\]", re.IGNORECASE)
META_VERSION = 1


@dataclass
class Entry:
    name: str
    set_code: str | None = None
    full_art: bool = False
    art_prompt: str | None = None
    qty: int = 1
    image: str | None = None      # user art file ([image: ...])
    fit: str | None = None        # crop | fit for the user art
    png: Path | None = None       # output PNG to reproduce
    runs: int | None = None       # art generations to make ([runs: N]; else --runs)
    repro: dict | None = field(default=None, repr=False)

    def key(self):
        return (self.name.lower(), self.set_code, self.full_art, self.art_prompt,
                self.image, self.fit, str(self.png) if self.png else None, self.runs)


def _take(pattern: re.Pattern, line: str) -> tuple[str | None, str]:
    m = pattern.search(line)
    if not m:
        return None, line
    return m.group(1), line[: m.start()] + line[m.end():]


def parse_entry(line: str) -> Entry | None:
    """Parse one list line. Returns None for blanks/comments/section headers.
    Tags: `[full]` full-art, `[art: prompt]` custom prompt, `[image: path | fit]` your own art,
    `[runs: N]` N separate art generations.
    A path to a slopxy output PNG reproduces that image."""
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("//"):
        return None
    if line.lower().rstrip(":") in SECTION_HEADERS:
        return None
    art_prompt, line = _take(ART_PROMPT_RE, line)
    image, line = _take(IMAGE_RE, line)
    runs, line = _take(RUNS_RE, line)
    runs = max(1, int(runs)) if runs else None
    fit = None
    if image and "|" in image:
        image, mode = (s.strip() for s in image.rsplit("|", 1))
        if mode.lower() in FIT_MODES:
            fit = mode.lower()
    image = image.strip().strip('"') if image else None
    line, n_tags = FULL_ART_RE.subn("", line)
    qty = 1
    m = QTY_RE.match(line)
    if m:
        qty = max(1, int(m.group(1)))
        line = line[m.end():]
    bare = line.strip().strip('"')
    if bare.lower().endswith(".png"):
        return Entry(bare, qty=qty, png=Path(bare), runs=runs)
    set_code = None
    if "|" in line:
        line, set_code = (s.strip() for s in line.split("|", 1))
    else:
        m = ARENA_RE.search(line)
        if m:
            set_code = m.group(1)
            line = line[: m.start()]
    line = line.strip()
    if not line:
        return None
    return Entry(line, set_code or None, n_tags > 0, art_prompt or None, qty, image or None, fit,
                 runs=runs)


def safe_filename(name: str) -> str:
    name = name.replace(" // ", " - ")
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name)
    return name.strip(" .") or "card"


def _interactive() -> list[str]:
    print("Enter card names (optionally Name|SET, [full] for full-art, [art: prompt] for a "
          "custom art prompt, [image: file] for your own art) or slopxy PNGs to reproduce, "
          "one per line. Blank line to finish.")
    lines = []
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    return lines


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(argv)
    except KeyboardInterrupt:
        say("Interrupted.", err=True)
        if threading.active_count() > 1:  # requests still in flight: don't wait for them
            os._exit(130)
        return 130


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="slopxy", description="Generate MTG proxy card images.")
    ap.add_argument("cards", nargs="*",
                    help='card names, e.g. "Lightning Bolt", "Lightning Bolt|M10", "Sol Ring [full]", '
                         'or a slopxy output PNG to reproduce')
    ap.add_argument("-f", "--file", help="text file with one card per line (decklist formats accepted)")
    ap.add_argument("-o", "--out", default="output", help="output directory (default: ./output)")
    ap.add_argument("--data", help="path to Scryfall bulk JSONL (default: newest default-cards-*.jsonl here)")
    ap.add_argument("--full-art", action="store_true",
                    help="render every card full-art (only the title bar over the art)")
    ap.add_argument("--art-fit", choices=FIT_MODES, default="crop",
                    help="how [image: file] art fills the window when its shape differs: crop "
                         "(fill, trim edges; default) or fit (whole image, blurred fill)")
    g = ap.add_argument_group("AI art (a local generator or a hosted API; see --list-backends)")
    g.add_argument("--art", action="store_true", help="generate art for the art window")
    g.add_argument("--art-backend", metavar="NAME",
                   help="image backend or config profile: comfyui, a1111, openai, gemini, "
                        "placeholder, or a [profiles.NAME] from slopxy.toml (default: the "
                        "config's [art] image)")
    g.add_argument("--art-model", metavar="MODEL",
                   help="model for the image backend (comfyui: a preset such as krea2-turbo, "
                        "sdxl-turbo, flux; openai/gemini: a model id)")
    g.add_argument("--checkpoint", help="model file for comfyui (same family as the preset) or "
                                        "a1111")
    g.add_argument("--art-url", metavar="URL", help="address of a local backend (comfyui, a1111)")
    g.add_argument("--comfy-url", metavar="URL",
                   help="ComfyUI's address; also picks comfyui when no backend is chosen")
    g.add_argument("--describe-backend", metavar="NAME",
                   help="who writes scene descriptions: anthropic, openai, gemini, comfyui, or "
                        "a profile (default: the config's [art] describe; comfyui describes "
                        "for itself)")
    g.add_argument("--describe-model", metavar="MODEL", help="model for --describe-backend")
    g.add_argument("--config", metavar="FILE",
                   help="profiles file (default: ./slopxy.toml, then the user config file)")
    g.add_argument("--list-backends", action="store_true",
                   help="show the backends, what each needs, and the configured profiles")
    g.add_argument("--yes", action="store_true",
                   help="don't ask before making many paid requests")
    g.add_argument("--keep-art", action="store_true",
                   help="also save each generated artwork as '<image>.art.png', so the PNG "
                        "can be reproduced exactly even with backends that have no seeds")
    g.add_argument("--prompts", help="file of `Card Name = prompt` lines overriding auto prompts")
    g.add_argument("--no-describe", action="store_true",
                   help="skip the language-model scene description; use the template prompt")
    g.add_argument("--seed", type=int, help="fixed seed (default: random each run)")
    g.add_argument("--runs", type=int, default=1, metavar="N",
                   help="generate art N times per card, one image each (per card: [runs: N])")
    g.add_argument("--redescribe", action="store_true",
                   help="when reproducing a PNG, write a new scene description")
    g.add_argument("--new-seed", action="store_true",
                   help="when reproducing a PNG, use a new random seed")
    sg = ap.add_argument_group("Printable sheets")
    sg.add_argument("--sheet", action="store_true",
                    help="write a print-ready PDF of all cards (quantities honored) instead of "
                         "individual images")
    sg.add_argument("--sheet-name", metavar="FILE.pdf",
                    help="PDF file name (default: <list file>.pdf or proxies.pdf in --out)")
    sg.add_argument("--paper", default="letter",
                    help=f"{', '.join(PAPER_SIZES)}, or custom like 210x297mm / 8.5x11in "
                         "(default: letter)")
    sg.add_argument("--orientation", choices=["auto", "portrait", "landscape"], default="auto",
                    help="page orientation; auto picks whichever fits more cards (default)")
    sg.add_argument("--margin", type=float, default=0.2, help="page margin in inches (default 0.2)")
    sg.add_argument("--gap", type=float, default=1.0, help="space between cards in mm (default 1)")
    sg.add_argument("--cut-lines", choices=["lines", "marks", "none"], default="lines",
                    help="thin outlines around cards, crop marks in the margin, or none (default lines)")
    sg.add_argument("--card-scale", type=float, default=100.0,
                    help="card size in percent of 2.5x3.5in (default 100)")
    sg.add_argument("--also-images", action="store_true",
                    help="with --sheet, also write the individual card PNGs")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing output files instead of writing 'Name (1).png' copies")
    ap.add_argument("--rebuild-index", action="store_true", help="force rebuilding the card index cache")
    args = ap.parse_args(argv)
    if args.runs < 1:
        ap.error("--runs must be at least 1")
    if args.list_backends:
        try:
            print(list_backends(load_config(args.config)))
        except ArtError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2
        return 0

    data = Path(args.data) if args.data else find_data_file(Path.cwd())
    if not data or not data.exists():
        print("Card data file not found. Pass --data path/to/default-cards.jsonl", file=sys.stderr)
        return 2
    db = CardDB.load(data, rebuild=args.rebuild_index)

    lines = list(args.cards)
    if args.file:
        lines += Path(args.file).read_text(encoding="utf-8").splitlines()
    if not lines and not args.rebuild_index:
        lines = _interactive()

    missing = []
    merged: dict[tuple, Entry] = {}
    for line in lines:
        e = parse_entry(line)
        if not e:
            continue
        if e.png:
            if not _load_repro(e, missing):
                continue
        else:
            e.full_art = e.full_art or args.full_art
        if e.key() in merged:  # merge duplicate lines, adding up quantities
            merged[e.key()].qty += e.qty
        else:
            merged[e.key()] = e
    entries = list(merged.values())
    if any((e.runs or args.runs) > 1 for e in entries) and not (
            args.art or any(e.repro for e in entries)):
        print("Note: runs only apply to generated art (--art or a reproduced PNG); "
              "making one image per card.", file=sys.stderr)
    if not entries:
        _report_missing(missing)
        return 1 if missing else 0

    sheet_opts = sheet_lay = None
    if args.sheet or args.sheet_name:
        sheet_opts = SheetOptions(paper=args.paper, orientation=args.orientation,
                                  margin_in=args.margin, gap_mm=args.gap,
                                  cut_lines=args.cut_lines, card_scale=args.card_scale)
        try:
            sheet_lay = compute_layout(sheet_opts)  # validate paper/margins before any work
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2

    config = Config()
    if args.art or any(e.repro for e in entries):  # only art needs the profiles file
        try:
            config = load_config(args.config)
        except ArtError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2
    session = ArtSession(args, config)
    prompts = load_prompt_file(args.prompts) if args.prompts else {}
    if args.art:
        try:
            session.main()  # connect / check the model and key before doing any work
        except ArtError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 3

    # 1. Look up the cards and decide where each art window's art comes from.
    jobs = []
    for e in entries:
        try:
            card, printing = db.find(e.name, e.set_code)
        except LookupError as err:
            missing.append((e.name, err.args[0] if err.args else [],
                            err.args[1] if len(err.args) > 1 else ""))
            continue
        try:
            jobs += _jobs(e, card, printing, session, args, prompts)
        except ArtUnavailable as err:  # e.g. the backend of a reproduced PNG isn't reachable
            print(f"Error: {err}", file=sys.stderr)
            return 3

    # 2. Plan every generation (scene descriptions first, all of them), after asking if
    #    that means many paid requests.
    windows = [(j, w) for j in jobs for w in j.windows or ()]
    paid_images = [w.gen.image for _, w in windows if w.gen and w.gen.image.paid]
    paid_texts = [w.gen.text for _, w in windows
                  if w.gen and w.will_describe(args) and w.gen.text.paid]
    if not confirm_paid(paid_images, paid_texts, args.yes):
        return 4
    planning = [(j, w) for j, w in windows if w.kind in ("new", "repro")]
    for (j, w), result in ordered(planning, lambda jw: _plan(*jw, prompts, args),
                                  session.workers("text")):
        if isinstance(result, Exception):
            w.error = result
        else:
            w.spec = result

    # 3. Generate (in parallel where the backend allows), compose and write, in order.
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_images = sheet_opts is None or args.also_images
    written, art_failed, kept_paths = [], [], []
    sheet_cards: list = []
    renders: dict = {}
    for j, arts in ordered(jobs, lambda j: _generate(j, session),
                           session.workers("image")):
        for stem, img, meta, kept in _compose_job(j, arts, renders, art_failed, args):
            if write_images:
                target = out / f"{stem}.png"  # never overwrite a file from this same run
                arts = [k for k, _ in kept]  # the card's name must be free for its art too
                path = _write(target, lambda p: _save_card(img, p, meta, kept),
                              args.overwrite and target not in written,
                              related=lambda p: [_art_path(p, k) for k in arts])
                written.append(path)
                say(f"  wrote {path}  [{j.printing['set']} #{j.printing['collector_number']}]")
            elif kept:
                for key, art in kept:  # no card PNG to sit next to: just the art
                    target = out / f"{stem}.art{key}.png"
                    path = _write(target, art.save, args.overwrite and target not in kept_paths)
                    kept_paths.append(path)
                    say(f"  kept art {path}")
            if sheet_opts:
                # Keep only the flattened, sheet-sized copy; the full RGBA image can go.
                sheet_cards += [prepare_card(img, sheet_lay, sheet_opts)] * j.entry.qty
                if not write_images:
                    say(f"  {j.entry.qty} x {stem}")

    if written:
        print(f"\n{len(written)} image(s) written to {out.resolve()}")
    if sheet_opts and sheet_cards:
        if args.sheet_name:
            pdf = Path(args.sheet_name)
            if pdf.suffix.lower() != ".pdf":
                pdf = pdf.with_name(pdf.name + ".pdf")
            if not pdf.is_absolute() and pdf.parent == Path("."):
                pdf = out / pdf
        else:
            pdf = out / f"{Path(args.file).stem if args.file else 'proxies'}.pdf"
        lay = sheet_lay
        n_pages = page_count(len(sheet_cards), lay)
        pdf = _write(pdf, lambda p: save_pdf(iter_pages(sheet_cards, sheet_opts, lay), p),
                     args.overwrite)
        print(f"\n{len(sheet_cards)} card(s) on {n_pages} page(s), {lay.cols}x{lay.rows} per page, "
              f"{sheet_opts.paper} {lay.orientation} -> {pdf.resolve()}")
        print("Print at 100% / actual size (not 'fit to page').")
    for stem, err in art_failed:
        print(f"  NO ART: {stem} ({err})", file=sys.stderr)
    _report_missing(missing)
    return 1 if missing or art_failed else 0


def _report_missing(missing):
    for name, sugg, msg in missing:
        hint = f" {msg}" if msg else f" Did you mean: {', '.join(sugg)}?" if sugg else ""
        print(f"  NOT FOUND: {name}.{hint}", file=sys.stderr)


def _load_repro(e: Entry, missing: list) -> bool:
    """Fill an entry from the metadata of the PNG it names."""
    if not e.png.exists():
        missing.append((str(e.png), [], "file does not exist"))
        return False
    try:
        meta = read_meta(e.png)
    except (OSError, ValueError) as err:
        missing.append((str(e.png), [], f"could not read it ({err})"))
        return False
    if not meta or "card" not in meta:
        missing.append((str(e.png), [], "it has no slopxy metadata to reproduce from"))
        return False
    e.name, e.set_code = meta["card"], meta.get("set")
    e.full_art, e.art_prompt, e.repro = bool(meta.get("full_art")), meta.get("custom_prompt"), meta
    return True


def _free_path(path: Path, taken=None) -> Path:
    """First of 'Name (1).ext', 'Name (2).ext', ... that isn't taken (doesn't exist yet)."""
    taken = taken or (lambda p: p.exists())
    n = 1
    while True:
        candidate = path.with_name(f"{path.stem} ({n}){path.suffix}")
        if not taken(candidate):
            return candidate
        n += 1


def _write(path: Path, writer, overwrite: bool, related=None) -> Path:
    """Write via writer(path) without clobbering existing files (unless overwrite); related(p)
    lists files that writing p also creates, which must be free too.
    If the target is locked (e.g. open in a viewer), fall back to a numbered copy."""
    def taken(p: Path) -> bool:
        return p.exists() or any(q.exists() for q in (related(p) if related else ()))

    if taken(path) and not overwrite:
        path = _free_path(path, taken)
    try:
        writer(path)
    except PermissionError:
        alt = _free_path(path, taken)
        print(f"  {path.name} is in use; writing {alt.name} instead", file=sys.stderr)
        writer(alt)
        path = alt
    return path


@dataclass
class Window:
    """One art window of one image to write, and where its art comes from."""
    rect: tuple
    face: object
    key: str
    label: str
    kind: str = "none"  # none | image (a file) | kept (--keep-art file) | new | repro | failed
    spec: dict | None = None
    gen: object = None       # ArtGenerator for new / repro
    override: str | None = None   # custom prompt for new art
    hint: str | None = None       # per-run direction for the scene description
    error: Exception | None = None

    def will_describe(self, args) -> bool:
        if self.gen is None or self.gen.text is None:
            return False
        return (self.kind == "new" and self.gen.describe and not self.override) or (
            self.kind == "repro" and args.redescribe)


@dataclass
class Job:
    """One image to write (one face, one run), or for cards with no art to plan, all of a
    card's faces (windows is None)."""
    entry: Entry
    card: dict
    printing: dict
    suffix: str | None = None
    stem: str = ""
    run: int = 0
    runs: bool = False       # one of several runs
    windows: list | None = None


def _jobs(e: Entry, card, printing, session: ArtSession, args, prompts=None) -> list[Job]:
    """The images an entry turns into, with each art window's source decided (no requests
    made yet, apart from connecting to the backends involved)."""
    if not ((args.art and not e.image) or e.repro is not None):
        return [Job(e, card, printing)]
    prompts = prompts or {}
    tag = " (full art)" if e.full_art else ""
    generated = args.art or any(a.get("source") == "generated"
                                for a in (e.repro or {}).get("art", []))
    n_runs = (e.runs or args.runs) if generated and not e.image else 1
    only = e.repro.get("suffix", "") if e.repro is not None else None
    jobs = []
    for r in render_card(card, printing, full_art=e.full_art, only=only or None):
        if e.repro is not None and r.suffix != e.repro.get("suffix", ""):
            continue  # reproducing one face of a double-faced card
        stem = f"{safe_filename(card['name'])}{tag}{r.suffix}"
        face = r.windows[0].face if r.windows else None
        hints = variation_hints(n_runs, face) if n_runs > 1 else None
        for run in range(n_runs):
            windows = [_decide(e, w, stem + w.key + (f" run {run + 1}" if hints else ""),
                               card, session, args, prompts, hints[run] if hints else None)
                       for w in r.windows]
            jobs.append(Job(e, card, printing, r.suffix, stem, run, bool(hints), windows))
    return jobs


def _decide(e: Entry, w, label, card, session: ArtSession, args, prompts, hint) -> Window:
    """Where one window's art comes from: user image, kept art, reproduced spec, new art."""
    win = Window(w.rect, w.face, w.key, label, hint=hint)
    if e.image:
        win.kind, win.spec = "image", {"source": "image", "path": str(Path(e.image).resolve()),
                                       "fit": e.fit or args.art_fit}
        return win
    if e.repro is not None:
        saved = next((a for a in e.repro.get("art", []) if a.get("window") == w.key), None)
        if saved is None:
            return win
        spec = {k: v for k, v in saved.items() if k not in ("window", "face")}
        if spec.get("source") != "generated":
            win.kind, win.spec = "image", spec
            return win
        kept = spec.pop("kept_art", None)
        backend = spec.get("backend", "comfyui")
        if args.seed is not None and spec.get("seed") is None:
            session.note(f"seed:{backend}", f"  --seed doesn't apply to {backend}: it takes no "
                                            "seed")
        changing = (args.redescribe or args.new_seed or hint is not None
                    or (args.seed is not None and spec.get("seed") is not None))
        if kept and not changing:
            path = e.png.parent / kept
            if path.exists():
                spec["_art_path"] = str(path)
                win.kind, win.spec = "kept", spec
                return win
            session.note(f"kept:{path}", f"  kept art {path} is missing; generating it again")
        try:
            gen = session.for_spec(spec)
        except ArtUnavailable:
            raise
        except ArtError as err:  # e.g. its model file is gone: this card only, as before
            win.kind, win.spec, win.error = "failed", spec, err
            return win
        if args.redescribe and gen.text is None:
            raise ArtUnavailable("--redescribe needs a scene description writer: pass "
                                 "--describe-backend (anthropic, openai, gemini, comfyui, ...) "
                                 "or set describe under [art] in slopxy.toml")
        win.kind, win.spec, win.gen = "repro", spec, gen
        if spec.get("seed") is None and not kept and not (
                args.redescribe or args.new_seed or hint is not None):
            session.note(f"noseed:{backend}", f"  {backend} takes no seed, so reproducing makes "
                                              "a new image from the same prompt (--keep-art "
                                              "keeps art for exact copies)")
        return win
    if args.art:
        win.kind, win.gen = "new", session.main()
        if args.seed is not None and not win.gen.image.caps.seed:
            session.note(f"seed:{win.gen.image.name}",
                         f"  --seed doesn't apply to {win.gen.image.name}: it takes no seed")
        win.override = (e.art_prompt or prompts.get(w.face.name.lower())
                        or prompts.get(card["name"].lower()))
    return win


def _plan(j: Job, w: Window, prompts, args):
    """The full spec for a new or reproduced window (writes scene descriptions). Returns the
    spec, or the exception that stopped it."""
    try:
        if w.kind == "repro":
            spec = dict(w.spec)
            if args.redescribe:
                spec = w.gen.restyle(spec, w.face, w.label, w.hint)
            if spec.get("seed") is not None:
                if args.seed is not None:
                    spec["seed"] = args.seed + j.run
                elif args.new_seed or (j.runs and not args.redescribe):
                    spec["seed"] = new_seed()
            return spec
        spec = w.gen.plan(w.face, w.rect, w.override, w.label, w.hint)
        if args.seed is not None and spec.get("seed") is not None:
            spec["seed"] = args.seed + j.run
        return spec
    except Exception as e:  # noqa: BLE001 - one window failing must not stop the run
        return e


def _generate(j: Job, session: ArtSession):
    """Art for each window of a job: (image, fit mode), None, or the exception."""
    if j.windows is None:
        return None
    results = []
    for w in j.windows:
        try:
            if w.error is not None:
                raise w.error
            if w.kind == "none" or w.spec is None:
                results.append(None)
            elif w.kind == "image":
                results.append((load_image_art(w.spec["path"]), w.spec.get("fit", "crop")))
            elif w.kind == "kept":
                results.append((load_image_art(w.spec["_art_path"]), "crop"))
            else:
                with session.limit(w.gen.image):
                    results.append((w.gen.generate(w.spec, w.label), "crop"))
        except Exception as e:  # noqa: BLE001
            results.append(e)
    return results


def _failure(e: Exception) -> str:
    return str(e) if isinstance(e, (ArtError, OSError)) else f"{type(e).__name__}: {e}"


def _compose_job(j: Job, arts, renders: dict, art_failed: list, args):
    """Yield (file stem, final image, metadata, art to keep) for each image of a job."""
    e = j.entry
    if j.windows is None:  # nothing planned: every face, with the entry's own image if any
        tag = " (full art)" if e.full_art else ""
        for r in render_card(j.card, j.printing, full_art=e.full_art):
            stem = f"{safe_filename(j.card['name'])}{tag}{r.suffix}"
            layers, art_meta = [], []
            for w in r.windows if e.image else ():
                spec = {"source": "image", "path": str(Path(e.image).resolve()),
                        "fit": e.fit or args.art_fit}
                try:
                    layers.append((load_image_art(spec["path"]), w.rect, spec["fit"]))
                    art_meta.append({"window": w.key, "face": w.face.name, **spec})
                except OSError as err:
                    art_failed.append((stem + w.key, str(err)))
                    say(f"  art failed for {stem + w.key}: {err}", err=True)
            img = compose(r.image, layers) if layers else r.image
            yield stem, img, _meta(e, j.card, j.printing, r.suffix, art_meta), []
        return
    key = (id(e), j.suffix)
    if key not in renders:  # the runs of one face share one render
        renders.clear()
        renders[key] = next(r for r in render_card(j.card, j.printing, full_art=e.full_art,
                                                    only=j.suffix or None)
                            if r.suffix == j.suffix)
    r = renders[key]
    layers, art_meta, kept = [], [], []
    for w, res in zip(j.windows, arts):
        if isinstance(res, Exception):
            art_failed.append((w.label, _failure(res)))
            say(f"  art failed for {w.label}: {_failure(res)}", err=True)
            continue
        if res is None:
            continue
        art, mode = res
        layers.append((art, w.rect, mode))
        entry = {"window": w.key, "face": w.face.name,
                 **{k: v for k, v in w.spec.items() if not k.startswith("_")}}
        if w.kind == "kept" and not args.keep_art:
            entry["kept_art"] = str(Path(w.spec["_art_path"]).resolve())
        if args.keep_art and w.kind in ("new", "repro", "kept"):
            kept.append((w.key, art))
        art_meta.append(entry)
    img = compose(r.image, layers) if layers else r.image
    yield j.stem, img, _meta(e, j.card, j.printing, r.suffix, art_meta), kept


def _art_path(card_png: Path, key: str) -> Path:
    """Where --keep-art puts a window's art: next to the card, as 'Name.art<window>.png'."""
    return card_png.with_name(f"{card_png.stem}.art{key}.png")


def _meta(e: Entry, card, printing, suffix, art_meta) -> dict:
    return {"version": META_VERSION, "card": card["name"], "set": printing["set"],
            "collector_number": printing["collector_number"], "full_art": e.full_art,
            "suffix": suffix, "custom_prompt": e.art_prompt, "art": art_meta}


def _save_card(img, path: Path, meta: dict, kept: list) -> None:
    """Save a card PNG; with kept art, also each artwork as '<name>.art<window>.png' next to
    it, recorded (relative to the PNG) in the metadata."""
    names = {key: _art_path(path, key).name for key, _ in kept}
    if names:
        meta = {**meta, "art": [{**a, "kept_art": names[a["window"]]}
                                if a["window"] in names else a for a in meta["art"]]}
    save(img, path, meta)
    for key, art in kept:
        (art.convert("RGB") if art.mode == "CMYK" else art).save(path.parent / names[key])
