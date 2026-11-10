"""Command line interface: card names in, proxy PNGs out."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .art import (FIT_MODES, ArtError, ArtGenerator, compose, load_image_art, load_prompt_file,
                  new_seed, variation_hints)
from .art.backends.comfyui import (DEFAULT_MODEL, DEFAULT_URL, PRESETS, ComfyClient,
                                   ComfyImageBackend, ComfyTextBackend)
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


class ArtContext:
    """Connects to ComfyUI on first use and keeps one generator per model/checkpoint."""

    def __init__(self, args):
        self.args = args
        self.client = None
        self.gens: dict[tuple, ArtGenerator] = {}

    def connect(self):
        if self.client is None:
            client = ComfyClient(self.args.comfy_url)
            client.check()
            self.client = client

    def gen(self, model: str, checkpoint: str | None) -> ArtGenerator:
        key = (model, checkpoint)
        if key not in self.gens:
            self.connect()
            self.gens[key] = ArtGenerator(ComfyImageBackend(self.client, model, checkpoint),
                                          ComfyTextBackend(self.client), seed=self.args.seed,
                                          describe=not self.args.no_describe)
        return self.gens[key]


def main(argv: list[str] | None = None) -> int:
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
    g = ap.add_argument_group("AI art (needs ComfyUI Desktop running)")
    g.add_argument("--art", action="store_true", help="generate art with ComfyUI for the art window")
    g.add_argument("--art-model", choices=sorted(PRESETS), default=DEFAULT_MODEL,
                   help=f"generation preset (default: {DEFAULT_MODEL})")
    g.add_argument("--checkpoint", help="override the model file (same model family as the preset)")
    g.add_argument("--comfy-url", default=DEFAULT_URL, help=f"ComfyUI address (default: {DEFAULT_URL})")
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

    ctx = ArtContext(args)
    needs_comfy = args.art or any(
        a.get("source") == "generated" for e in entries if e.repro for a in e.repro.get("art", []))
    prompts = load_prompt_file(args.prompts) if args.prompts else {}
    if needs_comfy:
        try:
            ctx.connect()
            if args.art:
                ctx.gen(args.art_model, args.checkpoint)  # validate the model up front
        except ArtError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 3

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_images = sheet_opts is None or args.also_images
    written, art_failed = [], []
    sheet_cards: list = []
    for e in entries:
        try:
            card, printing = db.find(e.name, e.set_code)
        except LookupError as err:
            missing.append((e.name, err.args[0] if err.args else [],
                            err.args[1] if len(err.args) > 1 else ""))
            continue
        for stem, img, meta in _card_images(e, card, printing, ctx, prompts, art_failed, args):
            if write_images:
                target = out / f"{stem}.png"  # never overwrite a file from this same run
                path = _write(target, lambda p: save(img, p, meta),
                              args.overwrite and target not in written)
                written.append(path)
                print(f"  wrote {path}  [{printing['set']} #{printing['collector_number']}]")
            if sheet_opts:
                # Keep only the flattened, sheet-sized copy; the full RGBA image can go.
                sheet_cards += [prepare_card(img, sheet_lay, sheet_opts)] * e.qty
                if not write_images:
                    print(f"  {e.qty} x {stem}")

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


def _free_path(path: Path) -> Path:
    """First of 'Name (1).ext', 'Name (2).ext', ... that doesn't exist yet."""
    n = 1
    while True:
        candidate = path.with_name(f"{path.stem} ({n}){path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def _write(path: Path, writer, overwrite: bool) -> Path:
    """Write via writer(path) without clobbering existing files (unless overwrite).
    If the target is locked (e.g. open in a viewer), fall back to a numbered copy."""
    if path.exists() and not overwrite:
        path = _free_path(path)
    try:
        writer(path)
    except PermissionError:
        alt = _free_path(path)
        print(f"  {path.name} is in use; writing {alt.name} instead", file=sys.stderr)
        writer(alt)
        path = alt
    return path


def _window_spec(e: Entry, w, label, card, ctx: ArtContext, prompts, args,
                 run: int = 0, hints: list[str] | None = None) -> dict | None:
    """How to fill one art window: user image, reproduced spec, new generation, or nothing.
    With several runs, each one is planned separately (new description and seed); a fixed
    --seed becomes seed, seed+1, ...; reproduced PNGs get new seeds unless --redescribe.
    hints (one per run) steer each run's scene description somewhere different."""
    hint = hints[run] if hints else None
    if e.image:
        return {"source": "image", "path": str(Path(e.image).resolve()), "fit": e.fit or args.art_fit}
    if e.repro is not None:
        saved = next((a for a in e.repro.get("art", []) if a.get("window") == w.key), None)
        if saved is None:
            return None
        spec = {k: v for k, v in saved.items() if k not in ("window", "face")}
        if spec.get("source") == "generated":
            gen = ctx.gen(spec["model"], spec["checkpoint"])
            if args.redescribe:
                spec = gen.restyle(spec, w.face, label, hint)
            if args.seed is not None:
                spec["seed"] = args.seed + run
            elif args.new_seed or (hints and not args.redescribe):
                spec["seed"] = new_seed()
        return spec
    if args.art:
        override = (e.art_prompt or prompts.get(w.face.name.lower())
                    or prompts.get(card["name"].lower()))
        spec = ctx.gen(args.art_model, args.checkpoint).plan(w.face, w.rect, override, label, hint)
        if args.seed is not None:
            spec["seed"] = args.seed + run
        return spec
    return None


def _card_images(e: Entry, card, printing, ctx: ArtContext, prompts, art_failed, args):
    """Yield (file stem, final image, metadata) for each image the card renders to."""
    tag = " (full art)" if e.full_art else ""
    generated = args.art or bool(e.repro and any(
        a.get("source") == "generated" for a in e.repro.get("art", [])))
    n_runs = (e.runs or args.runs) if generated and not e.image else 1
    only = e.repro.get("suffix", "") if e.repro is not None else None
    for r in render_card(card, printing, full_art=e.full_art, only=only or None):
        if e.repro is not None and r.suffix != e.repro.get("suffix", ""):
            continue  # reproducing one face of a double-faced card
        stem = f"{safe_filename(card['name'])}{tag}{r.suffix}"
        face = r.windows[0].face if r.windows else None
        hints = variation_hints(n_runs, face) if n_runs > 1 else None
        for run in range(n_runs):
            yield stem, *_compose_run(e, r, stem, card, printing, ctx, prompts, art_failed,
                                      args, run, hints)


def _compose_run(e, r, stem, card, printing, ctx, prompts, art_failed, args, run, hints):
    """(final image, metadata) for one rendered face, filling its art windows."""
    img = r.image
    layers, art_meta = [], []
    for w in r.windows:
        label = stem + w.key + (f" run {run + 1}" if hints else "")
        try:
            spec = _window_spec(e, w, label, card, ctx, prompts, args, run, hints)
            if not spec:
                continue
            if spec["source"] == "image":
                art, mode = load_image_art(spec["path"]), spec.get("fit", "crop")
            else:
                art, mode = ctx.gen(spec["model"], spec["checkpoint"]).generate(spec, label), "crop"
            layers.append((art, w.rect, mode))
            art_meta.append({"window": w.key, "face": w.face.name, **spec})
        except (ArtError, OSError) as err:
            art_failed.append((label, str(err)))
            print(f"  art failed for {label}: {err}", file=sys.stderr)
    if layers:
        img = compose(img, layers)
    meta = {"version": META_VERSION, "card": card["name"], "set": printing["set"],
            "collector_number": printing["collector_number"], "full_art": e.full_art,
            "suffix": r.suffix, "custom_prompt": e.art_prompt, "art": art_meta}
    return img, meta
