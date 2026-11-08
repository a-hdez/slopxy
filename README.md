# proxygen — MTG proxy card generator

Builds proxy card images from scratch (frame, mana symbols, rules text) using the
local Scryfall bulk file `default-cards-*.jsonl`. The art window is transparent, or filled
with AI art from a local ComfyUI (see below).

Mana symbols (in the title bar and in rules text) use the badges in
`proxygen/assets/*_badge.svg` (star W, droplet U, skull B, fireball R, tree G, diamond C, phi for
Phyrexian). Hybrid costs combine two badges on a split circle. Generic numbers, X, snow and tap
are still drawn by `proxygen/symbols.py`.

## Setup
Requires Python 3.10+.
```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Download the **Default Cards** bulk file from https://scryfall.com/docs/api/bulk-data and put
the `default-cards-*.jsonl` file in the folder you run proxygen from (or pass `--data PATH`).
It is not in the repository because of its size (~650 MB).

Fonts: Palatino and Arial from Windows are used when present; otherwise Georgia/Times or
DejaVu are used as fallbacks, so cards look slightly different on other systems.

## Usage
```
.venv\Scripts\python.exe -m proxygen "Lightning Bolt" "Fire // Ice"
.venv\Scripts\python.exe -m proxygen -f decklist.txt -o output
.venv\Scripts\python.exe -m proxygen            # interactive: type names, blank line to finish
```

- `Name|SET` picks a specific printing (e.g. `Lightning Bolt|M10`); otherwise the latest
  English paper printing is used.
- List files accept `4 Lightning Bolt`, `4x Lightning Bolt`, Arena-style
  `1 Brainstorm (ICE) 61`, `#` comments and `Deck`/`Sideboard` headers. Quantities are used
  by printable sheets; individual images are written once per card.
- Full-art: add `[full]` to a card (`Sol Ring [full]`, `4 Forest [full]`) or pass `--full-art`
  for every card. Only the title bar is drawn; the whole area inside the black border is a
  transparent art window. Files are named `Name (full art).png`.
- The art window is transparent in every image, so artwork can go on a layer underneath.
- Double-faced cards produce `_front` and `_back` images; split, flip and adventure cards
  render on one image.
- Output: 750×1050 PNG (2.5″×3.5″ at 300 DPI).
- Existing files are never overwritten: a new copy is written as `Name (1).png`, `Name (2).png`…
  (same for sheet PDFs). Pass `--overwrite` to replace them; if a file is open/locked, a numbered
  copy is written instead with a warning.
- The first run builds `.cache/card_index.pkl` (a few seconds). It rebuilds automatically
  when the data file changes, or on `--rebuild-index`.

## Printable sheets
Add `--sheet` to get a print-ready multi-page PDF (quantities honored) instead of single images:
```
.venv\Scripts\python.exe -m proxygen -f decklist.txt --sheet
.venv\Scripts\python.exe -m proxygen -f decklist.txt --sheet --paper a4 --cut-lines marks
.venv\Scripts\python.exe -m proxygen -f decklist.txt --sheet --art --sheet-name modern.pdf
```
- Cards are true size (2.5×3.5in); print at **100% / actual size**, not "fit to page".
- `--paper letter|a4|legal|a3|tabloid` or custom `210x297mm` / `8.5x11in` (default letter).
  Letter and A4 hold 3×3 cards; `--orientation auto` (default) picks whichever fits more.
- `--cut-lines lines` (default: thin gray outlines), `marks` (crop marks in the margin) or `none`.
- `--gap` mm between cards (default 1), `--margin` inches (default 0.2),
  `--card-scale` percent (e.g. 98 for tight sleeves).
- Double-faced cards are placed as two separate cards (front and back).
- The PDF goes to `--out` as `<list file>.pdf` (or `proxies.pdf`); `--sheet-name` changes it,
  `--also-images` also writes the single-card PNGs.

## AI art (ComfyUI)
Open ComfyUI Desktop first (it serves on `http://127.0.0.1:8189`), then add `--art`:
```
.venv\Scripts\python.exe -m proxygen --art "Lightning Bolt" "Sol Ring [full]"
.venv\Scripts\python.exe -m proxygen --art -f decklist.txt --prompts prompts.txt
```
- Prompts: before each image, the Qwen3-VL language model (the Krea 2 text encoder, run
  through ComfyUI's built-in Generate Text node) reads the card's name, type, colors, rules
  and flavor text and writes one concrete scene description. That description plus the
  model's style text is the image prompt, so cards stop looking alike. `--no-describe` uses
  the simpler template prompt (name, type, color mood) instead. Override per card with
  `Sol Ring [full] [art: a glowing golden ring on a stone altar]`, or with a `--prompts` file
  of `Card Name = prompt` lines (face names work for double-faced/split cards).
- Basic lands (Plains, Island, Swamp, Mountain, Forest, Wastes, and snow-covered ones) are
  always pure landscapes. Each image picks a scene from a catalog for its land type
  (`LAND_SCENES` in `proxygen/art.py`), plus a time of day or weather. Forests are seen from
  above, from afar or from under the canopy; Islands show real coastlines and islands;
  Mountains show ranges, volcanoes and lone peaks; Swamps are dead, drowned bogs. The prompt
  bans people, buildings, ruins, carvings, runes and magic effects. With `--runs`, each run
  uses a different scene. Edit `LAND_SCENES` and `LAND_RULES` to change the mix.
- Nothing is cached: every `--art` run picks a random seed (unless `--seed N`) and writes new
  art. Everything needed to recreate an image (card, printing, full-art flag, and per art
  window the scene description, prompt, model, checkpoint, seed and size) is embedded in the
  output PNG as a `proxygen` text chunk.
- `--art-model`:
  - `krea2-turbo` (default): Krea 2 Turbo int8 (`diffusion_models/krea2_turbo_int8_convrot`)
    with the `qwen3vl_4b_fp8_scaled` text encoder and `qwen_image_vae`.
  - `sdxl-turbo`: DreamShaper XL Turbo.
  - `flux`: FLUX.1 dev fp8 (slow on 8 GB).
- `--checkpoint NAME` swaps in another model file of the same family, `--comfy-url` points
  elsewhere.

## Several generations per card
`--runs N` generates art N times for every card, one image per run; `[runs: N]` does it for
one line of a list:
```
.venv\Scripts\python.exe -m proxygen --art "Forest [full]" --runs 10
.venv\Scripts\python.exe -m proxygen --art "Forest [full] [runs: 10]" "Sol Ring"
```
- Each run gets its own scene description and seed, saved as `Forest (full art).png`,
  `Forest (full art) (1).png`, ... (with `--overwrite`, runs still never replace each other).
- To push the runs apart, each description is given a different random direction (view,
  time of day, weather or season). With `--seed N`, the runs use seeds N, N+1, N+2, ...
- With a reproduced PNG, runs keep its scene and get new seeds (or new scenes with
  `--redescribe`). On a sheet, each run's image is placed as many times as the line's quantity.
- Runs apply only to generated art; `[image: ...]` and art-less cards render once.

## Reproducing a card
Give a proxygen output PNG instead of a card name (on the command line or in a list file,
with a quantity if you like) and it is rebuilt exactly from its embedded metadata:
```
.venv\Scripts\python.exe -m proxygen "output/Sol Ring.png"
.venv\Scripts\python.exe -m proxygen "output/Sol Ring.png" --new-seed     # same scene, new image
.venv\Scripts\python.exe -m proxygen "output/Sol Ring.png" --redescribe   # new scene, same seed
```
- `--seed N` sets the seed explicitly. Generated art needs ComfyUI open; `--art` isn't needed.
- For a double-faced card, each face's PNG reproduces just that face.

## Your own art
Add `[image: path]` to a card to use an image file as its art (no ComfyUI needed):
```
.venv\Scripts\python.exe -m proxygen "Forest [full] [image: C:\pics\forest.jpg]"
.venv\Scripts\python.exe -m proxygen "Island [image: pics/island.png | fit]"
```
- When the image's shape doesn't match the art window: `crop` (default) fills the window and
  trims the overflow; `fit` shows the whole image over a blurred fill. Set it per card with
  `| crop` / `| fit`, or for all cards with `--art-fit`.
- The image path is stored in the PNG metadata, so the card can be reproduced while the file
  still exists.
- Workflows live in `proxygen/workflows/*.json` (ComfyUI API format, core nodes only).

Requires Pillow (`pip install -r requirements.txt`).
