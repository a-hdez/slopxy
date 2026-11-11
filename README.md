# slopxy — MTG proxy card generator

Builds proxy card images from scratch (frame, mana symbols, rules text) using the
local Scryfall bulk file `default-cards-*.jsonl`. The art window is transparent, or filled
with AI art from a local generator (ComfyUI, Automatic1111/Forge) or a hosted API (OpenAI,
Gemini; see below).

Mana symbols (in the title bar and in rules text) use the badges in
`assets/*_badge.svg` (star W, droplet U, skull B, fireball R, tree G, diamond C, phi for
Phyrexian). Hybrid costs combine two badges on a split circle. Generic numbers, X, snow and tap
are still drawn by `symbols.py`.

## Setup
Requires Python 3.10+. From the repository folder:
```
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```
This installs the `slopxy` command (in the venv; activate it in each new terminal). For the
hosted art backends add their SDKs: `pip install -e ".[openai]"`, `".[gemini]"`,
`".[anthropic]"` or `".[all]"`. Without
activating, run `.venv\Scripts\slopxy.exe` or `.venv\Scripts\python.exe -m slopxy` instead.
The repository root is the `slopxy` package itself, so the install step is required:
`python -m slopxy` does not work from a fresh checkout without it. With `-e`, code edits take
effect immediately.

Download the **Default Cards** bulk file from https://scryfall.com/docs/api/bulk-data and put
the `default-cards-*.jsonl` file in the folder you run slopxy from (or pass `--data PATH`).
It is not in the repository because of its size (~650 MB).

Fonts: Palatino and Arial from Windows are used when present; otherwise Georgia/Times or
DejaVu are used as fallbacks, so cards look slightly different on other systems.

## Usage
```
slopxy "Lightning Bolt" "Fire // Ice"
slopxy -f decklist.txt -o output
slopxy            # interactive: type names, blank line to finish
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
- The first run builds `.cache/card_index.pkl` and `.cache/card_sets.pkl` (a few seconds). It rebuilds automatically
  when the data file changes, or on `--rebuild-index`.

## Printable sheets
Add `--sheet` to get a print-ready multi-page PDF (quantities honored) instead of single images:
```
slopxy -f decklist.txt --sheet
slopxy -f decklist.txt --sheet --paper a4 --cut-lines marks
slopxy -f decklist.txt --sheet --art --sheet-name modern.pdf
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

## AI art
Add `--art` and choose who makes the images with `--art-backend` (or once, in a config file):

| Backend | Images | Scene descriptions | Needs |
|---|---|---|---|
| `comfyui` | yes | yes (its own Qwen3-VL) | ComfyUI running (`http://127.0.0.1:8189` by default) |
| `a1111` | yes | | Automatic1111 / Forge / SD.Next started with `--api` (`http://127.0.0.1:7860`) |
| `openai` | yes | yes | `pip install slopxy[openai]` and `OPENAI_API_KEY`; or any OpenAI-compatible server |
| `gemini` | yes | yes | `pip install slopxy[gemini]` and `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) |
| `anthropic` | | yes | `pip install slopxy[anthropic]` and `ANTHROPIC_API_KEY` |
| `placeholder` | yes | yes | nothing: draws the prompt, for trying layouts and runs for free |

```
slopxy --art --art-backend comfyui "Lightning Bolt" "Sol Ring [full]"
slopxy --art --art-backend openai --describe-backend anthropic -f decklist.txt
slopxy --list-backends      # what each backend needs, what's installed, your profiles
```
`pip install slopxy[all]` installs every hosted SDK. Nothing is assumed: `--art` without a
backend (and without a config default) stops and says how to pick one.

- Prompts: before each image, a language model reads the card's name, type, colors, rules
  and flavor text and writes one concrete scene description. That description plus the
  model's style text is the image prompt, so cards stop looking alike. ComfyUI writes its own
  (Qwen3-VL through its Generate Text node); for other image backends choose a writer with
  `--describe-backend` (`anthropic`, `openai`, `gemini`, `comfyui` or a profile), otherwise
  they use the simpler template prompt (name, type, color mood), as `--no-describe` does.
  Override per card with `Sol Ring [full] [art: a glowing golden ring on a stone altar]`, or
  with a `--prompts` file of `Card Name = prompt` lines (face names work for double-faced and
  split cards). If a model declines to describe one card, that card gets the template prompt.
- All scene descriptions are written before the first image is generated, so a local GPU
  doesn't swap between the language model and the image model for every card. Hosted
  backends then generate several images at once (4 by default, `concurrency` in a profile).
- Hosted backends charge per request: a run that would make more than 20 paid requests
  (images plus descriptions) asks first; `--yes` skips the question.
- Basic lands (Plains, Island, Swamp, Mountain, Forest, Wastes, and snow-covered ones) are
  always pure landscapes. Each image picks a scene from a catalog for its land type
  (`LAND_SCENES` in `art/prompts.py`), plus a time of day or weather. Forests are seen from
  above, from afar or from under the canopy; Islands show real coastlines and islands;
  Mountains show ranges, volcanoes and lone peaks; Swamps are dead, drowned bogs. The prompt
  bans people, buildings, ruins, carvings, runes and magic effects. With `--runs`, each run
  uses a different scene. Edit `LAND_SCENES` and `LAND_RULES` to change the mix.
- Nothing is cached: every `--art` run picks a random seed (unless `--seed N`) and writes new
  art. Everything needed to recreate an image (card, printing, full-art flag, and per art
  window the backend, model, scene description, prompt, seed and size) is embedded in the
  output PNG as a `slopxy` text chunk. API keys and server addresses are never stored there.
- `--art-model` picks the backend's model:
  - `comfyui`: a preset. `krea2-turbo` (default): Krea 2 Turbo int8
    (`diffusion_models/krea2_turbo_int8_convrot`) with the `qwen3vl_4b_fp8_scaled` text
    encoder and `qwen_image_vae`; `sdxl-turbo`: DreamShaper XL Turbo; `flux`: FLUX.1 dev fp8
    (slow on 8 GB). `--checkpoint NAME` swaps in another model file of the same family.
  - `a1111`: the checkpoint (default: whatever the WebUI has loaded).
  - `openai`: default `gpt-image-2.6-sunburst` (`gpt-image-2.6-flare` is faster).
  - `gemini`: default `gemini-3.1-flash-image` (e.g. `gemini-nano-banana-2.1`,
    `gemini-3-pro-image`).
  - `--describe-model` does the same for the description writer: `anthropic` defaults to
    `claude-opus-5-5`, `openai` to `gpt-6.1-terra`, `gemini` to `gemini-3.6-flash`.
- `--art-url` points `comfyui` or `a1111` at another address. `--comfy-url URL` still works,
  and picks `comfyui` when no backend is chosen, so `slopxy --art --comfy-url URL ...` runs as
  it used to.
- OpenAI and Gemini take no seed, so their images can't be regenerated exactly; keep the
  art itself with `--keep-art` (see Reproducing a card).

### Profiles (slopxy.toml)
Settings you'd otherwise repeat go in `slopxy.toml` in the folder you run slopxy from (or the
user config file: `%APPDATA%\slopxy\config.toml` on Windows, `~/.config/slopxy/config.toml`
elsewhere, or `--config FILE`):
```toml
[art]
image = "local"        # used by --art unless --art-backend says otherwise
describe = "claude"    # writes the scene descriptions unless --describe-backend says otherwise

[profiles.local]
backend = "comfyui"
url = "http://127.0.0.1:8189"
model = "krea2-turbo"

[profiles.forge]
backend = "a1111"
url = "http://127.0.0.1:7860"
steps = 30
cfg_scale = 5.5
sampler = "DPM++ 2M"
scheduler = "Karras"

[profiles.fast]
backend = "openai"
model = "gpt-image-2.6-flare"
quality = "high"
api_key_env = "MY_OPENAI_KEY"   # the variable holding the key (default OPENAI_API_KEY)

[profiles.banana]
backend = "gemini"
image_size = "2K"

[profiles.claude]
backend = "anthropic"
effort = "low"

[profiles.ollama]              # any OpenAI-compatible server, local or hosted
backend = "openai"
base_url = "http://127.0.0.1:11434/v1"
model = "qwen3:8b"
```
- A profile name works wherever a backend name does (`--art-backend forge`), and a backend
  name works as a profile with its defaults.
- Keys never go in the file: a profile names the environment variable that holds its key
  (`api_key_env`), or the vendor's usual variable is used. A file with `api_key` is refused.
- Every setting a backend reads is listed at the top of its module in `art/backends/`.

## Several generations per card
`--runs N` generates art N times for every card, one image per run; `[runs: N]` does it for
one line of a list:
```
slopxy --art "Forest [full]" --runs 10
slopxy --art "Forest [full] [runs: 10]" "Sol Ring"
```
- Each run gets its own scene description and seed, saved as `Forest (full art).png`,
  `Forest (full art) (1).png`, ... (with `--overwrite`, runs still never replace each other).
- To push the runs apart, each description is given a different random direction (view,
  time of day, weather or season). With `--seed N`, the runs use seeds N, N+1, N+2, ...
- With a reproduced PNG, runs keep its scene and get new seeds (or new scenes with
  `--redescribe`). On a sheet, each run's image is placed as many times as the line's quantity.
- Runs apply only to generated art; `[image: ...]` and art-less cards render once.

## Reproducing a card
Give a slopxy output PNG instead of a card name (on the command line or in a list file,
with a quantity if you like) and it is rebuilt from its embedded metadata:
```
slopxy "output/Sol Ring.png"
slopxy "output/Sol Ring.png" --new-seed     # same scene, new image
slopxy "output/Sol Ring.png" --redescribe   # new scene, same seed
```
- `--seed N` sets the seed explicitly. Generated art needs the backend that made it (ComfyUI
  open, the API key set); `--art` isn't needed. Reproducing doesn't write new scene
  descriptions unless you pass `--redescribe`.
- With seeds (ComfyUI, Automatic1111, placeholder) the same image comes back. OpenAI and
  Gemini have no seeds, so they make a new image from the same prompt, unless the art was
  kept: with `--keep-art`, each generated artwork is also saved next to its card as
  `Name.art.png` (`Name.art_1.png`, ... for split cards) and the PNG records it. Reproducing
  then reuses that file, exactly and without the backend; `--new-seed` or `--redescribe`
  generate again.
- For a double-faced card, each face's PNG reproduces just that face.

## Your own art
Add `[image: path]` to a card to use an image file as its art (no backend needed):
```
slopxy "Forest [full] [image: C:\pics\forest.jpg]"
slopxy "Island [image: pics/island.png | fit]"
```
- When the image's shape doesn't match the art window: `crop` (default) fills the window and
  trims the overflow; `fit` shows the whole image over a blurred fill. Set it per card with
  `| crop` / `| fit`, or for all cards with `--art-fit`.
- The image path is stored in the PNG metadata, so the card can be reproduced while the file
  still exists.
- ComfyUI workflows live in `art/backends/workflows/*.json` (API format, core nodes only).
  Each backend is one module in `art/backends/`; a new one only needs registering in
  `BACKENDS` in `art/backends/__init__.py`.


## License
Public domain, under the [Unlicense](LICENSE). Magic: The Gathering card names and rules text
come from the Scryfall data file you download; they are not part of this repository.
