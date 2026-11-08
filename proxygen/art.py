"""AI art: prompt building, ComfyUI generation and compositing."""

from __future__ import annotations

import copy
import io
import json
import math
import random
import re
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

from .comfy import ComfyClient, ComfyError

WORKFLOW_DIR = Path(__file__).parent / "workflows"

PRESETS = {
    # Krea 2 follows prompts literally, renders quoted text and ignores negatives (cfg 1),
    # so it gets plain-language style, no flavor quotes and no "card" wording.
    "krea2-turbo": {"workflow": "krea2_turbo.json",
                    "checkpoint": "krea2_turbo_int8_convrot.safetensors",
                    "style": ("epic fantasy digital painting, full-bleed illustration filling the "
                              "entire image, dramatic lighting, highly detailed. The image contains "
                              "no text, no letters, no border and no frame"),
                    "flavor": False},
    "sdxl-turbo": {"workflow": "sdxl_turbo.json",
                   "checkpoint": "dreamshaperXL_v21TurboDPMSDE.safetensors"},
    "flux": {"workflow": "flux_dev.json", "checkpoint": "flux1-dev-fp8.safetensors"},
}
DEFAULT_MODEL = "krea2-turbo"

# Model loader node -> the input that names the model file (node "1" in every workflow).
LOADER_INPUT = {"CheckpointLoaderSimple": "ckpt_name", "UNETLoader": "unet_name"}

# SDXL/Flux-friendly resolutions (~1 megapixel)
BUCKETS = [(1024, 1024), (1152, 896), (1216, 832), (1344, 768), (1536, 640),
           (896, 1152), (832, 1216), (768, 1344), (640, 1536)]

STYLE = ("fantasy illustration, trading card game art, detailed digital painting, "
         "dramatic lighting, highly detailed")
NEGATIVE = ("text, letters, words, caption, watermark, signature, logo, card frame, border, "
            "blurry, lowres, jpeg artifacts, deformed, bad anatomy, extra limbs")

MOOD = {
    "W": "radiant holy light, white and gold tones",
    "U": "arcane blue tones, mist and water",
    "B": "dark shadowy atmosphere, ominous, deathly",
    "R": "fiery red and orange tones, volcanic, intense",
    "G": "lush green nature, primal wilderness",
}


# ---------------------------------------------------------------- basic lands
# Basic lands get pure landscapes: a scene picked from a catalog per land type (so runs vary),
# land-specific must-haves, and rules against people, buildings, carvings and magic effects.

LAND_SCENES = {
    "Plains": [
        "golden grassland rolling in long waves to a distant horizon under an enormous sky",
        "a wildflower meadow on gentle hills, seen from a low angle among the grass",
        "a windswept steppe with a single wind-bent tree far away",
        "a high plateau of short grass ending at a sheer cliff edge, with plains far below",
        "an aerial view of vast prairie crossed by a winding river",
        "a savanna of tall pale grass with scattered flat-topped trees",
    ],
    "Island": [
        "an aerial view of a tropical archipelago: green islands, white beaches, turquoise shallows",
        "a lone rocky island with steep cliffs rising from the open sea, seen from the water",
        "a rugged coastline of sea cliffs and sea stacks with waves breaking against them",
        "a sandy cove seen from a headland, with small islands scattered offshore",
        "a coral atoll ringing a calm lagoon, seen from high above",
        "a chain of misty mountainous islands stretching across a calm sea to the horizon",
        "a black volcanic island with surf crashing on dark beaches",
    ],
    "Swamp": [
        "a stagnant black bog with dead, leafless trees and grey moss hanging from bare branches",
        "a fog-choked marsh with rotting stumps and animal bones half sunk in the mud",
        "a flooded dead forest of pale, drowned trunks standing in still dark water",
        "a putrid swamp with sickly yellow-green haze over oily water and floating decay",
        "an aerial view of a vast dark mire of murky pools, reeds and dead trees",
        "a low view across the water surface of a swamp, roots and carrion birds on dead snags",
    ],
    "Mountain": [
        "a jagged mountain range of sharp peaks stretching to the horizon",
        "an erupting volcano pouring rivers of lava down its slopes under an ash-filled sky",
        "a single lone mountain rising high above a flat plain",
        "snow-capped peaks rising above a sea of clouds, seen from high altitude",
        "a looming cliff face seen from the base of the mountain, looking straight up",
        "a broad cone-shaped volcano with a glowing crater and a column of ash, rising from a "
        "barren field of black, cracked lava rock",
        "a deep valley between steep mountain walls, with ridges fading into the distance",
    ],
    "Forest": [
        "an aerial view over an endless forest canopy rolling across hills",
        "looking up from the forest floor past towering trunks to the leaf canopy and sky",
        "a forest seen from afar across a valley, its treeline climbing distant slopes",
        "the dim interior of an old-growth forest with shafts of light through the canopy",
        "a river winding through dense forest, seen from a high ridge",
        "a misty conifer forest covering steep mountain slopes",
        "a mossy forest floor of ferns, roots and fallen logs under the canopy",
    ],
    "Wastes": [
        "a barren grey wasteland of cracked earth stretching to the horizon",
        "wind-scoured stone ridges in an empty, colorless desert",
        "a vast pale salt flat under a dull sky",
    ],
}
LAND_RULES = {
    "Plains": "Open, sunlit grassland is the main subject; keep the land wide and open.",
    "Island": "Land and sea must meet: show an actual island or coastline above the water, "
              "never an underwater scene, a ship or a sea creature.",
    "Swamp": "The scene must look swampy and feel like death: waterlogged ground, stagnant water, "
             "dead or dying trees, rot and decay, gloomy light. No healthy green forest canopy.",
    "Mountain": "Mountains, peaks or volcanoes are the main subject, large and dominant in the frame.",
    "Forest": "Trees and forest are the main subject.",
    "Wastes": "Empty, lifeless, colorless land is the main subject.",
}
LAND_CONDITIONS = ["at dawn", "at midday", "at dusk", "at night under moonlight", "in rain",
                   "in fog or mist", "under storm clouds", "in autumn", "on a clear day"]
LAND_STYLE = ("an untouched natural wilderness, completely empty of people and figures, with no "
              "buildings, ruins, statues, carvings, runes, symbols or magic effects")
LAND_NEGATIVE = (", people, person, figure, character, silhouette, building, ruins, tower, "
                 "statue, runes, symbols, glowing magic")

LAND_INSTRUCTION = """\
You are an art director writing the prompt for a landscape painting. Describe ONE natural \
landscape: {land}.

Scene: {scene}.
{rule}

Rules:
- This is pure nature. No people, figures, silhouettes, travelers, boats or ships. Animals only \
as small, distant birds.
- No buildings, ruins, towers, bridges, roads, statues, altars, standing stones, carvings, runes, \
glowing symbols, magic effects or writing.
- Describe the landforms, vegetation, water, sky, the camera view, the lighting and the color \
palette. Make it dramatic and grand in scale, but natural.
- Use literal, concrete language with no similes or metaphors.
- Write one paragraph of 50 to 80 words. Output only the description."""


def basic_land_type(face) -> str | None:
    """'Forest', 'Island', ... for a basic land face, else None."""
    main, _, sub = (face.type_line or "").partition("—")
    if "basic" not in main.lower() or "land" not in main.lower():
        return None
    for word in sub.split():
        if word in LAND_SCENES:
            return word
    return "Wastes" if face.name == "Wastes" else None


def land_hints(face, n: int) -> list[str]:
    """n scene directions for a basic land, using different catalog scenes where possible."""
    land = basic_land_type(face)
    scenes = random.sample(LAND_SCENES[land], min(n, len(LAND_SCENES[land])))
    scenes += random.choices(LAND_SCENES[land], k=n - len(scenes))
    snow = "snow" in (face.type_line or "").lower()
    return [f"{s}, " + ("covered in snow and ice" if snow else random.choice(LAND_CONDITIONS))
            for s in scenes]


# ---------------------------------------------------------------- prompts

def _clean_flavor(flavor: str | None) -> str:
    if not flavor:
        return ""
    text = re.split(r"\n?\s*—", flavor)[0]  # drop "—Attribution"
    text = text.replace("\n", " ").strip(" \"“”")
    return text[:220]


def build_prompt(face, override: str | None = None, style: str = STYLE,
                 flavor: bool = True) -> tuple[str, str]:
    """Return (positive, negative) prompts for a card face."""
    land = basic_land_type(face)
    if land:  # pure landscape, whatever the prompt source
        subject = override or f"a vast {land.lower()} landscape: {land_hints(face, 1)[0]}"
        return f"{subject}, {LAND_STYLE}, {style}", NEGATIVE + LAND_NEGATIVE
    if override:
        return f"{override}, {style}", NEGATIVE
    name = face.name
    type_line = face.type_line or ""
    main, _, sub = type_line.partition("—")
    sub = sub.strip().lower()
    main_l = main.lower()
    if "planeswalker" in main_l:
        subject = f"{name}, a powerful planeswalker mage casting magic, heroic portrait"
    elif "creature" in main_l:
        subject = f"{name}, a {sub} creature" if sub else f"{name}, a fantasy creature"
    elif "vehicle" in sub:
        subject = f"{name}, a fantastical war machine vehicle"
    elif "land" in main_l:
        subject = f"{name}, a sweeping fantasy landscape location"
    elif "artifact" in main_l:
        subject = f"{name}, an ornate magical artifact" + (f", {sub}" if sub else "")
    elif "enchantment" in main_l:
        subject = f"{name}, a magical enchantment, mystical aura"
    elif "instant" in main_l or "sorcery" in main_l:
        subject = f"{name}, a magic spell being cast, dynamic action scene"
    elif "battle" in main_l:
        subject = f"{name}, an epic battle on a fantasy world"
    else:
        subject = name
    colors = [c for c in "WUBRG" if c in (face.colors or [])]
    if colors:
        mood = ", ".join(MOOD[c] for c in colors)
    elif "land" in main_l:
        mood = ""
    else:
        mood = "metallic tones, ancient craftsmanship"
    parts = [subject, mood, _clean_flavor(face.flavor_text) if flavor else "", style]
    negative = NEGATIVE
    if "land" in main_l and "creature" not in main_l:
        negative += ", people, person, character, figure"
    return ", ".join(p for p in parts if p), negative


def load_prompt_file(path: str | Path) -> dict[str, str]:
    """Lines of `Card Name = prompt`; '#' comments allowed. Keys are lowercase."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip().lower()] = v.strip()
    return out


# ---------------------------------------------------------------- scene descriptions (LLM)

COLOR_WORDS = {"W": "white", "U": "blue", "B": "black", "R": "red", "G": "green"}
SYMBOL_WORDS = {"T": "tap", "Q": "untap", "C": "colorless mana", "W": "white mana",
                "U": "blue mana", "B": "black mana", "R": "red mana", "G": "green mana",
                "S": "snow mana", "E": "energy", "X": "X mana"}

DESCRIBE_INSTRUCTION = """\
You are an art director writing the prompt for a fantasy illustration of a Magic: The Gathering card. \
Read the card below and describe ONE specific scene that shows what this card is and what it does.

Rules:
- Name the main subject and describe how it looks: shape, materials, size, anatomy or clothing.
- Choose a specific setting, a camera view or composition, and the lighting and color palette.
- Make the scene distinctive to this card. Take ideas from its name, rules text and flavor text; \
avoid generic fantasy filler.
- Never show playing cards, trading cards, card games, dice or other game pieces. Turn game \
ideas into the fantasy world: a library or deck becomes tomes, scrolls or visions of the \
future; a hand becomes secrets or plans; a graveyard becomes a crypt or battlefield remains.
- Use literal, concrete language. Avoid similes and metaphors ("like a spellbook"), because the \
image model draws them literally.
- Describe visuals only. Do not mention cards, game rules, mana, numbers or the card's name, \
and do not put any words, letters or writing in the image.
- Write one paragraph of 50 to 80 words. Output only the description.

Card:
"""


def _symbols_to_words(text: str) -> str:
    def word(m):
        sym = m.group(1).upper()
        if sym.isdigit():
            return f"{sym} generic mana"
        return SYMBOL_WORDS.get(sym, sym.replace("/", " or ").lower() + " mana")
    return re.sub(r"\{([^}]+)\}", word, text)


def card_brief(face) -> str:
    """Plain-text summary of a face for the language model."""
    lines = [f"Name: {face.name}", f"Type: {face.type_line}"]
    colors = [COLOR_WORDS[c] for c in "WUBRG" if c in (face.colors or [])]
    lines.append(f"Colors: {', '.join(colors) if colors else 'colorless'}")
    if face.oracle_text:
        lines.append(f"Rules text: {_symbols_to_words(face.oracle_text)}")
    if face.flavor_text:
        lines.append(f"Flavor text: {face.flavor_text}")
    return "\n".join(lines)


def _clean_description(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^\s*(description|scene|prompt)\s*:\s*", "", text.strip(), flags=re.I)
    return " ".join(text.split()).strip(" \"\u201c\u201d")


# ---------------------------------------------------------------- geometry / compositing

FIT_MODES = ("crop", "fit")


def bucket_for(w: float, h: float) -> tuple[int, int]:
    target = math.log(w / h)
    return min(BUCKETS, key=lambda b: abs(math.log(b[0] / b[1]) - target))


def fit_art(art: Image.Image, size: tuple[int, int], mode: str = "crop") -> Image.Image:
    """Size art to a window. crop: fill the window, trimming the overflow (centered).
    fit: show the whole image, filling the leftover space with a blurred, darkened copy."""
    art = art.convert("RGBA")
    cover = ImageOps.fit(art, size, Image.LANCZOS)
    if mode != "fit":
        return cover
    bg = cover.filter(ImageFilter.GaussianBlur(max(size) / 30))
    bg = Image.blend(bg, Image.new("RGBA", size, (0, 0, 0, 255)), 0.35)
    inner = ImageOps.contain(art, size, Image.LANCZOS)
    bg.alpha_composite(inner, ((size[0] - inner.width) // 2, (size[1] - inner.height) // 2))
    return bg


def compose(card: Image.Image, layers: list[tuple[Image.Image, tuple, str]]) -> Image.Image:
    """Put each art image under the card at its window rect: [(art, rect, fit_mode), ...]."""
    base = Image.new("RGBA", card.size, (0, 0, 0, 0))
    for art, rect, mode in layers:
        x0, y0, x1, y1 = (int(round(v)) for v in rect)
        base.paste(fit_art(art, (x1 - x0 + 1, y1 - y0 + 1), mode), (x0, y0))
    base.alpha_composite(card)
    return base


def load_image_art(path: str | Path) -> Image.Image:
    img = Image.open(path)
    img.load()
    return img


# Directions that push several runs of the same card toward different scenes.
VARIATION_VIEWS = ["a wide establishing view", "a close, intimate view", "a low angle looking up",
                   "a high view looking down", "a view from far away", "an unusual off-center view"]
VARIATION_CONDITIONS = ["at dawn", "at midday", "at dusk", "at night", "in rain", "in fog or mist",
                        "in snow", "in autumn", "under a storm", "in a different region or climate"]


def variation_hints(n: int, face=None) -> list[str]:
    """n distinct random directions for the scene description, one per run."""
    if face is not None and basic_land_type(face):
        return land_hints(face, n)
    combos = [(v, c) for v in VARIATION_VIEWS for c in VARIATION_CONDITIONS]
    picks = random.sample(combos, min(n, len(combos)))
    picks += random.choices(combos, k=n - len(picks))
    return [f"Show it from {v}, {c}. Choose a subject or composition different from the most "
            "obvious one." for v, c in picks]


def new_seed() -> int:
    return random.randrange(2**32)


# ---------------------------------------------------------------- generator

class ArtGenerator:
    """Generates art with one model preset. Nothing is cached: every call to plan() picks a
    seed (random unless fixed) and the resulting spec is stored in the output PNG instead."""

    def __init__(self, client: ComfyClient, model: str = DEFAULT_MODEL,
                 checkpoint: str | None = None, seed: int | None = None,
                 describe: bool = True, log=print):
        self.client = client
        self.preset = PRESETS[model]
        self.model = model
        self.seed = seed
        self.log = log
        self.template = json.loads((WORKFLOW_DIR / self.preset["workflow"]).read_text())
        self.loader = self.template["1"]["class_type"]
        self.loader_input = LOADER_INPUT[self.loader]
        self.checkpoint = self._resolve_checkpoint(checkpoint or self.preset["checkpoint"])
        self.describe = describe
        self.describe_template = json.loads((WORKFLOW_DIR / "describe.json").read_text())

    def _resolve_checkpoint(self, wanted: str) -> str:
        available = self.client.model_files(self.loader, self.loader_input)
        key = wanted.lower().replace("\\", "/")
        for name in available:
            n = name.lower().replace("\\", "/")
            if n == key or n.rsplit("/", 1)[-1] == key.rsplit("/", 1)[-1]:
                return name
        raise ComfyError(f"Model '{wanted}' not found by ComfyUI's {self.loader}. Available: "
                         + ", ".join(available))

    def _seed(self) -> int:
        return self.seed if self.seed is not None else new_seed()

    def prompt_for(self, face, override: str | None = None, label: str = "",
                   hint: str | None = None) -> tuple[str, str, dict]:
        """(positive, negative, info): user override, else an LLM scene description,
        else the template prompt. info records where the prompt came from."""
        style = self.preset.get("style", STYLE)
        info = {"prompt_source": "custom" if override else "template"}
        if not override and self.describe:
            if hint is None and basic_land_type(face):
                hint = land_hints(face, 1)[0]
            try:
                override = self.describe_face(face, label, hint)
                info = {"prompt_source": "description", "description": override}
                if hint:
                    info["direction"] = hint
            except ComfyError as e:
                self.log(f"  scene description failed ({e}); using the template prompt. "
                         "Pass --no-describe to skip this step.")
                self.describe = False
        positive, negative = build_prompt(face, override, style, self.preset.get("flavor", True))
        return positive, negative, info

    def describe_face(self, face, label: str = "", hint: str | None = None) -> str:
        """Scene description written by the language model (random seed unless --seed)."""
        wf = copy.deepcopy(self.describe_template)
        land = basic_land_type(face)
        if land:
            wf["2"]["inputs"]["prompt"] = LAND_INSTRUCTION.format(
                land=f"a {land.lower()} landscape" if land != "Wastes" else "a barren wasteland",
                scene=hint or land_hints(face, 1)[0], rule=LAND_RULES[land])
        else:
            wf["2"]["inputs"]["prompt"] = DESCRIBE_INSTRUCTION + card_brief(face) + (
                f"\n\nFor this image: {hint}" if hint else "")
        wf["2"]["inputs"]["sampling_mode.seed"] = self._seed()
        self.log(f"  describing scene for {label or face.name}...")
        text = _clean_description(self.client.run_text(wf))
        if not text:
            raise ComfyError("the language model returned an empty description")
        return text

    def plan(self, face, rect, override: str | None = None, label: str = "",
             hint: str | None = None) -> dict:
        """Everything needed to generate (and later reproduce) art for one window."""
        positive, negative, info = self.prompt_for(face, override, label, hint)
        w, h = bucket_for(rect[2] - rect[0], rect[3] - rect[1])
        return {"source": "generated", "model": self.model, "checkpoint": self.checkpoint,
                "positive": positive, "negative": negative, "seed": self._seed(),
                "width": w, "height": h, **info}

    def restyle(self, spec: dict, face, label: str = "", hint: str | None = None) -> dict:
        """A copy of spec with a fresh scene description (for --redescribe)."""
        description = self.describe_face(face, label, hint)
        positive, negative = build_prompt(face, description, self.preset.get("style", STYLE),
                                          self.preset.get("flavor", True))
        return {**spec, "positive": positive, "negative": negative,
                "prompt_source": "description", "description": description}

    def generate(self, spec: dict, label: str = "") -> Image.Image:
        wf = copy.deepcopy(self.template)
        wf["1"]["inputs"][self.loader_input] = spec["checkpoint"]
        wf["2"]["inputs"]["text"] = spec["positive"]
        if wf["3"]["class_type"] == "CLIPTextEncode":
            wf["3"]["inputs"]["text"] = spec["negative"]
        wf["4"]["inputs"]["width"], wf["4"]["inputs"]["height"] = spec["width"], spec["height"]
        wf["5"]["inputs"]["seed"] = spec["seed"]
        self.log(f"  generating art for {label} ({spec['width']}x{spec['height']}, "
                 f"{spec['model']}, seed {spec['seed']})...")
        img = Image.open(io.BytesIO(self.client.run(wf)))
        img.load()
        return img
