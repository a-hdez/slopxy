"""Prompt text for AI art: template prompts, basic-land landscapes, the scene-description
instruction for the language model, and per-run variation hints. No backend code here."""

from __future__ import annotations

import random
import re
from pathlib import Path

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


def clean_description(text: str) -> str:
    """Tidy the language model's output into a single-paragraph description."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^\s*(description|scene|prompt)\s*:\s*", "", text.strip(), flags=re.I)
    return " ".join(text.split()).strip(" \"\u201c\u201d")


def describe_prompt(face, hint: str | None = None) -> str:
    """The instruction sent to the language model to write a scene description for a face."""
    land = basic_land_type(face)
    if land:
        return LAND_INSTRUCTION.format(
            land=f"a {land.lower()} landscape" if land != "Wastes" else "a barren wasteland",
            scene=hint or land_hints(face, 1)[0], rule=LAND_RULES[land])
    return DESCRIBE_INSTRUCTION + card_brief(face) + (
        f"\n\nFor this image: {hint}" if hint else "")


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
