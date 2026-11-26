"""Shared caption logic used by scripts 08-11 and the Streamlit app.

  * object crops + prompt filling
  * parsing model output (JSON or free text) into caption + proposed conditions
  * rule-based "template" captions built only from the structured labels
  * automatic consistency checks (material / category mentions, unsupported objects, brands)
  * merging raw generations + human review events into the annotation records
"""
from __future__ import annotations

import json
import re
from collections import Counter as Counter_, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image

from common import ROOT, read_jsonl

CAPTIONS = ROOT / "data" / "captions"
REVIEWS = ROOT / "data" / "reviews"
CAPTION_CFG_PATH = ROOT / "configs" / "captioning.yaml"
for _d in (CAPTIONS, REVIEWS):
    _d.mkdir(parents=True, exist_ok=True)


def load_caption_cfg() -> dict:
    with open(CAPTION_CFG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Lexicons: words in a caption -> taxonomy values
# ---------------------------------------------------------------------------
MATERIAL_WORDS = {
    "plastic": ["plastic", "styrofoam", "polystyrene", "foam", "pet", "nylon", "polythene"],
    "metal": ["metal", "metallic", "aluminium", "aluminum", "tin", "steel", "iron", "foil"],
    "glass": ["glass"],
    "paper_cardboard": ["paper", "cardboard", "paperboard", "newspaper", "carton"],
    "organic": ["food waste", "food scraps", "organic", "fruit", "peel", "leftover", "vegetable"],
    "textile_rubber": ["cloth", "fabric", "textile", "rubber", "rope", "leather", "string"],
    "e_waste": ["battery", "batteries", "electronic"],
}
CATEGORY_WORDS = {
    "bottle_jar": ["bottle", "bottles", "jar", "jars"],
    # bare "can" is excluded: "we can see ..." is the most common caption opener
    "can": ["cans", "a can", "the can", "tin can", "soda can", "drink can", "beer can", "metal can",
            "aluminium can", "aluminum can", "food can", "aerosol"],
    "carton_box": ["carton", "cartons", "box", "boxes", "cardboard"],
    "cup": ["cup", "papercup", "mug", "wine glass", "glasses"],
    "bag_wrapper": ["bag", "bags", "wrapper", "wrappers", "packet", "packets", "film", "wrapping", "sachet"],
    "container_packaging": ["container", "containers", "tub", "tray", "packaging", "package", "blister"],
    "cap_lid": ["cap", "caps", "lid", "lids"],
    "paper": ["paper", "papers", "newspaper", "magazine", "tissue", "napkin", "receipt"],
    "straw_utensil": ["straw", "straws", "fork", "spoon", "knife", "utensil", "cutlery"],
    "cigarette": ["cigarette", "cigarettes", "cigarette butt", "butt", "butts"],
    "food_waste": ["food waste", "food scraps", "fruit", "peel", "banana", "leftovers", "vegetable"],
    "glass_fragment": ["shard", "shards", "broken glass", "glass pieces"],
    "hazardous": ["battery", "batteries"],
}
CONDITION_WORDS = {
    "crushed": ["crushed", "flattened", "squashed", "compressed"],
    "dented": ["dented", "deformed", "bent"],
    "crumpled": ["crumpled", "wrinkled", "scrunched"],
    "torn": ["torn", "ripped", "shredded"],
    "broken": ["broken", "shattered", "cracked", "smashed"],
    "open": ["open", "opened", "uncapped", "lid off"],
    "soiled": ["dirty", "soiled", "stained", "muddy", "greasy", "filthy"],
    "wet": ["wet", "damp", "soaked"],
    "burnt": ["burnt", "burned", "charred"],
    "faded": ["faded", "weathered", "sun-bleached", "discoloured", "discolored"],
    "intact": ["intact", "undamaged", "unopened", "sealed", "good condition", "new condition"],
}
# People / animals: never labelled in the waste datasets, so in TACO / ZeroWaste a mention is usually a
# hallucination ("the body of a man", "a dead animal on the sand"). COCO scenes can contain real people.
LIVING_WORDS = ["man", "men", "woman", "women", "person", "people", "child", "children", "boy", "girl", "body",
                "animal", "dog", "cat", "bird", "fish", "horse", "cow", "insect", "stuffed animal"]
# Category words that carry no evidence on their own for unspecified / other_litter
VAGUE_CATEGORIES = {"unspecified", "other_litter"}
QUOTED_TEXT_RE = re.compile(r"[\"“”][^\"“”]{1,40}[\"“”]")       # quoted words read off a label
BRAND_WORD_RE = re.compile(r"\b(brand|logo|made by|trademark)\b", re.I)
VALID_CONDITIONS = set(CONDITION_WORDS) | {"partially_occluded", "heavily_occluded"}


# compound nouns whose first word is only a modifier ("bottle cap" is a cap, not a bottle)
COMPOUNDS = {"bottle cap": "cap", "bottle top": "cap", "bottle lid": "lid", "can tab": "pop tab",
             "pull tab": "pop tab", "paper cup": "papercup", "cigarette pack": "packet"}
# a "composite" label (paper cup, crisp packet, Tetra Pak) is compatible with these material words
COMPOSITE_PARTS = {"composite", "paper_cardboard", "plastic", "metal"}


def _has(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}(?:s|es)?\b", text) is not None     # plural-tolerant


def _normalise(text: str) -> str:
    t = text.lower()
    for a, b in COMPOUNDS.items():
        t = re.sub(rf"\b{a}(s?)\b", b + r"\1", t)
    return t


def mentions(text: str, lexicon: dict[str, list[str]]) -> list[str]:
    t = _normalise(text)
    return sorted(k for k, words in lexicon.items() if any(_has(t, w) for w in words))


def has_brand_or_text(text: str) -> bool:
    return bool(QUOTED_TEXT_RE.search(text) or BRAND_WORD_RE.search(text))


NEGATION_RE = re.compile(r"\b(not|no|never|without)\s+(\w+\s+){0,2}$")


def extract_conditions(text: str) -> list[str]:
    """Condition words in the text, skipping negated ones ("not fully open", "no dents")."""
    t = _normalise(text)
    found = set()
    for cond, words in CONDITION_WORDS.items():
        for w in words:
            for m in re.finditer(rf"\b{re.escape(w)}(?:s|es)?\b", t):
                if not NEGATION_RE.search(t[max(0, m.start() - 30):m.start()]):
                    found.add(cond)
    return sorted(found)


# ---------------------------------------------------------------------------
# Crops and prompts
# ---------------------------------------------------------------------------
def crop_object(img: Image.Image, bbox, pad: float = 0.15, min_side: int = 32):
    """Crop a COCO box enlarged by `pad` on each side. Returns (crop, [x0, y0, x1, y1])."""
    x, y, w, h = bbox
    x0, y0 = max(0, x - pad * w), max(0, y - pad * h)
    x1, y1 = min(img.width, x + w + pad * w), min(img.height, y + h + pad * h)
    box = [round(x0), round(y0), round(x1), round(y1)]
    crop = img.crop(box)
    if min(crop.size) < min_side:
        s = min_side / min(crop.size)
        crop = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))), Image.BICUBIC)
    return crop, box


def object_label(obj: dict, source: str) -> str:
    """Human-readable label used in prompts and template captions."""
    if source == "zerowaste":
        return {"cardboard": "piece of cardboard", "soft_plastic": "piece of soft plastic film",
                "rigid_plastic": "rigid plastic item", "metal": "metal item"}[obj["source_category"]]
    return obj["source_category"].lower()


def fill_prompt(template: str, rec: dict | None = None, obj: dict | None = None, taxonomy: dict | None = None) -> str:
    vals = {"conditions": ", ".join(taxonomy["conditions"]) if taxonomy else ""}
    if rec is not None and obj is None:      # image level: the annotated counts, used by count-aware prompts
        vals.update(n_items=len(rec["objects"]), item_listing=count_listing(rec, digits=True))
    if obj is not None:
        vals.update(label=object_label(obj, rec["source"]),
                    material=obj["material"].replace("_", " / " if obj["material"] == "paper_cardboard" else " "),
                    region=obj["location"]["region"].replace("-", " "))
    return template.strip().format(**vals) if "{" in template else template.strip()


def parse_object_output(raw: str) -> tuple[str, list[str], bool]:
    """Model output -> (caption, proposed_conditions, parsed_as_json)."""
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            cap = str(d.get("caption", "")).strip()
            if cap and "<" not in cap:
                conds = [c for c in d.get("condition", []) if isinstance(c, str) and c in VALID_CONDITIONS]
                return cap, sorted(set(conds) | set(extract_conditions(cap))), True
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
    cap = raw.strip().strip('"').strip()
    return cap, extract_conditions(cap), False


MAX_SENTENCES = {"image": 2, "object": 1}


def clean_caption(text: str, level: str) -> str:
    """Keep at most N complete sentences and drop a fragment cut off by max_new_tokens.
    Small chat models ignore "one sentence" and ramble until the token limit."""
    text = " ".join(text.split())
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"])", text)
    if len(sents) > 1 and not re.search(r"[.!?\"]$", sents[-1]):
        sents = sents[:-1]                      # truncated last sentence
    return " ".join(sents[:MAX_SENTENCES[level]])


def make_caption_row(model: str, mcfg: dict, rec: dict, obj: dict | None, prompt_id: str, prompt: str | None,
                     raw: str, k: int, runtime: float, gen: dict, crop_box=None) -> dict:
    """One raw generation as stored in data/captions/<model>.jsonl."""
    level = "object" if obj else "image"
    target = obj["object_id"] if obj else rec["image_id"]
    row = {
        "caption_id": f"{target}__{model}__{prompt_id}__{k}",
        "image_id": rec["image_id"], "object_id": obj["object_id"] if obj else None, "level": level,
        "model": model, "hf_id": mcfg.get("hf_id"), "prompt_id": prompt_id, "prompt": prompt,
        "raw_text": raw, "text": raw, "clip_score": None, "gen": gen,
        "runtime_s": round(runtime, 3), "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if obj:
        cap, conds, as_json = parse_object_output(raw)
        cap = clean_caption(cap, level)
        if not as_json:                 # free text: only conditions stated in the kept caption count
            conds = extract_conditions(cap)
        row.update(text=cap, crop_box=crop_box, proposed={"condition": conds, "parsed_json": as_json})
    elif mcfg.get("backend") == "chat":
        row["text"] = clean_caption(raw, level)
    return row


# ---------------------------------------------------------------------------
# Rule-based template captions (no VLM): the "structured language" reference
# ---------------------------------------------------------------------------
ENV_PHRASES = {
    "indoor_manmade": "indoors", "pavement_urban": "on pavement", "sand_dirt_pebbles": "on sand or dirt",
    "vegetation": "among vegetation", "water": "in or near water", "trash_pile": "on a trash pile",
    "clean_background": "on a clean background", "conveyor_belt": "on a recycling conveyor belt",
    "household_tabletop": "on a tabletop",
}
NUM_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


MASS_NOUNS = ("litter", "foil", "paper", "glass", "plastic", "cardboard", "film")


def plural(label: str) -> str:
    if label.startswith("piece of "):
        return "pieces of " + label[9:]
    if label.endswith(MASS_NOUNS):
        return f"pieces of {label}"
    if label.endswith(("s", "x", "ch")):
        return label if label.endswith("s") else label + "es"
    return label + "s"


def template_object_caption(obj: dict, source: str) -> str:
    label = object_label(obj, source)
    words = [c.replace("_", " ") for c in obj["condition"] if c not in ("intact",)]
    mat = obj["material"]
    if mat not in ("unknown", "composite") and not any(w in label for w in MATERIAL_WORDS[mat]):
        words.append(mat.replace("paper_cardboard", "cardboard"))
    return f"{' '.join(words + [label])} in the {obj['location']['region'].replace('-', ' ')} of the image"


def join_and(parts: list[str]) -> str:
    return ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1] if parts else ""


def label_groups(rec: dict) -> list[tuple[str, list[dict]]]:
    """Objects grouped by their readable label, largest group first."""
    groups = defaultdict(list)
    for o in rec["objects"]:
        groups[object_label(o, rec["source"])].append(o)
    return sorted(groups.items(), key=lambda kv: -len(kv[1]))


def count_phrase(label: str, n: int, digits: bool = False) -> str:
    if n == 1:
        return f"{'an' if label[0] in 'aeiou' else 'a'} {label}"
    return f"{n if digits or n >= 10 else NUM_WORDS[n]} {plural(label)}"


def count_listing(rec: dict, digits: bool = False, max_groups: int = 5) -> str:
    groups = label_groups(rec)
    parts = [count_phrase(label, len(objs), digits) for label, objs in groups[:max_groups]]
    if len(groups) > max_groups:
        parts.append("other items")
    return join_and(parts)


def template_image_caption(rec: dict) -> str:
    env = [ENV_PHRASES[e] for e in rec["scene"]["environment"] if e in ENV_PHRASES]
    n = rec["scene"]["object_count"]
    noun = "waste item" if rec["scene"]["is_discarded"] == "yes" else "item"    # COCO objects may be in use
    head = f"{NUM_WORDS[n] if n < 10 else n} {noun}{'s' if n > 1 else ''}"
    where = f" {' and '.join(env[:2])}" if env else ""
    return f"{head.capitalize()}{where}: {count_listing(rec)}."


# ---------------------------------------------------------------------------
# Detailed grounded captions (ZeroWaste): counts per material, position, colour, size,
# coverage, overlap and sorting bin - every fact comes from the annotation or the object's pixels
# ---------------------------------------------------------------------------
SORT_PHRASES = {"recycle_paper": "paper recycling", "recycle_plastic": "plastic recycling",
                "recycle_metal": "metal recycling", "recycle_glass": "glass recycling", "compost": "compost",
                "landfill": "general waste", "hazardous_dropoff": "special collection",
                "needs_review": "a bin decided on review"}
REGION_WORDS = {"top-left": "in the top left", "top-center": "at the top", "top-right": "in the top right",
                "middle-left": "on the left", "center": "in the centre", "middle-right": "on the right",
                "bottom-left": "in the bottom left", "bottom-center": "at the bottom",
                "bottom-right": "in the bottom right"}


def colour_name(rgb_pixels) -> str | None:
    """Name the median colour of a set of RGB pixels (coarse, 11 colour words)."""
    import colorsys
    import statistics
    if len(rgb_pixels) < 20:
        return None
    r, g, b = (statistics.median(p[i] for p in rgb_pixels) / 255 for i in range(3))
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h *= 360
    if v < 0.22:
        return "black"
    if s < 0.18:
        return "white" if v > 0.78 else "grey" if v > 0.4 else "dark grey"
    if h < 15 or h >= 340:
        return "red" if v > 0.45 else "brown"
    if h < 45:
        return "brown" if v < 0.75 or s < 0.45 else "orange"
    if h < 70:
        return "yellow" if s > 0.4 and v > 0.6 else "beige" if v > 0.6 else "brown"
    if h < 170:
        return "green"
    if h < 260:
        return "blue"
    return "purple" if h < 300 else "pink"


def object_colour(img: Image.Image, obj: dict) -> str | None:
    """Median colour inside the object's segmentation polygon (bbox if there is none)."""
    from PIL import ImageDraw
    x, y, w, h = obj["bbox"]
    x0, y0, x1, y1 = int(x), int(y), int(x + w) + 1, int(y + h) + 1
    crop = img.crop((x0, y0, x1, y1)).convert("RGB")
    mask = Image.new("L", crop.size, 0)
    seg = obj.get("segmentation")
    if isinstance(seg, list) and seg:
        d = ImageDraw.Draw(mask)
        for poly in seg:
            if len(poly) >= 6:
                d.polygon([(poly[i] - x0, poly[i + 1] - y0) for i in range(0, len(poly) - 1, 2)], fill=255)
    else:
        mask.paste(255, (0, 0) + crop.size)
    step = max(1, int((crop.width * crop.height / 4000) ** 0.5))     # sample ~4k pixels
    small = crop.resize((max(1, crop.width // step), max(1, crop.height // step)))
    msmall = mask.resize(small.size)
    px = [p for p, m in zip(small.getdata(), msmall.getdata()) if m > 127]
    return colour_name(px)


def where_phrase(objs: list[dict]) -> str:
    regions = Counter_(REGION_WORDS[o["location"]["region"]] for o in objs)
    if len(objs) == 1:
        return next(iter(regions))
    if len(regions) == 1:
        return f"all {next(iter(regions))}"
    if len(regions) >= 4 and len(objs) >= 4:
        return "spread across the belt"
    (top, k), = regions.most_common(1)
    if k > len(objs) / 2:
        return f"mostly {top}"
    return join_and([r for r, _ in regions.most_common(3)])


def _iou_small(a, b) -> float:
    """Intersection over the smaller box (detects one item lying on another)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    return iw * ih / max(1e-6, min(aw * ah, bw * bh))


def detailed_image_caption(rec: dict, img: Image.Image | None = None) -> str:
    """Multi-sentence grounded caption: counts, positions, largest item + colour, coverage, overlap, bins."""
    objs = rec["objects"]
    n = len(objs)
    groups = label_groups(rec)
    env = [ENV_PHRASES[e] for e in rec["scene"]["environment"] if e in ENV_PHRASES]
    setting = "A recycling conveyor belt" if "conveyor_belt" in rec["scene"]["environment"] else "A scene"
    if env and setting == "A scene":
        setting = f"A scene {env[0]}"
    big = max(objs, key=lambda o: o["area"])
    colour = object_colour(img, big) if img is not None else None
    desc = " ".join([big["location"]["size"]] + ([colour] if colour else []) + [object_label(big, rec["source"])])
    where_big = REGION_WORDS[big["location"]["region"]]
    if n == 1:
        s = [f"{setting} with 1 waste item: a {desc} {where_big}."]
    else:
        s = [f"{setting} with {n} waste items: "
             f"{join_and([count_phrase(l, len(g), digits=True) for l, g in groups])}."]
        pos = join_and([f"the {plural(l) if len(g) > 1 else l} {'are' if len(g) > 1 else 'is'} {where_phrase(g)}"
                        for l, g in groups])
        s.append(pos[0].upper() + pos[1:] + ".")
        s.append(f"The largest item is a {desc} {where_big}.")
    cover = 100 * sum(o["area"] for o in objs) / (rec["width"] * rec["height"])
    occluded = sum("possibly_occluded" in o["flags"] for o in objs)
    overlap = any(_iou_small(a["bbox"], b["bbox"]) > 0.3 for i, a in enumerate(objs) for b in objs[i + 1:])
    pct = f"{round(cover / 5) * 5}%" if cover >= 2.5 else "less than 5%"
    density = (f"The items cover about {pct} of the belt" if n > 1 else f"It covers {pct} of the belt")
    if occluded:
        density += f" and {occluded} {'is' if occluded == 1 else 'are'} partly hidden under other items"
    elif overlap:
        density += " and some of them overlap"
    s.append(density + ".")
    conds = Counter_(c for o in objs for c in o["condition"] if c != "intact")
    if conds:
        s.append("Visible condition: " + join_and([f"{k} {c.replace('_', ' ')}" for c, k in conds.most_common(3)]) + ".")
    bins = Counter_(o["sorting_action"] for o in objs)
    if bins:
        s.append("For sorting, " + join_and([f"{k} {'goes' if k == 1 else 'go'} to {SORT_PHRASES.get(b, b)}"
                                             for b, k in bins.most_common()]) + ".")
    return " ".join(s)


def detailed_object_caption(obj: dict, source: str, img: Image.Image | None = None) -> str:
    colour = object_colour(img, obj) if img is not None else None
    words = [obj["location"]["size"]] + ([colour] if colour else []) + \
            [c.replace("_", " ") for c in obj["condition"] if c != "intact"]
    label = object_label(obj, source)
    head = f"{' '.join(words)} {label}"
    tail = f" {REGION_WORDS[obj['location']['region']]} of the belt"
    if "possibly_occluded" in obj["flags"]:
        tail += ", partly hidden"
    return f"{head[0].upper()}{head[1:]}{tail}; it goes to {SORT_PHRASES.get(obj['sorting_action'], obj['sorting_action'])}."


# ---------------------------------------------------------------------------
# Automatic checks (signals for the reviewer, not ground truth)
# ---------------------------------------------------------------------------
def image_checks(text: str, rec: dict) -> dict:
    gt_cats = set(rec["scene"]["category_counts"])
    gt_mats = {o["material"] for o in rec["objects"]} - {"unknown"}
    if "composite" in gt_mats:
        gt_mats |= COMPOSITE_PARTS
    cats, mats = set(mentions(text, CATEGORY_WORDS)), set(mentions(text, MATERIAL_WORDS))
    vague = bool(gt_cats & VAGUE_CATEGORIES)
    specific = gt_cats - VAGUE_CATEGORIES
    # "paper" vs carton etc. are near-synonyms: a mention of a sibling category is not counted as unsupported
    siblings = {"paper": {"carton_box"}, "carton_box": {"paper"}, "cup": {"bottle_jar"}, "bottle_jar": {"cup"},
                "container_packaging": {"cup", "carton_box", "bag_wrapper"}, "glass_fragment": {"bottle_jar", "cup"}}
    unsupported = sorted(c for c in cats - gt_cats if not (siblings.get(c, set()) & gt_cats))
    return {
        "n_words": len(text.split()),
        "empty": not text.strip(),
        "mentioned_categories": sorted(cats),
        "category_recall": round(len(cats & specific) / len(specific), 3) if specific else None,
        "unsupported_categories": [] if vague else unsupported,   # hallucination candidates
        "mentioned_materials": sorted(mats),
        "material_recall": round(len(mats & gt_mats) / len(gt_mats), 3) if gt_mats else None,
        "material_conflict": bool(gt_mats) and bool(mats - gt_mats - {"organic"}) and not (mats & gt_mats),
        "mentions_material": bool(mats),
        "mentions_condition": bool(extract_conditions(text)),
        "brand_or_text": has_brand_or_text(text),
        "mentions_people_animals": any(_has(_normalise(text), w) for w in LIVING_WORDS),
        **quality_checks(text, rec),
    }


# ---------------------------------------------------------------------------
# Caption quality parameters (count accuracy, hallucination, repetition, detail) + 0-100 score
# ---------------------------------------------------------------------------
NUMBER_WORDS = {w: i for i, w in enumerate(NUM_WORDS)} | {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "single": 1, "couple": 2, "pair": 2}
ITEM_NOUNS = {"item", "items", "object", "objects", "things", "pieces", "piece", "waste", "objects"}
VAGUE_QUANTITY = ["several", "many", "various", "a few", "a pile", "a lot", "lots of", "multiple", "some",
                  "a bunch", "a heap", "numerous", "a variety", "assorted"]
NOT_COUNTABLE_AFTER_A = {"pile", "lot", "lots", "bunch", "heap", "variety", "mix", "number", "few", "couple",
                         "stack", "collection", "group", "range", "recycling", "conveyor", "belt", "scene",
                         "close", "photo", "picture", "cluttered", "messy"}
# objects that never appear in the waste labels: a mention is a likely hallucination
NON_WASTE_OBJECTS = ["keyboard", "mouse", "book", "computer", "laptop", "phone", "chair", "car", "house",
                     "tree", "flower", "toy", "shoe", "bed", "window", "door", "television", "clock", "lamp"]
# setting words that contradict a conveyor-belt scene
WRONG_SETTING = {"conveyor_belt": ["table", "floor", "street", "parking", "beach", "grass", "jungle", "room",
                                   "kitchen", "road", "garden", "forest", "sidewalk", "neighborhood", "desk"]}
SETTING_WORDS = {"conveyor_belt": ["conveyor", "belt", "recycling facility", "sorting line", "recycling plant"]}
LOCATION_WORDS = ["left", "right", "top", "bottom", "centre", "center", "middle", "corner", "foreground",
                  "background", "edge", "upper", "lower"]
COLOUR_WORDS = ["black", "white", "grey", "gray", "brown", "red", "orange", "yellow", "beige", "green",
                "blue", "purple", "pink", "transparent", "clear", "silver", "golden", "dark", "light"]


def _num(tok: str) -> int | None:
    return int(tok) if tok.isdigit() else NUMBER_WORDS.get(tok)


def _material_of(window: list[str]) -> str | None:
    t = " ".join(window)
    found = mentions(t, MATERIAL_WORDS)
    return found[0] if found else None


def stated_counts(text: str) -> tuple[int | None, dict[str, int]]:
    """Counts written in a caption -> (total item count, {material: count}).

    "9 waste items: 7 pieces of cardboard and two rigid plastic items" -> (9, {paper_cardboard: 7, plastic: 2})
    A material's count is summed inside the first sentence that gives it ("3 pieces of soft plastic film and
    a rigid plastic item" -> plastic 4); later sentences repeating it ("7 go to paper recycling") are ignored.
    """
    total, mats, mat_sentence = None, defaultdict(int), {}
    for si, sent in enumerate(re.split(r"(?<=[.!?;])\s+", _normalise(text))):
        toks = re.findall(r"\d+|[a-z]+|[,:;]", sent)
        for i, tok in enumerate(toks):
            n = 1 if tok in ("a", "an", "one") else _num(tok)
            if n is None:
                continue
            window = []
            for w in toks[i + 1:i + 8]:
                if w in (",", ":", ";", "and", "with", "go", "goes", "is", "are") or _num(w) is not None:
                    break
                window.append(w)
            if not window or (tok in ("a", "an") and set(window[:3]) & NOT_COUNTABLE_AFTER_A):
                continue
            mat = _material_of(window)
            if mat:
                if mat_sentence.setdefault(mat, si) == si:
                    mats[mat] += n
            elif total is None and tok not in ("a", "an") and set(window) & ITEM_NOUNS:
                total = n
    return total, dict(mats)


def repetition_ratio(text: str) -> float:
    """Share of word 4-grams that are repeats (0 = none; "a plastic cover, a plastic cover, ..." -> high)."""
    w = re.findall(r"[a-z0-9]+", text.lower())
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return round(1 - len(set(grams)) / len(grams), 3) if grams else 0.0


def quality_checks(text: str, rec: dict) -> dict:
    t = _normalise(text)
    gt_total = len(rec["objects"])
    gt_mats = Counter_(o["material"] for o in rec["objects"] if o["material"] not in ("unknown", "composite"))
    total, mats = stated_counts(text)
    implied = sum(mats.values()) if mats and set(mats) >= set(gt_mats) else None
    total_used = total if total is not None else implied
    mat_correct = sum(mats.get(m) == k for m, k in gt_mats.items())
    env = rec["scene"]["environment"]
    wrong_setting = sorted({w for e in env for w in WRONG_SETTING.get(e, []) if _has(t, w)})
    non_waste = sorted(w for w in NON_WASTE_OBJECTS if _has(t, w))
    living = rec["source"] != "coco" and any(_has(t, w) for w in LIVING_WORDS)
    rep = repetition_ratio(text)
    details = {
        "count": total is not None or bool(mats),
        "material": bool(mentions(text, MATERIAL_WORDS)),
        "location": any(_has(t, w) for w in LOCATION_WORDS),
        "colour": any(_has(t, w) for w in COLOUR_WORDS),
        "condition": bool(extract_conditions(text)),
        "setting": any(_has(t, w) for e in env for w in SETTING_WORDS.get(e, [])) or (
            not any(e in SETTING_WORDS for e in env) and any(_has(t, w) for w in ENV_WORDS)),
    }
    return {
        "stated_total": total,
        "gt_total": gt_total,
        "count_correct": None if total_used is None else total_used == gt_total,
        "count_error": None if total_used is None else abs(total_used - gt_total),
        "stated_material_counts": mats,
        "material_count_accuracy": round(mat_correct / len(gt_mats), 3) if gt_mats else None,
        "vague_quantity": any(_has(t, w) for w in VAGUE_QUANTITY) and total_used is None,
        "non_waste_objects": non_waste,
        "wrong_setting": wrong_setting,
        "repetition": rep,
        "repetitive": rep > 0.15,
        "details": sorted(k for k, v in details.items() if v),
        "detail_score": round(sum(details.values()) / len(details), 3),
    }


ENV_WORDS = ["sand", "beach", "grass", "street", "pavement", "road", "water", "table", "floor", "ground",
             "indoors", "room", "pile", "background", "soil", "dirt", "plants", "bushes"]
DEFAULT_QUALITY_WEIGHTS = {"count": 25, "material": 20, "no_hallucination": 20, "detail": 15, "clip": 10,
                           "fluency": 10}


def is_hallucinating(a: dict) -> bool:
    return bool(a.get("unsupported_categories") or a.get("mentions_people_animals") or a.get("non_waste_objects")
                or a.get("material_conflict") or a.get("wrong_setting"))


def quality_score(a: dict, clip: float | None, weights: dict | None = None) -> dict:
    """Combine the checks into a 0-100 caption quality score (weights in captioning.yaml -> quality_score).

    count            50% total count correct + 50% per-material counts correct (0 if no count is given)
    material         share of the annotated materials that the caption names
    no_hallucination 1 if no people/animals, non-waste objects, unsupported categories, wrong setting or
                     material conflict; 0 otherwise
    detail           share of {count, material, location, colour, condition, setting} mentioned
    clip             CLIPScore mapped from [0.5, 0.9] to [0, 1]
    fluency          1 - 3 x repeated-4-gram ratio (empty caption -> 0)
    """
    w = weights or DEFAULT_QUALITY_WEIGHTS
    if a.get("empty"):
        return {"quality_score": 0.0, "quality_parts": {k: 0.0 for k in w}}
    parts = {
        "count": 0.5 * bool(a.get("count_correct")) + 0.5 * (a.get("material_count_accuracy") or 0.0),
        "material": a.get("material_recall") if a.get("material_recall") is not None else 1.0,
        "no_hallucination": 0.0 if is_hallucinating(a) else 1.0,
        "detail": a.get("detail_score", 0.0),
        "clip": min(1.0, max(0.0, ((clip or 0.0) - 0.5) / 0.4)),
        "fluency": max(0.0, 1 - 3 * a.get("repetition", 0.0)),
    }
    score = sum(w[k] * parts[k] for k in w) / sum(w.values()) * 100
    return {"quality_score": round(score, 1), "quality_parts": {k: round(v, 3) for k, v in parts.items()}}


def object_checks(text: str, obj: dict) -> dict:
    mats = set(mentions(text, MATERIAL_WORDS))
    cats = set(mentions(text, CATEGORY_WORDS))
    gt_mat = obj["material"]
    known = gt_mat not in ("unknown", "composite")     # composite items legitimately mention several
    return {
        "n_words": len(text.split()),
        "empty": not text.strip(),
        "mentioned_materials": sorted(mats),
        "material_match": (gt_mat in mats) if known else None,
        "material_conflict": known and bool(mats - {gt_mat}) and gt_mat not in mats,
        "category_match": (obj["category"] in cats) if obj["category"] not in VAGUE_CATEGORIES else None,
        "mentions_condition": bool(extract_conditions(text)),
        "brand_or_text": has_brand_or_text(text),
    }


# ---------------------------------------------------------------------------
# Loading generations + reviews and merging them into records
# ---------------------------------------------------------------------------
def load_generations(models: list[str] | None = None) -> list[dict]:
    rows = []
    for p in sorted(CAPTIONS.glob("*.jsonl")):
        if models is None or p.stem in models:
            rows += read_jsonl(p)
    return rows


def load_review_events() -> list[dict]:
    p = REVIEWS / "reviews.jsonl"
    return read_jsonl(p) if p.exists() else []


def append_review_event(event: dict) -> None:
    with open(REVIEWS / "reviews.jsonl", "a") as f:
        f.write(json.dumps(event) + "\n")


CAPTION_KEYS = ("caption_id", "text", "model", "hf_id", "prompt_id", "prompt", "clip_score", "created_at")


def merge_records(records: list[dict], generations: list[dict], events: list[dict]) -> list[dict]:
    """Attach caption candidates (image + object level) and apply review events, in place."""
    by_img, by_obj = defaultdict(list), defaultdict(list)
    for g in generations:
        (by_obj[g["object_id"]] if g["level"] == "object" else by_img[g["image_id"]]).append(g)
    cap_events, obj_events, img_events = {}, {}, {}
    for e in events:                     # later events win
        if e["type"] == "caption":
            cap_events.setdefault(e["caption_id"], {}).update(e)
        elif e["type"] == "object":
            obj_events.setdefault(e["object_id"], {}).update(e["fields"])
        elif e["type"] == "image":
            img_events.setdefault(e["image_id"], {}).update(e)
    human_new = defaultdict(list)        # captions written from scratch by a reviewer
    for cid, e in cap_events.items():
        if e.get("new"):
            human_new[e.get("object_id") or e["image_id"]].append(e)

    def build(g, review):
        c = {k: g.get(k) for k in CAPTION_KEYS}
        c.update(status="pending", review_score=None, hallucinated_objects=[], reviewer=None)
        if g.get("proposed"):
            c["proposed"] = g["proposed"]
        if review:
            if review.get("text") and review["text"] != c["text"]:
                c["original_text"], c["text"] = c["text"], review["text"]
            for k in ("status", "review_score", "hallucinated_objects", "reviewer", "updated_at"):
                if k in review:
                    c[k] = review[k]
        return c

    weights = load_caption_cfg().get("quality_score", {}).get("weights")
    for rec in records:
        caps = [build(g, cap_events.get(g["caption_id"])) for g in by_img.get(rec["image_id"], [])]
        caps += [build({**e, "model": "human"}, e) for e in human_new.get(rec["image_id"], [])]
        for c in caps:
            c["auto_checks"] = image_checks(c["text"], rec)
            c["auto_checks"].update(quality_score(c["auto_checks"], c["clip_score"], weights))
        rec["captions"] = caps
        for o in rec["objects"]:
            if o["object_id"] in obj_events:
                o.update({k: v for k, v in obj_events[o["object_id"]].items() if k != "caption"})
            ocaps = [build(g, cap_events.get(g["caption_id"])) for g in by_obj.get(o["object_id"], [])]
            ocaps += [build({**e, "model": "human"}, e) for e in human_new.get(o["object_id"], [])]
            for c in ocaps:
                c["auto_checks"] = object_checks(c["text"], o)
            o["caption_candidates"] = ocaps
            approved = [c for c in ocaps if c["status"] in ("approved", "edited")]
            o["caption"] = approved[-1]["text"] if approved else None
        if any(o["object_id"] in obj_events for o in rec["objects"]):   # a reviewer re-labelled objects
            counts = defaultdict(int)
            for o in rec["objects"]:
                counts[o["category"]] += 1
            rec["scene"]["category_counts"] = dict(counts)
        ev = img_events.get(rec["image_id"])
        if ev:
            for k, v in ev.get("scene", {}).items():
                rec["scene"][k] = v
            if "status" in ev:
                rec["review"] = {"status": ev["status"], "reviewer": ev.get("reviewer"),
                                 "notes": ev.get("notes"), "updated_at": ev.get("updated_at")}
    return records


def read_jsonl_safe(path: Path) -> list[dict]:
    return read_jsonl(path) if path.exists() else []
