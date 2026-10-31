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
from collections import defaultdict
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


def template_image_caption(rec: dict) -> str:
    groups = defaultdict(int)
    for o in rec["objects"]:
        groups[object_label(o, rec["source"])] += 1
    parts = []
    for label, n in sorted(groups.items(), key=lambda kv: -kv[1])[:5]:
        num = NUM_WORDS[n] if n < 10 else str(n)
        parts.append(f"{num} {plural(label)}" if n > 1 else f"{'an' if label[0] in 'aeiou' else 'a'} {label}")
    if len(groups) > 5:
        parts.append("other items")
    env = [ENV_PHRASES[e] for e in rec["scene"]["environment"] if e in ENV_PHRASES]
    n = rec["scene"]["object_count"]
    noun = "waste item" if rec["scene"]["is_discarded"] == "yes" else "item"    # COCO objects may be in use
    head = f"{NUM_WORDS[n] if n < 10 else n} {noun}{'s' if n > 1 else ''}"
    where = f" {' and '.join(env[:2])}" if env else ""
    listing = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
    return f"{head.capitalize()}{where}: {listing}."


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
    }


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

    for rec in records:
        caps = [build(g, cap_events.get(g["caption_id"])) for g in by_img.get(rec["image_id"], [])]
        caps += [build({**e, "model": "human"}, e) for e in human_new.get(rec["image_id"], [])]
        for c in caps:
            c["auto_checks"] = image_checks(c["text"], rec)
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
