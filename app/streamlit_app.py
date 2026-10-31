"""CaptionGene - Waste Caption Dataset Curation App (Week 2 version).

    streamlit run app/streamlit_app.py

Reads the annotation records (data/processed/manifest_all.jsonl), the raw caption candidates
(data/captions/*.jsonl) and the review log (data/reviews/reviews.jsonl). Every reviewer action is
appended to the review log; nothing in data/processed is edited by the app. Run
scripts/10_merge_captions.py to write the reviewed state back into the dataset files.
"""
import csv
import io
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from captionlib import (CAPTIONS, append_review_event, crop_object, fill_prompt, load_caption_cfg,  # noqa: E402
                        load_generations, load_review_events, make_caption_row, merge_records, read_jsonl_safe)
from common import PROCESSED, ROOT, load_taxonomy, read_jsonl  # noqa: E402

st.set_page_config(page_title="CaptionGene curation", layout="wide")
CFG = load_caption_cfg()
TAX = load_taxonomy()
STATUS_ICON = {"pending": "⏳", "approved": "✅", "edited": "✏️", "rejected": "❌"}

if "ver" not in st.session_state:
    st.session_state.ver = 0
if "idx" not in st.session_state:
    st.session_state.idx = 0


def bump():
    st.session_state.ver += 1


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- data
@st.cache_resource(show_spinner="Loading annotations ...", max_entries=2)
def state(ver):
    """Records with candidates + review events merged in. `ver` invalidates the cache after writes.
    cache_resource (not cache_data) so the ~2.4k records are not copied on every rerun."""
    recs = read_jsonl(PROCESSED / "manifest_all.jsonl")
    merge_records(recs, load_generations(), load_review_events())
    return recs


@st.cache_resource(show_spinner="Loading model ...")
def captioner(name):
    from vlm import load_captioner
    return load_captioner(name, {**CFG["models"][name],
                                 "no_repeat_ngram_size": CFG["generation"]["no_repeat_ngram_size"]})


@st.cache_resource(show_spinner="Loading CLIP ...")
def clip_scorer():
    from vlm import ClipScorer
    return ClipScorer()


def load_image(rec):
    return Image.open(ROOT / "data" / rec["file_name"]).convert("RGB")


def draw_overlay(img, objects, highlight=None, masks=False):
    img = img.copy()
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d, do = ImageDraw.Draw(img), ImageDraw.Draw(over)
    lw = max(2, round(max(img.size) / 400))
    for o in objects:
        hl = o["object_id"] == highlight
        color = (255, 215, 0) if hl else (255, 60, 60)
        if masks and o.get("segmentation"):
            for poly in o["segmentation"]:
                if len(poly) >= 6:
                    do.polygon(list(zip(poly[::2], poly[1::2])), fill=color + (90,))
        x, y, w, h = o["bbox"]
        d.rectangle([x, y, x + w, y + h], outline=color, width=lw * (2 if hl else 1))
        d.text((x + 3, y + 2), o["object_id"].rsplit("_", 1)[-1], fill=color)
    return Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")


def best_clip(rec):
    s = [c["clip_score"] for c in rec["captions"] if c["clip_score"] is not None]
    return max(s) if s else None


def generate_into_store(model, rec, obj, prompt_id, prompt, n):
    """Run a VLM on one image / object, append candidates to data/captions/<model>.jsonl, score with CLIP."""
    mcfg = CFG["models"][model]
    img = load_image(rec)
    crop_box = None
    if obj:
        img, crop_box = crop_object(img, obj["bbox"], CFG["object_crop"]["context_pad"], CFG["object_crop"]["min_side"])
    level = "object" if obj else "image"
    g = CFG["generation"]
    t = time.time()
    outs = captioner(model).generate([img], [prompt], g[f"max_new_tokens_{level}"], n, g["temperature"])[0]
    runtime = (time.time() - t) / len(outs)
    path = CAPTIONS / f"{model}.jsonl"
    existing = {r["caption_id"] for r in read_jsonl_safe(path)}
    target = obj["object_id"] if obj else rec["image_id"]
    gen = {"max_new_tokens": g[f"max_new_tokens_{level}"], "num_candidates": n, "do_sample": n > 1,
           "temperature": g["temperature"] if n > 1 else None}
    rows, k = [], 0
    for raw in outs:
        while f"{target}__{model}__{prompt_id}__{k}" in existing:
            k += 1
        rows.append(make_caption_row(model, mcfg, rec, obj, prompt_id, prompt, raw, k, runtime, gen, crop_box))
        existing.add(rows[-1]["caption_id"])
    try:
        scores = clip_scorer().score([img] * len(rows), [r["text"] or " " for r in rows])
        for r, s in zip(rows, scores):
            r["clip_score"] = s
    except Exception as e:  # CLIP not downloaded / no disk: keep the caption, score later with script 09
        st.warning(f"CLIPScore skipped: {e}")
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    bump()
    return rows


# ---------------------------------------------------------------- sidebar: filters
recs = state(st.session_state.ver)
st.sidebar.title("♻️ CaptionGene")
reviewer = st.sidebar.text_input("Reviewer name", value=st.session_state.get("reviewer", ""), key="reviewer")

pilot_path = CAPTIONS / "pilot_ids.txt"
pilot = set(pilot_path.read_text().split()) if pilot_path.exists() else set()
with st.sidebar.expander("Filters", expanded=True):
    only_captioned = st.checkbox("Only images with captions", value=bool(pilot))
    splits = st.multiselect("Split", ["train", "val", "test"])
    sources = st.multiselect("Source", ["taco", "zerowaste", "coco"])
    cats = st.multiselect("Contains category", list(TAX["categories"]))
    statuses = st.multiselect("Image review status", ["unreviewed", "in_progress", "approved", "rejected"])
    has_pending = st.checkbox("Has pending captions")
    flag_unsupported = st.checkbox("Auto-check: unsupported object mentioned")
    flag_brand = st.checkbox("Auto-check: brand / label text")
    min_clip = st.slider("Min best CLIPScore", 0.0, 1.2, 0.0, 0.05)


def keep(r):
    if only_captioned and not r["captions"]:
        return False
    if splits and r["split"] not in splits or sources and r["source"] not in sources:
        return False
    if cats and not set(cats) & set(r["scene"]["category_counts"]):
        return False
    if statuses and r["review"]["status"] not in statuses:
        return False
    if has_pending and not any(c["status"] == "pending" for c in r["captions"]):
        return False
    if flag_unsupported and not any(c["auto_checks"].get("unsupported_categories") for c in r["captions"]):
        return False
    if flag_brand and not any(c["auto_checks"]["brand_or_text"] for c in r["captions"]):
        return False
    if min_clip > 0 and (best_clip(r) or 0) < min_clip:
        return False
    return True


view = [r for r in recs if keep(r)]
st.sidebar.caption(f"{len(view)} of {len(recs)} images match")
if not view:
    st.info("No image matches the filters.")
    st.stop()
st.session_state.idx = min(st.session_state.idx, len(view) - 1)
c1, c2 = st.sidebar.columns(2)
if c1.button("◀ Prev", use_container_width=True):
    st.session_state.idx = (st.session_state.idx - 1) % len(view)
if c2.button("Next ▶", use_container_width=True):
    st.session_state.idx = (st.session_state.idx + 1) % len(view)
ids = [r["image_id"] for r in view]
chosen = st.sidebar.selectbox("Image", ids, index=st.session_state.idx)
st.session_state.idx = ids.index(chosen)
rec = view[st.session_state.idx]

# ---------------------------------------------------------------- main
st.title(f"{rec['image_id']}")
tab_review, tab_objects, tab_generate, tab_stats, tab_export = st.tabs(
    ["🖼️ Image captions", "📦 Objects", "⚙️ Generate", "📊 Statistics", "⬇️ Export"])


def caption_card(c, rec, obj=None, key_prefix=""):
    """One candidate: text, scores, auto-checks, and approve / edit / reject controls."""
    a = c["auto_checks"]
    badges = [f"CLIP **{c['clip_score']:.2f}**" if c["clip_score"] is not None else "CLIP –"]
    if a.get("mentioned_materials"):
        badges.append("materials: " + ", ".join(a["mentioned_materials"]))
    if a.get("unsupported_categories"):
        badges.append("⚠️ unsupported: " + ", ".join(a["unsupported_categories"]))
    if a.get("material_conflict"):
        badges.append("⚠️ material conflict")
    if a["brand_or_text"]:
        badges.append("⚠️ brand/text")
    if a.get("mentions_people_animals") and rec["source"] != "coco":
        badges.append("⚠️ people/animals")
    if c.get("proposed", {}).get("condition"):
        badges.append("proposed condition: " + ", ".join(c["proposed"]["condition"]))
    st.markdown(f"{STATUS_ICON[c['status']]} {c['text'] or '*(empty)*'}")
    st.caption(" · ".join(badges) + (f" · score {c['review_score']}/5" if c["review_score"] else ""))
    with st.expander("Review", expanded=False):
        k = key_prefix + c["caption_id"]
        text = st.text_area("Caption", c["text"], key=f"t_{k}", height=80)
        score = st.slider("Review score (5 = complete and correct, 1 = wrong / hallucinated)", 1, 5,
                          c["review_score"] or 3, key=f"s_{k}")
        known = sorted(set(a.get("mentioned_categories", [])) | set(a.get("unsupported_categories", [])))
        halluc = st.multiselect("Hallucinated objects", list(TAX["categories"]),
                                default=c["hallucinated_objects"] or [], key=f"h_{k}",
                                help=f"auto-check mentions: {known}")
        b1, b2, b3 = st.columns(3)
        base = {"type": "caption", "caption_id": c["caption_id"], "image_id": rec["image_id"],
                "object_id": obj["object_id"] if obj else None, "review_score": score,
                "hallucinated_objects": halluc, "reviewer": reviewer or None, "updated_at": now()}
        if b1.button("✅ Approve", key=f"a_{k}"):
            ev = {**base, "status": "approved"}
            if text != c["text"]:
                ev.update(status="edited", text=text)
            append_review_event(ev)
            bump(); st.rerun()
        if b2.button("💾 Save edit", key=f"e_{k}", disabled=text == c["text"]):
            append_review_event({**base, "status": "edited", "text": text})
            bump(); st.rerun()
        if b3.button("❌ Reject", key=f"r_{k}"):
            append_review_event({**base, "status": "rejected", "review_score": 1})
            bump(); st.rerun()


def new_caption_form(rec, obj=None):
    target = obj["object_id"] if obj else rec["image_id"]
    with st.form(f"new_{target}", clear_on_submit=True):
        text = st.text_input("Write a caption yourself")
        if st.form_submit_button("Add as approved human caption") and text.strip():
            append_review_event({"type": "caption", "new": True, "caption_id": f"{target}__human__{int(time.time())}",
                                 "image_id": rec["image_id"], "object_id": obj["object_id"] if obj else None,
                                 "level": "object" if obj else "image", "text": text.strip(), "status": "approved",
                                 "review_score": 5, "reviewer": reviewer or None, "updated_at": now()})
            bump(); st.rerun()


def regenerate_button(model, rec, obj=None, key=""):
    if model in ("template", "human") or model not in CFG["models"]:
        return
    mcfg = CFG["models"][model]
    if st.button(f"🔁 Regenerate ({model})", key=f"regen_{key}{model}"):
        pid = mcfg["prompts"]["object" if obj else "image"]
        prompt = fill_prompt(CFG["prompts"][pid], rec, obj, TAX)
        with st.spinner(f"Running {model} ..."):
            generate_into_store(model, rec, obj, pid, prompt, n=2)
        st.rerun()


# ---- tab 1: image captions, side by side
with tab_review:
    left, right = st.columns([3, 2])
    with left:
        o1, o2 = st.columns(2)
        show_boxes = o1.toggle("Boxes", True)
        show_masks = o2.toggle("Masks", False)
        img = load_image(rec)
        st.image(draw_overlay(img, rec["objects"], masks=show_masks) if show_boxes else img, use_container_width=True)
    with right:
        sc = rec["scene"]
        st.markdown(f"**Source** {rec['source']} · **split** {rec['split']} · {rec['width']}×{rec['height']} px")
        st.markdown(f"**Objects** {sc['object_count']} ({sc['clutter_level']}) · "
                    + ", ".join(f"{k} ×{v}" for k, v in sc["category_counts"].items()))
        st.markdown(f"**Environment** {', '.join(sc['environment'])} · **discarded** {sc['is_discarded']}")
        if rec.get("quality", {}).get("flags"):
            st.markdown(f"**Quality flags** {', '.join(rec['quality']['flags'])}")
        st.caption(f"{rec['license']} · [source]({rec['url']})")
        with st.form(f"scene_{rec['image_id']}"):
            env = st.multiselect("Scene environment", TAX["scene_environments"], default=sc["environment"])
            disc = st.radio("Items are discarded?", ["yes", "no", "unknown"], horizontal=True,
                            index=["yes", "no", "unknown"].index(sc["is_discarded"]))
            desc = st.text_area("Scene context (free text)", sc.get("description") or "", height=70)
            status = st.selectbox("Image review status", ["unreviewed", "in_progress", "approved", "rejected"],
                                  index=["unreviewed", "in_progress", "approved", "rejected"].index(rec["review"]["status"]))
            notes = st.text_input("Notes", rec["review"].get("notes") or "")
            if st.form_submit_button("Save scene fields"):
                append_review_event({"type": "image", "image_id": rec["image_id"], "status": status, "notes": notes or None,
                                     "scene": {"environment": env, "is_discarded": disc, "description": desc or None},
                                     "reviewer": reviewer or None, "updated_at": now()})
                bump(); st.rerun()

    st.subheader("Image-level captions")
    present = sorted({c["model"] for c in rec["captions"]})
    default_pair = [m for m in CFG["compare_pair"] if m in present] or \
                   [m for m in CFG["pilot_compare_pair"] if m in present] or present[:2]
    pair = st.multiselect("Models side by side", sorted(set(present) | set(CFG["models"])), default=default_pair[:2],
                          max_selections=3)
    cols = st.columns(max(1, len(pair)))
    for col, m in zip(cols, pair):
        with col:
            st.markdown(f"#### {CFG['models'].get(m, {}).get('label', m)}")
            caps = [c for c in rec["captions"] if c["model"] == m]
            if not caps:
                st.caption("no captions yet")
            for c in caps:
                caption_card(c, rec)
            regenerate_button(m, rec)
    others = [c for c in rec["captions"] if c["model"] not in pair]
    if others and st.toggle(f"Show other candidates ({len(others)})"):
        for c in others:
            st.markdown(f"**{c['model']}**")
            caption_card(c, rec, key_prefix="o_")
    new_caption_form(rec)

# ---- tab 2: objects with structured fields
with tab_objects:
    objs = sorted(rec["objects"], key=lambda o: -o["bbox"][2] * o["bbox"][3])
    sel = st.selectbox("Object", [o["object_id"] for o in objs],
                       format_func=lambda i: next(f"{i} - {o['source_category']} ({o['location']['size']}, "
                                                  f"{o['location']['region']}){' [tiny]' if 'tiny' in o['flags'] else ''}"
                                                  for o in objs if o["object_id"] == i))
    obj = next(o for o in objs if o["object_id"] == sel)
    img = load_image(rec)
    a, b = st.columns([2, 3])
    with a:
        st.image(draw_overlay(img, rec["objects"], highlight=sel, masks=True), use_container_width=True)
        crop, _ = crop_object(img, obj["bbox"], CFG["object_crop"]["context_pad"], CFG["object_crop"]["min_side"])
        st.image(crop, caption="crop shown to the VLM", width=260)
    with b:
        st.markdown(f"**Source label** {obj['source_category']} · **flags** {', '.join(obj['flags']) or '–'} · "
                    f"bbox {obj['bbox']}")
        proposed = sorted({k for c in obj["caption_candidates"] for k in c.get("proposed", {}).get("condition", [])
                           if c["status"] != "rejected"})
        with st.form(f"obj_{sel}"):
            f1, f2 = st.columns(2)
            cat = f1.selectbox("Object category", list(TAX["categories"]), index=list(TAX["categories"]).index(obj["category"]))
            mat = f2.selectbox("Material", list(TAX["materials"]), index=list(TAX["materials"]).index(obj["material"]))
            cond = st.multiselect("Condition", TAX["conditions"], default=obj["condition"],
                                  help=f"VLM proposed: {', '.join(proposed) or 'nothing'}")
            act = st.selectbox("Sorting action", list(TAX["sorting_actions"]),
                               index=list(TAX["sorting_actions"]).index(obj["sorting_action"]))
            same = sum(1 for o in rec["objects"] if o["category"] == obj["category"])
            st.caption(f"Count of '{obj['category']}' in this image: {same}")
            s1, s2 = st.columns(2)
            save = s1.form_submit_button("Save structured fields")
            take = s2.form_submit_button(f"Accept VLM conditions ({', '.join(proposed) or '–'})", disabled=not proposed)
            if save or take:
                fields = {"category": cat, "material": mat, "sorting_action": act,
                          "condition": sorted(set(cond) | (set(proposed) if take else set()))}
                append_review_event({"type": "object", "object_id": sel, "image_id": rec["image_id"], "fields": fields,
                                     "reviewer": reviewer or None, "updated_at": now()})
                bump(); st.rerun()
        st.markdown("##### Object caption candidates")
        if not obj["caption_candidates"]:
            st.caption("No object captions (tiny objects and objects beyond the per-image limit are skipped).")
        for m in sorted({c["model"] for c in obj["caption_candidates"]}):
            st.markdown(f"**{m}**")
            for c in obj["caption_candidates"]:
                if c["model"] == m:
                    caption_card(c, rec, obj)
            regenerate_button(m, rec, obj, key=sel)
        new_caption_form(rec, obj)

# ---- tab 3: generate with any model + custom prompt
with tab_generate:
    neural = [m for m, c in CFG["models"].items() if c["backend"] != "template"]
    g1, g2, g3 = st.columns([2, 1, 1])
    model = g1.selectbox("Captioning model", neural, format_func=lambda m: f"{m} - {CFG['models'][m]['label']}",
                         index=neural.index("smolvlm-256m") if "smolvlm-256m" in neural else 0)
    level = g2.radio("Level", ["image", "object"], horizontal=True)
    n = g3.number_input("Candidates", 1, 5, 1, help=">1 uses sampling (temperature from the config)")
    obj = None
    if level == "object":
        oid = st.selectbox("Object", [o["object_id"] for o in rec["objects"]], key="gen_obj")
        obj = next(o for o in rec["objects"] if o["object_id"] == oid)
    mcfg = CFG["models"][model]
    pid = st.selectbox("Prompt template", list(CFG["prompts"]), index=list(CFG["prompts"]).index(mcfg["prompts"][level]))
    st.caption("Placeholders {label}, {material}, {region}, {conditions} are filled from the object's annotation.")
    tmpl = st.text_area("Prompt", CFG["prompts"][pid].strip(), height=130, key=f"tmpl_{pid}")
    prompt = fill_prompt(tmpl, rec, obj, TAX) if obj else tmpl.strip()
    with st.expander("Prompt sent to the model"):
        st.code(prompt or "(empty prompt: unconditional caption)")
    if mcfg["hf_id"] and mcfg.get("dtype") in ("float16", "bfloat16") and not __import__("torch").cuda.is_available():
        st.warning(f"{model} is a large model meant for a GPU; on this machine use blip-base or smolvlm-256m.")
    if st.button("Generate", type="primary"):
        with st.spinner(f"Running {model} ..."):
            rows = generate_into_store(model, rec, obj, pid if tmpl.strip() == CFG["prompts"][pid].strip() else "custom",
                                       prompt, int(n))
        for r in rows:
            st.success(f"{r['text']}  (CLIPScore {r['clip_score']})")
        st.caption("Saved as pending candidates; review them in the other tabs.")

# ---- tab 4: statistics
with tab_stats:
    allcaps = [(r, None, c) for r in recs for c in r["captions"]] + \
              [(r, o, c) for r in recs for o in r["objects"] for c in o["caption_candidates"]]
    st_counts = Counter(c["status"] for *_, c in allcaps)
    reviewed = st_counts["approved"] + st_counts["edited"] + st_counts["rejected"]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Images", len(recs))
    m2.metric("Images with captions", sum(1 for r in recs if r["captions"]))
    m3.metric("Caption candidates", len(allcaps))
    m4.metric("Reviewed", reviewed)
    m5.metric("Approval rate", f"{100 * (st_counts['approved'] + st_counts['edited']) / reviewed:.0f}%" if reviewed else "–")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**Objects per category (all images)**")
        st.bar_chart(pd.Series(Counter(o["category"] for r in recs for o in r["objects"])).sort_values(ascending=False))
        st.markdown("**Review status of candidates**")
        st.bar_chart(pd.Series(st_counts))
    with s2:
        st.markdown("**Object captions per category** (candidates / approved)")
        per = pd.DataFrame([{"category": o["category"], "approved": c["status"] in ("approved", "edited")}
                            for r, o, c in allcaps if o is not None])
        if len(per):
            st.bar_chart(per.groupby("category").agg(candidates=("approved", "size"), approved=("approved", "sum")))
        st.markdown("**Mean CLIPScore per model and level**")
        df = pd.DataFrame([{"model": c["model"], "level": "object" if o else "image", "clip": c["clip_score"]}
                           for r, o, c in allcaps if c["clip_score"] is not None])
        if len(df):
            st.dataframe(df.pivot_table(index="model", columns="level", values="clip", aggfunc="mean").round(3))

# ---- tab 5: export approved annotations
with tab_export:
    ok_status = st.multiselect("Include caption status", ["approved", "edited", "pending", "rejected"],
                               default=["approved", "edited"])
    ex_splits = st.multiselect("Splits", ["train", "val", "test"], default=["train", "val", "test"])
    rows_img, rows_obj = [], []
    for r in recs:
        if r["split"] not in ex_splits:
            continue
        for c in r["captions"]:
            if c["status"] in ok_status:
                rows_img.append({"image_id": r["image_id"], "file_name": r["file_name"], "split": r["split"],
                                 "source": r["source"], "caption": c["text"], "model": c["model"], "status": c["status"],
                                 "clip_score": c["clip_score"], "review_score": c["review_score"],
                                 "environment": "|".join(r["scene"]["environment"]),
                                 "clutter_level": r["scene"]["clutter_level"], "object_count": r["scene"]["object_count"]})
        for o in r["objects"]:
            for c in o["caption_candidates"]:
                if c["status"] in ok_status:
                    rows_obj.append({"image_id": r["image_id"], "object_id": o["object_id"], "split": r["split"],
                                     "caption": c["text"], "model": c["model"], "status": c["status"],
                                     "category": o["category"], "material": o["material"],
                                     "condition": "|".join(o["condition"]), "sorting_action": o["sorting_action"],
                                     "bbox": o["bbox"], "region": o["location"]["region"], "clip_score": c["clip_score"]})
    st.write(f"{len(rows_img)} image captions, {len(rows_obj)} object captions match.")

    def to_csv(rows):
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return buf.getvalue()

    jsonl = "\n".join(json.dumps({**r, "level": "image"}) for r in rows_img) + "\n" + \
            "\n".join(json.dumps({**r, "level": "object"}) for r in rows_obj)
    e1, e2, e3 = st.columns(3)
    e1.download_button("JSONL (image + object)", jsonl, "captiongene_captions.jsonl", "application/json")
    e2.download_button("CSV image captions", to_csv(rows_img), "captiongene_image_captions.csv", "text/csv")
    e3.download_button("CSV object captions", to_csv(rows_obj), "captiongene_object_captions.csv", "text/csv")
    if rows_img:
        st.dataframe(pd.DataFrame(rows_img).head(50), use_container_width=True)
