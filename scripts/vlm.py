"""Thin wrappers around Hugging Face vision-language models used for captioning.

Two backends cover every model in configs/captioning.yaml:
  * BlipCaptioner  - BLIP / BLIP-2: image + optional text prefix -> continuation
  * ChatCaptioner  - instruction-tuned VLMs (Qwen2.5-VL, LLaVA, SmolVLM) via the chat template

Both expose  generate(images, prompts, max_new_tokens, n) -> list[list[str]]
(one list of n candidate strings per input image).
"""
from __future__ import annotations

import re

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BaseCaptioner:
    def __init__(self, name: str, cfg: dict, device: str | None = None):
        self.name, self.cfg = name, cfg
        self.hf_id = cfg["hf_id"]
        self.device = device or pick_device()
        dtype = DTYPES[cfg.get("dtype", "float32")]
        if self.device == "cpu" or (self.device == "mps" and dtype == torch.bfloat16):
            dtype = torch.float32       # half precision is slow or unsupported there
        self.dtype = dtype

    def _gen_kwargs(self, max_new_tokens: int, n: int, temperature: float) -> dict:
        kw = {"max_new_tokens": max_new_tokens, "num_return_sequences": n}
        if n > 1:
            kw.update(do_sample=True, temperature=temperature, top_p=0.9)
        else:
            kw.update(do_sample=False, num_beams=1)
        return kw

    @staticmethod
    def _group(texts: list[str], n: int) -> list[list[str]]:
        return [texts[i:i + n] for i in range(0, len(texts), n)]


class BlipCaptioner(BaseCaptioner):
    def __init__(self, name, cfg, device=None):
        super().__init__(name, cfg, device)
        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self.model = AutoModelForImageTextToText.from_pretrained(self.hf_id, dtype=self.dtype)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def generate(self, images: list[Image.Image], prompts: list[str], max_new_tokens=40, n=1,
                 temperature=0.7) -> list[list[str]]:
        kw = {"images": images, "return_tensors": "pt"}
        if any(prompts):
            kw.update(text=prompts, padding=True)
        inputs = self.processor(**kw).to(self.device, self.dtype)
        out = self.model.generate(**inputs, **self._gen_kwargs(max_new_tokens, n, temperature),
                                  no_repeat_ngram_size=self.cfg.get("no_repeat_ngram_size", 3))
        texts = self.processor.batch_decode(out, skip_special_tokens=True)
        # BLIP returns prompt + continuation; strip the prompt so only the caption remains
        prompts_rep = [p for p in prompts for _ in range(n)]
        cleaned = []
        for t, p in zip(texts, prompts_rep):
            t = re.sub(r"\s+-\s+", "-", t.strip())    # BERT tokenizer decodes "close-up" as "close - up"
            if p and t.lower().startswith(p.lower()):
                t = t[len(p):].strip()
            cleaned.append(t)
        return self._group(cleaned, n)


class ChatCaptioner(BaseCaptioner):
    def __init__(self, name, cfg, device=None):
        super().__init__(name, cfg, device)
        proc_kw = {}
        if "max_pixels" in cfg:
            proc_kw["max_pixels"] = cfg["max_pixels"]
        if "image_longest_edge" in cfg:
            proc_kw["size"] = {"longest_edge": cfg["image_longest_edge"]}
        self.processor = AutoProcessor.from_pretrained(self.hf_id, **proc_kw)
        self.processor.tokenizer.padding_side = "left"   # required for batched generation
        self.model = AutoModelForImageTextToText.from_pretrained(self.hf_id, dtype=self.dtype)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def generate(self, images: list[Image.Image], prompts: list[str], max_new_tokens=60, n=1,
                 temperature=0.7) -> list[list[str]]:
        texts = []
        for p in prompts:
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}]
            texts.append(self.processor.apply_chat_template(msgs, add_generation_prompt=True))
        inputs = self.processor(text=texts, images=[[im] for im in images], padding=True,
                                return_tensors="pt").to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        out = self.model.generate(**inputs, **self._gen_kwargs(max_new_tokens, n, temperature))
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        decoded = [t.strip() for t in self.processor.batch_decode(new_tokens, skip_special_tokens=True)]
        return self._group(decoded, n)


def load_captioner(name: str, cfg: dict, device: str | None = None) -> BaseCaptioner:
    backend = cfg["backend"]
    if backend == "blip":
        return BlipCaptioner(name, cfg, device)
    if backend == "chat":
        return ChatCaptioner(name, cfg, device)
    raise ValueError(f"model '{name}' has backend '{backend}', which has no neural captioner")


class ClipScorer:
    """Reference-free CLIPScore (Hessel et al., 2021): 2.5 * max(cos(image, "A photo depicts " + text), 0)."""
    CLIP_ID = "openai/clip-vit-base-patch32"
    PREFIX = "A photo depicts "

    def __init__(self, device: str | None = None):
        from transformers import CLIPModel, CLIPProcessor
        self.device = device or pick_device()
        self.model = CLIPModel.from_pretrained(self.CLIP_ID, use_safetensors=True).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(self.CLIP_ID)

    @torch.inference_mode()
    def score(self, images: list[Image.Image], texts: list[str]) -> list[float]:
        inp = self.processor(text=[self.PREFIX + t for t in texts], images=images, return_tensors="pt",
                             padding=True, truncation=True, max_length=77).to(self.device)
        img = self.model.get_image_features(pixel_values=inp["pixel_values"])
        txt = self.model.get_text_features(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
        # transformers 5 returns an output object; the projected embedding is its pooler_output
        img, txt = (getattr(x, "pooler_output", x) for x in (img, txt))
        img = img / img.norm(dim=-1, keepdim=True)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        return [round(s, 4) for s in (2.5 * (img * txt).sum(-1).clamp(min=0)).tolist()]
