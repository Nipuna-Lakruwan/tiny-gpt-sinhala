"""Samples, curves, attention maps and a generation GIF.

    uv run python src/figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tokenizers import Tokenizer

from common import CKPT, DATA, RESULTS, device
from model import GPT, GPTConfig

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
FONT_PATHS = ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
              "/System/Library/Fonts/Supplemental/Sinhala Sangam MN.ttc",
              "/usr/share/fonts/truetype/noto/NotoSansSinhala-Regular.ttf"]

PROMPTS = {
    "si": ["ශ්‍රී ලංකාවේ", "කොළඹ නගරය", "බුදුදහම"],
    "en": ["The history of", "Sri Lanka is", "In mathematics,"],
}


def load(dev):
    s = torch.load(CKPT / "gpt.pt", map_location=dev)
    m = GPT(GPTConfig(**s["cfg"])).to(dev)
    m.load_state_dict(s["model"])
    return m.eval()


def samples(m, tok, dev):
    out = {}
    g = torch.manual_seed(0)
    end = tok.token_to_id("<|end|>")
    for lang, ps in PROMPTS.items():
        tag = tok.token_to_id(f"<|{lang}|>")
        for p in ps:
            for temp in (0.7, 1.0):
                ids = torch.tensor([[tag] + tok.encode(p).ids], device=dev)
                o = m.generate(ids, 80, temperature=temp, top_k=50, stop_id=end)[0, 1:].tolist()
                out.setdefault(lang, []).append({"prompt": p, "temperature": temp,
                                                 "text": tok.decode([i for i in o if i != end])})
    (RESULTS / "samples.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def curves():
    log = json.loads((RESULTS / "train_log.json").read_text())
    base = json.loads((RESULTS / "baselines.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    tr = log["train"]
    a1.plot([r["iter"] for r in tr], [r["loss"] for r in tr], lw=0.8)
    a1.set_xlabel("iteration"); a1.set_ylabel("train loss (nats / token)"); a1.grid(alpha=0.3)
    a1.set_title("Next-token cross-entropy")
    v = log["val"]
    for lang, c in (("si", "C0"), ("en", "C1")):
        a2.plot([r["iter"] for r in v], [r[f"{lang}_bpc"] for r in v], color=c, label=f"GPT, {lang}")
        a2.axhline(base[lang]["bigram_bpc"], color=c, ls="--", lw=1, label=f"bigram, {lang}")
    a2.set_xlabel("iteration"); a2.set_ylabel("validation bits per character"); a2.grid(alpha=0.3)
    a2.set_title("Validation (lower is better)"); a2.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(RESULTS / "curves.png"); plt.close(fig)


@torch.no_grad()
def attention(m, tok, dev):
    text = "The cat sat on the mat because it was tired."
    ids = [tok.token_to_id("<|en|>")] + tok.encode(text).ids
    labels = ["<en>"] + [tok.decode([i]).strip() or "_" for i in ids[1:]]
    _, _, atts = m(torch.tensor([ids], device=dev), return_attn=True)
    L = len(atts)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    picks = [(0, 0), (0, 1), (L // 2, 0), (L // 2, 2), (L - 1, 0), (L - 1, 3)]
    for ax, (l, h) in zip(axes.flat, picks):
        a = atts[l][0, h].cpu().numpy()
        ax.imshow(a, cmap="magma", vmin=0, vmax=min(1, a.max()))
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_title(f"layer {l + 1}, head {h + 1}")
    fig.suptitle("Causal self-attention: row = token being predicted from, column = token it looks at", fontsize=10)
    fig.tight_layout(); fig.savefig(RESULTS / "attention.png"); plt.close(fig)


def font(size):
    for p in FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


@torch.no_grad()
def generation_gif(m, tok, dev, prompt="Sri Lanka is", lang="en", n=40):
    """Token-by-token generation with the top-5 next-token probabilities at each step."""
    torch.manual_seed(3)
    ids = torch.tensor([[tok.token_to_id(f"<|{lang}|>")] + tok.encode(prompt).ids], device=dev)
    f, fs = font(22), font(16)
    frames = []
    for _ in range(n):
        logits, _ = m(ids[:, -m.c.ctx:])
        p = (logits[0, -1] / 0.8).softmax(-1)
        top = torch.topk(p, 5)
        nxt = torch.multinomial(p, 1)
        text = tok.decode(ids[0, 1:].tolist()).replace("\n", " ")
        im = Image.new("RGB", (900, 330), (14, 14, 18))
        d = ImageDraw.Draw(im)
        # wrap text
        words, lines, cur = text.split(" "), [], ""
        for w in words:
            if d.textlength(cur + " " + w, font=f) > 860:
                lines.append(cur); cur = w
            else:
                cur = (cur + " " + w).strip()
        lines.append(cur)
        for k, ln in enumerate(lines[-4:]):
            d.text((20, 20 + 30 * k), ln, font=f, fill=(235, 235, 235))
        d.text((20, 160), "next token, top 5 (temperature 0.8):", font=fs, fill=(150, 150, 160))
        for k, (pr, ti) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
            y = 188 + 26 * k
            chosen = ti == nxt.item()
            d.rectangle([20, y, 20 + int(pr * 600), y + 18], fill=(90, 160, 255) if chosen else (70, 70, 85))
            d.text((630, y - 1), f"{tok.decode([ti])!r}  {pr:.2f}", font=fs, fill=(235, 235, 235))
        frames.append(im)
        ids = torch.cat([ids, nxt[None]], 1)
    frames += [frames[-1]] * 8
    frames[0].save(RESULTS / "generation.gif", save_all=True, append_images=frames[1:], duration=350, loop=0)


def main():
    dev = device()
    tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    m = load(dev)
    curves()
    out = samples(m, tok, dev)
    for lang, rows in out.items():
        for r in rows[:2]:
            print(lang, r["temperature"], r["text"][:200])
    attention(m, tok, dev)
    generation_gif(m, tok, dev)


if __name__ == "__main__":
    main()
