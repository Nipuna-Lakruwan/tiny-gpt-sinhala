"""Count-based baselines on the same tokens, and a tokenizer comparison.

  unigram  p(next) = count(next) / N
  bigram   p(next | prev), interpolated with the unigram (lambda tuned on a held-out slice of train)

Both are reported in bits per character, like the GPT.

    uv run python src/baselines.py
"""
import json
import math

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from common import DATA, RESULTS

V = 8192


def bigram_counts(ids):
    pair = ids[:-1].astype(np.int64) * V + ids[1:]
    keys, counts = np.unique(pair, return_counts=True)
    return keys, counts


def score(ids, uni, keys, counts, prev_tot, lam):
    pair = ids[:-1].astype(np.int64) * V + ids[1:]
    pos = np.searchsorted(keys, pair)
    pos = np.clip(pos, 0, len(keys) - 1)
    c = np.where(keys[pos] == pair, counts[pos], 0)
    p_bi = c / np.maximum(prev_tot[ids[:-1]], 1)
    p = lam * p_bi + (1 - lam) * uni[ids[1:]]
    return -np.log(p).mean(), -np.log(uni[ids[1:]]).mean()


def main():
    train = np.fromfile(DATA / "train.bin", dtype=np.uint16)
    stats = json.loads((DATA / "stats.json").read_text())
    tune, fit = train[-500_000:], train[:-500_000]
    uni = (np.bincount(fit, minlength=V) + 1) / (len(fit) + V)  # add-one smoothing
    keys, counts = bigram_counts(fit)
    prev_tot = np.bincount(fit[:-1], minlength=V)
    best = min((score(tune, uni, keys, counts, prev_tot, l)[0], l) for l in (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995))
    lam = best[1]
    out = {"bigram_lambda": float(lam)}
    for lang in ("si", "en"):
        ids = np.fromfile(DATA / f"val_{lang}.bin", dtype=np.uint16)
        bi, un = score(ids, uni, keys, counts, prev_tot, lam)
        tpc = stats[f"val_{lang}_tokens"] / stats[f"val_{lang}_chars"]
        out[lang] = {"unigram_bpc": un / math.log(2) * tpc, "bigram_bpc": bi / math.log(2) * tpc,
                     "bigram_ppl": math.exp(bi)}

    # tokenizer efficiency: ours vs GPT-2's (trained almost only on English)
    ours = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    gpt2 = Tokenizer.from_file(hf_hub_download("openai-community/gpt2", "tokenizer.json"))
    samples = {
        "si": "ශ්‍රී ලංකාව ඉන්දියන් සාගරයේ පිහිටි දිවයිනකි. එහි අගනුවර ශ්‍රී ජයවර්ධනපුර කෝට්ටේ වේ.",
        "en": "Sri Lanka is an island country in the Indian Ocean. Its capital is Sri Jayawardenepura Kotte.",
    }
    tok = {}
    for lang, s in samples.items():
        e = ours.encode(s)
        tok[lang] = {"chars": len(s), "ours": len(e.ids), "gpt2": len(gpt2.encode(s).ids),
                     "ours_pieces": [s[a:b] for a, b in e.offsets]}
    out["tokenizer"] = tok
    print(json.dumps(out, indent=2, ensure_ascii=False))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "baselines.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
