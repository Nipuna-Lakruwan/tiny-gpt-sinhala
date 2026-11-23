"""Download Sinhala + Simple-English Wikipedia, clean it, train a BPE tokenizer, write token files.

    uv run python src/prepare.py

Sources (both CC BY-SA 4.0): wikimedia/wikipedia, dump 20231101, configs "si" and "simple".
Split is by *article* (not by line), so no article appears in both train and validation.
"""
import json
import random
import re
import unicodedata

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from common import DATA

RAW = DATA / "raw"
VOCAB = 8192
SPECIAL = ["<|si|>", "<|en|>", "<|end|>"]


def fetch(cfg):
    local = RAW / ("siwiki_20231101.parquet" if cfg == "si" else "simplewiki_20231101.parquet")
    if not local.exists():
        p = hf_hub_download("wikimedia/wikipedia", f"20231101.{cfg}/train-00000-of-00001.parquet",
                            repo_type="dataset", local_dir=RAW / "hf")
        local.symlink_to(p)
    return pq.read_table(local, columns=["title", "text"]).to_pylist()


def clean(text):
    text = unicodedata.normalize("NFC", text)
    lines = []
    for ln in text.split("\n"):
        ln = ln.strip()
        if len(ln) < 40:  # drop headings, list stubs, empty lines
            continue
        if ln.lower().startswith(("references", "external links", "මූලාශ්‍ර", "බාහිර සබැඳි")):
            break
        lines.append(re.sub(r"\s+", " ", ln))
    return "\n".join(lines)


def sinhala_share(s):
    letters = [c for c in s if c.isalpha()]
    return sum("඀" <= c <= "෿" for c in letters) / max(1, len(letters))


def main():
    random.seed(0)
    RAW.mkdir(parents=True, exist_ok=True)
    docs = {}
    for lang, cfg in (("si", "si"), ("en", "simple")):
        arts = []
        for a in fetch(cfg):
            t = clean(a["text"])
            if len(t) < 200:
                continue
            if lang == "si" and sinhala_share(t) < 0.5:  # many si articles are mostly English/tables
                continue
            arts.append(t)
        random.shuffle(arts)
        docs[lang] = arts
        print(lang, "articles", len(arts), "chars", sum(map(len, arts)))

    # balance by characters: use all Sinhala, and the same number of characters of English
    si_chars = sum(map(len, docs["si"]))
    en, n = [], 0
    for a in docs["en"]:
        if n >= si_chars:
            break
        en.append(a); n += len(a)
    docs["en"] = en

    splits = {}
    for lang, arts in docs.items():
        k = max(50, int(0.02 * len(arts)))
        splits[lang] = {"val": arts[:k], "train": arts[k:]}

    # ---- tokenizer: byte-level BPE trained on our own text -------------------------
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    # GPT-2's split regex uses \p{L}+ for words, and Sinhala vowel signs (pili) are \p{M}, not \p{L},
    # so every word got cut at each vowel sign ("ප | ි | හ | ි | ට | ි"). Keep letters, marks and the
    # zero-width joiner (used in conjuncts like ශ්‍රී) in one piece.
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"'(?:s|t|re|ve|m|ll|d)| ?[\p{L}\p{M}\x{200D}]+| ?\p{N}+"
                                   r"| ?[^\s\p{L}\p{M}\p{N}\x{200D}]+|\s+(?!\S)|\s+"), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=SPECIAL, min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    def corpus():
        for lang in ("si", "en"):
            yield from splits[lang]["train"]

    tok.train_from_iterator(corpus(), trainer)
    tok.save(str(DATA / "tokenizer.json"))

    stats = {"vocab": tok.get_vocab_size()}
    for split in ("train", "val"):
        ids = []
        for lang in ("si", "en"):
            tag, end = tok.token_to_id(f"<|{lang}|>"), tok.token_to_id("<|end|>")
            arts = splits[lang][split]
            enc = tok.encode_batch(arts)
            ntok = 0
            per_lang = []
            for e in enc:
                per_lang += [tag] + e.ids + [end]
                ntok += len(e.ids)
            stats[f"{split}_{lang}_articles"] = len(arts)
            stats[f"{split}_{lang}_tokens"] = ntok
            stats[f"{split}_{lang}_chars"] = sum(map(len, arts))
            if split == "val":  # per-language validation files, so we can report each separately
                np.array(per_lang, dtype=np.uint16).tofile(DATA / f"val_{lang}.bin")
            ids += per_lang
        if split == "train":
            np.array(ids, dtype=np.uint16).tofile(DATA / "train.bin")
    print(json.dumps(stats, indent=2))
    (DATA / "stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
