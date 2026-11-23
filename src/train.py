"""Train the GPT on random 256-token windows of train.bin.

    uv run python src/train.py --iters 4000

Validation is reported per language in *bits per character* (bpc), not per token, because the
tokenizer splits Sinhala into ~2.3x more tokens than English for the same text. Per-token loss
would make the two languages look different for the wrong reason.
"""
import argparse
import json
import math
import time

import numpy as np
import torch

from common import CKPT, DATA, RESULTS, device
from model import GPT, GPTConfig


def batches(data, bs, ctx, dev, gen):
    ix = torch.randint(len(data) - ctx - 1, (bs,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + ctx].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + ctx].astype(np.int64)) for i in ix])
    return x.to(dev), y.to(dev)


@torch.no_grad()
def evaluate(model, val, stats, ctx, dev, max_windows=200):
    """Mean next-token loss over non-overlapping windows of each validation file -> bpc."""
    model.eval()
    out = {}
    for lang, data in val.items():
        n = min(max_windows, (len(data) - 1) // ctx)
        losses = []
        for s in range(0, n, 32):
            idx = range(s, min(n, s + 32))
            x = torch.stack([torch.from_numpy(data[i * ctx:(i + 1) * ctx].astype(np.int64)) for i in idx]).to(dev)
            y = torch.stack([torch.from_numpy(data[i * ctx + 1:(i + 1) * ctx + 1].astype(np.int64)) for i in idx]).to(dev)
            _, loss = model(x, y)
            losses.append(loss.item() * len(idx))
        nats_per_tok = sum(losses) / n
        tok_per_char = stats[f"val_{lang}_tokens"] / stats[f"val_{lang}_chars"]
        out[lang] = {"loss": nats_per_tok, "ppl": math.exp(nats_per_tok),
                     "bpc": nats_per_tok / math.log(2) * tok_per_char}
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = device()
    cfg = GPTConfig(n_layer=args.n_layer, d=args.d, n_head=args.d // 64)
    model = GPT(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params / 1e6:.2f}M on {dev}")

    train = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
    val = {l: np.fromfile(DATA / f"val_{l}.bin", dtype=np.uint16) for l in ("si", "en")}
    stats = json.loads((DATA / "stats.json").read_text())

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0}],
                            lr=args.lr, betas=(0.9, 0.95))

    def lr_at(it):  # linear warmup, then cosine decay to 10%
        if it < args.warmup:
            return args.lr * (it + 1) / args.warmup
        p = (it - args.warmup) / max(1, args.iters - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    CKPT.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    ck = CKPT / "last.pt"
    start, log = 0, {"train": [], "val": []}
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        start, log = s["iter"] + 1, s["log"]
        print(f"resumed at iter {start}")
    gen = torch.Generator().manual_seed(args.seed + start)

    t0 = time.time()
    for it in range(start, args.iters + 1):
        if it % args.eval_every == 0 or it == args.iters:
            ev = evaluate(model, val, stats, cfg.ctx, dev)
            log["val"].append({"iter": it, **{f"{l}_{k}": v for l, d in ev.items() for k, v in d.items()}})
            print(f"iter {it:5d}  val bpc  si {ev['si']['bpc']:.3f}  en {ev['en']['bpc']:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iter": it, "log": log,
                        "cfg": cfg.__dict__, "args": vars(args)}, ck)
            if it == args.iters:
                break
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        for _ in range(args.accum):
            x, y = batches(train, args.bs, cfg.ctx, dev, gen)
            _, loss = model(x, y)
            (loss / args.accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if it % 25 == 0:
            log["train"].append({"iter": it, "loss": loss.item()})

    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, CKPT / "gpt.pt")
    (RESULTS / "train_log.json").write_text(json.dumps(log))


if __name__ == "__main__":
    main()
