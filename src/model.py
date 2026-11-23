"""A decoder-only Transformer (GPT), written out in full.

    token ids -> token embedding + position embedding
              -> N x [ x + Attention(LN(x)) ;  x + MLP(LN(x)) ]
              -> LN -> linear head (tied to the token embedding) -> next-token logits

Attention(Q, K, V) = softmax(Q K^T / sqrt(d_head) + causal_mask) V
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab: int = 8192
    ctx: int = 256
    n_layer: int = 6
    n_head: int = 6
    d: int = 384
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.h = c.n_head
        self.qkv = nn.Linear(c.d, 3 * c.d)
        self.proj = nn.Linear(c.d, c.d)
        self.drop = c.dropout
        self.register_buffer("mask", torch.tril(torch.ones(c.ctx, c.ctx, dtype=torch.bool)), persistent=False)

    def forward(self, x, return_attn=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))  # (B, h, T, dh)
        if return_attn:  # explicit version, so we can look at the attention pattern
            att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
            att = att.masked_fill(~self.mask[:T, :T], float("-inf")).softmax(-1)
            y = att @ v
        else:
            att = None
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), att


class Block(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d)
        self.attn = CausalSelfAttention(c)
        self.ln2 = nn.LayerNorm(c.d)
        self.mlp = nn.Sequential(nn.Linear(c.d, 4 * c.d), nn.GELU(), nn.Linear(4 * c.d, c.d), nn.Dropout(c.dropout))
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, return_attn=False):
        a, att = self.attn(self.ln1(x), return_attn)
        x = x + self.drop(a)
        x = x + self.mlp(self.ln2(x))
        return x, att


class GPT(nn.Module):
    def __init__(self, c: GPTConfig):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.ctx, c.d)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)
        for n, p in self.named_parameters():  # GPT-2: scale residual projections by depth
            if n.endswith("proj.weight") or n.endswith("mlp.2.weight"):
                nn.init.normal_(p, 0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, idx, targets=None, return_attn=False):
        B, T = idx.shape
        x = self.drop(self.tok(idx) + self.pos(torch.arange(T, device=idx.device)))
        atts = []
        for b in self.blocks:
            x, att = b(x, return_attn)
            atts.append(att)
        logits = self.head(self.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return (logits, loss, atts) if return_attn else (logits, loss)

    @torch.no_grad()
    def generate(self, idx, n_new, temperature=1.0, top_k=None, stop_id=None):
        for _ in range(n_new):
            logits, _ = self(idx[:, -self.c.ctx:])
            logits = logits[:, -1] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(logits.softmax(-1), 1)
            idx = torch.cat([idx, nxt], 1)
            if stop_id is not None and (nxt == stop_id).all():
                break
        return idx
