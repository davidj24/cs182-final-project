"""Minimal GPT-2 style causal transformer in plain torch (no transformers dependency).

Written from scratch following the structure used by Garg et al. (2022), "What can
transformers learn in-context?": pre-LayerNorm blocks, GELU MLP (4x), learned position
embeddings, no token-embedding table (inputs are continuous 5-dim tokens projected by a
linear layer) and a linear read-out. Nothing is vendored, so third_party/ is unused.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

N_PUSHES = 11
SEQ_LEN = 2 * N_PUSHES  # push, outcome, push, outcome, ...
TOKEN_DIM = 5


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int = 12
    n_embd: int = 256
    n_head: int = 8
    seq_len: int = SEQ_LEN
    token_dim: int = TOKEN_DIM

    def to_dict(self) -> dict:
        return asdict(self)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.n_embd
        self.n_head = cfg.n_head
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.out = nn.Linear(4 * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = (z.view(b, t, self.n_head, d // self.n_head).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class PushGPT(nn.Module):
    """tokens (B, L, 5) with L <= 22 -> predictions (B, L); entry t is the readout at token t.

    Only the entries at push-token positions (even t) are meaningful predictions.
    """

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        self.inp = nn.Linear(c.token_dim, c.n_embd)
        self.pos = nn.Embedding(c.seq_len, c.n_embd)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, 1)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        t = tokens.shape[1]
        x = self.inp(tokens) + self.pos(torch.arange(t, device=tokens.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x)).squeeze(-1)

    def push_predictions(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, 22, 5) -> (B, 11): readouts at the push tokens (positions 0, 2, ..., 20)."""
        return self(tokens)[:, 0::2]


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
