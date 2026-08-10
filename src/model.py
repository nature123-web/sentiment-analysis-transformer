"""A transformer encoder written from scratch for sequence classification.

``nn.MultiheadAttention`` is deliberately not used. Attention is implemented
explicitly so the mask handling, the head split, and the scaling are all visible
and testable -- and so attention weights can be returned for visualisation,
which the fused implementation makes awkward.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import PAD_ID


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product attention over ``n_heads`` parallel subspaces."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None,
                need_weights: bool = False):
        b, length, d = x.shape
        qkv = self.qkv(x).view(b, length, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)         # each (B, H, L, head_dim)

        # Scaling by sqrt(head_dim) keeps the pre-softmax variance at ~1. Without
        # it, dot products of d-dimensional vectors grow like sqrt(d), the
        # softmax saturates, and gradients vanish for large d.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if pad_mask is not None:
            # pad_mask: (B, L), True where the token is padding.
            scores = scores.masked_fill(
                pad_mask[:, None, None, :], torch.finfo(scores.dtype).min
            )

        weights = F.softmax(scores, dim=-1)
        attended = self.dropout(weights) @ v
        attended = attended.transpose(1, 2).reshape(b, length, d)
        out = self.out_proj(attended)
        return (out, weights) if need_weights else (out, None)


class EncoderLayer(nn.Module):
    """Pre-norm transformer block.

    Pre-norm (normalise *before* the sublayer) rather than the original
    post-norm: it leaves a clean residual path from input to output, so deep
    stacks train without a learning-rate warmup crutch and are far less prone to
    diverging early.
    """

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None,
                need_weights: bool = False):
        attended, weights = self.attention(
            self.norm1(x), pad_mask, need_weights
        )
        x = x + self.dropout(attended)
        x = x + self.dropout(self.feedforward(self.norm2(x)))
        return x, weights


class SentimentTransformer(nn.Module):
    """Transformer encoder with attention pooling for text classification."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        max_length: int = 512,
        dropout: float = 0.1,
        num_classes: int = 2,
        pool: str = "attention",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pool = pool
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        # Learned rather than sinusoidal: reviews are short and the corpus is
        # large enough to fit them, and learned positions consistently edge out
        # fixed ones on classification.
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        if pool == "attention":
            self.pool_query = nn.Linear(d_model, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                # The pad embedding must stay at zero; it is masked out anyway,
                # but a nonzero value leaks into pooling if a mask is missed.
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def encode(self, input_ids: torch.Tensor, need_weights: bool = False):
        b, length = input_ids.shape
        pad_mask = input_ids == PAD_ID

        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.embed_dropout(x + self.position_embedding(positions))

        all_weights = []
        for layer in self.layers:
            x, weights = layer(x, pad_mask, need_weights)
            if need_weights:
                all_weights.append(weights)
        return self.norm(x), pad_mask, all_weights

    def pool_sequence(self, x: torch.Tensor, pad_mask: torch.Tensor):
        """Collapse (B, L, D) to (B, D), ignoring padded positions."""
        valid = (~pad_mask).unsqueeze(-1).float()
        if self.pool == "attention":
            scores = self.pool_query(x).masked_fill(
                pad_mask.unsqueeze(-1), torch.finfo(x.dtype).min
            )
            weights = F.softmax(scores, dim=1)
            return (x * weights).sum(dim=1), weights.squeeze(-1)
        if self.pool == "mean":
            # Divide by the true token count, not the padded length, or long
            # sequences get systematically shrunk toward zero.
            return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1), None
        if self.pool == "max":
            return (x.masked_fill(pad_mask.unsqueeze(-1),
                                  torch.finfo(x.dtype).min)).max(dim=1).values, None
        if self.pool == "cls":
            return x[:, 0], None
        raise ValueError(f"unknown pool: {self.pool}")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x, pad_mask, _ = self.encode(input_ids)
        pooled, _ = self.pool_sequence(x, pad_mask)
        return self.classifier(pooled)

    def forward_with_attention(self, input_ids: torch.Tensor):
        """Logits plus per-layer attention and the pooling weights."""
        x, pad_mask, layer_weights = self.encode(input_ids, need_weights=True)
        pooled, pool_weights = self.pool_sequence(x, pad_mask)
        return self.classifier(pooled), layer_weights, pool_weights


@torch.no_grad()
def attention_rollout(layer_weights: list[torch.Tensor]) -> torch.Tensor:
    """Aggregate attention across layers into token-to-token influence.

    Raw last-layer attention is a poor explanation because the values it attends
    over have already been mixed by every layer below. Rollout accounts for that
    by adding the residual connection as an identity matrix, renormalising, and
    multiplying the per-layer matrices together (Abnar & Zuidema, 2020).
    """
    rollout = None
    for weights in layer_weights:
        # Average the heads, then fold in the residual stream.
        attn = weights.mean(dim=1)                       # (B, L, L)
        eye = torch.eye(attn.size(-1), device=attn.device).unsqueeze(0)
        attn = attn + eye
        attn = attn / attn.sum(dim=-1, keepdim=True)
        rollout = attn if rollout is None else attn @ rollout
    return rollout
