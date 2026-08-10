"""Classify text and visualise which tokens the model attended to.

    python -m src.predict --checkpoint runs/base/best.pt --text "a superb film"
    python -m src.predict --checkpoint runs/base/best.pt --file reviews.txt \
        --explain
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import SentimentTransformer, attention_rollout
from .tokenizer import PAD_ID, BPETokenizer


def load_model(checkpoint: str | Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    m = cfg["model"]
    model = SentimentTransformer(
        vocab_size=ckpt["vocab_size"], d_model=m["d_model"], n_heads=m["n_heads"],
        n_layers=m["n_layers"], dim_feedforward=m["dim_feedforward"],
        max_length=cfg["data"]["max_length"], dropout=m["dropout"],
        num_classes=m["num_classes"], pool=m["pool"],
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    tokenizer_path = Path(checkpoint).parent / "tokenizer.json"
    tokenizer = BPETokenizer.load(tokenizer_path) if tokenizer_path.exists() else None
    return model, cfg, ckpt["classes"], tokenizer


@torch.no_grad()
def explain(model, tokenizer, text: str, max_length: int, device: torch.device):
    """Return per-token importance from attention rollout.

    Rollout is used rather than raw last-layer attention because by the last
    layer every position's value vector already mixes information from
    everywhere else, so attending to a token no longer means depending on it.
    """
    ids = torch.tensor(
        [tokenizer.encode(text, max_length)], dtype=torch.long, device=device
    )
    logits, layer_weights, pool_weights = model.forward_with_attention(ids)
    rollout = attention_rollout(layer_weights)[0]        # (L, L)

    valid = (ids[0] != PAD_ID).cpu().numpy()
    if pool_weights is not None:
        # Weight each token by how much the pooling step drew on it.
        importance = (pool_weights[0].unsqueeze(0) @ rollout).squeeze(0)
    else:
        importance = rollout.mean(dim=0)

    importance = importance.cpu().numpy() * valid
    tokens = [tokenizer.id_to_token[int(i)] for i in ids[0].cpu()]
    return logits[0].cpu(), tokens, importance, valid


def render_importance(tokens, importance, valid, top_k: int = 12) -> str:
    """Text bar chart of the most influential tokens."""
    order = np.argsort(-importance)
    lines = []
    shown = 0
    for i in order:
        if not valid[i] or shown >= top_k:
            continue
        weight = importance[i] / max(importance.max(), 1e-9)
        bar = "█" * int(round(weight * 30))
        lines.append(f"  {tokens[i]:<18} {importance[i]:.4f} {bar}")
        shown += 1
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", default=None)
    parser.add_argument("--file", default=None, help="One document per line.")
    parser.add_argument("--explain", action="store_true",
                        help="Show per-token attention rollout importance.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.text and not args.file:
        parser.error("provide --text or --file")

    device = torch.device(args.device)
    model, cfg, classes, tokenizer = load_model(args.checkpoint, device)
    if tokenizer is None:
        raise FileNotFoundError(
            "tokenizer.json not found next to the checkpoint; it is written by "
            "src/train.py and is required to encode text the same way"
        )
    max_length = cfg["data"]["max_length"]

    texts = (
        [args.text] if args.text
        else [l.strip() for l in Path(args.file).read_text(encoding="utf-8")
              .splitlines() if l.strip()]
    )

    for text in texts:
        logits, tokens, importance, valid = explain(
            model, tokenizer, text, max_length, device
        )
        probs = F.softmax(logits, dim=-1)
        idx = int(probs.argmax())
        preview = text if len(text) <= 90 else text[:87] + "..."
        print(f"\n{preview}")
        print(f"  -> {classes[idx]}  ({float(probs[idx]):.4f})")
        if args.explain:
            print("  most influential tokens:")
            print(render_importance(tokens, importance, valid))


if __name__ == "__main__":
    main()
