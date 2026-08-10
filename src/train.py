"""Train the sentiment transformer.

    python -m src.train --config configs/base.yaml
    python -m src.train --config configs/base.yaml --dataset imdb --epochs 8
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_datasets
from .metrics import evaluate, format_report
from .model import SentimentTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    logits, targets = [], []
    for ids, y in loader:
        logits.append(model(ids.to(device)).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(targets), np.concatenate(logits)


def tfidf_baseline(train_ds, test_ds) -> float:
    """TF-IDF + logistic regression, for comparison.

    Reporting this next to the transformer is the honest thing to do. On short,
    single-domain sentiment data a linear bag-of-ngrams model is a genuinely
    strong baseline, and a transformer that fails to beat it is not working.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    pipeline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000),
        LogisticRegression(max_iter=2000, C=4.0),
    )
    pipeline.fit(train_ds.texts, train_ds.labels)
    return float(pipeline.score(test_ds.texts, test_ds.labels))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--dataset", default=None, choices=["imdb", "synthetic"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig"))
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.out_dir:
        cfg["out_dir"] = args.out_dir

    set_seed(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  out_dir={out_dir}")

    train_ds, val_ds, test_ds, tokenizer = build_datasets(cfg, cfg["seed"])
    tokenizer.save(out_dir / "tokenizer.json")
    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    loaders = {
        name: DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                         shuffle=(name == "train"),
                         num_workers=cfg["train"]["num_workers"])
        for name, ds in (("train", train_ds), ("val", val_ds), ("test", test_ds))
    }

    m = cfg["model"]
    model = SentimentTransformer(
        vocab_size=len(tokenizer), d_model=m["d_model"], n_heads=m["n_heads"],
        n_layers=m["n_layers"], dim_feedforward=m["dim_feedforward"],
        max_length=cfg["data"]["max_length"], dropout=m["dropout"],
        num_classes=m["num_classes"], pool=m["pool"],
    ).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # No weight decay on norms and biases -- decaying them shrinks the scale the
    # network relies on and consistently costs accuracy.
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if param.ndim <= 1 else decay).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["train"]["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["train"]["lr"], betas=(0.9, 0.98),
    )
    total_steps = cfg["train"]["epochs"] * max(1, len(loaders["train"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: cosine_with_warmup(s, cfg["train"]["warmup_steps"], total_steps),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])

    classes = ["negative", "positive"]
    history, best_acc, patience = [], -1.0, 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        total, seen = 0.0, 0
        for ids, y in tqdm(loaders["train"], desc=f"epoch {epoch}", leave=False):
            ids, y = ids.to(device), y.to(device)
            loss = criterion(model(ids), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["train"]["grad_clip"]
            )
            optimizer.step()
            scheduler.step()
            total += float(loss) * ids.size(0)
            seen += ids.size(0)

        y_true, logits = collect(model, loaders["val"], device)
        val = evaluate(y_true, logits, classes)
        print(f"epoch {epoch:3d}  loss {total/seen:.4f}  "
              f"val_acc {val['accuracy']:.4f}  val_f1 {val['macro_f1']:.4f}  "
              f"ECE {val['ece']:.4f}")
        history.append({"epoch": epoch, "loss": total / seen, "val": val})
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        if val["accuracy"] > best_acc:
            best_acc, patience = val["accuracy"], 0
            torch.save({"model": model.state_dict(), "config": cfg,
                        "vocab_size": len(tokenizer), "classes": classes},
                       out_dir / "best.pt")
        else:
            patience += 1
            if patience >= cfg["train"]["early_stopping_patience"]:
                print(f"early stopping after {epoch} epochs")
                break

    ckpt = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    y_true, logits = collect(model, loaders["test"], device)
    results = evaluate(y_true, logits, classes)
    print("\n" + format_report(results, classes))
    (out_dir / "test_metrics.json").write_text(json.dumps(results, indent=2))

    if not args.skip_baseline:
        print("\nfitting TF-IDF + logistic regression baseline")
        baseline = tfidf_baseline(train_ds, test_ds)
        print(f"  baseline accuracy    {baseline:.4f}")
        print(f"  transformer accuracy {results['accuracy']:.4f}")
        if results["accuracy"] < baseline:
            print("  NOTE: the transformer is not beating bag-of-ngrams here.")


if __name__ == "__main__":
    main()
