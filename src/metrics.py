"""Classification metrics with calibration, shared shape with the other repos."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray,
                               n_bins: int = 15) -> float:
    """Average gap between confidence and accuracy across confidence bins."""
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(y_true)).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if lo == 0.0:
            in_bin |= confidence == 0.0
        if in_bin.any():
            ece += in_bin.mean() * abs(
                correct[in_bin].mean() - confidence[in_bin].mean()
            )
    return float(ece)


def evaluate(y_true: np.ndarray, logits: np.ndarray,
             classes: Sequence[str]) -> Dict[str, object]:
    y_true = np.asarray(y_true)
    probs = softmax(np.asarray(logits))
    y_pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, len(classes))

    per_class = {}
    f1_scores = []
    for i, name in enumerate(classes):
        tp = cm[i, i]
        support, predicted = cm[i].sum(), cm[:, i].sum()
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall else 0.0)
        per_class[name] = {"precision": float(precision), "recall": float(recall),
                           "f1": float(f1), "support": int(support)}
        f1_scores.append(f1)

    return {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(np.mean(f1_scores)),
        "ece": expected_calibration_error(y_true, probs),
        "mean_confidence": float(probs.max(axis=1).mean()),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def format_report(results: Dict[str, object], classes: Sequence[str]) -> str:
    lines = [
        f"accuracy         {results['accuracy']:.4f}",
        f"macro F1         {results['macro_f1']:.4f}",
        f"ECE              {results['ece']:.4f}",
        f"mean confidence  {results['mean_confidence']:.4f}",
        "",
        f"{'class':<12}{'prec':>8}{'recall':>8}{'f1':>8}{'n':>7}",
    ]
    per_class = results["per_class"]           # type: ignore[index]
    for name in classes:
        m = per_class[name]                    # type: ignore[index]
        lines.append(f"{name:<12}{m['precision']:>8.3f}{m['recall']:>8.3f}"
                     f"{m['f1']:>8.3f}{m['support']:>7d}")
    return "\n".join(lines)
