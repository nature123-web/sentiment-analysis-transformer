"""IMDB loading and a synthetic sentiment corpus for offline runs."""

from __future__ import annotations

import re
import tarfile
import urllib.request
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import BPETokenizer

IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# Sentiment-bearing vocabulary for the synthetic corpus.
POSITIVE_WORDS = [
    "brilliant", "gripping", "heartfelt", "luminous", "superb", "inventive",
    "charming", "riveting", "masterful", "delightful", "poignant", "elegant",
]
NEGATIVE_WORDS = [
    "tedious", "clumsy", "shallow", "muddled", "lifeless", "derivative",
    "grating", "incoherent", "wooden", "plodding", "tiresome", "clichéd",
]
NEUTRAL_WORDS = [
    "the", "a", "film", "movie", "story", "director", "cast", "scene",
    "and", "with", "of", "was", "it", "this", "that", "for", "runtime",
    "cinematography", "plot", "ending", "opening", "sequence",
]
INTENSIFIERS = ["very", "utterly", "somewhat", "fairly", "remarkably"]
NEGATIONS = ["not", "hardly", "never"]


def clean_text(text: str) -> str:
    """Strip HTML and collapse whitespace. IMDB reviews contain raw <br /> tags."""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def download_imdb(root: str | Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    extracted = root / "aclImdb"
    if extracted.exists():
        return extracted
    archive = root / "aclImdb_v1.tar.gz"
    if not archive.exists():
        print(f"downloading IMDB to {archive} (~80 MB)")
        urllib.request.urlretrieve(IMDB_URL, archive)
    print("extracting")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    return extracted


def load_imdb(root: str | Path, split: str = "train"
              ) -> Tuple[List[str], np.ndarray]:
    folder = download_imdb(root) / split
    texts, labels = [], []
    for label, name in ((1, "pos"), (0, "neg")):
        for path in sorted((folder / name).glob("*.txt")):
            texts.append(clean_text(path.read_text(encoding="utf-8")))
            labels.append(label)
    return texts, np.array(labels, dtype=np.int64)


def make_synthetic_reviews(
    n_samples: int, seed: int = 0, min_words: int = 20, max_words: int = 90
) -> Tuple[List[str], np.ndarray]:
    """Generate reviews whose sentiment depends on negation, not word counts.

    Roughly a third of sentiment words are negated ("not brilliant"), which flips
    their contribution. That makes bag-of-words genuinely insufficient and gives
    the attention mechanism something real to learn -- a corpus where sentiment
    is a simple word count would be solved by logistic regression and would say
    nothing about whether the transformer works.
    """
    rng = np.random.default_rng(seed)
    texts, labels = [], []

    for _ in range(n_samples):
        length = int(rng.integers(min_words, max_words))
        target = int(rng.integers(0, 2))
        words: List[str] = []
        score = 0

        while len(words) < length:
            roll = rng.random()
            if roll < 0.22:
                # Bias the draw toward the target sentiment, then let negation
                # decide the actual contribution.
                want_positive = rng.random() < (0.75 if target == 1 else 0.25)
                pool = POSITIVE_WORDS if want_positive else NEGATIVE_WORDS
                polarity = 1 if want_positive else -1

                if rng.random() < 0.33:
                    words.append(str(rng.choice(NEGATIONS)))
                    polarity *= -1
                if rng.random() < 0.4:
                    words.append(str(rng.choice(INTENSIFIERS)))
                words.append(str(rng.choice(pool)))
                score += polarity
                # Always separate consecutive sentiment words with a neutral
                # token. Without this the generator can emit "never tedious
                # clumsy", where a reader cannot tell whether the negation
                # scopes over one word or both -- the label would then depend on
                # generator state that is invisible in the text, which is not a
                # learnable target.
                words.append(str(rng.choice(NEUTRAL_WORDS)))
            else:
                words.append(str(rng.choice(NEUTRAL_WORDS)))

        # The label is the *realised* sentiment, so it is always consistent with
        # the text even when the random draw fought the intended target.
        if score == 0:
            words.append(str(rng.choice(POSITIVE_WORDS if target else NEGATIVE_WORDS)))
            score = 1 if target else -1
        texts.append(" ".join(words) + ".")
        labels.append(1 if score > 0 else 0)

    return texts, np.array(labels, dtype=np.int64)


class TextDataset(Dataset):
    """Encodes text to fixed-length id tensors on access."""

    def __init__(self, texts: Sequence[str], labels: np.ndarray,
                 tokenizer: BPETokenizer, max_length: int = 256) -> None:
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        ids = self.tokenizer.encode(self.texts[idx], self.max_length)
        return torch.tensor(ids, dtype=torch.long), int(self.labels[idx])


def build_datasets(cfg: dict, seed: int = 0):
    """Load data, train a tokenizer on the training split only, build datasets.

    The tokenizer is fitted on training text alone. Fitting it on the full
    corpus would leak test-set vocabulary statistics into the merge table -- a
    subtle but real form of contamination that inflates reported accuracy.
    """
    d = cfg["data"]
    if d["dataset"] == "imdb":
        train_texts, train_labels = load_imdb(d["root"], "train")
        test_texts, test_labels = load_imdb(d["root"], "test")
    else:
        train_texts, train_labels = make_synthetic_reviews(d["n_synthetic"], seed)
        test_texts, test_labels = make_synthetic_reviews(
            max(400, d["n_synthetic"] // 4), seed + 5000
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_texts))
    train_texts = [train_texts[i] for i in order]
    train_labels = train_labels[order]

    n_val = int(len(train_texts) * d["val_fraction"])
    val_texts, val_labels = train_texts[:n_val], train_labels[:n_val]
    train_texts, train_labels = train_texts[n_val:], train_labels[n_val:]

    print(f"training BPE tokenizer (vocab {d['vocab_size']}) on "
          f"{len(train_texts)} documents")
    tokenizer = BPETokenizer(d["vocab_size"], d["min_frequency"]).train(train_texts)
    print(f"  vocabulary: {len(tokenizer)} tokens, {len(tokenizer.merges)} merges")

    make = lambda t, l: TextDataset(t, l, tokenizer, d["max_length"])  # noqa: E731
    return (
        make(train_texts, train_labels),
        make(val_texts, val_labels),
        make(test_texts, test_labels),
        tokenizer,
    )
