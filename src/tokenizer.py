"""Byte-pair encoding, trained from scratch.

BPE starts from a character vocabulary and repeatedly merges the most frequent
adjacent pair. The result sits between word-level and character-level
tokenisation: frequent words survive as single tokens, while rare and unseen
words decompose into subwords instead of collapsing to a single ``<unk>``. For
sentiment work that matters more than it might seem -- negation and intensity
often ride on morphology ("disappoint", "disappointing", "disappointingly"),
and a word-level vocabulary throws that structure away.

Words are kept separate during merging by tracking them as symbol tuples with an
end-of-word marker, so a merge can never span a word boundary.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

END_OF_WORD = "</w>"
_WORD_RE = re.compile(r"[a-z0-9']+|[^\sa-z0-9]", re.IGNORECASE)


def basic_tokenize(text: str) -> List[str]:
    """Lowercase and split into words and standalone punctuation.

    Punctuation is kept as its own token rather than stripped: exclamation marks
    and ellipses carry real sentiment signal.
    """
    return _WORD_RE.findall(text.lower())


class BPETokenizer:
    """Byte-pair encoding with an explicit, inspectable merge list."""

    def __init__(self, vocab_size: int = 8000, min_frequency: int = 2) -> None:
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.merges: List[Tuple[str, str]] = []
        self.ranks: Dict[Tuple[str, str], int] = {}
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: List[str] = []
        self._cache: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, texts: Iterable[str], verbose: bool = False) -> "BPETokenizer":
        word_counts: Counter[str] = Counter()
        for text in texts:
            word_counts.update(basic_tokenize(text))

        # Each word becomes a tuple of characters plus an end-of-word marker.
        # The marker is what stops "in" inside "inside" merging with a following
        # word, and lets the decoder know where to reinsert spaces.
        splits: Dict[str, List[str]] = {
            word: list(word) + [END_OF_WORD] for word in word_counts
        }

        alphabet = sorted({ch for word in word_counts for ch in word})
        vocabulary = SPECIAL_TOKENS + alphabet + [END_OF_WORD]

        while len(vocabulary) < self.vocab_size:
            pair_counts: Counter[Tuple[str, str]] = Counter()
            for word, count in word_counts.items():
                symbols = splits[word]
                for a, b in zip(symbols, symbols[1:]):
                    pair_counts[(a, b)] += count

            if not pair_counts:
                break
            best, frequency = pair_counts.most_common(1)[0]
            if frequency < self.min_frequency:
                break

            merged = best[0] + best[1]
            for word in word_counts:
                splits[word] = _merge_pair(splits[word], best, merged)

            self.merges.append(best)
            vocabulary.append(merged)
            if verbose and len(self.merges) % 500 == 0:
                print(f"  {len(self.merges)} merges, vocab {len(vocabulary)}")

        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.id_to_token = vocabulary
        self.token_to_id = {tok: i for i, tok in enumerate(vocabulary)}
        self._cache.clear()
        return self

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #

    def tokenize_word(self, word: str) -> List[str]:
        """Apply the learned merges to one word, lowest rank first."""
        if word in self._cache:
            return self._cache[word]

        symbols = list(word) + [END_OF_WORD]
        while len(symbols) > 1:
            pairs = [(a, b) for a, b in zip(symbols, symbols[1:])]
            ranked = [(self.ranks[p], p) for p in pairs if p in self.ranks]
            if not ranked:
                break
            # Always apply the *earliest-learned* merge available, which is what
            # reproduces the training-time merge order. Applying merges
            # left-to-right instead would give a different, inconsistent split.
            _, best = min(ranked)
            symbols = _merge_pair(symbols, best, best[0] + best[1])

        self._cache[word] = symbols
        return symbols

    def tokenize(self, text: str) -> List[str]:
        tokens: List[str] = []
        for word in basic_tokenize(text):
            tokens.extend(self.tokenize_word(word))
        return tokens

    def encode(self, text: str, max_length: int | None = None,
               add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        ids = [self.token_to_id.get(t, UNK_ID) for t in tokens]
        if add_special_tokens:
            ids = [BOS_ID] + ids + [EOS_ID]
        if max_length is not None:
            if len(ids) > max_length:
                # Truncate the middle, not the tail: the opening and closing
                # sentences of a review carry most of its sentiment, and cutting
                # only the end throws away the conclusion.
                keep = max_length - 1
                head = keep // 2
                tail = keep - head
                # Drop the trailing EOS before slicing: it is the last element
                # of `ids`, so a tail slice taken from `ids` directly would
                # include it, and appending EOS_ID afterwards would then
                # duplicate it -- wasting one of the tail slots on a repeated
                # token instead of real content.
                body = ids[:-1]
                ids = body[:head] + body[len(body) - tail:] + [EOS_ID]
            else:
                ids = ids + [PAD_ID] * (max_length - len(ids))
        return ids

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        pieces = []
        for i in ids:
            token = self.id_to_token[i] if i < len(self.id_to_token) else UNK
            if skip_special and token in SPECIAL_TOKENS:
                continue
            pieces.append(token)
        return "".join(pieces).replace(END_OF_WORD, " ").strip()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.id_to_token)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "merges": [list(m) for m in self.merges],
            "vocab": self.id_to_token,
        }))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        blob = json.loads(Path(path).read_text())
        tok = cls(blob["vocab_size"], blob["min_frequency"])
        tok.merges = [tuple(m) for m in blob["merges"]]
        tok.ranks = {pair: i for i, pair in enumerate(tok.merges)}
        tok.id_to_token = blob["vocab"]
        tok.token_to_id = {t: i for i, t in enumerate(tok.id_to_token)}
        return tok


def _merge_pair(symbols: List[str], pair: Tuple[str, str], merged: str
                ) -> List[str]:
    """Replace every non-overlapping occurrence of ``pair`` with ``merged``."""
    out: List[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out
