# Sentiment Analysis with a Transformer Built From Scratch

A transformer encoder for text classification with **no** `nn.MultiheadAttention`
and **no** `tokenizers` library. Byte-pair encoding, scaled dot-product attention,
multi-head projection, pre-norm residual blocks and attention rollout are all
implemented here, so every part of the pipeline is inspectable and tested.

```
raw text
   │  BPE — merges learned from the training split only
   ▼
token ids ──► embedding + learned positions
   │
   ▼  N × pre-norm encoder block
       ├─ multi-head self-attention  (masked at padding)
       └─ GELU feed-forward
   │
   ▼  attention pooling → classifier
positive / negative  +  per-token attribution
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

```bash
# Synthetic corpus — no download, runs in a couple of minutes
python -m src.train --config configs/base.yaml

# IMDB (50k reviews, ~80 MB download)
python -m src.train --config configs/base.yaml --dataset imdb --epochs 8

# Classify, and show what drove the decision
python -m src.predict --checkpoint runs/base/best.pt \
    --text "not brilliant, and utterly tedious" --explain

pytest    # 40 tests
```

## Why byte-pair encoding

A word-level vocabulary maps every unseen word to `<unk>`, which for sentiment is
actively harmful — the rare words are often the most loaded ones. BPE starts from
characters and merges the most frequent adjacent pair repeatedly, so common words
end up as single tokens while rare ones decompose into meaningful subwords:

```
brilliant    → ['brilliant</w>']          # frequent, merged whole
brilliantly  → ['brilliant', 'ly</w>']    # unseen, but morphology survives
```

The merge table is a plain list you can print and read. Two details that are easy
to get wrong and are pinned by tests:

- Merges are applied in **learned-rank order**, not left to right. Applying them
  positionally gives a different segmentation from the one training produced.
- The end-of-word marker prevents a merge from spanning a word boundary.

## Why the synthetic corpus uses negation

A generated corpus where sentiment is a simple count of positive minus negative
words is solved perfectly by logistic regression and proves nothing about the
model. Here roughly a third of sentiment words are negated:

> ... **not** brilliant ... **hardly** gripping ... utterly tedious ...

so word order carries real information and a bag-of-words ceiling exists. The
test suite asserts that a naive count-based classifier stays below 90%, which
keeps the dataset honest as the generator evolves.

## The baseline is reported, not hidden

`src/train.py` fits a **TF-IDF + logistic regression** baseline at the end of
every run and prints it next to the transformer's score. On short single-domain
sentiment data a linear bag-of-ngrams model is genuinely strong, often within a
point or two of a small transformer. A project that omits this comparison is
hiding its most informative number; if the transformer is not winning, that is
worth knowing.

Skip it with `--skip-baseline` on large corpora where the fit is slow.

## Explanations: rollout, not raw attention

`--explain` reports per-token importance via **attention rollout** rather than
last-layer attention weights. The distinction matters: by the final layer, every
position's representation already mixes information from the whole sequence
through the residual stream, so "attends to token *i*" no longer means "depends
on token *i*". Rollout multiplies the per-layer attention matrices together after
adding the residual as an identity, recovering an approximate token-to-token
influence map.

Output format (values depend on your trained checkpoint):

```
not brilliant, and utterly tedious
  -> negative  (0.94)
  most influential tokens:
    tedious            0.1834 ██████████████████████████████
    not                0.1120 ██████████████████
    brilliant          0.0908 ███████████████
    utterly            0.0641 ██████████
```

## Configuration

```yaml
data:
  dataset: synthetic     # synthetic | imdb
  vocab_size: 8000
  max_length: 256
model:
  d_model: 256
  n_heads: 8
  n_layers: 4
  pool: attention        # attention | mean | max | cls
train:
  lr: 0.0003
  warmup_steps: 200
  label_smoothing: 0.05
```

All four pooling modes are tested to be padding-invariant, so `pool` can be
changed freely without silently corrupting long-sequence batches.

## Layout

```
src/
  tokenizer.py   BPE training, encoding, save/load
  data.py        IMDB loader, synthetic corpus with negation
  model.py       attention, encoder blocks, pooling, rollout
  metrics.py     accuracy, macro F1, calibration error
  train.py       training loop + TF-IDF baseline
  predict.py     inference and token attribution
tests/           pytest suite
```

## License

MIT
