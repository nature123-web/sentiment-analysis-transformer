"""Tests for the from-scratch transformer, attention masking, and metrics."""

import numpy as np
import pytest
import torch

from src.data import make_synthetic_reviews
from src.metrics import evaluate, expected_calibration_error
from src.model import (
    MultiHeadSelfAttention,
    SentimentTransformer,
    attention_rollout,
)
from src.tokenizer import PAD_ID


def make_model(**kwargs):
    defaults = dict(
        vocab_size=100, d_model=32, n_heads=4, n_layers=2,
        dim_feedforward=64, max_length=64, dropout=0.0, num_classes=2,
    )
    defaults.update(kwargs)
    return SentimentTransformer(**defaults)


def test_forward_shape():
    model = make_model().eval()
    ids = torch.randint(4, 100, (3, 20))
    assert model(ids).shape == (3, 2)


def test_d_model_must_divide_by_heads():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadSelfAttention(d_model=30, n_heads=4)


def test_attention_weights_sum_to_one():
    attn = MultiHeadSelfAttention(d_model=32, n_heads=4, dropout=0.0).eval()
    x = torch.randn(2, 10, 32)
    _, weights = attn(x, need_weights=True)
    assert weights.shape == (2, 4, 10, 10)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 10), atol=1e-5)


def test_attention_assigns_no_weight_to_padding():
    """The mask must actually zero out padded keys, not merely downweight them."""
    attn = MultiHeadSelfAttention(d_model=32, n_heads=4, dropout=0.0).eval()
    x = torch.randn(1, 8, 32)
    pad_mask = torch.zeros(1, 8, dtype=torch.bool)
    pad_mask[0, 5:] = True

    _, weights = attn(x, pad_mask, need_weights=True)
    assert torch.allclose(weights[0, :, :, 5:], torch.zeros(4, 8, 3), atol=1e-7)


def test_padding_does_not_change_real_token_predictions():
    """Appending padding must leave the classification untouched."""
    torch.manual_seed(0)
    model = make_model().eval()
    ids = torch.randint(4, 100, (1, 12))
    padded = torch.cat([ids, torch.full((1, 10), PAD_ID)], dim=1)
    with torch.no_grad():
        assert torch.allclose(model(ids), model(padded), atol=1e-5)


@pytest.mark.parametrize("pool", ["attention", "mean", "max", "cls"])
def test_all_pooling_modes_work_and_ignore_padding(pool):
    torch.manual_seed(0)
    model = make_model(pool=pool).eval()
    ids = torch.randint(4, 100, (1, 12))
    padded = torch.cat([ids, torch.full((1, 8), PAD_ID)], dim=1)
    with torch.no_grad():
        a, b = model(ids), model(padded)
    assert a.shape == (1, 2)
    assert torch.allclose(a, b, atol=1e-5), f"{pool} pooling leaks padding"


def test_unknown_pool_raises():
    model = make_model(pool="nonsense").eval()
    with pytest.raises(ValueError, match="unknown pool"):
        model(torch.randint(4, 100, (1, 8)))


def test_pad_embedding_is_zero_at_init():
    model = make_model()
    assert torch.allclose(
        model.token_embedding.weight[PAD_ID],
        torch.zeros(model.d_model),
    )


def test_attention_scaling_keeps_scores_bounded():
    """Without the 1/sqrt(d) factor the softmax saturates for wide heads."""
    attn = MultiHeadSelfAttention(d_model=256, n_heads=4, dropout=0.0).eval()
    x = torch.randn(1, 16, 256)
    _, weights = attn(x, need_weights=True)
    # A saturated softmax would put essentially all mass on one key.
    assert weights.max() < 0.99


def test_gradients_reach_every_layer():
    model = make_model()
    ids = torch.randint(4, 100, (2, 16))
    torch.nn.functional.cross_entropy(model(ids), torch.tensor([0, 1])).backward()
    for name, param in model.named_parameters():
        if "position_embedding" in name or param.grad is None:
            continue
        assert param.grad.abs().sum() > 0 or name.startswith("token_embedding"), name


def test_attention_rollout_rows_are_distributions():
    model = make_model().eval()
    ids = torch.randint(4, 100, (2, 12))
    with torch.no_grad():
        _, layer_weights, _ = model.forward_with_attention(ids)
    rollout = attention_rollout(layer_weights)
    assert rollout.shape == (2, 12, 12)
    assert torch.allclose(rollout.sum(dim=-1), torch.ones(2, 12), atol=1e-4)


def test_rollout_of_a_single_layer_includes_the_residual():
    """One layer of rollout is the attention averaged with the identity."""
    weights = torch.zeros(1, 2, 3, 3)
    weights[:, :, :, 0] = 1.0            # every token attends only to token 0
    rollout = attention_rollout([weights])
    # Row 1 is attention [1,0,0] plus identity [0,1,0], renormalised: the
    # residual puts exactly half the mass back on the diagonal even though
    # attention ignored it entirely.
    assert rollout[0, 1, 1] > 0
    assert rollout[0, 1] == pytest.approx([0.5, 0.5, 0.0], abs=1e-6)
    # Token 0 attends to itself, so both paths coincide and it keeps all mass.
    assert rollout[0, 0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)


def test_forward_with_attention_returns_per_layer_weights():
    model = make_model(n_layers=3).eval()
    ids = torch.randint(4, 100, (1, 10))
    with torch.no_grad():
        logits, layer_weights, pool_weights = model.forward_with_attention(ids)
    assert logits.shape == (1, 2)
    assert len(layer_weights) == 3
    assert pool_weights is not None and pool_weights.shape == (1, 10)


# --------------------------------------------------------------------------- #
# Data and metrics
# --------------------------------------------------------------------------- #

def test_synthetic_reviews_are_balanced():
    _, labels = make_synthetic_reviews(1000, seed=0)
    assert 0.35 < labels.mean() < 0.65


def test_synthetic_labels_match_the_realised_sentiment():
    """Labels are derived from the text that was actually generated."""
    texts, labels = make_synthetic_reviews(400, seed=1)
    from src.data import NEGATIONS, NEGATIVE_WORDS, POSITIVE_WORDS

    for text, label in zip(texts[:60], labels[:60]):
        words = text.rstrip(".").split()
        score = 0
        for i, word in enumerate(words):
            if word in POSITIVE_WORDS or word in NEGATIVE_WORDS:
                polarity = 1 if word in POSITIVE_WORDS else -1
                # Look back past an optional intensifier for a negation.
                window = words[max(0, i - 2): i]
                if any(w in NEGATIONS for w in window):
                    polarity *= -1
                score += polarity
        assert (score > 0) == bool(label), text


def test_negation_makes_the_task_non_trivial_for_word_counts():
    """A naive positive-minus-negative word count should be clearly imperfect."""
    texts, labels = make_synthetic_reviews(800, seed=2)
    from src.data import NEGATIVE_WORDS, POSITIVE_WORDS

    naive = np.array([
        sum(w in POSITIVE_WORDS for w in t.split())
        - sum(w in NEGATIVE_WORDS for w in t.split()) > 0
        for t in texts
    ]).astype(int)
    accuracy = (naive == labels).mean()
    assert accuracy < 0.9, f"ignoring negation still reaches {accuracy:.3f}"


def test_evaluate_reports_accuracy_and_macro_f1():
    logits = np.array([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0]])
    results = evaluate(np.array([0, 1, 0, 1]), logits, ["negative", "positive"])
    assert results["accuracy"] == 1.0
    assert results["macro_f1"] == 1.0


def test_ece_detects_overconfidence():
    probs = np.tile([0.95, 0.05], (100, 1))
    y = np.array([0] * 50 + [1] * 50)
    assert expected_calibration_error(y, probs) == pytest.approx(0.45, abs=0.02)


def test_render_importance_is_printable_on_a_legacy_console_codepage():
    """render_importance used to build its bar with the Unicode block
    character '█', which crashes with UnicodeEncodeError the moment it
    reaches print() on a console still using cp1252 -- the Windows default
    outside Windows Terminal. The whole point of the report is that it
    gets printed, so the string this returns must survive that encoding.
    """
    from src.predict import render_importance

    tokens = ["the", "movie", "was", "great"]
    importance = np.array([0.1, 0.8, 0.05, 0.9])
    valid = np.array([True, True, True, True])
    text = render_importance(tokens, importance, valid, top_k=4)
    text.encode("cp1252")  # raises UnicodeEncodeError on a regression
