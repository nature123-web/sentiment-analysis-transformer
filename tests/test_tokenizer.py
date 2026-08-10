"""Tests for the from-scratch BPE tokenizer."""

import pytest

from src.tokenizer import (
    BOS_ID,
    END_OF_WORD,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    UNK_ID,
    BPETokenizer,
    basic_tokenize,
)

CORPUS = [
    "the film was brilliant and utterly gripping",
    "a brilliant story with a brilliant cast",
    "the movie was tedious and clumsy",
    "utterly tedious plot and a clumsy ending",
    "gripping cinematography and a heartfelt story",
] * 6


def train_tokenizer(vocab_size=200, min_frequency=2):
    return BPETokenizer(vocab_size, min_frequency).train(CORPUS)


def test_basic_tokenize_splits_punctuation():
    assert basic_tokenize("Great film!") == ["great", "film", "!"]


def test_basic_tokenize_keeps_apostrophes():
    """Contractions matter for sentiment: don't must not become do + n't + t."""
    assert basic_tokenize("it isn't good") == ["it", "isn't", "good"]


def test_special_tokens_occupy_the_first_ids():
    tok = train_tokenizer()
    assert tok.id_to_token[:4] == SPECIAL_TOKENS
    assert (PAD_ID, UNK_ID, BOS_ID, EOS_ID) == (0, 1, 2, 3)


def test_vocabulary_respects_the_size_limit():
    tok = BPETokenizer(vocab_size=60, min_frequency=1).train(CORPUS)
    assert len(tok) <= 60


def test_merges_are_learned():
    tok = train_tokenizer()
    assert len(tok.merges) > 0


def test_frequent_word_becomes_a_single_token():
    """'brilliant' appears often enough that BPE should merge it whole."""
    tok = BPETokenizer(vocab_size=300, min_frequency=2).train(CORPUS)
    assert tok.tokenize_word("brilliant") == ["brilliant" + END_OF_WORD]


def test_unseen_word_decomposes_instead_of_becoming_unk():
    """The main reason to use BPE: rare words keep their morphology."""
    tok = train_tokenizer()
    pieces = tok.tokenize_word("brilliantly")
    assert len(pieces) > 1
    ids = [tok.token_to_id.get(p, UNK_ID) for p in pieces]
    assert ids.count(UNK_ID) < len(ids), "the whole word collapsed to <unk>"


def test_tokenize_reconstructs_the_original_characters():
    tok = train_tokenizer()
    for word in ("brilliant", "clumsy", "unseenword"):
        joined = "".join(tok.tokenize_word(word)).replace(END_OF_WORD, "")
        assert joined == word


def test_encode_decode_roundtrip():
    tok = train_tokenizer()
    text = "the film was brilliant"
    assert tok.decode(tok.encode(text)) == text


def test_encode_adds_bos_and_eos():
    tok = train_tokenizer()
    ids = tok.encode("brilliant film", add_special_tokens=True)
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID


def test_encode_pads_to_max_length():
    tok = train_tokenizer()
    ids = tok.encode("brilliant", max_length=32)
    assert len(ids) == 32
    assert ids[-1] == PAD_ID


def test_encode_truncates_to_max_length():
    tok = train_tokenizer()
    ids = tok.encode("the film was brilliant and utterly gripping " * 20,
                     max_length=24)
    assert len(ids) == 24
    assert ids[-1] == EOS_ID, "truncation must preserve the end-of-sequence token"


def test_truncation_keeps_both_ends_of_the_document():
    """Middle-out truncation: the opening and the conclusion both survive."""
    tok = BPETokenizer(vocab_size=300, min_frequency=1).train(
        CORPUS + ["alpha " * 5 + "omega"]
    )
    text = "alpha " * 60 + "omega"
    ids = tok.encode(text, max_length=30)
    decoded = tok.decode(ids)
    assert "alpha" in decoded
    assert "omega" in decoded, "the tail of the document was discarded"


def test_tokenization_is_deterministic():
    tok = train_tokenizer()
    assert tok.tokenize("a brilliant film") == tok.tokenize("a brilliant film")


def test_merge_order_is_by_rank_not_left_to_right():
    """Applying the earliest-learned merge first is what training did."""
    tok = train_tokenizer()
    word = "gripping"
    pieces = tok.tokenize_word(word)
    # Every produced piece must be a real vocabulary entry.
    for piece in pieces:
        assert piece in tok.token_to_id, f"{piece!r} is not in the vocabulary"


def test_save_and_load_roundtrip(tmp_path):
    tok = train_tokenizer()
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)

    assert len(loaded) == len(tok)
    assert loaded.merges == tok.merges
    text = "a brilliant and gripping film"
    assert loaded.encode(text) == tok.encode(text)


def test_loaded_tokenizer_handles_unseen_text(tmp_path):
    tok = train_tokenizer()
    path = tmp_path / "t.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert isinstance(loaded.encode("completely novel wording here"), list)


def test_min_frequency_stops_rare_merges():
    tok = BPETokenizer(vocab_size=5000, min_frequency=1000).train(CORPUS)
    assert tok.merges == []


def test_empty_text_encodes_to_specials_only():
    tok = train_tokenizer()
    assert tok.encode("") == [BOS_ID, EOS_ID]
