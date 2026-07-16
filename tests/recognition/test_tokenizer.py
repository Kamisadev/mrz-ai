"""Text to indices and back."""

from __future__ import annotations

import random

import numpy as np
import pytest

from mrz_ai.parser import serialize
from mrz_ai.parser.charset import ALPHABET
from mrz_ai.recognition.tokenizer import LABEL_LENGTH, NUM_CLASSES, decode, decode_batch, encode
from mrz_ai.synthetic.identity import random_identity

SPECIMEN_LINE = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"[:44].ljust(44, "<")


def test_a_line_round_trips() -> None:
    line = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    assert decode(encode(line)) == line


def test_every_generated_line_round_trips() -> None:
    for seed in range(200):
        for line in serialize(random_identity(random.Random(seed))).split("\n"):
            assert decode(encode(line)) == line


def test_encoding_gives_one_index_per_character() -> None:
    encoded = encode("A" * 44)
    assert encoded.shape == (LABEL_LENGTH,)
    assert encoded.dtype == np.int64


@pytest.mark.parametrize("char", list(ALPHABET))
def test_every_character_of_the_alphabet_survives(char: str) -> None:
    assert decode(encode(char * 44)) == char * 44


def test_indices_stay_inside_the_alphabet() -> None:
    encoded = encode("".join(ALPHABET[i % NUM_CLASSES] for i in range(44)))
    assert encoded.min() >= 0
    assert encoded.max() < NUM_CLASSES


def test_the_filler_is_a_real_class_not_padding() -> None:
    # '<' carries meaning in an MRZ, so it must be a class like any other.
    assert encode("<" * 44).tolist() == [ALPHABET.index("<")] * 44


def test_a_short_line_is_rejected() -> None:
    with pytest.raises(ValueError, match="44"):
        encode("P<UTO")


def test_a_non_mrz_character_is_rejected() -> None:
    with pytest.raises(ValueError, match="not an MRZ character"):
        encode("a" * 44)


def test_decoding_the_wrong_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="44"):
        decode(np.zeros(10, dtype=np.int64))


def test_decoding_an_out_of_range_index_is_rejected() -> None:
    # Silently mapping a bad index would hide a bug upstream.
    with pytest.raises(ValueError, match="class index"):
        decode(np.full(44, NUM_CLASSES, dtype=np.int64))


def test_a_batch_decodes() -> None:
    lines = ["A" * 44, "<" * 44, "0" * 44]
    assert decode_batch(np.stack([encode(line) for line in lines])) == lines


def test_decode_batch_rejects_a_flat_array() -> None:
    with pytest.raises(ValueError, match="2-D"):
        decode_batch(np.zeros(44, dtype=np.int64))
