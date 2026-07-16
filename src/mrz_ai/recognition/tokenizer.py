"""Mapping between MRZ text and class indices.

Scene-text tokenizers carry [BOS], [EOS] and [PAD] because a word may be any
length. A TD3 line is always exactly 44 characters and the filler '<' is a real
character rather than padding, so none of that machinery is needed: the label is
44 indices into a 37-character alphabet, always.

That is worth being deliberate about. Adding an [EOS] class the model could emit
would let it produce a line shorter than 44 characters, which is not a thing a
TD3 MRZ can be — the model would have been handed a way to be wrong that the
document format does not allow.
"""

from __future__ import annotations

import numpy as np

from ..parser import fields as F
from ..parser.charset import ALPHABET, CHAR_TO_INDEX, INDEX_TO_CHAR, is_mrz_char

Array = np.ndarray

__all__ = ["encode", "decode", "decode_batch", "ALPHABET", "NUM_CLASSES", "LABEL_LENGTH"]

NUM_CLASSES = len(ALPHABET)
LABEL_LENGTH = F.LINE_LENGTH


def encode(text: str) -> Array:
    """Turn one 44-character MRZ line into an array of class indices."""
    if len(text) != LABEL_LENGTH:
        raise ValueError(f"a TD3 line is {LABEL_LENGTH} characters, got {len(text)}")
    for char in text:
        if not is_mrz_char(char):
            raise ValueError(f"{char!r} is not an MRZ character")
    return np.array([CHAR_TO_INDEX[char] for char in text], dtype=np.int64)


def decode(indices: Array) -> str:
    """Turn class indices back into text.

    Accepts anything index-like of the right length; out-of-range values raise
    rather than being silently mapped, since a bad index means a bug upstream.
    """
    values = np.asarray(indices).reshape(-1)
    if values.size != LABEL_LENGTH:
        raise ValueError(f"expected {LABEL_LENGTH} indices, got {values.size}")
    out = []
    for value in values.tolist():
        if value not in INDEX_TO_CHAR:
            raise ValueError(f"{value} is not a class index (0-{NUM_CLASSES - 1})")
        out.append(INDEX_TO_CHAR[value])
    return "".join(out)


def decode_batch(indices: Array) -> list[str]:
    """Decode a (batch, 44) array of indices."""
    values = np.asarray(indices)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D (batch, {LABEL_LENGTH}) array, got shape {values.shape}")
    return [decode(row) for row in values]
