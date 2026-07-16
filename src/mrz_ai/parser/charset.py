"""MRZ character set (ICAO 9303).

The MRZ alphabet is 37 characters: A-Z, 0-9 and the filler '<'.
"""

from __future__ import annotations

FILLER = "<"

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"

#: Canonical ordering used by the recognition model's classification head.
ALPHABET = LETTERS + DIGITS + FILLER

CHAR_TO_INDEX = {c: i for i, c in enumerate(ALPHABET)}
INDEX_TO_CHAR = {i: c for c, i in CHAR_TO_INDEX.items()}

ALPHABET_SIZE = len(ALPHABET)


def is_mrz_char(char: str) -> bool:
    return char in CHAR_TO_INDEX


def char_value(char: str) -> int:
    """Numeric value of an MRZ character for check-digit arithmetic.

    Digits map to themselves, A-Z map to 10-35, and the filler maps to 0.
    """
    if char in DIGITS:
        return ord(char) - 48
    if char in LETTERS:
        return ord(char) - 55
    if char == FILLER:
        return 0
    raise ValueError(f"character {char!r} is not in the MRZ alphabet")
