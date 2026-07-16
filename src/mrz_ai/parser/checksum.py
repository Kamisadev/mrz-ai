"""ICAO 9303 check-digit arithmetic.

Every character is weighted by a repeating 7-3-1 pattern; the check digit is the
sum of the weighted values modulo 10.
"""

from __future__ import annotations

from .charset import FILLER, char_value

WEIGHTS = (7, 3, 1)


def compute_check_digit(value: str) -> str:
    """Return the check digit for ``value`` as a single character '0'-'9'."""
    total = sum(
        char_value(char) * WEIGHTS[i % len(WEIGHTS)] for i, char in enumerate(value)
    )
    return str(total % 10)


def verify_check_digit(value: str, digit: str, *, allow_filler: bool = False) -> bool:
    """Check that ``digit`` is the correct check digit for ``value``.

    ICAO permits the filler '<' in place of the check digit when the field it
    guards is itself unused. ``allow_filler`` opts into that leniency; the
    filler is only accepted when ``value`` is entirely filler.
    """
    if allow_filler and digit == FILLER and set(value) <= {FILLER}:
        return True
    return digit == compute_check_digit(value)
