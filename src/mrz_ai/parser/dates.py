"""Century resolution and calendar checks for MRZ ``YYMMDD`` dates.

The MRZ carries only two year digits, so the century is a policy decision rather
than something the string can tell us. Both helpers take an explicit reference
year instead of reading the clock, which keeps the parser pure and lets tests
pin the behaviour.
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional

from .charset import DIGITS


def is_calendar_date(yy: int, mm: int, dd: int, century: int) -> bool:
    try:
        _dt.date(century + yy, mm, dd)
    except ValueError:
        return False
    return True


def parse_yymmdd(value: str) -> Optional[tuple[int, int, int]]:
    """Split a 6-digit MRZ date, or return ``None`` if it is not 6 digits."""
    if len(value) != 6 or any(c not in DIGITS for c in value):
        return None
    return int(value[0:2]), int(value[2:4]), int(value[4:6])


def resolve_birth_year(yy: int, reference_year: int) -> int:
    """Resolve a birth ``YY`` to a full year using a sliding pivot.

    A two-digit birth year that is greater than the current two-digit year must
    belong to the previous century, since nobody is born in the future.
    """
    reference_yy = reference_year % 100
    century = (reference_year // 100) * 100
    return century + yy if yy <= reference_yy else century - 100 + yy


def resolve_expiry_year(yy: int, reference_year: int) -> int:
    """Resolve an expiry ``YY`` to a full year, within the reference century.

    Unlike a birth year, an expiry has no direction we can lean on: it may sit
    in the past (an expired passport) or the future, so there is no pivot that
    is right in every case. We simply anchor to the reference century.

    The cost is that this breaks down near a century boundary — a passport
    expiring in 2103, read in 2098, resolves to 2003. The alternative, rolling
    long-past years forward, gets that case right but then misreads every
    passport that expired more than ten years ago, which is an ordinary document
    to be handed today. We take the error that is 70 years away over the one
    that is here now.
    """
    return (reference_year // 100) * 100 + yy
