"""TD3 (passport) MRZ geometry: two lines of 44 characters.

Positions are fixed by ICAO 9303 Part 4, so every field is a constant slice.
Keeping them in one place lets the synthetic generator, the parser and the
recognition post-processor share a single source of truth.
"""

from __future__ import annotations

LINE_LENGTH = 44
LINE_COUNT = 2

# --- line 1 -----------------------------------------------------------------
DOC_CODE = slice(0, 2)  # 'P<' for an ordinary passport
ISSUING_STATE = slice(2, 5)
NAME = slice(5, 44)

NAME_LENGTH = NAME.stop - NAME.start  # 39
NAME_SEPARATOR = "<<"  # primary<<secondary

# --- line 2 -----------------------------------------------------------------
DOCUMENT_NUMBER = slice(0, 9)
DOCUMENT_NUMBER_CD = 9
NATIONALITY = slice(10, 13)
BIRTH_DATE = slice(13, 19)
BIRTH_DATE_CD = 19
SEX = 20
EXPIRY_DATE = slice(21, 27)
EXPIRY_DATE_CD = 27
OPTIONAL_DATA = slice(28, 42)
OPTIONAL_DATA_CD = 42
COMPOSITE_CD = 43

OPTIONAL_DATA_LENGTH = OPTIONAL_DATA.stop - OPTIONAL_DATA.start  # 14

#: Slices of line 2 that feed the composite check digit, in order:
#: document number + its check digit, birth date + its check digit,
#: expiry date + its check digit, optional data + its check digit.
COMPOSITE_PARTS = (slice(0, 10), slice(13, 20), slice(21, 43))

VALID_SEX = frozenset({"M", "F", "<"})

#: Positions of line 2 that no check digit protects: nationality (10-12) and sex
#: (20). ICAO deliberately leaves them out of the composite, so a misread here is
#: invisible to validation — an 'M' read as 'F' produces an MRZ that is entirely
#: well-formed. The candidate decoder must rank these positions on model
#: confidence alone, and line 1 (name and issuing state) is unprotected likewise.
UNPROTECTED_LINE2 = tuple(range(10, 13)) + (20,)

#: Positions of line 2 that a check digit does cover, the composite digit aside.
PROTECTED_LINE2 = tuple(
    i for i in range(LINE_LENGTH) if i not in UNPROTECTED_LINE2 and i != COMPOSITE_CD
)


def composite_payload(line2: str) -> str:
    """Concatenate the segments of ``line2`` covered by the composite digit."""
    return "".join(line2[part] for part in COMPOSITE_PARTS)
