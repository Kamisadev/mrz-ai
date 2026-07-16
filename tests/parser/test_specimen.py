"""Ground truth: the ICAO 9303 specimen passport.

If these pass, the check-digit arithmetic and the TD3 field geometry are both
correct against a published, known-good vector.
"""

from __future__ import annotations

import pytest

from mrz_ai.parser import compute_check_digit, parse, serialize, validate
from mrz_ai.parser.types import TD3Fields

SPECIMEN_LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
SPECIMEN_LINE2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
SPECIMEN = f"{SPECIMEN_LINE1}\n{SPECIMEN_LINE2}"

SPECIMEN_FIELDS = TD3Fields(
    issuing_state="UTO",
    primary_name="ERIKSSON",
    secondary_name="ANNA MARIA",
    document_number="L898902C3",
    nationality="UTO",
    birth_date="740812",
    sex="F",
    expiry_date="120415",
    optional_data="ZE184226B",
)


def test_specimen_lines_are_44_characters() -> None:
    assert len(SPECIMEN_LINE1) == 44
    assert len(SPECIMEN_LINE2) == 44


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("L898902C3", "6"),
        ("740812", "2"),
        ("120415", "9"),
        ("ZE184226B<<<<<", "1"),
        ("L898902C36" + "7408122" + "120415" + "9" + "ZE184226B<<<<<" + "1", "0"),
    ],
    ids=["document_number", "birth_date", "expiry_date", "optional_data", "composite"],
)
def test_specimen_check_digits(payload: str, expected: str) -> None:
    assert compute_check_digit(payload) == expected


def test_serialize_reproduces_the_specimen() -> None:
    # The specimen carries a real optional-data value, so its check digit is a
    # digit rather than a filler and the flag makes no difference here.
    assert serialize(SPECIMEN_FIELDS) == SPECIMEN


def test_parse_recovers_the_specimen_fields() -> None:
    document = parse(SPECIMEN)

    assert document.fields == SPECIMEN_FIELDS
    assert document.check_digits.document_number == "6"
    assert document.check_digits.birth_date == "2"
    assert document.check_digits.expiry_date == "9"
    assert document.check_digits.optional_data == "1"
    assert document.check_digits.composite == "0"


def test_specimen_validates() -> None:
    # 2015 keeps the specimen's 2012 expiry in the past without it being absurd,
    # and pins the birth-year pivot so the test does not rot.
    result = validate(parse(SPECIMEN), reference_year=2015)
    assert result.is_valid, [str(issue) for issue in result.issues]
