"""The invariant the synthetic engine rests on.

Phase 1 generates a random identity, serializes it, and ships the string as a
training label. If serialize ever emits an MRZ that fails validation, every
sample built from it teaches the model something false. This fuzzes that
contract over the whole input space.
"""

from __future__ import annotations

import dataclasses
import random
import string

import pytest

from mrz_ai.parser import ALPHABET, char_value, compute_check_digit, parse, serialize, validate
from mrz_ai.parser import fields as F
from mrz_ai.parser.countries import VALID_CODES
from mrz_ai.parser.types import TD3Fields

REFERENCE_YEAR = 2026
ALNUM = string.ascii_uppercase + string.digits
COUNTRY_CODES = sorted(VALID_CODES)


def random_fields(rng: random.Random) -> TD3Fields:
    # Birth years are drawn from 1930-1999 and 2000-2015 so that the holder is
    # always born before the passport expires, and both centuries of the pivot
    # get exercised. Years 2016-2026 are skipped: against a 2016+ expiry they
    # would describe someone born after their own document ran out.
    birth_yy = rng.choice([rng.randint(30, 99), rng.randint(0, 15)])
    return TD3Fields(
        issuing_state=rng.choice(COUNTRY_CODES),
        primary_name="".join(rng.choices(string.ascii_uppercase, k=rng.randint(1, 20))),
        secondary_name="".join(rng.choices(string.ascii_uppercase + " ", k=rng.randint(0, 20))),
        document_number="".join(rng.choices(ALNUM, k=rng.randint(1, 9))),
        nationality=rng.choice(COUNTRY_CODES),
        birth_date=f"{birth_yy:02d}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}",
        sex=rng.choice(["M", "F", "<"]),
        expiry_date=f"{rng.randint(16, 36):02d}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}",
        optional_data="".join(rng.choices(ALNUM, k=rng.randint(0, 14))),
    )


@pytest.mark.parametrize("seed", range(200))
def test_a_random_identity_serializes_to_a_valid_mrz(seed: int) -> None:
    rng = random.Random(seed)
    fields = random_fields(rng)

    mrz = serialize(fields)
    line1, line2 = mrz.splitlines()

    assert len(line1) == F.LINE_LENGTH
    assert len(line2) == F.LINE_LENGTH

    result = validate(parse(mrz), reference_year=REFERENCE_YEAR)
    assert result.is_valid, [str(issue) for issue in result.issues]


@pytest.mark.parametrize("seed", range(200))
def test_serialization_is_deterministic(seed: int) -> None:
    # Reproducible training data requires that the same identity always encodes
    # to the same bytes.
    fields = random_fields(random.Random(seed))
    assert serialize(fields) == serialize(fields)


def _corrupt_line2(mrz: str, index: int, char: str) -> str:
    line1, line2 = mrz.splitlines()
    return f"{line1}\n{line2[:index] + char + line2[index + 1:]}"


def blind_class(char: str) -> int:
    """The equivalence class of characters no check digit can separate.

    A substitution shifts a check digit by ``delta * weight`` mod 10. Every
    weight in the 7-3-1 pattern is coprime with 10, so the digit is unchanged
    exactly when ``delta`` is a multiple of 10 — making two characters
    indistinguishable when their values agree mod 10, whatever their position.
    """
    return char_value(char) % 10


#: The blind classes in full. Note that '<', '0', 'A', 'K' and 'U' all sit in
#: class 0, so a filler and a zero are equivalent to every check digit — which is
#: why ICAO can allow either as the unused optional-data check digit.
BLIND_CLASSES = {
    cls: [c for c in ALPHABET if blind_class(c) == cls] for cls in range(10)
}


@pytest.mark.parametrize("seed", range(100))
def test_flipping_a_single_character_of_line_2_is_detected(seed: int) -> None:
    """A one-character error must break a check digit, blind classes aside.

    This is what the candidate decoder trades on: an OCR hypothesis with a
    single-character error should not pass validation. The name field carries no
    check digit, so only line 2 offers this guarantee.
    """
    rng = random.Random(seed)
    mrz = serialize(random_fields(rng), filler_optional_cd=False)
    line2 = mrz.splitlines()[1]

    index = rng.choice(F.PROTECTED_LINE2)
    original = line2[index]
    candidates = [c for c in ALPHABET if blind_class(c) != blind_class(original)]
    replacement = rng.choice(candidates)

    result = validate(
        parse(_corrupt_line2(mrz, index, replacement)), reference_year=REFERENCE_YEAR
    )
    assert not result.is_valid, (
        f"corrupting index {index} ({original!r}->{replacement!r}) went undetected"
    )


def test_sex_is_not_protected_by_any_check_digit() -> None:
    """'M' misread as 'F' yields a perfectly valid MRZ. Pin it: it is a spec fact.

    ICAO leaves position 20 out of the composite, so no amount of validation can
    catch this. The decoder cannot recover it and the API must not imply that a
    passing checksum vouches for the sex field.
    """
    mrz = serialize(dataclasses.replace(random_fields(random.Random(0)), sex="M"))
    flipped = _corrupt_line2(mrz, F.SEX, "F")

    assert validate(parse(flipped), reference_year=REFERENCE_YEAR).is_valid
    assert parse(flipped).fields.sex == "F"


def test_a_misread_nationality_is_caught_only_by_the_allowlist() -> None:
    """The composite skips positions 10-12, so the allowlist is the sole defence.

    'AUS' misread as 'AUZ' is caught, but only because no such country exists —
    no check digit moved.
    """
    mrz = serialize(dataclasses.replace(random_fields(random.Random(1)), nationality="AUS"))
    flipped = _corrupt_line2(mrz, 12, "Z")

    result = validate(parse(flipped), reference_year=REFERENCE_YEAR)
    assert [issue.field for issue in result.issues] == ["nationality"]


def test_a_misread_nationality_that_lands_on_a_real_country_is_undetectable() -> None:
    """'AUS' -> 'AUT' passes everything: Australia becomes Austria, silently.

    The allowlist cannot help when the substitution produces another valid code,
    and no check digit covers the field. This is the residual hole the decoder
    must cover with model confidence.
    """
    mrz = serialize(dataclasses.replace(random_fields(random.Random(1)), nationality="AUS"))
    flipped = _corrupt_line2(mrz, 12, "T")

    assert validate(parse(flipped), reference_year=REFERENCE_YEAR).is_valid
    assert parse(flipped).fields.nationality == "AUT"


@pytest.mark.parametrize("cls", range(10))
def test_the_check_digits_cannot_separate_a_blind_class(cls: int) -> None:
    """Pin the checksum's blind spot so Phase 4 does not assume it away.

    Any two characters whose values agree mod 10 ('0', 'A', 'K', 'U' and '<')
    produce byte-identical check digits in every position, so ICAO validation
    alone can never rank those hypotheses against each other. Two things save
    us: the pairs are not visually confusable in OCR-B, and the numeric fields
    reject letters outright. Where neither applies — the document number — the
    decoder must fall back on model confidence rather than on validity.
    """
    members = BLIND_CLASSES[cls]
    digits = {compute_check_digit(f"{c}23456789") for c in members}
    assert len(digits) == 1, f"class {cls} ({members}) is separable after all"


def test_a_filler_and_a_zero_are_interchangeable_to_the_checksum() -> None:
    assert blind_class("<") == blind_class("0")


def test_visually_confusable_ocr_pairs_are_caught() -> None:
    """The substitutions OCR actually makes are the ones checksums do catch."""
    from mrz_ai.parser import compute_check_digit

    for a, b in [("0", "O"), ("1", "I"), ("8", "B"), ("5", "S"), ("2", "Z")]:
        assert compute_check_digit(f"{a}23456789") != compute_check_digit(f"{b}23456789"), (
            f"{a!r}/{b!r} confusion is undetectable"
        )
