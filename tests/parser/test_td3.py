"""Encoding edge cases and validation behaviour."""

from __future__ import annotations

import pytest

from mrz_ai.parser import (
    MRZFieldError,
    MRZFormatError,
    decode_name,
    encode_name,
    parse,
    serialize,
    validate,
)
from mrz_ai.parser import fields as F
from mrz_ai.parser.types import TD3Fields

from .test_specimen import SPECIMEN, SPECIMEN_FIELDS


def _fields(**overrides: object) -> TD3Fields:
    from dataclasses import replace

    return replace(SPECIMEN_FIELDS, **overrides)  # type: ignore[arg-type]


# --- names -----------------------------------------------------------------

def test_encode_name_pads_to_the_full_field() -> None:
    assert len(encode_name("SMITH", "JOHN")) == F.NAME_LENGTH


def test_encode_name_maps_spaces_and_hyphens_to_filler() -> None:
    assert encode_name("VAN DER BERG", "MARY-JANE").startswith("VAN<DER<BERG<<MARY<JANE")


def test_encode_name_truncates_an_over_long_name() -> None:
    encoded = encode_name("A" * 30, "B" * 30)
    assert len(encoded) == F.NAME_LENGTH
    assert not encoded.endswith("<")  # the field is full, nothing left to pad


def test_name_round_trips() -> None:
    assert decode_name(encode_name("ERIKSSON", "ANNA MARIA")) == ("ERIKSSON", "ANNA MARIA")


def test_name_without_a_secondary_component_round_trips() -> None:
    assert decode_name(encode_name("CHER", "")) == ("CHER", "")


# --- optional data / check-digit policy ------------------------------------

def test_unused_optional_data_gets_a_filler_check_digit_by_default() -> None:
    line2 = serialize(_fields(optional_data="")).splitlines()[1]
    assert line2[F.OPTIONAL_DATA_CD] == "<"


def test_unused_optional_data_can_use_a_zero_check_digit() -> None:
    line2 = serialize(_fields(optional_data=""), filler_optional_cd=False).splitlines()[1]
    assert line2[F.OPTIONAL_DATA_CD] == "0"


def test_both_optional_check_digit_forms_validate() -> None:
    # '<' and '0' both carry the value 0, so the composite digit is identical
    # and each form must be accepted on the way back in.
    for filler in (True, False):
        mrz = serialize(_fields(optional_data=""), filler_optional_cd=filler)
        assert validate(parse(mrz), reference_year=2015).is_valid


def test_the_two_optional_check_digit_forms_agree_on_the_composite() -> None:
    with_filler = serialize(_fields(optional_data=""), filler_optional_cd=True)
    with_zero = serialize(_fields(optional_data=""), filler_optional_cd=False)
    assert with_filler.splitlines()[1][F.COMPOSITE_CD] == with_zero.splitlines()[1][F.COMPOSITE_CD]


# --- serialize round-trip --------------------------------------------------

@pytest.mark.parametrize("sex", ["M", "F", "<"])
def test_every_sex_value_round_trips(sex: str) -> None:
    document = parse(serialize(_fields(sex=sex)))
    assert document.fields.sex == sex
    assert validate(document, reference_year=2015).is_valid


def test_serialize_round_trips_through_parse() -> None:
    assert parse(serialize(SPECIMEN_FIELDS)).fields == SPECIMEN_FIELDS


def test_serialize_rejects_an_over_long_document_number() -> None:
    with pytest.raises(MRZFieldError, match="document_number"):
        serialize(_fields(document_number="X" * 10))


def test_serialize_rejects_a_non_mrz_character() -> None:
    with pytest.raises(MRZFieldError, match="not an MRZ character"):
        serialize(_fields(primary_name="MÜLLER"))


def test_serialize_rejects_a_malformed_sex() -> None:
    with pytest.raises(MRZFieldError, match="sex"):
        serialize(_fields(sex="X"))


# --- parse rejects malformed input -----------------------------------------

def test_parse_rejects_a_short_line() -> None:
    line1, line2 = SPECIMEN.splitlines()
    with pytest.raises(MRZFormatError, match="44"):
        parse(f"{line1[:-1]}\n{line2}")


def test_parse_rejects_the_wrong_line_count() -> None:
    with pytest.raises(MRZFormatError, match="2 lines"):
        parse(SPECIMEN.splitlines()[0])


def test_parse_rejects_a_character_outside_the_alphabet() -> None:
    line1, line2 = SPECIMEN.splitlines()
    with pytest.raises(MRZFormatError, match="not an MRZ character"):
        parse(f"{line1[:-1]}!\n{line2}")


# --- validation catches OCR-shaped corruption ------------------------------

def _corrupt(mrz: str, index: int, char: str) -> str:
    line1, line2 = mrz.splitlines()
    line2 = line2[:index] + char + line2[index + 1 :]
    return f"{line1}\n{line2}"


def test_a_single_character_error_is_caught_by_a_check_digit() -> None:
    # '8' misread as 'B' is the canonical OCR confusion; the check digits exist
    # precisely so the candidate decoder can reject it.
    corrupted = _corrupt(SPECIMEN, 1, "B")
    result = validate(parse(corrupted), reference_year=2015)

    assert not result.is_valid
    assert {issue.field for issue in result.issues} >= {"document_number_cd"}


def test_a_corrupted_check_digit_fails_the_composite_too() -> None:
    result = validate(parse(_corrupt(SPECIMEN, F.BIRTH_DATE_CD, "9")), reference_year=2015)
    assert {issue.field for issue in result.issues} == {"birth_date_cd", "composite_cd"}


def test_validation_reports_every_issue_rather_than_the_first() -> None:
    corrupted = _corrupt(_corrupt(SPECIMEN, 1, "B"), F.SEX, "X")
    result = validate(parse(corrupted), reference_year=2015)
    assert len(result.issues) >= 2


def test_an_impossible_date_is_rejected() -> None:
    fields = _fields(birth_date="740899")  # day 99
    mrz = serialize(fields)
    result = validate(parse(mrz), reference_year=2015)
    assert any(issue.field == "birth_date" for issue in result.issues)


def test_expiry_before_birth_is_rejected() -> None:
    # Both pivots have to resolve into the same century for the comparison to
    # mean anything: born 2015, expiring 2005.
    mrz = serialize(_fields(birth_date="150601", expiry_date="050101"))
    result = validate(parse(mrz), reference_year=2015)
    assert any("precedes" in issue.message for issue in result.issues)


def test_an_expiry_year_resolves_within_the_reference_century() -> None:
    from mrz_ai.parser.dates import resolve_expiry_year

    assert resolve_expiry_year(12, 2015) == 2012


def test_a_valid_mrz_has_no_issues_and_is_truthy() -> None:
    assert validate(parse(SPECIMEN), reference_year=2015)
