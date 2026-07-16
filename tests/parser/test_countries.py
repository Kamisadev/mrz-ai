"""The country-code allowlist.

Nationality and issuing state carry no check digit, so this list is the only
thing standing between an OCR error and a silently wrong field.
"""

from __future__ import annotations

import pytest

from mrz_ai.parser.countries import ICAO_SPECIAL, ISO_3166_1_ALPHA_3, VALID_CODES, is_valid_code


@pytest.mark.parametrize("code", ["USA", "THA", "GBR", "DEU", "JPN", "AUS", "AUT"])
def test_ordinary_countries_are_accepted(code: str) -> None:
    assert is_valid_code(code)


def test_the_specimen_country_is_accepted() -> None:
    # 'UTO' (Utopia) is fictional and ICAO-specimen-only, but rejecting it would
    # fail our own ground-truth test.
    assert is_valid_code("UTO")


@pytest.mark.parametrize("code", ["XXA", "XXB", "XXC", "XXX", "UNO", "UNK", "EUE", "RKS"])
def test_icao_special_codes_are_accepted(code: str) -> None:
    assert is_valid_code(code)


def test_germany_is_accepted_in_its_padded_form() -> None:
    # Germany writes its code as 'D<<' rather than 'DEU' on the MRZ.
    assert is_valid_code("D<<")


@pytest.mark.parametrize("code", ["QQQ", "AUZ", "US", "ZZZ", "123", "<<<"])
def test_codes_that_do_not_exist_are_rejected(code: str) -> None:
    assert not is_valid_code(code)


def test_the_iso_list_looks_complete() -> None:
    # ISO 3166-1 defines 249 alpha-3 codes; guard against a truncated paste.
    assert len(ISO_3166_1_ALPHA_3) > 240


def test_every_iso_code_is_three_letters() -> None:
    assert all(len(code) == 3 and code.isalpha() for code in ISO_3166_1_ALPHA_3)


def test_the_special_codes_do_not_shadow_iso_codes() -> None:
    # A code appearing in both lists would mean one of them is wrong.
    assert not (ICAO_SPECIAL & ISO_3166_1_ALPHA_3)


def test_valid_codes_is_the_union() -> None:
    assert VALID_CODES == ISO_3166_1_ALPHA_3 | ICAO_SPECIAL
