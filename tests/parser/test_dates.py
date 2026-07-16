"""Century resolution for the MRZ's two-digit years."""

from __future__ import annotations

import pytest

from mrz_ai.parser.dates import (
    is_calendar_date,
    parse_yymmdd,
    resolve_birth_year,
    resolve_expiry_year,
)


@pytest.mark.parametrize(
    ("yy", "reference_year", "expected"),
    [
        (74, 2015, 1974),  # in the past, previous century
        (15, 2015, 2015),  # born this year is possible
        (16, 2015, 1916),  # cannot be born next year, so it is a centenarian
        (0, 2015, 2000),
        (99, 2015, 1999),
    ],
)
def test_resolve_birth_year(yy: int, reference_year: int, expected: int) -> None:
    assert resolve_birth_year(yy, reference_year) == expected


def test_a_birth_year_never_resolves_into_the_future() -> None:
    for yy in range(100):
        assert resolve_birth_year(yy, 2026) <= 2026


@pytest.mark.parametrize(
    ("yy", "reference_year", "expected"),
    [
        (12, 2015, 2012),  # recently expired
        (25, 2015, 2025),  # still valid
        (10, 2026, 2010),  # long expired, and still resolved in this century
        (95, 2098, 2095),
    ],
)
def test_resolve_expiry_year(yy: int, reference_year: int, expected: int) -> None:
    assert resolve_expiry_year(yy, reference_year) == expected


def test_an_expiry_year_always_lands_in_the_reference_century() -> None:
    for yy in range(100):
        assert 2000 <= resolve_expiry_year(yy, 2026) <= 2099


def test_a_long_expired_passport_is_not_thrown_a_century_forward() -> None:
    # Handing over a passport that expired fifteen years ago is unremarkable;
    # resolving it to 2110 would be.
    assert resolve_expiry_year(10, 2026) == 2010


@pytest.mark.parametrize("value", ["740812", "000229", "991231"])
def test_parse_yymmdd_accepts_six_digits(value: str) -> None:
    assert parse_yymmdd(value) is not None


@pytest.mark.parametrize("value", ["7408", "7408123", "74081A", "<<<<<<"])
def test_parse_yymmdd_rejects_anything_else(value: str) -> None:
    assert parse_yymmdd(value) is None


def test_leap_day_is_a_real_date_in_a_leap_year() -> None:
    assert is_calendar_date(0, 2, 29, 2000)  # 2000 was a leap year


def test_leap_day_is_not_a_real_date_otherwise() -> None:
    assert not is_calendar_date(1, 2, 29, 1900)  # 1901 was not


@pytest.mark.parametrize(("mm", "dd"), [(13, 1), (0, 1), (1, 32), (1, 0), (99, 99)])
def test_impossible_month_and_day_combinations_are_rejected(mm: int, dd: int) -> None:
    assert not is_calendar_date(74, mm, dd, 1900)
