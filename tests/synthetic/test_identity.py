"""The generator's contract with the parser.

Every identity must serialize to an MRZ that validates. A generator that emits
even a small fraction of invalid labels poisons training silently, so this is
checked in bulk rather than on a handful of examples.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from mrz_ai.parser import ALPHABET, parse, serialize, validate
from mrz_ai.parser.countries import is_valid_code
from mrz_ai.parser.dates import resolve_birth_year, resolve_expiry_year
from mrz_ai.synthetic.identity import IdentityConfig, random_identity

REFERENCE_YEAR = 2026


def _identities(count: int, config: IdentityConfig | None = None):
    return [random_identity(random.Random(seed), config) for seed in range(count)]


def test_every_identity_serializes_to_a_valid_mrz() -> None:
    for seed, fields in enumerate(_identities(2000)):
        result = validate(parse(serialize(fields)), reference_year=REFERENCE_YEAR)
        assert result.is_valid, f"seed {seed}: {[str(i) for i in result.issues]}"


def test_an_identity_is_reproducible_from_its_seed() -> None:
    # Reproducible training runs depend on this.
    assert random_identity(random.Random(7)) == random_identity(random.Random(7))


def test_different_seeds_give_different_identities() -> None:
    assert len({serialize(f) for f in _identities(200)}) > 190


def test_a_passport_never_expires_before_its_holder_is_born() -> None:
    """Clamping the year alone once let a passport expire in March of a December
    birth year, invalidating roughly 1% of identities.

    The comparison has to resolve centuries first: a birth of '360710' and an
    expiry of '360216' share a two-digit year but are a hundred years apart —
    born 1936, expiring 2036 — and that identity is entirely valid.
    """
    for fields in _identities(5000):
        birth = (
            resolve_birth_year(int(fields.birth_date[:2]), REFERENCE_YEAR),
            fields.birth_date[2:],
        )
        expiry = (
            resolve_expiry_year(int(fields.expiry_date[:2]), REFERENCE_YEAR),
            fields.expiry_date[2:],
        )
        assert expiry > birth, f"{fields.birth_date} -> {fields.expiry_date}"


def test_the_whole_alphabet_appears() -> None:
    # A character the model never sees in training is one it cannot read.
    seen: Counter[str] = Counter()
    for fields in _identities(4000):
        seen.update(serialize(fields).replace("\n", ""))
    assert {c for c in ALPHABET} <= set(seen), sorted(set(ALPHABET) - set(seen))


def test_country_codes_are_always_real() -> None:
    for fields in _identities(500):
        assert is_valid_code(fields.issuing_state)
        assert is_valid_code(fields.nationality)


@pytest.mark.parametrize(
    ("attribute", "probability"),
    [("empty_optional_probability", 0.45), ("expired_probability", 0.15)],
)
def test_the_awkward_cases_appear_at_roughly_their_configured_rate(
    attribute: str, probability: float
) -> None:
    # These exist so that rare-but-legal documents are not rare in training.
    import dataclasses

    config = dataclasses.replace(IdentityConfig(), **{attribute: probability})
    identities = _identities(3000, config)

    if attribute == "empty_optional_probability":
        hits = sum(1 for f in identities if not f.optional_data)
    else:
        hits = sum(1 for f in identities if int(f.expiry_date[:2]) + 2000 < REFERENCE_YEAR)

    assert probability - 0.06 < hits / len(identities) < probability + 0.06


def test_sex_is_sometimes_unspecified() -> None:
    assert any(f.sex == "<" for f in _identities(1000))


def test_names_are_sometimes_truncated_to_the_field_width() -> None:
    # The 39-character name field is a real constraint and must be exercised.
    assert any(len(serialize(f).split("\n")[0][5:].rstrip("<")) == 39 for f in _identities(2000))
