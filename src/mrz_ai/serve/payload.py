"""A `Reading` as the page needs to see it.

The shaping here is mostly obvious — fields out of the document, dates with a
century, filler stripped. One part is not, and it is the reason this module
exists rather than a `dataclasses.asdict` call at the route.

A passport MRZ is not uniformly trustworthy. Line 2's fields are each covered by
a check digit, so a wrong one is almost always caught. The names on line 1 are
covered by nothing: 'ERIKSSQN' is exactly as valid an MRZ as 'ERIKSSON', and no
amount of validation will ever say otherwise. The country codes sit in between,
guarded only by an allowlist that catches most substitutions and no
transpositions.

So every field carries what guards it and whether that guard passed. A page that
showed one green tick for the document would be making a claim the standard does
not support — that the name was checked — and it would be most convincing exactly
when it was wrong, since a misread name never fails validation. `guard` is how
the page can be honest instead: verified means something checked it, not that
nobody objected.

Kept free of torch, and of the web framework, so it can be tested as what it is:
data shaping.
"""

from __future__ import annotations

from typing import Any

from ..inference.candidates import Reading
from ..parser.charset import FILLER
from ..parser.dates import (
    is_calendar_date,
    parse_yymmdd,
    resolve_birth_year,
    resolve_expiry_year,
)

__all__ = ["as_payload", "field_of"]

#: How a field is protected, if at all.
#:
#: ``checksum`` — a check digit covers it; a misread is caught.
#: ``allowlist`` — only a fixed set of values exists; most misreads are caught.
#: ``none`` — nothing covers it. A misread here is invisible, forever.
Guard = str

#: field key -> (label, guard, issue fields that condemn it)
_FIELDS: tuple[tuple[str, str, Guard, tuple[str, ...]], ...] = (
    ("document_code", "Document type", "allowlist", ("document_code",)),
    ("document_number", "Document number", "checksum", ("document_number_cd",)),
    ("issuing_state", "Issuing state", "allowlist", ("issuing_state",)),
    ("primary_name", "Surname", "none", ()),
    ("secondary_name", "Given names", "none", ()),
    ("nationality", "Nationality", "allowlist", ("nationality",)),
    ("sex", "Sex", "allowlist", ("sex",)),
    ("birth_date", "Date of birth", "checksum", ("birth_date_cd", "birth_date")),
    ("expiry_date", "Date of expiry", "checksum", ("expiry_date_cd", "expiry_date")),
    ("optional_data", "Optional data", "checksum", ("optional_data_cd",)),
)


def _clean(value: str) -> str:
    """Filler is padding, not content: a page should render it as absence."""
    return value.replace(FILLER, " ").strip()


def _iso(value: str, *, reference_year: int, birth: bool) -> str | None:
    """YYMMDD as a full date, or None if it is not one.

    The century is not in the MRZ. It is inferred from when the question is
    asked, and differently for the two date fields: a birth is in the past, an
    expiry is usually not.

    Six digits is not enough to format: '999999' is six digits and no date. It
    must come back as None so the caller shows the digits that were read, rather
    than a well-formed '1999-99-99' that the model never saw and nobody typed.
    """
    parts = parse_yymmdd(value)
    if parts is None:
        return None
    yy, mm, dd = parts
    year = resolve_birth_year(yy, reference_year) if birth else resolve_expiry_year(
        yy, reference_year
    )
    if not is_calendar_date(yy, mm, dd, year - yy):
        return None
    return f"{year:04d}-{mm:02d}-{dd:02d}"


def as_payload(reading: Reading, *, reference_year: int) -> dict[str, Any]:
    """The reading as JSON-ready data for the page.

    ``reference_year`` anchors the two-digit years, the same way `validate` does.
    """
    fields = reading.document.fields
    failed = {issue.field for issue in reading.issues}

    birth_iso = _iso(fields.birth_date, reference_year=reference_year, birth=True)
    expiry_iso = _iso(fields.expiry_date, reference_year=reference_year, birth=False)
    rendered = {"birth_date": birth_iso, "expiry_date": expiry_iso}

    # The composite digit is a checksum over every checksummed field on line 2 at
    # once. When it disagrees, the line is internally inconsistent, and a field's
    # own check digit passing no longer means much: a single digit catches most
    # single-character errors, not all of them. Observed, not theorised — a real
    # misread of document number '1028225<<' as '13X8225<<' satisfies the very
    # same check digit '8', and only the composite noticed. Reporting that field
    # as verified would put a tick beside a wrong number, which is precisely the
    # failure this page exists to not commit.
    composite_ok = "composite_cd" not in failed

    out: list[dict[str, Any]] = []
    for key, label, guard, condemning in _FIELDS:
        raw = _clean(getattr(fields, key))
        # A date that failed to parse has no ISO form; showing the raw digits is
        # the point — a human is about to retype them.
        value = rendered.get(key) or raw
        broken = bool(failed & set(condemning))

        if broken:
            status = "failed"
        elif guard == "none" or (guard == "checksum" and not composite_ok):
            status = "unverified"
        else:
            status = "verified"

        out.append(
            {
                "key": key,
                "label": label,
                "value": value,
                "raw": raw,
                "guard": guard,
                "status": status,
            }
        )

    return {
        "valid": reading.is_valid,
        # The model's belief that it read every one of the 88 characters right.
        # Not a probability that the passport is genuine, and not a checksum.
        "confidence": reading.confidence,
        "mrz": {"line1": reading.line1, "line2": reading.line2},
        "fields": out,
        "issues": [{"field": issue.field, "message": issue.message} for issue in reading.issues],
        # The composite digit covers line 2 as a whole; it belongs to the
        # document rather than to any one field.
        "composite_ok": composite_ok,
        "expired": expiry_iso is not None and expiry_iso < f"{reference_year:04d}-01-01",
    }


def field_of(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """The one field with ``key``. A helper for callers and tests."""
    for field in payload["fields"]:
        found: dict[str, Any] = field
        if found["key"] == key:
            return found
    raise KeyError(key)
