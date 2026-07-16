"""Serialize, parse and validate TD3 (passport) machine-readable zones.

``serialize`` is the direction the synthetic data engine depends on: it turns
field values into an ICAO-correct MRZ string with real check digits, so every
generated training label is well-formed by construction. ``parse`` is the
inverse, used at inference time, and ``validate`` is what lets the candidate
decoder reject OCR hypotheses that cannot be genuine documents.
"""

from __future__ import annotations


from . import fields as F
from .charset import FILLER, is_mrz_char
from .countries import is_valid_code
from .checksum import compute_check_digit, verify_check_digit
from .dates import is_calendar_date, parse_yymmdd, resolve_birth_year, resolve_expiry_year
from .errors import MRZFieldError, MRZFormatError
from .types import CheckDigits, Issue, TD3Document, TD3Fields, ValidationResult

__all__ = ["serialize", "parse", "validate", "encode_name", "decode_name"]


# --------------------------------------------------------------------------
# encoding helpers
# --------------------------------------------------------------------------

def _pad(value: str, length: int, name: str) -> str:
    if len(value) > length:
        raise MRZFieldError(f"{name} is {len(value)} characters, maximum is {length}")
    return value.ljust(length, FILLER)


def _normalize(value: str, name: str) -> str:
    """Uppercase a value and map spaces and hyphens onto the filler."""
    normalized = value.upper().replace(" ", FILLER).replace("-", FILLER)
    for char in normalized:
        if not is_mrz_char(char):
            raise MRZFieldError(f"{name} contains {char!r}, which is not an MRZ character")
    return normalized


def encode_name(primary: str, secondary: str) -> str:
    """Pack a name into the 39-character name field.

    Per ICAO 9303, an over-long name is truncated rather than rejected: the
    field is filled to its full width and the remainder is dropped.
    """
    primary = _normalize(primary, "primary_name")
    secondary = _normalize(secondary, "secondary_name")

    if secondary:
        packed = f"{primary}{F.NAME_SEPARATOR}{secondary}"
    else:
        packed = primary

    if len(packed) > F.NAME_LENGTH:
        packed = packed[: F.NAME_LENGTH]
    return packed.ljust(F.NAME_LENGTH, FILLER)


def decode_name(name_field: str) -> tuple[str, str]:
    """Split the name field back into (primary, secondary).

    Fillers become spaces. A truncated name cannot be recovered, so this is
    lossy in exactly the way the document itself is.
    """
    stripped = name_field.rstrip(FILLER)
    if F.NAME_SEPARATOR in stripped:
        primary, _, secondary = stripped.partition(F.NAME_SEPARATOR)
    else:
        primary, secondary = stripped, ""
    return (
        primary.replace(FILLER, " ").strip(),
        secondary.replace(FILLER, " ").strip(),
    )


# --------------------------------------------------------------------------
# serialize
# --------------------------------------------------------------------------

def serialize(fields: TD3Fields, *, filler_optional_cd: bool = True) -> str:
    """Render ``fields`` as a two-line TD3 MRZ with correct check digits.

    When the optional-data field is unused, ICAO allows its check digit to be
    written as a filler instead of '0'. ``filler_optional_cd`` selects that
    form. Both encode to the same numeric value, so the composite digit is
    unaffected either way.
    """
    document_code = _pad(_normalize(fields.document_code, "document_code"), 2, "document_code")
    issuing_state = _pad(_normalize(fields.issuing_state, "issuing_state"), 3, "issuing_state")
    line1 = document_code + issuing_state + encode_name(fields.primary_name, fields.secondary_name)

    document_number = _pad(_normalize(fields.document_number, "document_number"), 9, "document_number")
    nationality = _pad(_normalize(fields.nationality, "nationality"), 3, "nationality")
    birth_date = _normalize(fields.birth_date, "birth_date")
    expiry_date = _normalize(fields.expiry_date, "expiry_date")
    optional_data = _pad(
        _normalize(fields.optional_data, "optional_data"), F.OPTIONAL_DATA_LENGTH, "optional_data"
    )
    sex = _normalize(fields.sex or FILLER, "sex")

    for name, value, length in (
        ("birth_date", birth_date, 6),
        ("expiry_date", expiry_date, 6),
        ("sex", sex, 1),
    ):
        if len(value) != length:
            raise MRZFieldError(f"{name} must be exactly {length} characters, got {len(value)}")
    if sex not in F.VALID_SEX:
        raise MRZFieldError(f"sex must be one of M, F or {FILLER}, got {sex!r}")

    optional_cd = compute_check_digit(optional_data)
    if filler_optional_cd and set(optional_data) <= {FILLER}:
        optional_cd = FILLER

    line2 = (
        document_number
        + compute_check_digit(document_number)
        + nationality
        + birth_date
        + compute_check_digit(birth_date)
        + sex
        + expiry_date
        + compute_check_digit(expiry_date)
        + optional_data
        + optional_cd
    )
    line2 += compute_check_digit(F.composite_payload(line2))

    return f"{line1}\n{line2}"


# --------------------------------------------------------------------------
# parse
# --------------------------------------------------------------------------

def split_lines(mrz: str) -> tuple[str, str]:
    """Normalize arbitrary input into exactly two 44-character lines."""
    lines = [line.strip() for line in mrz.strip().splitlines() if line.strip()]
    if len(lines) != F.LINE_COUNT:
        raise MRZFormatError(f"TD3 requires {F.LINE_COUNT} lines, got {len(lines)}")
    for index, line in enumerate(lines, start=1):
        if len(line) != F.LINE_LENGTH:
            raise MRZFormatError(
                f"line {index} is {len(line)} characters, TD3 requires {F.LINE_LENGTH}"
            )
        for char in line:
            if not is_mrz_char(char):
                raise MRZFormatError(f"line {index} contains {char!r}, not an MRZ character")
    return lines[0], lines[1]


def parse(mrz: str) -> TD3Document:
    """Parse a TD3 MRZ into its fields and check digits.

    Parsing is purely positional and does not judge the content: an MRZ with a
    wrong check digit still parses. Use :func:`validate` to decide whether the
    result can be a real document.
    """
    line1, line2 = split_lines(mrz)

    primary, secondary = decode_name(line1[F.NAME])
    parsed_fields = TD3Fields(
        document_code=line1[F.DOC_CODE],
        issuing_state=line1[F.ISSUING_STATE],
        primary_name=primary,
        secondary_name=secondary,
        document_number=line2[F.DOCUMENT_NUMBER].rstrip(FILLER),
        nationality=line2[F.NATIONALITY],
        birth_date=line2[F.BIRTH_DATE],
        sex=line2[F.SEX],
        expiry_date=line2[F.EXPIRY_DATE],
        optional_data=line2[F.OPTIONAL_DATA].rstrip(FILLER),
    )
    check_digits = CheckDigits(
        document_number=line2[F.DOCUMENT_NUMBER_CD],
        birth_date=line2[F.BIRTH_DATE_CD],
        expiry_date=line2[F.EXPIRY_DATE_CD],
        optional_data=line2[F.OPTIONAL_DATA_CD],
        composite=line2[F.COMPOSITE_CD],
    )
    return TD3Document(fields=parsed_fields, check_digits=check_digits, line1=line1, line2=line2)


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------

def validate(document: TD3Document, *, reference_year: int) -> ValidationResult:
    """Report every way ``document`` fails to be a plausible passport MRZ.

    ``reference_year`` anchors the century of the two-digit dates; pass the
    current year in production. All issues are collected rather than raised on
    the first failure, because the candidate decoder scores hypotheses by how
    badly they fail, not merely whether they do.
    """
    issues: list[Issue] = []
    line2 = document.line2

    if not document.line1.startswith("P"):
        issues.append(Issue("document_code", f"expected a passport code starting with 'P', got {document.line1[F.DOC_CODE]!r}"))

    for label, payload, digit, lenient in (
        ("document_number", line2[F.DOCUMENT_NUMBER], document.check_digits.document_number, False),
        ("birth_date", line2[F.BIRTH_DATE], document.check_digits.birth_date, False),
        ("expiry_date", line2[F.EXPIRY_DATE], document.check_digits.expiry_date, False),
        ("optional_data", line2[F.OPTIONAL_DATA], document.check_digits.optional_data, True),
    ):
        if not verify_check_digit(payload, digit, allow_filler=lenient):
            issues.append(
                Issue(f"{label}_cd", f"check digit {digit!r} does not match {payload!r}")
            )

    composite = F.composite_payload(line2)
    if not verify_check_digit(composite, document.check_digits.composite):
        issues.append(
            Issue("composite_cd", f"composite check digit {document.check_digits.composite!r} does not match")
        )

    if document.fields.sex not in F.VALID_SEX:
        issues.append(Issue("sex", f"{document.fields.sex!r} is not M, F or {FILLER}"))

    birth = parse_yymmdd(document.fields.birth_date)
    if birth is None:
        issues.append(Issue("birth_date", f"{document.fields.birth_date!r} is not 6 digits"))
    else:
        yy, mm, dd = birth
        year = resolve_birth_year(yy, reference_year)
        if not is_calendar_date(yy, mm, dd, year - yy):
            issues.append(Issue("birth_date", f"{document.fields.birth_date!r} is not a real date"))

    expiry = parse_yymmdd(document.fields.expiry_date)
    if expiry is None:
        issues.append(Issue("expiry_date", f"{document.fields.expiry_date!r} is not 6 digits"))
    else:
        yy, mm, dd = expiry
        year = resolve_expiry_year(yy, reference_year)
        if not is_calendar_date(yy, mm, dd, year - yy):
            issues.append(Issue("expiry_date", f"{document.fields.expiry_date!r} is not a real date"))

    if birth is not None and expiry is not None:
        birth_year = resolve_birth_year(birth[0], reference_year)
        expiry_year = resolve_expiry_year(expiry[0], reference_year)
        if (expiry_year, expiry[1], expiry[2]) < (birth_year, birth[1], birth[2]):
            issues.append(Issue("expiry_date", "expiry date precedes the birth date"))

    # No check digit covers these two fields, so the allowlist is the only
    # error detection they get: a substituted letter usually lands on a code
    # that does not exist.
    for label, value in (
        ("issuing_state", document.fields.issuing_state),
        ("nationality", document.fields.nationality),
    ):
        if not is_valid_code(value):
            issues.append(Issue(label, f"{value!r} is not a known country code"))

    return ValidationResult(tuple(issues))
