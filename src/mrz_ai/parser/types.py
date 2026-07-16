"""Structured representations of a TD3 MRZ."""

from __future__ import annotations

from dataclasses import dataclass



@dataclass(frozen=True)
class TD3Fields:
    """The human-meaningful content of a TD3 MRZ, free of check digits.

    This is what the synthetic generator produces and what the inference
    pipeline hands back to callers. Check digits are derived, never stored.
    """

    issuing_state: str
    primary_name: str
    secondary_name: str
    document_number: str
    nationality: str
    birth_date: str  # YYMMDD
    sex: str  # 'M', 'F' or '<'
    expiry_date: str  # YYMMDD
    optional_data: str = ""
    document_code: str = "P<"


@dataclass(frozen=True)
class CheckDigits:
    """The five check digits carried by line 2, as read from the MRZ."""

    document_number: str
    birth_date: str
    expiry_date: str
    optional_data: str
    composite: str


@dataclass(frozen=True)
class Issue:
    """One reason an MRZ failed validation."""

    field: str
    message: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.field}: {self.message}"


@dataclass(frozen=True)
class ValidationResult:
    issues: tuple[Issue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def __bool__(self) -> bool:
        return self.is_valid


@dataclass(frozen=True)
class TD3Document:
    """A parsed MRZ: its fields, the check digits it carried, and the raw lines."""

    fields: TD3Fields
    check_digits: CheckDigits
    line1: str
    line2: str

    @property
    def mrz(self) -> str:
        return f"{self.line1}\n{self.line2}"
