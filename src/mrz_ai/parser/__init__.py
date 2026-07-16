"""ICAO 9303 TD3 machine-readable zone: encoding, decoding and validation.

Pure Python with no ML dependencies, so the synthetic data engine can import it
inside a dataloader worker without pulling in torch.
"""

from __future__ import annotations

from .charset import ALPHABET, ALPHABET_SIZE, FILLER, char_value
from .checksum import compute_check_digit, verify_check_digit
from .errors import MRZError, MRZFieldError, MRZFormatError
from .td3 import decode_name, encode_name, parse, serialize, validate
from .types import CheckDigits, Issue, TD3Document, TD3Fields, ValidationResult

__all__ = [
    "ALPHABET",
    "ALPHABET_SIZE",
    "FILLER",
    "char_value",
    "compute_check_digit",
    "verify_check_digit",
    "MRZError",
    "MRZFieldError",
    "MRZFormatError",
    "serialize",
    "parse",
    "validate",
    "encode_name",
    "decode_name",
    "TD3Fields",
    "TD3Document",
    "CheckDigits",
    "Issue",
    "ValidationResult",
]
