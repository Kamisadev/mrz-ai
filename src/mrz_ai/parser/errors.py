"""Exceptions raised by the MRZ parser."""

from __future__ import annotations


class MRZError(Exception):
    """Base class for every MRZ failure."""


class MRZFormatError(MRZError):
    """The input does not have the shape of a TD3 MRZ (length, lines, charset)."""


class MRZFieldError(MRZError):
    """A field value cannot be encoded into the MRZ."""
