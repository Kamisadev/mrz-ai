"""Pick the reading of a document that ICAO says could be real.

The recognizer is a good guesser that knows nothing about passports. It reads
'8' where a '9' was printed because that is what the ink looked like, and it has
no way to notice that the resulting document number contradicts a check digit
four characters later. The parser knows the rules and cannot read. This module
is where the two meet: the model proposes readings in order of belief, and ICAO
decides which of them could be a document at all.

The gain is asymmetric, and it is worth being exact about why. Every check digit
in a TD3 MRZ lives on line 2 — the document number's, the dates', the optional
data's, and the composite over all of them. Line 1 carries none. So a misread in
the document number is usually repairable, while a misread in a surname is
usually not: nothing in the standard contradicts 'ERIKSSQN'. Positions 0-4 of
line 1 are the exception, held by the document code and the issuing-state
allowlist. A reading of line 1 that validates is therefore not evidence that the
name is right; `Reading.confidence` reports the model's belief, not the
checksums' blessing, precisely so nothing downstream confuses the two.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np

from ..parser import parse, validate
from ..parser.errors import MRZError
from ..parser.types import Issue, TD3Document, ValidationResult
from .beam import Candidate, best_lines

Array = np.ndarray

__all__ = ["Reading", "best_reading"]


@dataclass(frozen=True)
class Reading:
    """One document the model and the standard both accept, or the least bad."""

    document: TD3Document
    validation: ValidationResult
    #: Joint log-probability of both lines under the model. Belief only: a valid
    #: reading with a low score is a document whose checksums happen to work out,
    #: which for a 9-character number is not the coincidence it sounds like.
    log_prob: float

    @property
    def line1(self) -> str:
        return self.document.line1

    @property
    def line2(self) -> str:
        return self.document.line2

    @property
    def mrz(self) -> str:
        return self.document.mrz

    @property
    def is_valid(self) -> bool:
        return self.validation.is_valid

    @property
    def issues(self) -> tuple[Issue, ...]:
        return self.validation.issues

    @property
    def confidence(self) -> float:
        """How likely the model thinks this exact reading is, in [0, 1]."""
        return math.exp(self.log_prob)


def _document_for(line1: str, line2: str) -> TD3Document | None:
    try:
        return parse(f"{line1}\n{line2}")
    except MRZError:
        # Every candidate is 44 characters of the MRZ alphabet, so this is not
        # expected — but a parser failure must not take down the pipeline for a
        # single unlucky crop.
        return None


def best_reading(
    line1_logits: Array,
    line2_logits: Array,
    *,
    k: int = 8,
    reference_year: int,
) -> Reading:
    """The best reading of a document from its two lines' ``(44, 37)`` logits.

    ``k`` candidates are taken per line and every pairing is scored, so the
    search covers the ``k`` most likely documents exactly: the lines are read
    independently, so a document in the global top-``k`` cannot contain a line
    outside its own line's top-``k``.

    Ranked by how badly a reading fails ICAO first, and by the model's belief
    only to break ties. That order is the whole point. A check digit is not a
    soft preference — a real MRZ always carries a correct one — so a candidate
    that fails one is not a document the model merely doubts, it is a document
    that cannot exist, whatever probability mass sits on it.

    Always returns a reading. When nothing validates, the least-bad one comes
    back carrying its issues, because a human correcting a rejected scan needs
    the best guess and the reason to distrust it, not a ``None``.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    first: list[Candidate] = best_lines(line1_logits, k=k)
    second: list[Candidate] = best_lines(line2_logits, k=k)

    best: Reading | None = None
    best_rank: tuple[int, float] | None = None

    for one, two in product(first, second):
        document = _document_for(one.text, two.text)
        if document is None:
            continue
        result = validate(document, reference_year=reference_year)
        log_prob = one.log_prob + two.log_prob
        # Fewer issues first, then higher likelihood. The second term is
        # currently redundant and is kept deliberately: no ICAO rule spans both
        # lines, so issue counts are additive and each line's candidates already
        # arrive best-first — which makes the first minimum-issue pairing found
        # the most likely one anyway. That is a property of today's rules, not of
        # this function. A single cross-line check would end it, and the tiebreak
        # is what keeps this correct when one appears. No test can distinguish
        # the two orderings until then.
        rank = (len(result.issues), -log_prob)
        if best_rank is None or rank < best_rank:
            best, best_rank = Reading(document, result, log_prob), rank

    if best is None:  # pragma: no cover - would need every pairing to be unparseable
        raise MRZError("no candidate pairing could be parsed")
    return best
