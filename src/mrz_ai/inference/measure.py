"""Does ICAO decoding actually beat reading the argmax?

Everything else in this package is tested against logits written by hand, which
proves the mechanism and nothing about the model. This is the harness that
answers the only question that matters: on a real checkpoint, does any of it
help — and by how much, and where.

Three things it is careful about, each of which would otherwise mislead:

*Documents, not lines.* The training loop evaluates lines independently, but a
reading needs both lines of one document together. Consecutive dataset indices
are the two lines of one document, which is the pairing used here.

*Line 2 reported apart from line 1.* Every TD3 check digit lives on line 2, so
that is the only line ICAO can repair. A single full-MRZ number would bury the
line-2 gain under line-1 name errors nothing can fix, and read as "barely
moved" when the part that can improve improved a lot.

*Net, bucketed by severity.* Decoding can pick a reading that validates and is
still wrong — rare, but likeliest where line 2 has several errors at once and
the truth falls outside the top-k. Reported per severity bucket so a regression
at the hard end cannot hide inside a flattering average.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..synthetic.dataset import DatasetConfig, MRZLineDataset
from ..parser import fields as F
from ..recognition.tokenizer import decode
from .candidates import best_reading

Array = np.ndarray

__all__ = ["Comparison", "Reader", "Tally", "compare", "outcome"]


class Reader(Protocol):
    """What this module needs of a reader, which is less than MRZReader is.

    A Protocol rather than the class itself: importing `MRZReader` would drag
    torch into a module that has no use for it, and the tests would then have to
    build a real model to check arithmetic.
    """

    k: int

    def logits_for(self, crops: list[Array]) -> Array: ...


@dataclass
class Tally:
    """Exact-match counts for one decoding strategy."""

    documents: int = 0
    line1_correct: int = 0
    line2_correct: int = 0
    both_correct: int = 0
    validates: int = 0

    def _rate(self, count: int) -> float:
        return count / self.documents if self.documents else 0.0

    @property
    def line1_accuracy(self) -> float:
        return self._rate(self.line1_correct)

    @property
    def line2_accuracy(self) -> float:
        """The number that matters: line 2 is the only line ICAO can repair."""
        return self._rate(self.line2_correct)

    @property
    def document_accuracy(self) -> float:
        return self._rate(self.both_correct)

    @property
    def validation_rate(self) -> float:
        """How often the reading passes ICAO — not how often it is right."""
        return self._rate(self.validates)

    def __str__(self) -> str:
        return (
            f"line1 {self.line1_accuracy:6.2%}  line2 {self.line2_accuracy:6.2%}  "
            f"document {self.document_accuracy:6.2%}  validates {self.validation_rate:6.2%}"
        )


@dataclass
class Comparison:
    """Greedy against ICAO decoding, overall and split by severity."""

    greedy: Tally = field(default_factory=Tally)
    decoded: Tally = field(default_factory=Tally)
    #: Line 2 was wrong under argmax and right after decoding.
    repairs: int = 0
    #: Line 2 was right under argmax and wrong after decoding. A false repair:
    #: some other reading validated and outscored the truth.
    regressions: int = 0
    #: severity bucket -> (greedy, decoded)
    buckets: dict[str, tuple[Tally, Tally]] = field(default_factory=dict)

    @property
    def line2_gain(self) -> float:
        return self.decoded.line2_accuracy - self.greedy.line2_accuracy

    def report(self) -> str:
        lines = [
            f"documents: {self.greedy.documents}",
            f"  greedy   {self.greedy}",
            f"  decoded  {self.decoded}",
            f"  line2 gain {self.line2_gain:+.2%}  "
            f"repairs {self.repairs}  regressions {self.regressions}",
        ]
        for name in sorted(self.buckets):
            greedy, decoded = self.buckets[name]
            gain = decoded.line2_accuracy - greedy.line2_accuracy
            lines.append(
                f"  severity {name:>9}: line2 {greedy.line2_accuracy:6.2%} -> "
                f"{decoded.line2_accuracy:6.2%} ({gain:+.2%})  n={greedy.documents}"
            )
        return "\n".join(lines)


def _bucket_of(severity: float) -> str:
    if severity < 0.25:
        return "0.00-0.25"
    if severity < 0.50:
        return "0.25-0.50"
    if severity < 0.75:
        return "0.50-0.75"
    return "0.75-1.00"


def outcome(greedy: str, decoded: str, truth: str) -> str:
    """What decoding did to one line: ``repair``, ``regression`` or ``unchanged``.

    Only against the truth, never against the greedy read. Decoding a line into
    something *different* is not decoding it into something *right* — a swap of
    one wrong reading for another wrong one that happens to validate is the
    failure mode this whole comparison exists to detect, and scoring it as a win
    would hide exactly what it was built to find.
    """
    if decoded == truth and greedy != truth:
        return "repair"
    if greedy == truth and decoded != truth:
        return "regression"
    return "unchanged"


def _record(tally: Tally, line1: str, line2: str, truth1: str, truth2: str, valid: bool) -> None:
    tally.documents += 1
    tally.line1_correct += line1 == truth1
    tally.line2_correct += line2 == truth2
    tally.both_correct += line1 == truth1 and line2 == truth2
    tally.validates += valid


def compare(
    reader: Reader,
    *,
    documents: int = 200,
    reference_year: int,
    dataset: DatasetConfig | None = None,
) -> Comparison:
    """Read ``documents`` synthetic documents both ways and count.

    ``reader`` is an :class:`~mrz_ai.inference.pipeline.MRZReader`. Taken as a
    parameter rather than imported so this module stays torch-free.

    The greedy baseline is the argmax of the very same logits the decoder sees,
    so the only difference measured is the decoding — not the model, the crops,
    or the batch.
    """
    stream = MRZLineDataset(dataset or DatasetConfig(severity_range=(0.0, 1.0)))
    comparison = Comparison()

    for index in range(documents):
        first = stream[index * F.LINE_COUNT]
        second = stream[index * F.LINE_COUNT + 1]

        logits = reader.logits_for([first.image, second.image])
        greedy1 = decode(logits[0].argmax(-1))
        greedy2 = decode(logits[1].argmax(-1))
        reading = best_reading(
            logits[0], logits[1], k=reader.k, reference_year=reference_year
        )

        # The greedy pair still gets an ICAO opinion, so `validates` compares
        # like with like: the question is what each strategy hands a caller.
        greedy_valid = _validates(greedy1, greedy2, reference_year)

        truth1, truth2 = first.text, second.text
        _record(comparison.greedy, greedy1, greedy2, truth1, truth2, greedy_valid)
        _record(comparison.decoded, reading.line1, reading.line2, truth1, truth2,
                reading.is_valid)

        verdict = outcome(greedy2, reading.line2, truth2)
        comparison.repairs += verdict == "repair"
        comparison.regressions += verdict == "regression"

        name = _bucket_of(first.severity)
        if name not in comparison.buckets:
            comparison.buckets[name] = (Tally(), Tally())
        bucket_greedy, bucket_decoded = comparison.buckets[name]
        _record(bucket_greedy, greedy1, greedy2, truth1, truth2, greedy_valid)
        _record(bucket_decoded, reading.line1, reading.line2, truth1, truth2, reading.is_valid)

    return comparison


def _validates(line1: str, line2: str, reference_year: int) -> bool:
    from ..parser import parse, validate
    from ..parser.errors import MRZError

    try:
        return validate(parse(f"{line1}\n{line2}"), reference_year=reference_year).is_valid
    except MRZError:
        return False
