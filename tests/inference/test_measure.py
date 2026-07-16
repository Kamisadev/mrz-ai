"""The harness must count honestly, including when the answer is unflattering.

A measurement nobody can trust is worse than no measurement: it would be quoted.
So the counting is checked against readers whose behaviour is known exactly —
a perfect one, a hopeless one, and one that misreads a single repairable digit.
"""

from __future__ import annotations

import numpy as np
import pytest

from mrz_ai.parser.charset import CHAR_TO_INDEX
from mrz_ai.recognition.tokenizer import LABEL_LENGTH, NUM_CLASSES
from mrz_ai.synthetic.dataset import DatasetConfig
from mrz_ai.inference.measure import Comparison, Tally, compare, outcome

YEAR = 2026
CLEAN = DatasetConfig(severity_range=(0.0, 0.0), dpi=100.0)


class ScriptedReader:
    """A reader that returns logits for whatever the truth is, optionally spoiled.

    Standing in for the model lets the harness be tested without one: what is
    under test is the arithmetic, and arithmetic checked against a trained
    network is arithmetic nobody can debug.
    """

    k = 8

    def __init__(self, truth: list[str], *, spoil: tuple[int, int, str] | None = None):
        self.truth = truth
        self.spoil = spoil
        self.calls = 0

    def logits_for(self, crops):
        out = []
        for offset in range(len(crops)):
            text = self.truth[self.calls * 2 + offset]
            logits = np.full((LABEL_LENGTH, NUM_CLASSES), -20.0, dtype=np.float32)
            for position, char in enumerate(text):
                logits[position, CHAR_TO_INDEX[char]] = 10.0
            if self.spoil is not None:
                line, position, wrong = self.spoil
                if offset == line:
                    # Prefer the wrong glyph, but leave the truth reachable.
                    logits[position, CHAR_TO_INDEX[wrong]] = 11.0
            out.append(logits)
        self.calls += 1
        return np.stack(out)


def truth_for(documents: int) -> list[str]:
    """The lines the dataset will actually produce, in the order compare reads them."""
    from mrz_ai.synthetic.dataset import MRZLineDataset

    stream = MRZLineDataset(CLEAN)
    return [stream[i].text for i in range(documents * 2)]


def test_a_perfect_reader_scores_everything() -> None:
    n = 4
    result = compare(ScriptedReader(truth_for(n)), documents=n, reference_year=YEAR,
                     dataset=CLEAN)

    assert result.greedy.documents == n
    assert result.greedy.line1_accuracy == 1.0
    assert result.greedy.line2_accuracy == 1.0
    assert result.decoded.document_accuracy == 1.0
    assert result.repairs == 0
    assert result.regressions == 0
    assert result.line2_gain == 0.0


def test_a_repairable_misread_is_counted_as_a_repair() -> None:
    """The headline number: greedy gets line 2 wrong, ICAO fixes it."""
    n = 4
    # Spoil position 2 of line 2 — inside the document number, so a check digit
    # contradicts it and the decoder can recover the truth.
    reader = ScriptedReader(truth_for(n), spoil=(1, 2, "8"))

    result = compare(reader, documents=n, reference_year=YEAR, dataset=CLEAN)

    assert result.greedy.line2_accuracy < 1.0, "the fixture must mislead the argmax"
    assert result.decoded.line2_accuracy > result.greedy.line2_accuracy
    assert result.repairs > 0
    assert result.regressions == 0
    assert result.line2_gain > 0


def test_an_unrepairable_misread_shows_no_gain() -> None:
    """A spoiled name must not produce a phantom improvement.

    If this reported a gain, the harness would be crediting ICAO with fixing
    something it provably cannot touch.
    """
    n = 4
    # Position 20 of line 1 is inside the name: no check digit covers it.
    reader = ScriptedReader(truth_for(n), spoil=(0, 20, "Q"))

    result = compare(reader, documents=n, reference_year=YEAR, dataset=CLEAN)

    assert result.greedy.line1_accuracy < 1.0
    assert result.decoded.line1_accuracy == result.greedy.line1_accuracy
    assert result.repairs == 0


def test_the_greedy_baseline_uses_the_same_logits_as_the_decoder() -> None:
    """One forward pass per document, or the comparison measures two models."""
    n = 3
    reader = ScriptedReader(truth_for(n))
    compare(reader, documents=n, reference_year=YEAR, dataset=CLEAN)

    assert reader.calls == n, "logits were computed more than once per document"


def test_documents_are_paired_not_mixed_across_passports() -> None:
    """Line 1 of one passport with line 2 of another would be a silent lie.

    Nothing downstream could catch it: ICAO puts no check digit on line 1 to tie
    it to line 2, so a mismatched pair still validates.
    """
    n = 5
    truth = truth_for(n)
    result = compare(ScriptedReader(truth), documents=n, reference_year=YEAR, dataset=CLEAN)

    # A perfect reader can only score 100% if each document's own two lines were
    # compared against each other.
    assert result.greedy.document_accuracy == 1.0
    assert result.decoded.validation_rate == 1.0


def test_severity_buckets_partition_the_documents() -> None:
    n = 12
    spread = DatasetConfig(severity_range=(0.0, 1.0), dpi=100.0)
    from mrz_ai.synthetic.dataset import MRZLineDataset

    stream = MRZLineDataset(spread)
    truth = [stream[i].text for i in range(n * 2)]
    result = compare(ScriptedReader(truth), documents=n, reference_year=YEAR, dataset=spread)

    counted = sum(greedy.documents for greedy, _ in result.buckets.values())
    assert counted == n, "every document must land in exactly one bucket"
    assert result.report()


def test_an_empty_tally_does_not_divide_by_zero() -> None:
    empty = Tally()
    assert empty.line2_accuracy == 0.0
    assert empty.document_accuracy == 0.0
    assert empty.validation_rate == 0.0


def test_validation_rate_is_not_accuracy() -> None:
    """A reading can validate and still be wrong; the report must not conflate them.

    This is the trap the whole phase is built around: line 1's name carries no
    check digit, so a document with a misread surname validates perfectly.
    """
    n = 4
    reader = ScriptedReader(truth_for(n), spoil=(0, 20, "Q"))
    result = compare(reader, documents=n, reference_year=YEAR, dataset=CLEAN)

    assert result.decoded.validation_rate == 1.0  # ICAO is happy
    assert result.decoded.line1_accuracy < 1.0  # and the name is still wrong
    assert result.decoded.document_accuracy < result.decoded.validation_rate


def test_the_report_renders() -> None:
    n = 2
    result = compare(ScriptedReader(truth_for(n)), documents=n, reference_year=YEAR,
                     dataset=CLEAN)
    text = result.report()

    assert "greedy" in text and "decoded" in text
    assert "line2 gain" in text
    assert isinstance(result, Comparison)


@pytest.mark.parametrize(
    "greedy, decoded, truth, expected",
    [
        ("WRONG", "RIGHT", "RIGHT", "repair"),
        ("RIGHT", "WRONG", "RIGHT", "regression"),
        ("RIGHT", "RIGHT", "RIGHT", "unchanged"),
        ("WRONG", "WRONG", "RIGHT", "unchanged"),
        # The one that matters: decoding swapped one wrong reading for a
        # different wrong reading. It changed the line, and changed nothing.
        ("WRONG", "OTHER", "RIGHT", "unchanged"),
    ],
)
def test_outcome_is_measured_against_the_truth_not_against_greedy(
    greedy: str, decoded: str, truth: str, expected: str
) -> None:
    """A false repair must never be scored as a win.

    Counting any change as a repair would make the harness flatter the decoder
    precisely where it fails: swapping a wrong reading for another wrong one
    that happens to satisfy the check digits.
    """
    assert outcome(greedy, decoded, truth) == expected
