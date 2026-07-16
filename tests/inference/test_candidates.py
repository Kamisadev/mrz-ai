"""Choosing a reading: the model proposes, ICAO disposes.

The fixtures here build logits by hand so that the *second* choice is the right
one, which is the only situation this module exists for. If the model's argmax
were always correct there would be nothing to decide.
"""

from __future__ import annotations

import numpy as np
import pytest

from mrz_ai.parser import parse, validate
from mrz_ai.parser.charset import CHAR_TO_INDEX
from mrz_ai.recognition.tokenizer import LABEL_LENGTH, NUM_CLASSES
from mrz_ai.inference.candidates import Reading, best_reading

YEAR = 2026
LINE1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
LINE2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"


def confident(text: str, margin: float = 12.0) -> np.ndarray:
    """Logits that read ``text`` with near-certainty."""
    logits = np.zeros((LABEL_LENGTH, NUM_CLASSES), dtype=np.float32)
    for position, char in enumerate(text):
        logits[position, CHAR_TO_INDEX[char]] = margin
    return logits


def torn(text: str, position: int, wrong: str, *, gap: float = 1.0) -> np.ndarray:
    """Logits that prefer ``wrong`` at ``position``, with the truth a close second.

    This is what a misread actually looks like: the model is confident about the
    line and hesitant about one glyph, with the right answer still in reach.
    """
    logits = confident(text)
    logits[position, CHAR_TO_INDEX[wrong]] = logits[position, CHAR_TO_INDEX[text[position]]] + gap
    return logits


def test_a_clean_read_is_returned_and_validates() -> None:
    reading = best_reading(confident(LINE1), confident(LINE2), reference_year=YEAR)

    assert reading.line1 == LINE1
    assert reading.line2 == LINE2
    assert reading.is_valid
    assert reading.document.fields.document_number == "L898902C3"


def test_the_check_digits_overrule_a_wrong_argmax() -> None:
    """The point of the whole module: a confident misread on line 2 is repaired.

    Greedy decoding returns the wrong character because the model prefers it.
    Only the checksum knows better.
    """
    # The model prefers '8' where the document says '9' — a classic confusion.
    logits2 = torn(LINE2, 2, "8")

    greedy = "".join(
        sorted(CHAR_TO_INDEX, key=lambda c: -logits2[p, CHAR_TO_INDEX[c]])[0]
        for p in range(LABEL_LENGTH)
    )
    assert greedy != LINE2, "the fixture must actually mislead a greedy decoder"

    reading = best_reading(confident(LINE1), logits2, k=8, reference_year=YEAR)

    assert reading.line2 == LINE2, "the check digit should have overruled the argmax"
    assert reading.is_valid


def test_a_misread_name_cannot_be_repaired_and_is_not_pretended_otherwise() -> None:
    """ICAO puts no check digit on line 1's name, so nothing can catch this.

    Worth pinning precisely because it is the module's honest limit. A reading
    can be fully valid and still wrong, and the confidence must not claim
    otherwise — the checksums simply have no opinion about these positions.
    """
    wrong_name = LINE1.replace("ERIKSSON", "ERIKSSQN")
    reading = best_reading(confident(wrong_name), confident(LINE2), reference_year=YEAR)

    assert reading.line1 == wrong_name  # the misread survives
    assert reading.is_valid  # and validation is perfectly happy
    assert validate(parse(reading.mrz), reference_year=YEAR).is_valid


def test_the_issuing_state_allowlist_repairs_line_one() -> None:
    """Line 1 is not entirely unprotected: positions 2-4 must be a real country."""
    # 'UTO' misread as 'UTX', which is not an issuing state that exists.
    logits1 = torn(LINE1, 4, "X")

    reading = best_reading(logits1, confident(LINE2), k=8, reference_year=YEAR)

    assert reading.line1 == LINE1
    assert reading.document.fields.issuing_state == "UTO"


def test_the_most_likely_valid_reading_wins_over_a_less_likely_one() -> None:
    """Among readings that validate, the model's belief is the tiebreak."""
    reading = best_reading(confident(LINE1), confident(LINE2), k=8, reference_year=YEAR)
    assert reading.is_valid
    # Nothing beats the argmax when the argmax already validates.
    assert reading.log_prob == pytest.approx(
        best_reading(confident(LINE1), confident(LINE2), k=1, reference_year=YEAR).log_prob
    )


def test_an_unrepairable_read_returns_the_least_bad_rather_than_nothing() -> None:
    """A caller always gets an answer, plus the evidence to distrust it.

    Returning None would throw away the best guess at exactly the moment a human
    reviewer needs something to correct. The issues travel with the reading
    instead, which is what the active-learning loop consumes.
    """
    noise = np.random.default_rng(0).normal(size=(LABEL_LENGTH, NUM_CLASSES)).astype(np.float32)

    reading = best_reading(noise, noise, k=4, reference_year=YEAR)

    assert isinstance(reading, Reading)
    assert not reading.is_valid
    assert reading.issues, "an invalid reading must say why"


def test_fewer_issues_beats_higher_likelihood() -> None:
    """A reading that cannot be a document loses to one that can.

    Check digits are not a soft preference: a real MRZ always carries correct
    ones. So a candidate that fails them is not a document the model merely
    doubts — it is one that cannot exist, however much probability mass it has.
    """
    # Make the corrupt line 2 substantially more likely than the truth.
    logits2 = torn(LINE2, 2, "8", gap=6.0)

    reading = best_reading(confident(LINE1), logits2, k=8, reference_year=YEAR)

    assert reading.line2 == LINE2
    assert reading.is_valid


def test_a_bigger_k_can_only_help() -> None:
    """More candidates must never produce a worse reading."""
    logits2 = torn(LINE2, 2, "8")

    small = best_reading(confident(LINE1), logits2, k=1, reference_year=YEAR)
    large = best_reading(confident(LINE1), logits2, k=8, reference_year=YEAR)

    assert not small.is_valid, "k=1 is greedy, and greedy is wrong here"
    assert large.is_valid
    assert len(large.issues) <= len(small.issues)


def test_confidence_is_a_probability() -> None:
    reading = best_reading(confident(LINE1), confident(LINE2), reference_year=YEAR)

    assert 0.0 <= reading.confidence <= 1.0
    # A near-certain read of both lines should be near-certain overall.
    assert reading.confidence > 0.9


def test_confidence_falls_when_the_model_is_unsure() -> None:
    sure = best_reading(confident(LINE1), confident(LINE2), reference_year=YEAR)
    unsure = best_reading(confident(LINE1, margin=1.0), confident(LINE2, margin=1.0),
                          reference_year=YEAR)

    assert unsure.confidence < sure.confidence


def test_the_reading_exposes_the_mrz_it_chose() -> None:
    reading = best_reading(confident(LINE1), confident(LINE2), reference_year=YEAR)

    assert reading.mrz == f"{LINE1}\n{LINE2}"
    assert reading.mrz == reading.document.mrz


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="k"):
        best_reading(confident(LINE1), confident(LINE2), k=0, reference_year=YEAR)
