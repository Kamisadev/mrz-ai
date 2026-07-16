"""The k-best enumerator returns exactly the K most likely lines, in order.

These tests build log-probs by hand rather than running the model: what is being
checked is the search, and a search whose correctness depends on a trained
network is a search nobody can debug.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from mrz_ai.parser.charset import CHAR_TO_INDEX, INDEX_TO_CHAR
from mrz_ai.recognition.tokenizer import LABEL_LENGTH, NUM_CLASSES
from mrz_ai.inference.beam import Candidate, best_lines


def logits_favouring(text: str, *, margin: float = 5.0) -> np.ndarray:
    """Logits whose argmax is ``text``, every other class equally unlikely."""
    logits = np.zeros((LABEL_LENGTH, NUM_CLASSES), dtype=np.float32)
    for position, char in enumerate(text):
        logits[position, CHAR_TO_INDEX[char]] = margin
    return logits


LINE = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"


def test_the_first_candidate_is_the_argmax() -> None:
    best = best_lines(logits_favouring(LINE), k=1)

    assert len(best) == 1
    assert best[0].text == LINE


def test_candidates_come_back_in_descending_likelihood() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(LABEL_LENGTH, NUM_CLASSES)).astype(np.float32)

    candidates = best_lines(logits, k=32)

    scores = [c.log_prob for c in candidates]
    assert scores == sorted(scores, reverse=True), "candidates are not ordered"


def test_no_candidate_is_repeated() -> None:
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(LABEL_LENGTH, NUM_CLASSES)).astype(np.float32)

    texts = [c.text for c in best_lines(logits, k=64)]

    assert len(set(texts)) == len(texts), "the same line was returned twice"


def test_the_enumeration_is_exact_not_approximate() -> None:
    """The whole reason not to use a beam: this is checkable against brute force.

    A beam of width W silently drops any candidate that was not in the top W at
    some intermediate step. With independent positions there are no intermediate
    steps to prune at, so the top-K is exactly computable — and here it is
    compared against every combination, on a toy problem small enough to enumerate.
    """
    # Three live positions of three classes each: 27 possible lines in total.
    live, classes = 3, 3
    logits = np.full((LABEL_LENGTH, NUM_CLASSES), -50.0, dtype=np.float32)
    rng = np.random.default_rng(2)
    for position in range(LABEL_LENGTH):
        if position < live:
            logits[position, :classes] = rng.normal(size=classes)
        else:
            logits[position, CHAR_TO_INDEX["<"]] = 0.0

    # Brute force: score every combination the same way the searcher must.
    log_probs = logits - _logsumexp(logits)
    exhaustive = []
    for combo in itertools.product(range(classes), repeat=live):
        text = "".join(INDEX_TO_CHAR[c] for c in combo) + "<" * (LABEL_LENGTH - live)
        score = sum(log_probs[p, c] for p, c in enumerate(combo))
        score += sum(log_probs[p, CHAR_TO_INDEX["<"]] for p in range(live, LABEL_LENGTH))
        exhaustive.append((score, text))
    exhaustive.sort(key=lambda pair: -pair[0])

    found = best_lines(logits, k=27)

    assert [c.text for c in found] == [text for _, text in exhaustive]
    for candidate, (score, _) in zip(found, exhaustive):
        assert candidate.log_prob == pytest.approx(score, abs=1e-4)


def _logsumexp(logits: np.ndarray) -> np.ndarray:
    peak = logits.max(axis=-1, keepdims=True)
    return peak + np.log(np.exp(logits - peak).sum(axis=-1, keepdims=True))


def test_the_score_is_a_joint_log_probability() -> None:
    """A candidate's score must be the sum of its positions, normalized.

    If the scores were raw logits, they would not be comparable across images and
    the ICAO stage could not use them to choose between candidates.
    """
    logits = logits_favouring(LINE, margin=10.0)
    best = best_lines(logits, k=1)[0]

    expected = float(sum((logits - _logsumexp(logits))[p, CHAR_TO_INDEX[c]]
                         for p, c in enumerate(LINE)))
    assert best.log_prob == pytest.approx(expected, abs=1e-4)
    # A confident model's best line is near-certain, so the joint is near 0.
    assert best.log_prob > -0.1
    # And a log probability is never positive.
    assert best.log_prob <= 0.0


def test_asking_for_more_candidates_than_exist_is_not_an_error() -> None:
    """A degenerate image must not crash the pipeline it feeds."""
    logits = np.full((LABEL_LENGTH, NUM_CLASSES), -50.0, dtype=np.float32)
    logits[:, CHAR_TO_INDEX["<"]] = 0.0
    # Only one line is meaningfully likely, but the space is still huge; the
    # searcher must simply return what was asked for without complaint.
    assert len(best_lines(logits, k=5)) == 5


def test_a_longer_k_extends_the_shorter_one() -> None:
    """Top-8 must be the first 8 of top-32, or the ordering is not a real ranking."""
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(LABEL_LENGTH, NUM_CLASSES)).astype(np.float32)

    short = [c.text for c in best_lines(logits, k=8)]
    long = [c.text for c in best_lines(logits, k=32)]

    assert long[:8] == short


def test_the_wrong_shape_is_refused() -> None:
    with pytest.raises(ValueError, match="44"):
        best_lines(np.zeros((10, NUM_CLASSES), dtype=np.float32), k=1)
    with pytest.raises(ValueError, match="37"):
        best_lines(np.zeros((LABEL_LENGTH, 5), dtype=np.float32), k=1)


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="k"):
        best_lines(logits_favouring(LINE), k=0)


def test_a_candidate_is_a_valid_mrz_line() -> None:
    """Every text returned must be 44 characters of the MRZ alphabet."""
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(LABEL_LENGTH, NUM_CLASSES)).astype(np.float32)

    for candidate in best_lines(logits, k=16):
        assert isinstance(candidate, Candidate)
        assert len(candidate.text) == LABEL_LENGTH
        assert all(char in CHAR_TO_INDEX for char in candidate.text)
