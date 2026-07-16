"""The K most likely readings of one MRZ line.

The name is a concession to the blueprint, which asks for beam search. This is
not one, and the difference is the point.

Beam search exists to decode a model whose next character depends on the last:
the joint probability cannot be factored, so a decoder walks the sequence and
keeps its W best prefixes, discarding anything that looks weak early. It is an
approximation, and its width is a knob trading accuracy for time.

`MRZRecognizer` has no such dependency. It emits all 44 positions from the image
in a single pass with no causal mask, so the joint log-probability of a line is
exactly the sum of its positions' log-probabilities, and the positions can be
reasoned about one at a time. That makes the K best lines *exactly* computable:
rank each position's classes, start from the argmax, and repeatedly take the
cheapest single-position downgrade of something already found. No width, no
approximation, no candidate silently pruned at an intermediate step — because
there are no intermediate steps.

Running a beam over this model would be slower and strictly worse: it would have
the same answers available and could still drop them.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from ..parser.charset import INDEX_TO_CHAR
from ..recognition.tokenizer import LABEL_LENGTH, NUM_CLASSES

Array = np.ndarray

__all__ = ["Candidate", "best_lines"]


@dataclass(frozen=True)
class Candidate:
    """One possible reading of a line, and how much the model believes it."""

    text: str
    #: Joint log-probability over all 44 positions: always <= 0, and comparable
    #: between images because the logits are normalized before summing. The ICAO
    #: stage needs that comparability to choose between candidates.
    log_prob: float


def log_softmax(logits: Array) -> Array:
    """Normalize logits into log-probabilities along the class axis."""
    peak = logits.max(axis=-1, keepdims=True)
    shifted = logits - peak
    normalized: Array = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return normalized


def best_lines(logits: Array, k: int = 8) -> list[Candidate]:
    """The ``k`` most likely lines for one crop's ``(44, 37)`` logits.

    Returned best-first. Exact: the i-th result is genuinely the i-th most
    likely of all 37^44 possible lines, not the best one some beam happened to
    keep.
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != LABEL_LENGTH:
        raise ValueError(
            f"expected ({LABEL_LENGTH}, {NUM_CLASSES}) logits, got shape {values.shape}"
        )
    if values.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"expected {NUM_CLASSES} classes per position, got {values.shape[1]}"
        )

    log_probs = log_softmax(values)

    # Each position's classes, best first. `ranked[p, r]` is the class index of
    # the r-th most likely character at position p, and `scores[p, r]` its
    # log-probability — so a line is a choice of one rank per position, and its
    # score is the sum of the chosen ranks' scores.
    ranked = np.argsort(-log_probs, axis=1)
    scores = np.take_along_axis(log_probs, ranked, axis=1)

    # The cost of a line, relative to the best one, is the sum of what each
    # position gives up by not taking its own argmax. Every step down a position
    # costs something non-negative, so the search only ever moves downhill: a
    # heap ordered by total cost pops lines in exactly descending likelihood.
    best_score = float(scores[:, 0].sum())
    penalty = scores[:, 0:1] - scores  # (44, 37), zero at rank 0, rising after

    start = (0,) * LABEL_LENGTH
    # (cost, ranks): cost is what this line gives up against the best line.
    heap: list[tuple[float, tuple[int, ...]]] = [(0.0, start)]
    seen: set[tuple[int, ...]] = {start}
    found: list[Candidate] = []

    while heap and len(found) < k:
        cost, ranks = heapq.heappop(heap)
        text = "".join(INDEX_TO_CHAR[int(ranked[p, r])] for p, r in enumerate(ranks))
        found.append(Candidate(text=text, log_prob=best_score - cost))

        # Every line reachable by demoting exactly one position by one rank. The
        # heap orders them, and `seen` stops a line arriving twice by different
        # routes — n downgrades can be applied in n! orders, and without this the
        # search would return duplicates and stall short of k.
        for position in range(LABEL_LENGTH):
            rank = ranks[position] + 1
            if rank >= NUM_CLASSES:
                continue
            successor = ranks[:position] + (rank,) + ranks[position + 1 :]
            if successor in seen:
                continue
            seen.add(successor)
            step = float(penalty[position, rank] - penalty[position, ranks[position]])
            heapq.heappush(heap, (cost + step, successor))

    return found
