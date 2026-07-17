"""The real set: photographs of actual passports, and what they actually say.

Every other number this project reports is graded by the engine that drew the
exam. `README.md` records what that was worth once: 99.5% line accuracy at full
severity, from a checkpoint that misread real passports badly, because the
generator used a single font and nothing measuring it could see outside the
generator. This module is the first thing here that can.

*What these are.* Specimen passports — real documents, real printing, real
typeface, fake identities. That combination is what makes the set worth having
and safe to have: the cut of OCR-B is whatever the issuer's press actually used,
which is the axis that broke the last checkpoint, while the name and number
belong to nobody. `docs/real.md` has the rules, and `real/` is ignored by git
regardless, since a public repository is no place to find out that an assumption
about "fake" was wrong.

*Measurement only, never training.* Not a precaution — the only thing the set is
good for. Fit anything to it and it stops being able to measure, immediately and
without saying so, and every number it reports afterwards means exactly what the
99.5% meant.

Watching it during a run is a softer version of the same thing: a number you look
at is a number you act on, and killing a run because this panel looked bad is
selection on the test set. With a handful of documents there is nothing to split,
so this set is a *dev* set and is honest about it. The sealed 50-100 that
`README.md` asks for is still a separate thing that does not exist yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..parser import fields as F
from ..parser.charset import ALPHABET
from ..serve.crop import Box, split_lines

Array = np.ndarray

__all__ = [
    "RealDocument",
    "RealResult",
    "REAL_DIR",
    "load_real_set",
    "measure_real",
]

#: Where the set lives. Ignored by git as a whole directory; see docs/real.md.
REAL_DIR = Path("real")
_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class RealDocument:
    """One photographed passport, and the truth about what its MRZ says."""

    name: str
    image: Array
    #: Where the two MRZ lines are. Defaults to the whole image, which is the
    #: right default: the box only has to contain the lines and nothing else, and
    #: `serve.crop` finds the ink inside it. A photograph cropped roughly around
    #: the MRZ needs no box at all.
    box: Box
    line1: str
    line2: str

    @property
    def truth(self) -> tuple[str, str]:
        return self.line1, self.line2


@dataclass(frozen=True)
class RealResult:
    """How the recognizer did on the real set.

    Counts, not rates, because the set is small enough that the counts are the
    honest presentation: 7/10 says what 70.0% conceals.
    """

    documents: int
    documents_read: int
    lines_read: int
    chars_read: int
    chars_total: int
    #: (wanted, got) -> times, commonest first. The reason to keep a real set at
    #: all: `0->O` is not a stupid model, it is a generator that never drew
    #: anything forcing the two apart.
    confusions: list[tuple[tuple[str, str], int]]
    #: Per document, in load order: its name and whether both lines were exact.
    per_document: list[tuple[str, bool]]

    @property
    def document_rate(self) -> float:
        return self.documents_read / max(self.documents, 1)

    @property
    def char_rate(self) -> float:
        return self.chars_read / max(self.chars_total, 1)

    def __str__(self) -> str:
        top = "  ".join(f"{want}->{got} x{count}" for (want, got), count in self.confusions[:5])
        line = (
            f"{self.documents_read}/{self.documents} documents  "
            f"{self.lines_read}/{2 * self.documents} lines  char {self.char_rate:.2%}"
        )
        return f"{line}\n    confusions: {top}" if top else line


def _validate(name: str, key: str, text: object) -> str:
    """A truth line, checked hard.

    A typo here is indistinguishable from a misreading — the model gets marked
    wrong for the transcriber's slip, and the run that follows chases a defect
    nobody has. So a bad truth file fails at load, loudly, rather than turning
    into a quiet point of accuracy.
    """
    if not isinstance(text, str):
        raise ValueError(f"{name}: {key} must be a string, got {type(text).__name__}")
    if len(text) != F.LINE_LENGTH:
        raise ValueError(
            f"{name}: {key} is {len(text)} characters, a TD3 line is {F.LINE_LENGTH}. "
            f"Pad the name line with '<' to the full width — a short line here would "
            f"score as a misreading forever.\n  {text!r}"
        )
    unknown = sorted(set(text) - set(ALPHABET))
    if unknown:
        raise ValueError(f"{name}: {key} holds characters no MRZ can contain: {unknown}")
    return text


def load_real_set(root: Path | str = REAL_DIR) -> list[RealDocument]:
    """Load `root/images/*` against `root/truth.json`.

    Every image must have truth and every truth an image. Neither is allowed to
    pass quietly: an image without truth would be dropped from the denominator
    and make the score better, and truth without an image is a transcription
    nobody is measuring. Both mean the set is not what its owner thinks it is,
    which is the one thing a measurement set cannot afford.
    """
    import cv2

    root = Path(root)
    images_dir = root / "images"
    truth_path = root / "truth.json"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"no images at {images_dir}. See docs/real.md.")
    if not truth_path.is_file():
        raise FileNotFoundError(
            f"no {truth_path}. An image without its MRZ text measures nothing — "
            "there is no way to be wrong. See docs/real.md."
        )

    truth = json.loads(truth_path.read_text())
    if not isinstance(truth, dict):
        raise ValueError(f"{truth_path} must be an object keyed by image filename")

    files = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in _SUFFIXES)
    named = {path.name for path in files}

    missing_truth = sorted(named - set(truth))
    if missing_truth:
        raise ValueError(
            f"no truth for {missing_truth} in {truth_path}. Transcribe them or move them "
            "out of images/ — an unmeasured image silently improves the score."
        )
    missing_image = sorted(set(truth) - named)
    if missing_image:
        raise ValueError(f"truth for {missing_image}, but no such image in {images_dir}")

    documents = []
    for path in files:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            raise ValueError(f"{path} is not an image OpenCV can read")

        entry = truth[path.name]
        if not isinstance(entry, dict):
            raise ValueError(f"{path.name}: truth must be an object with line1 and line2")
        for key in ("line1", "line2"):
            if key not in entry:
                raise ValueError(f"{path.name}: truth has no {key}")

        height, width = image.shape[:2]
        raw = entry.get("box")
        if raw is None:
            box = Box(0.0, 0.0, float(width), float(height))
        else:
            if not isinstance(raw, (list, tuple)) or len(raw) != 4:
                raise ValueError(f"{path.name}: box must be [x, y, width, height]")
            box = Box(*(float(value) for value in raw))

        documents.append(
            RealDocument(
                name=path.name,
                image=image,
                box=box,
                line1=_validate(path.name, "line1", entry["line1"]),
                line2=_validate(path.name, "line2", entry["line2"]),
            )
        )

    if not documents:
        raise ValueError(f"{images_dir} holds no .jpg or .png")
    return documents


class _Reading(Protocol):
    """The two lines. `Reading` carries a parsed document and a validation result
    as well, and none of it is read here — matching text is the measurement."""

    @property
    def line1(self) -> str: ...
    @property
    def line2(self) -> str: ...


class _Reader(Protocol):
    """What `measure_real` needs. A Protocol rather than `MRZReader`, for the same
    reason `measure` and the serve layer have theirs: a test gets to check the
    counting with a reader whose answers are known, instead of loading a 27MB
    checkpoint to assert on an integer."""

    def read(
        self, line1_crop: Array, line2_crop: Array, *, reference_year: int | None = ...
    ) -> _Reading: ...


def measure_real(
    reader: _Reader, documents: list[RealDocument], *, reference_year: int | None = None
) -> RealResult:
    """Read every document and count what came back exactly right.

    A document counts only when both lines match character for character. That is
    the number that matters and the one that flatters least: 99.57% per character
    over 88 characters is 72% per document, which is how a single-font model
    looked excellent and was not.

    A document that cannot be cropped at all counts as unread rather than
    raising. It *is* unread — the reader failing to find two lines in a real
    photograph is the failure being measured, not an error in the measuring.
    """
    documents_read = lines_read = chars_read = chars_total = 0
    confusions: dict[tuple[str, str], int] = {}
    per_document: list[tuple[str, bool]] = []

    for document in documents:
        try:
            first, second = split_lines(document.image, document.box)
            reading = reader.read(first, second, reference_year=reference_year)
            got = (reading.line1, reading.line2)
        except (ValueError, IndexError):
            got = ("", "")

        ok = got == document.truth
        documents_read += ok
        per_document.append((document.name, ok))

        for line, want in zip(got, document.truth):
            lines_read += line == want
            # Padded, so a reader that returned nothing is scored as 44 wrong
            # characters rather than silently shrinking the denominator.
            for actual, expected in zip(line.ljust(F.LINE_LENGTH), want):
                if actual == expected:
                    chars_read += 1
                elif actual != " ":
                    confusions[(expected, actual)] = confusions.get((expected, actual), 0) + 1
            chars_total += F.LINE_LENGTH

    return RealResult(
        documents=len(documents),
        documents_read=documents_read,
        lines_read=lines_read,
        chars_read=chars_read,
        chars_total=chars_total,
        confusions=sorted(confusions.items(), key=lambda item: -item[1]),
        per_document=per_document,
    )
