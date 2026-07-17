"""The real set: the only number here the generator does not grade.

The tests that matter are the ones about the truth file. A typo in truth.json is
indistinguishable from a misreading — the model is marked wrong for a
transcriber's slip, and the run that follows chases a defect nobody has. So the
loader is strict, and this is where that strictness is checked.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from mrz_ai.evaluation.real import RealDocument, load_real_set, measure_real
from mrz_ai.parser import serialize
from mrz_ai.serve.crop import Box
from mrz_ai.synthetic.identity import IdentityConfig, random_identity
from mrz_ai.synthetic.render import render_mrz

SPECIMEN = (
    "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
)


def a_set(tmp_path: Path, truth: dict | None = None, count: int = 2) -> Path:
    """A real set on disk, standing in for photographed specimens."""
    (tmp_path / "images").mkdir()
    entries = {}
    for index in range(count):
        mrz = serialize(random_identity(random.Random(500 + index), IdentityConfig()))
        rendered = render_mrz(mrz, dpi=250.0)
        name = f"{index:03d}_pass.png"
        cv2.imwrite(str(tmp_path / "images" / name), np.asarray(rendered.image))
        line1, line2 = mrz.split("\n")
        entries[name] = {"line1": line1, "line2": line2}
    (tmp_path / "truth.json").write_text(json.dumps(truth if truth is not None else entries))
    return tmp_path


@dataclass
class FakeReading:
    """Just the two lines. `Reading` wraps a parsed TD3Document and validation,
    and `measure_real` reads neither — building a real one here would tie these
    tests to the parser for nothing."""

    line1: str
    line2: str


class FakeReader:
    """Returns a fixed reading, so the counting is what is under test."""

    def __init__(self, line1: str, line2: str) -> None:
        self.line1, self.line2 = line1, line2

    def read(self, first, second, *, reference_year=None):
        return FakeReading(self.line1, self.line2)


def test_a_set_loads_with_its_truth(tmp_path: Path) -> None:
    documents = load_real_set(a_set(tmp_path))
    assert len(documents) == 2
    assert all(len(d.line1) == 44 and len(d.line2) == 44 for d in documents)


def test_the_whole_image_is_the_default_box(tmp_path: Path) -> None:
    """A photograph cropped roughly around the MRZ needs no box: crop.py finds the
    ink inside it, and being loose is free."""
    document = load_real_set(a_set(tmp_path))[0]
    height, width = document.image.shape[:2]
    assert (document.box.x, document.box.y) == (0.0, 0.0)
    assert (document.box.width, document.box.height) == (float(width), float(height))


def test_a_truth_line_of_the_wrong_length_is_refused(tmp_path: Path) -> None:
    """The one that pays for this file.

    A name line transcribed without its trailing fillers is 43 characters and
    looks exactly like a model that drops a character — forever, on every run,
    for a defect that is in the text file.
    """
    (tmp_path / "images").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "a_pass.png"), np.full((60, 900), 255, np.uint8))
    (tmp_path / "truth.json").write_text(
        json.dumps({"a_pass.png": {"line1": SPECIMEN[0][:-1], "line2": SPECIMEN[1]}})
    )
    with pytest.raises(ValueError, match="43 characters"):
        load_real_set(tmp_path)


def test_a_truth_line_with_impossible_characters_is_refused(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "a_pass.png"), np.full((60, 900), 255, np.uint8))
    (tmp_path / "truth.json").write_text(
        json.dumps({"a_pass.png": {"line1": "p" + SPECIMEN[0][1:], "line2": SPECIMEN[1]}})
    )
    with pytest.raises(ValueError, match="no MRZ can contain"):
        load_real_set(tmp_path)


def test_an_image_without_truth_is_refused(tmp_path: Path) -> None:
    """It would be dropped from the denominator, and the score would improve."""
    root = a_set(tmp_path)
    cv2.imwrite(str(root / "images" / "extra_pass.png"), np.full((60, 900), 255, np.uint8))
    with pytest.raises(ValueError, match="no truth for"):
        load_real_set(root)


def test_truth_without_an_image_is_refused(tmp_path: Path) -> None:
    entries = {"ghost_pass.png": {"line1": SPECIMEN[0], "line2": SPECIMEN[1]}}
    with pytest.raises(ValueError, match="no such image"):
        load_real_set(a_set(tmp_path, truth=entries, count=0))


def test_a_set_with_no_truth_file_says_why(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    cv2.imwrite(str(tmp_path / "images" / "a_pass.png"), np.full((60, 900), 255, np.uint8))
    with pytest.raises(FileNotFoundError, match="measures nothing"):
        load_real_set(tmp_path)


def test_a_document_counts_only_when_both_lines_are_exact() -> None:
    """Character accuracy flatters: 99.57%/char over 88 characters is 72%/document,
    which is how a single-font model looked excellent and was not."""
    document = RealDocument(
        name="a", image=np.full((60, 900), 255, np.uint8),
        box=Box(0.0, 0.0, 900.0, 60.0), line1=SPECIMEN[0], line2=SPECIMEN[1],
    )
    almost = SPECIMEN[1][:-1] + "9"  # one character out
    result = measure_real(FakeReader(SPECIMEN[0], almost), [document])

    assert result.documents_read == 0, "a document with a wrong character is a wrong document"
    assert result.lines_read == 1, "line 1 was exact"
    assert result.chars_read == 87


def test_the_confusions_name_the_pairs() -> None:
    document = RealDocument(
        name="a", image=np.full((60, 900), 255, np.uint8),
        box=Box(0.0, 0.0, 900.0, 60.0), line1="0" * 44, line2=SPECIMEN[1],
    )
    result = measure_real(FakeReader("O" * 44, SPECIMEN[1]), [document])
    assert result.confusions[0] == (("0", "O"), 44)


def test_an_uncroppable_document_counts_as_unread_rather_than_raising() -> None:
    """The reader failing on a real photograph is the failure being measured."""
    document = RealDocument(
        name="tiny", image=np.full((2, 2), 255, np.uint8),
        box=Box(0.0, 0.0, 2.0, 2.0), line1=SPECIMEN[0], line2=SPECIMEN[1],
    )
    result = measure_real(FakeReader(*SPECIMEN), [document])
    assert result.documents_read == 0
    assert result.chars_total == 88, "an unread document must stay in the denominator"
