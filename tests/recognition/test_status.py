"""The status file training writes for anything that wants to watch.

One direction only: training writes, the dashboard reads, and neither can hurt
the other. These tests are mostly about that promise holding under the ways a
45-minute rented pod actually ends — killed mid-write, killed without warning, or
watched by a dashboard running a different commit.

Lives beside test_training.py, which is where this package's training tests are.
"""

from __future__ import annotations

import json
from pathlib import Path

from mrz_ai.training.status import Status, StatusWriter, read_status


def test_a_status_survives_the_round_trip(tmp_path: Path) -> None:
    StatusWriter(tmp_path, Status(state="training", step=7, fonts=["OCR-B.ttf"])).write()

    back = read_status(tmp_path)
    assert back is not None
    assert (back.state, back.step, back.fonts) == ("training", 7, ["OCR-B.ttf"])


def test_updating_an_unknown_field_is_a_typo_not_a_new_field(tmp_path: Path) -> None:
    """`status.update(steps=...)` would otherwise write a field nothing reads, and
    the page would show zero for the rest of the run."""
    try:
        StatusWriter(tmp_path).update(stpe=3)
    except AttributeError as error:
        assert "stpe" in str(error)
    else:  # pragma: no cover
        raise AssertionError("a misspelt field was accepted")


def test_a_write_leaves_no_temporary_files_behind(tmp_path: Path) -> None:
    writer = StatusWriter(tmp_path)
    for step in range(5):
        writer.update(step=step)
    assert [path.name for path in tmp_path.iterdir()] == ["status.json"]


def test_the_run_does_not_die_when_the_status_cannot_be_written(tmp_path: Path) -> None:
    """The whole point of the try/except. Losing the dashboard is a bad afternoon;
    losing the run because a disk filled up is a worse one."""
    StatusWriter(tmp_path / "nope" / "\0bad").update(step=1)  # must not raise


def test_a_half_written_file_reads_as_nothing_rather_than_crashing(tmp_path: Path) -> None:
    """A reader polling every 3 seconds would eventually catch a partial file if
    the write were not atomic. It is — this covers truncation by anything else."""
    (tmp_path / "status.json").write_text('{"state": "trai')
    assert read_status(tmp_path) is None


def test_a_missing_file_reads_as_nothing(tmp_path: Path) -> None:
    assert read_status(tmp_path) is None


def test_a_field_this_version_does_not_know_is_ignored(tmp_path: Path) -> None:
    """A pod on one commit and a dashboard on another is normal. The page shows
    what it can rather than refusing the whole file."""
    (tmp_path / "status.json").write_text(json.dumps({"state": "training", "step": 3, "moon": 1}))
    back = read_status(tmp_path)
    assert back is not None and back.step == 3


def test_the_file_is_replaced_whole_never_appended(tmp_path: Path) -> None:
    writer = StatusWriter(tmp_path)
    writer.update(step=1)
    writer.update(step=2)
    assert json.loads((tmp_path / "status.json").read_text())["step"] == 2
