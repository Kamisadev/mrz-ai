"""What the run is doing, written to a file for anything that wants to watch.

One direction only: training writes, the dashboard reads, and they never speak.
That is the whole design, and it is the design because of what is on the other
side — a 45-minute rented pod that must not be brought down by a web server, a
port that did not bind, or a browser that stopped polling. If the dashboard
crashes, training does not notice. If training crashes, the last status stays on
disk saying where it got to.

Writes are atomic: a temporary file in the same directory, then a rename. Without
that, a reader eventually catches a half-written file and reports a broken run
that is fine. A rename inside one directory is atomic on every filesystem this
will meet, which is why the temporary file cannot simply go in /tmp.

Torch-free, so the dashboard can import it without a CUDA build.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["STATUS_FILE", "Status", "StatusWriter"]

#: Beside the checkpoint, so a run's status travels with its output.
STATUS_FILE = "status.json"


@dataclass
class Status:
    """Everything the dashboard shows. Serialised whole on every write.

    Whole rather than as a diff: it is a few kilobytes, and a reader that has to
    reconstruct state from a stream of updates is a reader that can be wrong
    about what is happening. This file is the state.
    """

    #: "starting" | "training" | "finished" | "failed"
    state: str = "starting"
    step: int = 0
    total_steps: int = 0
    stage: str = ""
    stage_index: int = 0
    stages: int = 0
    loss: float | None = None
    learning_rate: float | None = None

    # Health. A run can be perfectly healthy and useless, which is what the fonts
    # line is doing here: one cut of OCR-B is a run to throw away, and nothing in
    # the loss curve would ever say so.
    device: str = ""
    hardware: str = ""
    precision: str = ""
    parameters: float = 0.0
    workers: int = 0
    cores: int = 0
    fonts: list[str] = field(default_factory=list)
    rate: float | None = None
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None

    #: Synthetic eval over time: {"step", "stage", "loss", "char_accuracy", "line_accuracy"}.
    history: list[dict[str, Any]] = field(default_factory=list)

    #: The real set, when there is one. Counts and confusions only.
    real: dict[str, Any] | None = None
    #: Real accuracy over time, so the panel can show a trend rather than a point.
    real_history: list[dict[str, Any]] = field(default_factory=list)

    #: Set when state is "failed", so a dead run says why on the page rather than
    #: only in a terminal nobody is looking at.
    error: str | None = None

    started_at: float = 0.0
    updated_at: float = 0.0


class StatusWriter:
    """Holds a `Status` and puts it on disk whenever it changes.

    Never raises. A run must not die because its status file could not be
    written — the disk being full is a reason to lose the dashboard, not the
    forty minutes of GPU time.
    """

    def __init__(self, directory: Path | str, status: Status | None = None) -> None:
        self.path = Path(directory) / STATUS_FILE
        self.status = status or Status()

    def update(self, **fields: Any) -> None:
        """Set fields and write.

        The unknown-field check raises, unlike `write` below, and the difference
        is deliberate: `update(steps=...)` for `step` is a typo in this repository,
        not a condition of the machine, and silently writing a field nothing reads
        would leave the page showing zero for the rest of the run.
        """
        for key, value in fields.items():
            if not hasattr(self.status, key):
                raise AttributeError(f"Status has no field {key!r}")
            setattr(self.status, key, value)
        self.write()

    def write(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(asdict(self.status), indent=2, default=str)
            # Same directory as the target: rename is only atomic within a
            # filesystem, and /tmp is regularly a different one.
            handle, temporary = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
            try:
                with os.fdopen(handle, "w") as file:
                    file.write(payload)
                os.replace(temporary, self.path)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
        except Exception:
            # Every exception, not just OSError. The contract this class exists to
            # keep is that a run never dies of its own telemetry, and a path the
            # OS objects to raises ValueError before it ever reaches an OS call —
            # which a test found, having believed the narrower promise. A writer
            # that is allowed to be quietly useless must not also be allowed to be
            # loudly fatal.
            pass


def read_status(directory: Path | str) -> Status | None:
    """The last status written, or None if there is not a readable one.

    Tolerates the file being absent, half-gone, or written by a newer version
    with fields this one does not know: the dashboard's job is to show what it
    can, not to be right about the schema. A crash here would take out the
    window into the run at exactly the moment somebody opened it.
    """
    path = Path(directory) / STATUS_FILE
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    known = {key: value for key, value in raw.items() if key in Status.__dataclass_fields__}
    try:
        return Status(**known)
    except TypeError:  # pragma: no cover - a field changed type
        return None
