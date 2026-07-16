"""Reading a document: the recognizer proposes, ICAO disposes.

``beam`` and ``candidates`` are torch-free on purpose — the search and the ICAO
logic stay testable without a model and portable to the ONNX Runtime the
blueprint wants inference on. ``MRZReader`` needs torch, so it is left to an
explicit import from ``mrz_ai.inference.pipeline`` rather than re-exported here:
importing this package must not cost a caller the whole framework.
"""

from __future__ import annotations

from .beam import Candidate, best_lines
from .candidates import Reading, best_reading

__all__ = ["Candidate", "best_lines", "Reading", "best_reading"]
