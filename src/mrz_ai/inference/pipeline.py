"""Two line crops in, one validated document out.

This is the blueprint's candidate decoder with the detection step still missing:
a caller hands over the two line crops that detection would have produced, and
gets back the best reading ICAO will accept, its confidence, and — when nothing
validates — the reasons why not.

Torch appears here and nowhere else in this package. `beam` and `candidates` are
plain numpy, so the search and the ICAO logic stay testable without a model and
portable to ONNX Runtime later, which is what the blueprint asks inference to run
on. The boundary is the same one the synthetic engine keeps against torch, and
for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch

from ..recognition.geometry import INPUT, InputGeometry, ModelGeometry
from ..recognition.model import MRZRecognizer
from ..recognition.preprocess import prepare
from .candidates import Reading, best_reading

Array = np.ndarray

__all__ = ["MRZReader"]


@dataclass
class MRZReader:
    """Reads MRZ line crops with a trained recognizer.

    Construct from a checkpoint with :meth:`from_checkpoint`; the bare
    constructor takes a model directly, which is what the tests use.
    """

    model: MRZRecognizer
    input_geometry: InputGeometry = INPUT
    #: Candidates per line handed to the ICAO stage. 8 costs ~2ms and is where
    #: the repairable misreads live; the search is exact, so a larger k only ever
    #: adds less likely candidates rather than changing the ones already found.
    k: int = 8
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        self.model = self.model.to(self.device)
        # Dropout at inference would make the same crop read differently twice.
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        use_ema: bool = True,
        k: int = 8,
        device: str | torch.device = "cpu",
    ) -> MRZReader:
        """Load a reader from a checkpoint written by ``train_recognition``.

        The geometry comes out of the checkpoint rather than from the defaults:
        weights trained at one input size cannot read another, and a caller
        should not have to remember which.

        ``use_ema`` picks the averaged weights, which is why the trainer keeps
        them — they are reliably a little better, and defaulting to the raw ones
        would quietly discard that.
        """
        device = torch.device(device)
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)

        input_geometry = InputGeometry(**checkpoint["input_geometry"])
        model_geometry = ModelGeometry(**checkpoint["model_geometry"])
        model = MRZRecognizer(input_geometry, model_geometry)

        weights = checkpoint["ema"] if use_ema and "ema" in checkpoint else checkpoint["model"]
        model.load_state_dict(weights)

        return cls(model=model, input_geometry=input_geometry, k=k, device=device)

    @torch.no_grad()
    def _logits(self, crops: list[Array]) -> Array:
        """Run the model over crops, returning ``(n, 44, 37)`` logits."""
        batch = np.stack([prepare(crop, self.input_geometry) for crop in crops])
        tensor = torch.from_numpy(batch).to(self.device)
        logits: torch.Tensor = self.model(tensor)
        return logits.detach().cpu().numpy()

    def read(
        self,
        line1_crop: Array,
        line2_crop: Array,
        *,
        reference_year: int | None = None,
    ) -> Reading:
        """Read one document from its two line crops.

        The crops may be any size; they are resized to the model's input. The
        reading always comes back — check ``is_valid`` and ``issues`` rather than
        expecting an exception, since a scan that fails ICAO is a normal outcome
        and its best guess is what a human reviewer needs to correct.
        """
        return self.read_batch([(line1_crop, line2_crop)], reference_year=reference_year)[0]

    def read_batch(
        self,
        pairs: list[tuple[Array, Array]],
        *,
        reference_year: int | None = None,
    ) -> list[Reading]:
        """Read several documents in one forward pass.

        Purely a throughput decision: every document is decoded from its own
        logits afterwards, so a batched read returns what reading them one at a
        time would.
        """
        if not pairs:
            return []
        # Two-digit years mean the century depends on when the question is asked.
        year = reference_year if reference_year is not None else date.today().year

        flat = [crop for pair in pairs for crop in pair]
        logits = self._logits(flat)

        return [
            best_reading(logits[index * 2], logits[index * 2 + 1], k=self.k, reference_year=year)
            for index in range(len(pairs))
        ]
