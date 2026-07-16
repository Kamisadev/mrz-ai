"""End to end: two crops in, one validated document out.

The model here is untrained, so nothing checks accuracy — that needs a real
checkpoint and belongs in the evaluation notebook. What is checked is the
plumbing: that the geometry travels with the weights, that the EMA weights are
the ones used, that a crop of any size is accepted, and that batching does not
change an answer.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest
import torch

from mrz_ai.recognition.geometry import InputGeometry, ModelGeometry
from mrz_ai.recognition.model import MRZRecognizer
from mrz_ai.inference.candidates import Reading
from mrz_ai.inference.pipeline import MRZReader

TINY_INPUT = InputGeometry(height=16, width=176, patch_height=8, patch_width=8)
TINY_MODEL = ModelGeometry(embed_dim=32, encoder_depth=1, encoder_heads=2, decoder_heads=2)
YEAR = 2026


def crop(height: int = 40, width: int = 900) -> np.ndarray:
    """A greyscale line crop of whatever size detection happened to hand over."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(height, width), dtype=np.uint8)


def tiny_reader(**kwargs) -> MRZReader:
    model = MRZRecognizer(TINY_INPUT, TINY_MODEL)
    return MRZReader(model, input_geometry=TINY_INPUT, **kwargs)


def write_checkpoint(path, *, ema_offset: float | None = None):
    """A checkpoint shaped exactly like the trainer's."""
    model = MRZRecognizer(TINY_INPUT, TINY_MODEL)
    ema = {k: v.clone() for k, v in model.state_dict().items()}
    if ema_offset is not None:
        ema["head.bias"] = ema["head.bias"] + ema_offset
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema,
            "input_geometry": asdict(TINY_INPUT),
            "model_geometry": asdict(TINY_MODEL),
            "history": [],
            "final": {},
        },
        path,
    )
    return path


def test_reading_two_crops_returns_a_document() -> None:
    reading = tiny_reader().read(crop(), crop(), reference_year=YEAR)

    assert isinstance(reading, Reading)
    assert len(reading.line1) == 44
    assert len(reading.line2) == 44
    assert reading.mrz.count("\n") == 1
    assert 0.0 <= reading.confidence <= 1.0


def test_crops_of_any_size_are_accepted() -> None:
    """Detection boxes are loose and the crop is never a fixed size."""
    reader = tiny_reader()
    for shape in ((32, 704), (40, 900), (25, 600), (60, 1400)):
        reading = reader.read(crop(*shape), crop(*shape), reference_year=YEAR)
        assert len(reading.line1) == 44


def test_the_geometry_travels_with_the_weights(tmp_path) -> None:
    """Loading a checkpoint must not depend on the caller knowing its geometry.

    The default geometry is 32x704; this checkpoint is 16x176. If from_checkpoint
    used the default, the model would refuse the input it was handed.
    """
    reader = MRZReader.from_checkpoint(write_checkpoint(tmp_path / "c.pt"))

    assert reader.input_geometry.width == TINY_INPUT.width
    assert reader.input_geometry.height == TINY_INPUT.height
    reader.read(crop(), crop(), reference_year=YEAR)  # and it actually runs


def test_the_ema_weights_are_used_by_default(tmp_path) -> None:
    """EMA is why the trainer keeps a second copy; loading the raw weights and
    calling it done would quietly discard the better model."""
    path = write_checkpoint(tmp_path / "c.pt", ema_offset=5.0)

    ema_reader = MRZReader.from_checkpoint(path)
    raw_reader = MRZReader.from_checkpoint(path, use_ema=False)

    ema_bias = ema_reader.model.head.bias
    raw_bias = raw_reader.model.head.bias
    assert not torch.allclose(ema_bias, raw_bias), "use_ema had no effect"
    assert torch.allclose(ema_bias, raw_bias + 5.0)


def test_the_model_is_in_eval_mode() -> None:
    """Dropout at inference would make the same crop read differently twice."""
    reader = tiny_reader()
    assert not reader.model.training

    one = reader.read(crop(), crop(), reference_year=YEAR)
    two = reader.read(crop(), crop(), reference_year=YEAR)
    assert one.mrz == two.mrz
    assert one.log_prob == pytest.approx(two.log_prob)


def test_reading_a_batch_matches_reading_one_at_a_time() -> None:
    """Batching is a throughput decision and must not be an accuracy one."""
    reader = tiny_reader()
    pairs = [(crop(), crop()), (crop(30, 700), crop(30, 700)), (crop(50, 1000), crop(50, 1000))]

    batched = reader.read_batch(pairs, reference_year=YEAR)
    singles = [reader.read(a, b, reference_year=YEAR) for a, b in pairs]

    assert [r.mrz for r in batched] == [r.mrz for r in singles]
    for batch_reading, single in zip(batched, singles):
        assert batch_reading.log_prob == pytest.approx(single.log_prob, abs=1e-4)


def test_an_empty_batch_is_not_an_error() -> None:
    assert tiny_reader().read_batch([], reference_year=YEAR) == []


def test_no_gradients_are_tracked() -> None:
    """Inference that builds a graph leaks memory in a long-running service."""
    reader = tiny_reader()
    reading = reader.read(crop(), crop(), reference_year=YEAR)
    assert reading.confidence >= 0.0
    for parameter in reader.model.parameters():
        assert parameter.grad is None


def test_the_reference_year_defaults_to_the_current_one() -> None:
    """Dates carry two digits, so the century depends on when you ask."""
    from datetime import date

    reader = tiny_reader()
    explicit = reader.read(crop(), crop(), reference_year=date.today().year)
    implied = reader.read(crop(), crop())

    assert implied.mrz == explicit.mrz


def test_k_is_configurable_and_reaches_the_search() -> None:
    reader = tiny_reader(k=1)
    assert reader.k == 1
    reader.read(crop(), crop(), reference_year=YEAR)
