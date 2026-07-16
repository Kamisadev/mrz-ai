"""The recognizer's shape contract and the assumptions built into it."""

from __future__ import annotations

import pytest
import torch

from mrz_ai.recognition.geometry import INPUT, InputGeometry, ModelGeometry
from mrz_ai.recognition.model import MRZRecognizer, count_parameters


@pytest.fixture(scope="module")
def model() -> MRZRecognizer:
    return MRZRecognizer().eval()


def _blank(batch: int = 1) -> torch.Tensor:
    return torch.rand(batch, 1, INPUT.height, INPUT.width)


def test_the_output_is_one_distribution_per_character(model: MRZRecognizer) -> None:
    """The contract Phase 4 reads: (batch, 44, 37) logits, in reading order.

    Pinned here because the candidate decoder's beam search and ICAO validation
    are written against this shape.
    """
    with torch.no_grad():
        logits = model(_blank(3))
    assert logits.shape == (3, 44, 37)


def test_the_model_never_predicts_a_length(model: MRZRecognizer) -> None:
    # A TD3 line is always 44 characters, so there is no stop token and no way
    # for the model to emit a line that could not be a real one.
    assert model.model_geometry.max_label_length == 44
    assert model.model_geometry.num_classes == 37  # the alphabet, with no extras


def test_a_wrongly_sized_crop_is_rejected_rather_than_silently_resized(
    model: MRZRecognizer,
) -> None:
    with pytest.raises(ValueError, match="32x704"):
        model(torch.rand(1, 1, 32, 128))


def test_positions_are_predicted_from_the_image_not_from_each_other(
    model: MRZRecognizer,
) -> None:
    """No causal mask and no autoregression: every position sees the whole image.

    This is what makes the 44 outputs independent marginals, which is the input
    Phase 4's top-K-per-position search needs.
    """
    image = _blank()
    with torch.no_grad():
        first = model(image)
        second = model(image)
    torch.testing.assert_close(first, second)


def test_the_batch_dimension_is_independent(model: MRZRecognizer) -> None:
    # A sample's prediction must not depend on what it was batched with.
    a, b = _blank(), _blank()
    with torch.no_grad():
        separate = torch.cat([model(a), model(b)])
        together = model(torch.cat([a, b]))
    torch.testing.assert_close(separate, together, rtol=1e-4, atol=1e-4)


def test_the_model_stays_within_the_cpu_budget(model: MRZRecognizer) -> None:
    """The blueprint targets under 100ms on CPU for the whole pipeline, and that
    has to cover detection plus two line reads.

    Stock PARSeq's ViT-Small encoder measured 75.5ms per line at this width,
    which spends 151ms on the two lines before detection runs at all. Generous
    bound here: this is a smoke test against an accidental ViT-Small, not a
    benchmark.
    """
    assert count_parameters(model) < 8_000_000


def test_the_gradients_reach_the_whole_model() -> None:
    model = MRZRecognizer()
    model(_blank(2)).sum().backward()

    starved = [name for name, p in model.named_parameters() if p.grad is None]
    assert not starved, f"no gradient reached: {starved}"


def test_the_geometry_is_configurable() -> None:
    # Smaller geometry for a fast test, and proof nothing is hardcoded.
    model = MRZRecognizer(
        InputGeometry(height=16, width=176, patch_height=8, patch_width=8),
        ModelGeometry(embed_dim=32, encoder_depth=1, encoder_heads=2, decoder_heads=2),
    ).eval()
    with torch.no_grad():
        assert model(torch.rand(1, 1, 16, 176)).shape == (1, 44, 37)
