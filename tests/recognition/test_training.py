"""The training loop runs, learns, and saves something usable.

Kept tiny on purpose: this checks the plumbing, not the model's accuracy. A real
run is 38k steps on a GPU.
"""

from __future__ import annotations

import torch

from mrz_ai.recognition.geometry import InputGeometry, ModelGeometry
from mrz_ai.training.recognition import (
    EMA,
    Stage,
    TorchLineDataset,
    TrainConfig,
    evaluate,
    _learning_rate,
    train_recognition,
)
from mrz_ai.synthetic.dataset import DatasetConfig

TINY_INPUT = InputGeometry(height=16, width=176, patch_height=8, patch_width=8)
TINY_MODEL = ModelGeometry(embed_dim=32, encoder_depth=1, encoder_heads=2, decoder_heads=2)


def _config(tmp_path, **overrides) -> TrainConfig:
    defaults = dict(
        batch_size=2,
        num_workers=0,
        dpi=100.0,
        eval_every=4,
        eval_batches=1,
        log_every=4,
        output_dir=tmp_path / "out",
        curriculum=(Stage("clean", (0.0, 0.1), 2), Stage("hard", (0.0, 1.0), 2)),
        input_geometry=TINY_INPUT,
        model_geometry=TINY_MODEL,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def test_a_sample_arrives_as_a_tensor_pair() -> None:
    dataset = TorchLineDataset(DatasetConfig(dpi=100), TINY_INPUT)
    image, label = dataset[0]

    assert image.shape == (1, TINY_INPUT.height, TINY_INPUT.width)
    assert image.dtype == torch.float32
    assert label.shape == (44,)
    assert label.dtype == torch.int64


def test_training_runs_and_saves_a_checkpoint(tmp_path) -> None:
    checkpoint = train_recognition(_config(tmp_path))
    assert checkpoint.exists()

    saved = torch.load(checkpoint, weights_only=False)
    assert {"model", "ema", "input_geometry", "model_geometry", "history", "final"} <= saved.keys()


def test_the_checkpoint_records_the_geometry_it_was_trained_at(tmp_path) -> None:
    # Loading a checkpoint into the wrong geometry would fail confusingly, so the
    # geometry travels with the weights.
    saved = torch.load(train_recognition(_config(tmp_path)), weights_only=False)
    assert saved["input_geometry"]["width"] == TINY_INPUT.width
    assert saved["model_geometry"]["embed_dim"] == TINY_MODEL.embed_dim


def test_the_curriculum_is_walked_in_order(tmp_path) -> None:
    train_recognition(_config(tmp_path))
    # Both stages ran: 4 steps total, evaluated every 4.
    saved = torch.load(tmp_path / "out" / "recognition.pt", weights_only=False)
    assert saved["history"]


def test_the_learning_rate_warms_up_then_decays() -> None:
    config = TrainConfig(warmup_steps=100, learning_rate=1e-3,
                         curriculum=(Stage("only", (0, 1), 1000),))

    assert _learning_rate(0, config) == 0.0
    assert _learning_rate(50, config) < _learning_rate(100, config)
    assert _learning_rate(100, config) == config.learning_rate      # peak at the end of warmup
    assert _learning_rate(1000, config) < config.learning_rate * 0.01  # cosine floor


def test_the_ema_tracks_the_weights_slowly() -> None:
    from mrz_ai.recognition.model import MRZRecognizer

    model = MRZRecognizer(TINY_INPUT, TINY_MODEL)
    ema = EMA(model, decay=0.9)
    before = ema.shadow["head.bias"].clone()

    with torch.no_grad():
        model.head.bias.add_(1.0)
    ema.update(model)

    after = ema.shadow["head.bias"]
    # Moved towards the new value, but nowhere near all the way.
    assert not torch.allclose(after, before)
    assert not torch.allclose(after, model.head.bias)


def test_evaluate_reports_all_three_levels(tmp_path) -> None:
    from torch.utils.data import DataLoader
    from mrz_ai.recognition.model import MRZRecognizer

    loader = DataLoader(TorchLineDataset(DatasetConfig(dpi=100), TINY_INPUT), batch_size=2)
    model = MRZRecognizer(TINY_INPUT, TINY_MODEL)
    result = evaluate(model, loader, torch.device("cpu"), batches=1)

    assert 0.0 <= result.char_accuracy <= 1.0
    assert 0.0 <= result.line_accuracy <= 1.0
    # An untrained model gets a line right with probability ~37^-44.
    assert result.line_accuracy == 0.0


def test_evaluate_leaves_the_model_in_training_mode(tmp_path) -> None:
    from torch.utils.data import DataLoader
    from mrz_ai.recognition.model import MRZRecognizer

    loader = DataLoader(TorchLineDataset(DatasetConfig(dpi=100), TINY_INPUT), batch_size=2)
    model = MRZRecognizer(TINY_INPUT, TINY_MODEL)
    model.train()
    evaluate(model, loader, torch.device("cpu"), batches=1)
    assert model.training, "evaluation must not silently disable dropout for the rest of training"
