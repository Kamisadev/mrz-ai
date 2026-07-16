"""Training entrypoint for the MRZ line recognizer.

Everything the RunPod notebook needs is `train_recognition(TrainConfig(...))`.
The notebook stays a single paste-and-run cell because all the logic lives here,
where it can be tested and read.

The curriculum is the blueprint's five stages expressed as a severity sweep. The
generator's severity is one number, so a stage is just a range and a step count.
Stage 5 in the blueprint is "fine-tune on real images" and is absent here: there
are no real images. Nothing substitutes for it, and its absence is why the
accuracy this reports is not a claim about production.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from ..recognition.geometry import INPUT, MODEL, InputGeometry, ModelGeometry
from ..recognition.model import MRZRecognizer, count_parameters
from ..recognition.preprocess import prepare
from ..recognition.tokenizer import encode
from ..synthetic.dataset import DatasetConfig, MRZLineDataset


@dataclass(frozen=True)
class Stage:
    """One curriculum step: how hard the documents are, and for how long."""

    name: str
    severity: tuple[float, float]
    steps: int


#: The blueprint's curriculum. Each stage keeps the easier range beneath it —
#: sampling a range rather than a point stops the model forgetting how to read a
#: clean document once training gets hard.
DEFAULT_CURRICULUM = (
    Stage("clean", (0.0, 0.05), 2_000),
    Stage("light", (0.0, 0.35), 6_000),
    Stage("moderate", (0.0, 0.65), 10_000),
    Stage("heavy", (0.0, 1.0), 20_000),
)


@dataclass
class TrainConfig:
    batch_size: int = 128
    learning_rate: float = 7e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    #: Exponential moving average of the weights. Cheap, and reliably worth a
    #: little accuracy on this kind of task.
    ema_decay: float = 0.999
    num_workers: int = 8
    #: See DatasetConfig.dpi: 150 matches the model's 16px/char with the least work.
    dpi: float = 150.0
    seed: int = 0

    eval_every: int = 1_000
    eval_batches: int = 8
    log_every: int = 100

    output_dir: Path = Path("checkpoints/recognition")
    curriculum: tuple[Stage, ...] = DEFAULT_CURRICULUM
    input_geometry: InputGeometry = field(default_factory=lambda: INPUT)
    model_geometry: ModelGeometry = field(default_factory=lambda: MODEL)

    @property
    def total_steps(self) -> int:
        return sum(stage.steps for stage in self.curriculum)


class TorchLineDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Adapts the framework-free generator to torch.

    This is the only place torch meets the synthetic engine, which is why the
    engine never had to import it.
    """

    def __init__(self, config: DatasetConfig, geometry: InputGeometry) -> None:
        self.inner = MRZLineDataset(config)
        self.geometry = geometry

    def __len__(self) -> int:
        return len(self.inner)

    def set_epoch(self, epoch: int) -> None:
        self.inner.set_epoch(epoch)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.inner[index]
        image = prepare(sample.image, self.geometry)
        return torch.from_numpy(image), torch.from_numpy(encode(sample.text))


class EMA:
    """A shadow copy of the weights, averaged over training."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
            if param.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def state_dict(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        merged = {k: v.detach().clone() for k, v in model.state_dict().items()}
        merged.update(self.shadow)
        return merged


@dataclass
class EvalResult:
    """Accuracy at the three levels that matter, in rising order of difficulty."""

    char_accuracy: float
    line_accuracy: float
    loss: float

    def __str__(self) -> str:
        return (
            f"loss {self.loss:.4f}  char {self.char_accuracy:.2%}  line {self.line_accuracy:.2%}"
        )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    batches: int,
) -> EvalResult:
    model.eval()
    correct_chars = total_chars = correct_lines = total_lines = 0
    losses = []

    for index, (images, labels) in enumerate(loader):
        if index >= batches:
            break
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        losses.append(
            functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)).item()
        )
        predictions = logits.argmax(-1)

        correct_chars += (predictions == labels).sum().item()
        total_chars += labels.numel()
        # A line is only correct if every one of its 44 characters is: this is
        # the number that matters, since one wrong character is a wrong document.
        correct_lines += (predictions == labels).all(dim=1).sum().item()
        total_lines += labels.shape[0]

    model.train()
    return EvalResult(
        char_accuracy=correct_chars / max(total_chars, 1),
        line_accuracy=correct_lines / max(total_lines, 1),
        loss=float(np.mean(losses)) if losses else float("nan"),
    )


def _learning_rate(step: int, config: TrainConfig) -> float:
    """Linear warmup, then cosine decay — the blueprint's schedule."""
    if step < config.warmup_steps:
        return config.learning_rate * step / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(config.total_steps - config.warmup_steps, 1)
    return config.learning_rate * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def _loader(
    severity: tuple[float, float], config: TrainConfig, *, seed: int, train: bool
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = TorchLineDataset(
        DatasetConfig(
            severity_range=severity,
            dpi=config.dpi,
            target_height=config.input_geometry.height,
            seed=seed,
            epoch_size=config.batch_size * 200,
        ),
        config.input_geometry,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,  # the stream is already random; shuffling an index is pointless
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )


def train_recognition(config: TrainConfig | None = None) -> Path:
    """Train the recognizer through the curriculum. Returns the checkpoint path."""
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MRZRecognizer(config.input_geometry, config.model_geometry).to(device)
    ema = EMA(model, config.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    # Mixed precision on GPU only; on CPU it is a slowdown.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Held out by seed rather than by splitting: the generator is unbounded, so a
    # different seed is a genuinely disjoint stream and there is no leakage to
    # worry about. Evaluated at full severity, which is the honest test.
    validation = _loader((0.0, 1.0), config, seed=999_983, train=False)

    print(f"device: {device} | params: {count_parameters(model)/1e6:.2f}M")
    print(f"input: {config.input_geometry.height}x{config.input_geometry.width}, "
          f"{config.input_geometry.num_tokens} tokens, "
          f"{config.input_geometry.pixels_per_char:.0f}px/char")
    print(f"steps: {config.total_steps} over {len(config.curriculum)} stages\n")

    history: list[dict[str, object]] = []
    step = 0
    started = time.perf_counter()

    for stage in config.curriculum:
        print(f"--- stage {stage.name}: severity {stage.severity}, {stage.steps} steps")
        loader = _loader(stage.severity, config, seed=config.seed, train=True)
        stream: TorchLineDataset = loader.dataset  # type: ignore[assignment]
        stage_step = 0
        epoch = 0

        while stage_step < stage.steps:
            stream.set_epoch(epoch)
            for images, labels in loader:
                if stage_step >= stage.steps:
                    break
                step += 1
                stage_step += 1

                for group in optimizer.param_groups:
                    group["lr"] = _learning_rate(step, config)

                images, labels = images.to(device, non_blocking=True), labels.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(images)
                    loss = functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        labels.reshape(-1),
                        label_smoothing=config.label_smoothing,
                    )

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)

                if step % config.log_every == 0:
                    rate = step / (time.perf_counter() - started)
                    print(f"  step {step:6d}  loss {loss.item():.4f}  "
                          f"lr {optimizer.param_groups[0]['lr']:.2e}  {rate:.1f} it/s")

                if step % config.eval_every == 0:
                    result = evaluate(model, validation, device, config.eval_batches)
                    print(f"  eval  {result}")
                    history.append({"step": step, "stage": stage.name, **asdict(result)})
            epoch += 1

    result = evaluate(model, validation, device, config.eval_batches * 2)
    print(f"\nfinal (raw weights): {result}")

    checkpoint = output_dir / "recognition.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(model),
            "input_geometry": asdict(config.input_geometry),
            "model_geometry": asdict(config.model_geometry),
            "history": history,
            "final": asdict(result),
        },
        checkpoint,
    )
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"saved: {checkpoint}")
    return checkpoint
