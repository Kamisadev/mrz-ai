"""Online synthetic samples: a new document every time one is asked for.

Nothing is cached and there is no epoch boundary in the data — sample *i* of
epoch 2 is a different document from sample *i* of epoch 1. With an effectively
infinite training set the model cannot memorise it, which is the whole reason to
generate rather than collect.

Deliberately free of torch. Phase 2 wraps this in a ``torch.utils.data.Dataset``
in three lines; keeping the dependency out means the notebooks and the tests can
import it cheaply, and it is the same boundary the parser keeps.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np

from ..parser import serialize
from ..parser import fields as F
from .degrade import DegradeConfig, degrade
from .identity import IdentityConfig, random_identity
from .render import render_mrz

Array = np.ndarray


@dataclass(frozen=True)
class LineSample:
    """One MRZ line, cropped as detection would hand it over."""

    image: Array
    text: str
    line_index: int
    severity: float

    def __post_init__(self) -> None:
        if len(self.text) != F.LINE_LENGTH:
            raise ValueError(f"a TD3 line is {F.LINE_LENGTH} characters, got {len(self.text)}")


@dataclass(frozen=True)
class DatasetConfig:
    """How samples are drawn.

    ``severity_range`` is what the curriculum moves. Sampling a range rather than
    a fixed value keeps easy examples in the mix at every stage, which stops the
    model forgetting how to read a clean document once training gets hard.
    """

    severity_range: tuple[float, float] = (0.0, 1.0)
    dpi: float = 200.0
    #: Height every crop is resized to. PARSeq wants a fixed input height.
    target_height: int = 32
    #: Fraction of the line's height added as padding on each edge before the
    #: crop, drawn per-sample and per-edge. Real detection boxes are loose,
    #: tight and offset by turns; a recognizer that has only seen pixel-perfect
    #: crops meets none of that in production.
    max_crop_padding: float = 0.3
    #: How far the crop may be offset, as a fraction of the line's height.
    max_crop_offset: float = 0.15
    #: Skew left behind after deskewing, in degrees. A detector's deskew is
    #: never exact, so the recognizer must not assume perfectly level text.
    max_residual_skew_deg: float = 1.5
    epoch_size: int = 100_000
    seed: int = 0

    identity: IdentityConfig = IdentityConfig()
    degrade: DegradeConfig = DegradeConfig()


def extract_line(
    image: Array,
    quad: Array,
    *,
    target_height: int,
    padding: tuple[float, float, float, float],
    skew_deg: float,
) -> Array:
    """Straighten a line out of the image, the way a detector hands one over.

    Cropping the quad's upright bounding box does not work here. An MRZ line is
    about 44:1, so a few degrees of rotation make that box tall enough to contain
    the *neighbouring* line — the crop would show two lines while the label named
    one, and the model would have to guess which to read. Production does not
    work that way either: detection locates the line and straightens it. So we
    map the quadrilateral onto a rectangle, which yields exactly one line.

    ``padding`` is (left, top, right, bottom) in multiples of the line's height,
    applied along the line's own axes rather than the image's. ``skew_deg`` tilts
    the result slightly, standing in for a deskew that was not quite right.
    """
    top_left, top_right, bottom_right, bottom_left = quad.astype(np.float32)

    along = top_right - top_left
    down = bottom_left - top_left
    length = float(np.linalg.norm(along)) or 1.0
    height = float(np.linalg.norm(down)) or 1.0
    along_unit, down_unit = along / length, down / height

    pad_left, pad_top, pad_right, pad_bottom = (p * height for p in padding)

    source = np.array(
        [
            top_left - along_unit * pad_left - down_unit * pad_top,
            top_right + along_unit * pad_right - down_unit * pad_top,
            bottom_right + along_unit * pad_right + down_unit * pad_bottom,
            bottom_left - along_unit * pad_left + down_unit * pad_bottom,
        ],
        dtype=np.float32,
    )

    out_height = target_height
    padded_length = length + pad_left + pad_right
    padded_height = height + pad_top + pad_bottom
    out_width = max(int(round(padded_length * out_height / padded_height)), 1)
    destination = np.array(
        [[0, 0], [out_width, 0], [out_width, out_height], [0, out_height]], dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(source, destination)
    if skew_deg:
        rotation = cv2.getRotationMatrix2D((out_width / 2, out_height / 2), skew_deg, 1.0)
        matrix = np.vstack([rotation, [0, 0, 1]]).astype(np.float32) @ matrix

    return cv2.warpPerspective(
        image,
        matrix,
        (out_width, out_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


class MRZLineDataset:
    """An endless stream of MRZ line crops with their ground-truth text."""

    def __init__(self, config: DatasetConfig | None = None) -> None:
        self.config = config or DatasetConfig()
        self._epoch = 0

    def __len__(self) -> int:
        # An arbitrary number: the data is unbounded, but training loops want a
        # length. It only sets how often the epoch counter ticks.
        return self.config.epoch_size

    def set_epoch(self, epoch: int) -> None:
        """Move to a new epoch, so the same index yields a new document."""
        self._epoch = epoch

    def _seed_for(self, document_index: int) -> int:
        # Mixing the epoch in is what makes the stream endless. Without it every
        # epoch would replay the same documents and the model would memorise
        # them, which is exactly what online generation exists to prevent.
        return (self.config.seed * 1_000_003 + self._epoch * 7_919 + document_index) % (2**31)

    def __getitem__(self, index: int) -> LineSample:
        """Build one sample from scratch.

        Every source of randomness descends from the index and the epoch, so a
        sample is reproducible without any shared state between workers.
        """
        config = self.config
        # Consecutive indices are the two lines of one document, so each
        # document contributes both its lines and the two are always seen
        # equally often. Seeding from the index directly would instead pair
        # line 1 of one passport with line 2 of another — which nothing would
        # catch, since ICAO puts no check digit on line 1 to tie it to line 2.
        document_index, line_index = divmod(index, F.LINE_COUNT)
        seed = self._seed_for(document_index)

        # Three independent streams, split by what they belong to. Who the
        # document belongs to and how it was photographed are unrelated, and
        # sharing a stream made the identity depend on how many numbers the
        # camera happened to draw first — fragile, and impossible to state as an
        # invariant. The camera stream is keyed to the document, since both lines
        # were in the same photograph and must share its severity and blur; the
        # crop stream is keyed to the line, since a detector locates each line
        # separately and frames each one a little differently.
        identity_rng = random.Random(seed)
        camera_rng = random.Random(seed ^ 0x5F37_2C91)
        crop_rng = random.Random((seed ^ 0x5F37_2C91) * F.LINE_COUNT + line_index)
        np_rng = np.random.default_rng(seed)

        fields = random_identity(identity_rng, config.identity)
        mrz = serialize(fields)

        low, high = config.severity_range
        severity = float(camera_rng.uniform(low, high))
        rendered = render_mrz(mrz, dpi=config.dpi)
        clean = np.asarray(rendered.image)

        result = degrade(clean, np_rng, severity=severity, config=config.degrade)

        # Both lines are always rendered and degraded together so the document
        # stays coherent: shared lighting, shared blur, and a sliver of the
        # neighbouring line bleeding into the crop, exactly as a real one has.
        quad = result.locate_quad(rendered.line_boxes[line_index])

        offset = crop_rng.uniform(-config.max_crop_offset, config.max_crop_offset)
        padding = (
            crop_rng.uniform(0, config.max_crop_padding),
            crop_rng.uniform(0, config.max_crop_padding) + offset,
            crop_rng.uniform(0, config.max_crop_padding),
            crop_rng.uniform(0, config.max_crop_padding) - offset,
        )
        skew = crop_rng.uniform(-config.max_residual_skew_deg, config.max_residual_skew_deg)

        crop = extract_line(
            result.image,
            quad,
            target_height=config.target_height,
            padding=padding,
            skew_deg=skew,
        )

        return LineSample(
            image=crop,
            text=mrz.split("\n")[line_index],
            line_index=line_index,
            severity=severity,
        )
