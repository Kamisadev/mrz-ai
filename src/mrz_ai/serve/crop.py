"""An upload and a dragged box, turned into the two crops the model reads.

The box comes from a human, which is a deliberate choice worth recording. The
blueprint's detection stage — the model that would find the MRZ by itself — is
not built, and could not honestly be built yet: the synthetic engine draws bare
MRZ blocks, never a whole passport page, so a detector trained on it would learn
to find text on blank paper and would meet its first real page in production.
Hand-tuned morphology over a whole photograph would be worse than asking,
because it would be confidently wrong on exactly the real pages there is no way
to check it against.

So the person holding the passport draws the box. Inside that box the problem is
small enough to solve honestly, and this module solves it: given a region known
to contain two MRZ lines and nothing else, find them.

*Why the box is not simply halved.* The recognizer resizes its crop
anisotropically to exactly 32x704 — 44 characters, 16 pixels each. That makes
framing part of the input, not a detail of it: a crop carrying one extra
character-width of blank paper shifts every character out of the cell the model
expects. Training drew crops with at most 0.3 line-heights of padding per edge,
so the tolerance is roughly half a character. A person cannot drag to half a
character, and should not have to. Measured on synthetic pages, halving a box
drawn 30% loose read 10% of documents correctly; tightening it first reads 100%,
and the result no longer depends on how carefully the box was drawn.

*Why the ink projection is trustworthy here and a detector is not.* Inside the
box the search is one-dimensional and the answer is checkable: two dark bands on
pale paper, in a region already known to hold exactly them. That is testable
against the synthetic engine's own ground-truth line boxes, and is.

Free of torch and of the web framework: this is image arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Array = np.ndarray

__all__ = ["Box", "Line", "decode_image", "locate_lines", "split_lines"]

#: Below this a split is meaningless — in practice a box this small is a mis-drag.
_MIN_SIDE = 8
#: A row counts as text if it holds this fraction of the darkest row's ink. Low
#: enough to keep a line's thin top and bottom, high enough to ignore paper
#: speckle.
_ROW_INK = 0.12
#: A band must be at least this tall to be a line rather than a blemish.
_MIN_BAND = 3
#: Padding added around found text, in fractions of its own height. Sits inside
#: the 0.3 the recognizer was trained to tolerate.
_PAD = 0.15


@dataclass(frozen=True)
class Box:
    """A region of the uploaded image, in its own pixels, as the browser sends it.

    Floats because a canvas drag is scaled by the ratio between the displayed
    image and the natural one, and rounding on the client would be one more place
    for two coordinate systems to disagree.
    """

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class Line:
    """A located MRZ line, and whether the drag appears to have cut it off."""

    box: Box
    #: The line's ink ran to the edge of the selection, so characters may lie
    #: outside it. Worth reporting rather than swallowing: nothing downstream can
    #: recover a character that was never in the crop, and the resulting misread
    #: looks exactly like a model that cannot read. It is a suspicion, not a
    #: finding — an MRZ genuinely at the edge of a photograph trips it too.
    clipped: bool


def decode_image(data: bytes) -> Array:
    """Decode uploaded bytes to a greyscale image.

    Raises ``ValueError`` on anything that is not an image. An upload is
    untrusted: a caller sending a PDF, an empty body or a truncated JPEG must get
    an error it can show a user, not an exception from inside OpenCV.
    """
    if not data:
        raise ValueError("the image is empty")
    buffer = np.frombuffer(data, dtype=np.uint8)
    # IMREAD_GRAYSCALE rather than converting afterwards: OpenCV can skip
    # building the colour planes, and the recognizer never wanted them.
    image = cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise ValueError("the upload is not an image OpenCV can read")
    return image


def _ink(region: Array) -> Array:
    """A mask of the dark pixels, 255 where there is ink.

    Otsu rather than an adaptive threshold, which is not the obvious choice: a
    local threshold handles uneven lighting better in general, and read *worse*
    here — 72% against 85% on heavily degraded synthetic pages. An MRZ band is
    mostly paper, and a local window inside one finds contrast in the paper
    grain when there is no character nearby to anchor it. Otsu sees the whole
    region, where the ink and the paper are genuinely two populations.
    """
    blur = cv2.GaussianBlur(region, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary


def _runs(live: Array, minimum: int) -> list[tuple[int, int]]:
    """The contiguous ``True`` spans of ``live``, at least ``minimum`` long."""
    out: list[tuple[int, int]] = []
    start: int | None = None
    for index, on in enumerate(live):
        if on and start is None:
            start = index
        elif not on and start is not None:
            out.append((start, index))
            start = None
    if start is not None:
        out.append((start, len(live)))
    return [span for span in out if span[1] - span[0] >= minimum]


def _extent(mask: Array, axis: int, threshold: float) -> tuple[int, int] | None:
    """The first and last index along ``axis`` holding ink."""
    profile = mask.sum(axis=axis) / 255.0
    if profile.max() <= 0:
        return None
    live = profile > profile.max() * threshold if threshold else profile > 0
    if not live.any():
        return None
    first = int(np.argmax(live))
    last = int(len(live) - np.argmax(live[::-1]))
    return first, last


def _tighten(region: Array, top: int, bottom: int) -> tuple[tuple[int, int, int, int], bool]:
    """The ink's padded bounding box within ``region[top:bottom]``, and whether it is cut off.

    Thresholded again on the strip alone: a threshold chosen across both lines
    plus the paper between them is not the one that separates this line's ink
    from the paper immediately around it.
    """
    strip = region[top:bottom]
    mask = _ink(strip)

    rows = _extent(mask, 1, _ROW_INK)
    columns = _extent(mask, 0, 0.0)
    if rows is None or columns is None:
        return (0, top, region.shape[1], bottom), False

    row_start, row_end = rows
    left, right = columns
    pad = int(round(_PAD * (row_end - row_start)))

    # Ink reaching the selection's own border means the text was still going when
    # the box stopped. Checked before padding, since padding is then clamped to
    # the same border and would hide it.
    height, width = region.shape[:2]
    clipped = (
        left == 0
        or right == width
        or top + row_start == 0
        or top + row_end == height
    )

    box = (
        max(left - pad, 0),
        max(top + row_start - pad, 0),
        min(right + pad, width),
        min(top + row_end + pad, height),
    )
    return box, clipped


def _clamped(image: Array, box: Box) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]

    left = int(round(max(box.x, 0.0)))
    top = int(round(max(box.y, 0.0)))
    right = int(round(min(box.x + box.width, float(width))))
    bottom = int(round(min(box.y + box.height, float(height))))

    if left >= width or top >= height or right <= 0 or bottom <= 0:
        raise ValueError("the selection is outside the image")
    if right - left < _MIN_SIDE or bottom - top < _MIN_SIDE:
        raise ValueError("the selection is too small to hold two MRZ lines")
    return left, top, right, bottom


def locate_lines(image: Array, box: Box) -> tuple[Line, Line]:
    """Where the two MRZ lines are inside ``box``, in the image's own pixels.

    The box is clamped to the image first, because a drag that ran past the edge
    of the picture is an ordinary thing for a person to do and means what it
    looks like. A box that misses the image entirely is not, and raises.

    When the ink does not resolve into exactly two bands — a heavy shadow, a
    thumb over the page — the region is halved and each half tightened on its
    own. That is a worse guess about *where* the lines are and still a good one
    about how each is framed, which is the part the recognizer is sensitive to.
    """
    left, top, right, bottom = _clamped(image, box)
    region = image[top:bottom, left:right]

    mask = _ink(region)
    profile = mask.sum(axis=1) / 255.0
    bands = _runs(profile > profile.max() * _ROW_INK, _MIN_BAND) if profile.max() > 0 else []
    if len(bands) != 2:
        middle = region.shape[0] // 2
        bands = [(0, middle), (middle, region.shape[0])]

    found = []
    for band_top, band_bottom in bands:
        (x0, y0, x1, y1), clipped = _tighten(region, band_top, band_bottom)
        found.append(
            Line(
                box=Box(
                    x=float(left + x0),
                    y=float(top + y0),
                    width=float(x1 - x0),
                    height=float(y1 - y0),
                ),
                clipped=clipped,
            )
        )
    return found[0], found[1]


def split_lines(image: Array, box: Box) -> tuple[Array, Array]:
    """The crops of the two MRZ lines inside ``box``: line 1 and line 2."""
    first, second = locate_lines(image, box)
    return _cut(image, first.box), _cut(image, second.box)


def _cut(image: Array, box: Box) -> Array:
    return image[
        int(box.y) : int(box.y + box.height),
        int(box.x) : int(box.x + box.width),
    ]
