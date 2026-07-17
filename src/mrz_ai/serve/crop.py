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

*Why the tilt is corrected first.* That one-dimensional search assumes the text
is level, and the assumption is far more brittle than it looks. ICAO puts the two
lines 4.23mm apart with a cap height of 3.2mm, so barely 1mm of paper separates
them — across a line 111.76mm long. Tilt the page by more than
``asin(1.03/111.76)`` = **0.53 degrees** and the far end of line 1 sinks into the
rows of line 2: the projection sees one band where there are two, and there is no
angle of view from which a row profile can tell them apart again.

Measured, that was exactly the cliff. A clean render read 100% of documents level
and fell to 0% at 5 degrees, and the two-band search had in fact stopped working
at 0.75 — everything between was the halving fallback getting lucky and the
recognizer's own 1.5 degrees of trained skew tolerance absorbing the rest. Both
of those are accidents, not features, and a photograph taken by hand is not
level. So the skew is measured and removed before anything looks for a band.

Free of torch and of the web framework: this is image arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Array = np.ndarray

__all__ = [
    "Box",
    "Found",
    "Line",
    "decode_image",
    "deskew",
    "find_lines",
    "locate_lines",
    "skew_angle",
    "split_lines",
]

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

#: How far from level the text is allowed to be. A hand-held photograph of a page
#: on a desk is a few degrees out; a passport at 20 degrees is a different picture
#: than this tool is for, and searching that far mostly buys chances to lock onto
#: something that is not the MRZ.
_MAX_SKEW_DEG = 15.0
#: The coarse sweep's step, then the fine sweep's. The fine one is what matters:
#: the whole reason to do this is that half a degree separates two bands from one,
#: so an estimate good to a degree would not have been worth making.
_SKEW_COARSE_DEG = 1.0
_SKEW_FINE_DEG = 0.05
#: Below this, leave the pixels alone. Rotating an image resamples every pixel in
#: it, and the glyphs here are 16px wide — a level scan re-rendered to correct a
#: hundredth of a degree comes back very slightly blurrier and no straighter.
_SKEW_DEADBAND_DEG = 0.15
#: Fewer ink pixels than this and the estimate is being made from noise. Returning
#: zero is the honest answer: not "level", but "no evidence of tilt".
_MIN_INK_PX = 200
#: Width the region is shrunk to before the angle is measured, when it is wider.
#: An angle survives a uniform scale, and the sweep's cost is the ink pixel count
#: times the number of candidates: on a 12MP photograph, measuring at full size
#: cost 120ms against the 25ms the reading itself takes. A 44-character line at
#: this width still leaves ~22px per character, which is more than the recognizer
#: is given.
_SKEW_WORK_PX = 1000


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


@dataclass(frozen=True)
class Found:
    """The two lines, and the image whose pixels their boxes are counted in.

    The image travels with the boxes because it is not the one that was uploaded:
    it has been rotated to level the text. Handing back boxes alone would invite
    the caller to cut them out of the original photograph, where they address the
    wrong paper by exactly the tilt that was removed — and would do it silently,
    since a box is a box.
    """

    lines: tuple[Line, Line]
    image: Array
    #: Degrees of tilt taken out. Zero when the text was already level.
    skew_deg: float
    #: The ink resolved into exactly two bands. False means it did not, and the
    #: region was halved instead — a guess, and usually a bad one.
    #:
    #: Worth reporting rather than swallowing, because of what the alternative
    #: looks like. Point this at a whole passport page and the projection finds
    #: ink in the photo, the printed fields and the guilloche; the fallback
    #: halves the page and hands the recognizer two crops of nothing much.
    #: Measured on synthetic full pages, that reads 0 of 4 documents at 8.8% of
    #: characters — which is indistinguishable from a model that cannot read,
    #: and is nothing of the kind. The same pages with a box around the MRZ read
    #: 4 of 4 at 100%.
    banded: bool = True


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


def _sharpness(ys: Array, xs: Array, degrees: float, bins: int, offset: int) -> float:
    """How tightly the ink stacks into rows once sheared by ``degrees``.

    The sum of the squared row counts. It peaks when the ink piles into as few
    rows as possible, which for two straight lines of text is exactly when they
    are level — every character of a line lands in the same rows as every other,
    and the paper between the lines is left empty. Any tilt smears both lines
    across more rows and flattens the profile, so the sum of squares drops.

    Squared, not summed: the count is conserved whatever the angle, so anything
    linear in it is constant and measures nothing. It is the concentration that
    carries the signal.

    ``offset`` and ``bins`` must leave room for the largest shear the caller will
    ask about. Clamping instead of making room does not merely lose the overflow
    — it stacks it, and a pile of ink against the end of the profile scores like
    a beautifully sharp line. Written that way, this returned the widest angle in
    the sweep for every input, tilted or level.
    """
    shifted = ys - xs * np.tan(np.radians(degrees))
    binned = np.rint(shifted).astype(np.int64) + offset
    profile = np.bincount(binned, minlength=bins).astype(np.float64)
    return float((profile**2).sum())


def _search(ys: Array, xs: Array, span: tuple[int, int], low: float, high: float, step: float
            ) -> float:
    bins, offset = span
    angles = np.arange(low, high + step / 2, step)
    scores = [_sharpness(ys, xs, float(angle), bins, offset) for angle in angles]
    return float(angles[int(np.argmax(scores))])


def skew_angle(image: Array, box: Box) -> float:
    """How far the text inside ``box`` is tilted from level, in degrees.

    Positive means the line runs downhill to the right. Shears the ink's own
    coordinates rather than rotating the image once per candidate angle: the
    answer is identical for the small angles that matter, and it costs one
    ``nonzero`` call plus some arithmetic instead of a hundred image warps.

    Coarse to fine, because the score is not smooth enough to trust a hill climb
    — a sweep at ``_SKEW_COARSE_DEG`` cannot miss the peak, and a second sweep
    around it buys the precision this actually needs.
    """
    left, top, right, bottom = _clamped(image, box)
    region = image[top:bottom, left:right]
    if region.shape[1] > _SKEW_WORK_PX:
        # Uniformly, or the shrinking would itself shear the text and the answer
        # would come back scaled by the aspect ratio it was measured through.
        ratio = _SKEW_WORK_PX / region.shape[1]
        region = cv2.resize(
            region,
            (_SKEW_WORK_PX, max(int(round(region.shape[0] * ratio)), 1)),
            interpolation=cv2.INTER_AREA,
        )
    rows, columns = region.shape[:2]

    ys_int, xs_int = np.nonzero(_ink(region))
    if ys_int.size < _MIN_INK_PX:
        return 0.0

    ys = ys_int.astype(np.float64)
    # Shear about the region's middle, so the ink cannot slide off the profile's
    # end at one angle and be scored on fewer pixels than at another.
    xs = xs_int.astype(np.float64) - columns / 2.0

    # Room for the whole sweep. Half the width, levered by the steepest angle
    # allowed, is the furthest any pixel can travel — plus one for the rounding.
    reach = int(np.ceil(columns / 2.0 * np.tan(np.radians(_MAX_SKEW_DEG)))) + 1
    span = (rows + 2 * reach, reach)

    coarse = _search(ys, xs, span, -_MAX_SKEW_DEG, _MAX_SKEW_DEG, _SKEW_COARSE_DEG)
    fine = _search(
        ys, xs, span, coarse - _SKEW_COARSE_DEG, coarse + _SKEW_COARSE_DEG, _SKEW_FINE_DEG
    )
    return 0.0 if abs(fine) < _SKEW_DEADBAND_DEG else fine


def deskew(image: Array, box: Box) -> tuple[Array, float]:
    """The image rotated until the text in ``box`` is level, and the angle removed.

    Rotated about the box's own centre and kept at the original size, so every
    coordinate inside the box still means what it meant — the caller's ``box``
    goes on addressing the same paper, and only the paper has turned. Pixels near
    the frame's edge rotate out of view, which is of no consequence: the MRZ is
    at the centre of the selection by construction.

    Returns the image untouched when the text is already level. That is not an
    optimisation, it is the correctness case: a rotation resamples every pixel,
    and there is nothing to be won by softening a scan that was straight.
    """
    angle = skew_angle(image, box)
    if angle == 0.0:
        return image, 0.0

    centre = (box.x + box.width / 2.0, box.y + box.height / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
    height, width = image.shape[:2]
    leveled = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return leveled, angle


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
    """Where the two MRZ lines are inside ``box``. See ``_locate``."""
    first, second, _ = _locate(image, box)
    return first, second


def _locate(image: Array, box: Box) -> tuple[Line, Line, bool]:
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
    banded = len(bands) == 2
    if not banded:
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
    return found[0], found[1], banded


def find_lines(image: Array, box: Box) -> Found:
    """Level the text inside ``box``, then find its two lines. The whole job.

    ``locate_lines`` is kept separate and given a level region deliberately: it
    is the piece that can be checked against the synthetic engine's ground-truth
    boxes, and it stays checkable only while its assumption is somebody else's
    responsibility to satisfy.
    """
    leveled, angle = deskew(image, box)
    first, second, banded = _locate(leveled, box)
    return Found(lines=(first, second), image=leveled, skew_deg=angle, banded=banded)


def split_lines(image: Array, box: Box) -> tuple[Array, Array]:
    """The crops of the two MRZ lines inside ``box``: line 1 and line 2."""
    found = find_lines(image, box)
    return (
        _cut(found.image, found.lines[0].box),
        _cut(found.image, found.lines[1].box),
    )


def _cut(image: Array, box: Box) -> Array:
    return image[
        int(box.y) : int(box.y + box.height),
        int(box.x) : int(box.x + box.width),
    ]
