"""Turn a clean render into something that looks photographed.

This module is where the synthetic-to-real gap is won or lost, so two choices
are worth stating plainly.

**Severity is one number, not four pipelines.** The blueprint's curriculum —
clean, then blur, then reflection, then heavy — is a magnitude, so every effect
scales with a single ``severity`` in [0, 1] and the curriculum becomes a sweep.

**The effects are written on numpy and OpenCV rather than Albumentations.** Not
for speed alone: Albumentations draws from its own global RNG, which would
quietly break the per-seed reproducibility that the rest of this package
guarantees. Everything here draws from a ``numpy.random.Generator`` the caller
owns, and the OpenCV calls used are deterministic, so a seed really does pin the
output. Augraphy's paper and ink simulations are excellent but cost 100ms to
seconds per sample; they belong in an offline tier, not on a dataloader's hot
path where they would starve the GPU.

Effects are applied in the order the physical world applies them: the document
is printed, then posed, then imaged through a lens, then lit, then sampled by a
sensor, then compressed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class DegradeConfig:
    """Probability and magnitude of each effect at ``severity`` 1.0.

    Probabilities are scaled by severity too, so a low-severity sample is not
    merely a gently-degraded one — it is often a clean one, which is what the
    early curriculum stages want.
    """

    max_rotation_deg: float = 3.0
    max_perspective: float = 0.012
    max_motion_blur_px: int = 13
    max_defocus_px: float = 3.2
    #: The worst resolution loss, as a divisor: 3.0 takes a 30px character down
    #: to 10px before it is scaled back up.
    max_downscale: float = 3.0
    max_gaussian_noise: float = 18.0
    min_jpeg_quality: int = 28
    max_shadow_strength: float = 0.45
    max_glare_strength: float = 0.55
    max_scratches: int = 6
    max_dirt_spots: int = 40
    max_occlusions: int = 2

    p_rotate: float = 0.9
    p_perspective: float = 0.7
    p_motion_blur: float = 0.55
    p_defocus: float = 0.75
    p_resample: float = 0.85
    p_shadow: float = 0.5
    p_glare: float = 0.4
    p_dirt: float = 0.4
    p_scratches: float = 0.3
    p_occlusion: float = 0.15
    p_jpeg: float = 0.8


def _hit(rng: np.random.Generator, probability: float, severity: float) -> bool:
    return bool(rng.random() < probability * severity)


#: How much of an effect's range is still random at full severity. Drawing
#: uniformly from [0, max] would make the average sample only half-strength, so
#: severity 1.0 came out far milder than intended — visibly readable when it was
#: meant to be near the edge of legibility. Biasing the floor upwards makes
#: severity mean what it says while keeping variety.
_MAGNITUDE_FLOOR = 0.4


def _scaled(rng: np.random.Generator, magnitude: float, severity: float) -> float:
    """A random magnitude, biased towards ``magnitude`` as severity rises.

    At severity 1.0 this draws from [0.4, 1.0] x magnitude; at low severity it
    collapses towards zero.
    """
    spread = _MAGNITUDE_FLOOR + (1.0 - _MAGNITUDE_FLOOR) * rng.random()
    return float(spread * magnitude * severity)


# --------------------------------------------------------------------------
# printing
# --------------------------------------------------------------------------

def paper_background(
    shape: tuple[int, int], rng: np.random.Generator, severity: float
) -> Array:
    """A sheet of paper: tinted, unevenly lit, faintly textured.

    A flat 255 background is the tell of a synthetic sample. Real paper has a
    tint, a gradient from how it was lit, and a visible fibre texture.
    """
    height, width = shape
    base = 255.0 - _scaled(rng, 35.0, severity)
    paper = np.full((height, width), base, dtype=np.float32)

    # A broad linear gradient standing in for uneven illumination.
    angle = rng.random() * 2 * np.pi
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    ramp = (np.cos(angle) * xs / max(width, 1)) + (np.sin(angle) * ys / max(height, 1))
    paper += ramp * _scaled(rng, 28.0, severity)

    # Fibre texture: noise at a coarse scale, smoothed up.
    if severity > 0:
        small = rng.normal(0, 1, (max(height // 8, 1), max(width // 8, 1))).astype(np.float32)
        texture = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
        paper += texture * _scaled(rng, 6.0, severity)

    return paper


def compose_ink(clean: Array, paper: Array, rng: np.random.Generator, severity: float) -> Array:
    """Lay the rendered ink onto the paper.

    The render is ink-on-white; treat its darkness as a coverage mask so the
    paper shows through, and let the ink be something other than pure black —
    real MRZ ink is a dark grey that varies between printers.
    """
    coverage = 1.0 - clean.astype(np.float32) / 255.0
    ink_value = _scaled(rng, 60.0, severity)
    printed: Array = paper * (1.0 - coverage) + ink_value * coverage
    return printed


# --------------------------------------------------------------------------
# posing
# --------------------------------------------------------------------------

def warp(
    image: Array, rng: np.random.Generator, severity: float, config: DegradeConfig
) -> tuple[Array, Array]:
    """Rotate and skew the sheet, returning the image and the transform applied.

    The transform is returned rather than discarded because the caller cannot
    reconstruct where the text went without it. An MRZ line is about 44:1, so
    geometry that sounds negligible is not: rotating a 1320-pixel line by 3
    degrees drags its ends 34 pixels vertically, which is most of the 38-pixel
    line height. Cropping such a line with a padded axis-aligned box around its
    *original* position would slice the ends off. Mapping the corners through
    this matrix gives the real extent.

    Angles stay small regardless, because in production the recognizer is fed a
    crop that detection has already deskewed. Heavy perspective belongs to the
    page-level detection generator, not here.
    """
    height, width = image.shape
    result = image
    transform = np.eye(3, dtype=np.float32)

    if _hit(rng, config.p_rotate, severity):
        angle = (rng.random() * 2 - 1) * config.max_rotation_deg * severity
        affine = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        result = cv2.warpAffine(
            result, affine, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        transform = np.vstack([affine, [0, 0, 1]]).astype(np.float32) @ transform

    if _hit(rng, config.p_perspective, severity):
        d = config.max_perspective * severity
        source = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
        )
        offsets = (rng.random((4, 2)).astype(np.float32) * 2 - 1) * np.array(
            [width * d, height * d], dtype=np.float32
        )
        matrix = cv2.getPerspectiveTransform(source, source + offsets)
        result = cv2.warpPerspective(
            result, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        transform = matrix.astype(np.float32) @ transform

    return result, transform


def transform_quad(box: tuple[int, int, int, int], transform: Array) -> Array:
    """Map a box's four corners through a transform.

    Returns the corners as (4, 2) in the order top-left, top-right,
    bottom-right, bottom-left. The quadrilateral is the honest answer: a rotated
    rectangle is not a rectangle, and flattening it to an upright bounding box
    loses the information a caller needs to extract just that line.
    """
    left, top, right, bottom = box
    corners = np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float32
    )
    return cv2.perspectiveTransform(corners.reshape(1, -1, 2), transform).reshape(-1, 2)


def transform_box(
    box: tuple[int, int, int, int], transform: Array
) -> tuple[int, int, int, int]:
    """The smallest upright box containing ``box`` after the transform.

    Useful for the detection stage, which predicts axis-aligned boxes. It is the
    wrong tool for cropping a single MRZ line: at 44:1, a few degrees of rotation
    make a line's upright box tall enough to swallow its neighbour, so the crop
    would show two lines while the label named one. Use :func:`transform_quad`
    for that.
    """
    moved = transform_quad(box, transform)
    return (
        int(np.floor(moved[:, 0].min())),
        int(np.floor(moved[:, 1].min())),
        int(np.ceil(moved[:, 0].max())),
        int(np.ceil(moved[:, 1].max())),
    )


# --------------------------------------------------------------------------
# the lens
# --------------------------------------------------------------------------

def motion_blur(image: Array, rng: np.random.Generator, severity: float, config: DegradeConfig) -> Array:
    """Smear along one direction, as a moving camera does."""
    length = int(_scaled(rng, config.max_motion_blur_px, severity))
    if length < 2:
        return image
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    angle = rng.random() * 180.0
    matrix = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = np.asarray(cv2.warpAffine(kernel, matrix, (length, length)), dtype=np.float32)
    total = kernel.sum()
    if total <= 0:  # pragma: no cover - degenerate kernel
        return image
    return cv2.filter2D(image, -1, kernel / total)


def defocus_blur(image: Array, rng: np.random.Generator, severity: float, config: DegradeConfig) -> Array:
    """Soften everything, as a misfocused lens does."""
    sigma = _scaled(rng, config.max_defocus_px, severity)
    if sigma < 0.3:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)


def resample(image: Array, rng: np.random.Generator, severity: float, config: DegradeConfig) -> Array:
    """Throw away resolution, then restore the size.

    The most important effect here and the easiest to forget. We render at
    300dpi, where a character is 30 pixels wide; a phone photographing a passport
    from a normal distance yields something closer to 8-12. A model that has only
    ever seen crisp glyphs has never seen the input it will actually be given, and
    no amount of noise or blur substitutes for genuinely missing resolution.
    """
    factor = 1.0 - _scaled(rng, 1.0 - 1.0 / config.max_downscale, severity)
    height, width = image.shape
    small_h, small_w = max(int(height * factor), 4), max(int(width * factor), 44)
    if small_h >= height and small_w >= width:
        return image
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------
# the scene
# --------------------------------------------------------------------------

#: Shadows are built at this fraction of the full resolution. A soft shadow has
#: no high-frequency content, so blurring it at full size cost 2.8ms of a 4.3ms
#: budget for a result indistinguishable from this one.
_SHADOW_SCALE = 0.25


def add_shadow(image: Array, rng: np.random.Generator, severity: float, config: DegradeConfig) -> Array:
    """Darken a soft-edged region, as something between page and light does."""
    height, width = image.shape
    small_h = max(int(height * _SHADOW_SCALE), 8)
    small_w = max(int(width * _SHADOW_SCALE), 8)

    mask = np.zeros((small_h, small_w), dtype=np.float32)
    points = (
        rng.random((3, 2)) * np.array([small_w, small_h], dtype=np.float32)
    ).astype(np.int32)
    cv2.fillPoly(mask, [points], 1.0)
    blurred = np.asarray(
        cv2.GaussianBlur(mask, (0, 0), sigmaX=small_w * 0.05 + 1, sigmaY=small_h * 0.15 + 1),
        dtype=np.float32,
    )
    full = np.asarray(
        cv2.resize(blurred, (width, height), interpolation=cv2.INTER_LINEAR), dtype=np.float32
    )

    shaded: Array = image * (1.0 - full * _scaled(rng, config.max_shadow_strength, severity))
    return shaded


def add_glare(image: Array, rng: np.random.Generator, severity: float, config: DegradeConfig) -> Array:
    """A specular highlight: the laminate of a passport reflecting a light.

    This is the effect most likely to destroy real MRZ reads, so it must be
    strong enough to hurt — but the caller bounds how much of the zone it covers.
    """
    height, width = image.shape
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = rng.random() * width, rng.random() * height
    radius_x = (0.1 + rng.random() * 0.4) * width
    radius_y = (0.2 + rng.random() * 0.8) * height
    falloff = ((xs - cx) / radius_x) ** 2 + ((ys - cy) / radius_y) ** 2
    highlight = np.exp(-falloff * 2.0)
    strength = _scaled(rng, config.max_glare_strength, severity)
    lit: Array = image + (255.0 - image) * highlight * strength
    return lit


def add_dirt(image: Array, rng: np.random.Generator, severity: float, config: DegradeConfig) -> Array:
    """Specks of grime on the page."""
    height, width = image.shape
    count = int(_scaled(rng, config.max_dirt_spots, severity))
    result = image.copy()
    for _ in range(count):
        x, y = int(rng.random() * width), int(rng.random() * height)
        radius = int(1 + rng.random() * 3)
        value = float(rng.random() * 120)
        cv2.circle(result, (x, y), radius, value, -1)
    return result


def add_scratches(image: Array, rng: np.random.Generator, severity: float, config: DegradeConfig) -> Array:
    """Thin bright lines, as a worn laminate carries."""
    height, width = image.shape
    count = int(_scaled(rng, config.max_scratches, severity))
    result = image.copy()
    for _ in range(count):
        start = (int(rng.random() * width), int(rng.random() * height))
        end = (
            start[0] + int((rng.random() * 2 - 1) * width * 0.3),
            start[1] + int((rng.random() * 2 - 1) * height * 0.3),
        )
        value = float(180 + rng.random() * 75)
        cv2.line(result, start, end, value, 1)
    return result


def add_occlusion(image: Array, rng: np.random.Generator, severity: float, config: DegradeConfig) -> Array:
    """Cover a small part of the zone — a finger, a staple, a fold.

    Bounded on purpose. An occlusion that erases a character while the label
    still claims it teaches the recognizer to hallucinate from context, which
    puts a ceiling on the per-character accuracy it can ever reach. These stay
    narrow enough to damage a glyph rather than delete one.
    """
    height, width = image.shape
    count = int(round(_scaled(rng, config.max_occlusions, severity)))
    result = image.copy()
    for _ in range(count):
        patch_width = int((0.01 + rng.random() * 0.03) * width)
        patch_height = int((0.1 + rng.random() * 0.5) * height)
        x = int(rng.random() * max(width - patch_width, 1))
        y = int(rng.random() * max(height - patch_height, 1))
        value = float(rng.random() * 255)
        result[y : y + patch_height, x : x + patch_width] = value
    return result


# --------------------------------------------------------------------------
# the sensor
# --------------------------------------------------------------------------

def add_noise(image: Array, rng: np.random.Generator, severity: float, config: DegradeConfig) -> Array:
    """Sensor noise: a signal-dependent part and a constant part."""
    shot = rng.normal(0, 1, image.shape).astype(np.float32)
    shot *= np.sqrt(np.clip(image, 0, None)) * 0.06 * severity
    read = rng.normal(0, _scaled(rng, config.max_gaussian_noise, severity), image.shape)
    return image + shot + read.astype(np.float32)


def jpeg_artifacts(image: Array, rng: np.random.Generator, severity: float, config: DegradeConfig) -> Array:
    """Round-trip through JPEG to pick up its blocking and ringing."""
    quality = int(100 - _scaled(rng, 100 - config.min_jpeg_quality, severity))
    quality = max(config.min_jpeg_quality, min(100, quality))
    as_bytes = np.clip(image, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode(".jpg", as_bytes, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:  # pragma: no cover - encoder failure
        return image
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if decoded is None:  # pragma: no cover - decoder failure
        return image
    return np.asarray(decoded, dtype=np.float32)


# --------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DegradeResult:
    """A degraded image and the geometry needed to find the text in it.

    ``transform`` maps coordinates in the clean render onto the degraded image.
    Only the warp moves pixels around; every other effect is photometric, and
    the resampling stage restores the original size, so this one matrix is the
    whole story.
    """

    image: Array
    transform: Array

    def locate(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """The upright box a detector would predict for ``box``."""
        return transform_box(box, self.transform)

    def locate_quad(self, box: tuple[int, int, int, int]) -> Array:
        """The four corners ``box`` was moved to. Use this to crop a line."""
        return transform_quad(box, self.transform)


def degrade(
    clean: Array,
    rng: np.random.Generator,
    *,
    severity: float = 1.0,
    config: DegradeConfig | None = None,
) -> DegradeResult:
    """Photograph a clean render, badly.

    ``clean`` is 8-bit greyscale ink-on-paper; the image returned has the same
    shape and dtype. ``severity`` scales both how likely each effect is and how
    hard it hits, so 0.0 returns something nearly pristine and 1.0 something a
    human may struggle with.
    """
    config = config or DegradeConfig()
    severity = float(np.clip(severity, 0.0, 1.0))

    image = compose_ink(clean, paper_background(clean.shape, rng, severity), rng, severity)
    image, transform = warp(image, rng, severity, config)

    if _hit(rng, config.p_motion_blur, severity):
        image = motion_blur(image, rng, severity, config)
    if _hit(rng, config.p_defocus, severity):
        image = defocus_blur(image, rng, severity, config)

    if _hit(rng, config.p_shadow, severity):
        image = add_shadow(image, rng, severity, config)
    if _hit(rng, config.p_glare, severity):
        image = add_glare(image, rng, severity, config)
    if _hit(rng, config.p_dirt, severity):
        image = add_dirt(image, rng, severity, config)
    if _hit(rng, config.p_scratches, severity):
        image = add_scratches(image, rng, severity, config)
    if _hit(rng, config.p_occlusion, severity):
        image = add_occlusion(image, rng, severity, config)

    if _hit(rng, config.p_resample, severity):
        image = resample(image, rng, severity, config)

    image = add_noise(image, rng, severity, config)
    if _hit(rng, config.p_jpeg, severity):
        image = jpeg_artifacts(image, rng, severity, config)

    return DegradeResult(image=np.clip(image, 0, 255).astype(np.uint8), transform=transform)
