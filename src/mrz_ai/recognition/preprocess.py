"""Turn a line crop into the tensor the recognizer expects.

The dataset produces crops of a fixed height but variable width, because the
crop jitter that imitates a real detector changes how much padding surrounds the
text. The model needs exactly 32x704. Resizing anisotropically — squeezing width
without touching height — is deliberate: it is what scene-text recognizers do,
and it means the model always sees 44 characters spread across 704 pixels
regardless of how loosely the line was framed. Preserving the aspect ratio and
padding instead would put the characters at a different scale in every sample,
which is a harder problem for no benefit.

Kept free of torch so the same code runs in a dataloader worker, in ONNX export
and in the CPU inference path.
"""

from __future__ import annotations

import cv2
import numpy as np

from .geometry import INPUT, InputGeometry

Array = np.ndarray

__all__ = ["prepare", "prepare_batch"]


def prepare(image: Array, geometry: InputGeometry | None = None) -> Array:
    """Scale a greyscale crop to the model's input, as float32 in [0, 1].

    Returns ``(1, height, width)`` — channel first, ready to stack into a batch.
    """
    geometry = geometry or INPUT
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim != 2:
        raise ValueError(f"expected a greyscale image, got shape {image.shape}")
    if image.size == 0:
        raise ValueError("the image is empty")

    # INTER_AREA is the right filter when shrinking: it averages the pixels being
    # discarded rather than sampling one of them, which matters when a 900-pixel
    # crop of 44 characters becomes 704.
    shrinking = image.shape[1] > geometry.width
    resized = cv2.resize(
        image,
        (geometry.width, geometry.height),
        interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
    )
    return (resized.astype(np.float32) / 255.0)[None, ...]


def prepare_batch(images: list[Array], geometry: InputGeometry | None = None) -> Array:
    """Stack crops into ``(batch, 1, height, width)``."""
    if not images:
        raise ValueError("no images to prepare")
    return np.stack([prepare(image, geometry) for image in images])
