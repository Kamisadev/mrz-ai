"""The recognizer's input contract, pinned by measurement.

Every number here was chosen against a benchmark or a rendered sample rather
than by analogy to scene-text defaults, because stock PARSeq is wrong for an MRZ
on every axis at once.

Stock PARSeq expects 32x128 (4:1) images of at most 25 characters from a
36-character lowercase alphabet. A TD3 line is 44 characters at roughly 24:1
over a 37-character alphabet. Squeezing 44 characters into 128 pixels leaves 2.9
pixels each, which is not legible to anything.

The width follows from the alphabet, not from convention: 44 x 16 = 704, so each
character occupies exactly 16 pixels and exactly two patches. One patch per
character (a 16-pixel-wide patch) is cheaper, but the crop jitter that a real
detector produces would then misalign the grid by up to half a character and
smear every glyph across a patch boundary. Two patches per character absorbs
that.

The depth follows from the latency target. Measured single-line CPU latency:

    stock ViT-Small at 32x704, patch (4,8), 704 tokens   75.5 ms
    ViT-tiny        at 32x704, patch (4,8), 704 tokens   18.1 ms
    ViT-tiny        at 32x704, patch (8,8), 352 tokens    7.6 ms   <- chosen
    ViT-tiny        at 32x704, patch (8,16), 176 tokens   4.4 ms

The blueprint's target is under 100ms on CPU for the whole pipeline, and that
budget has to cover detection plus *two* line reads. Stock PARSeq alone would
spend 151ms on the two lines, so the ViT-Small encoder it ships with is not an
option here regardless of how well it reads.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..parser import fields as F
from ..parser.charset import ALPHABET_SIZE


@dataclass(frozen=True)
class InputGeometry:
    """Shape of the image the recognizer accepts."""

    height: int = 32
    #: 44 characters x 16 pixels. Keeping this an exact multiple of the line
    #: length is what makes the patch grid align to character cells.
    width: int = 704

    #: (height, width) of a patch. Half a character wide, so a character always
    #: spans two of them.
    patch_height: int = 8
    patch_width: int = 8

    def __post_init__(self) -> None:
        if self.width % F.LINE_LENGTH:
            raise ValueError(
                f"width {self.width} is not a multiple of the {F.LINE_LENGTH}-character line"
            )
        if self.height % self.patch_height or self.width % self.patch_width:
            raise ValueError("the patch grid must divide the image exactly")

    @property
    def pixels_per_char(self) -> float:
        return self.width / F.LINE_LENGTH

    @property
    def patches_per_char(self) -> float:
        return self.pixels_per_char / self.patch_width

    @property
    def grid(self) -> tuple[int, int]:
        """Patch grid as (rows, columns)."""
        return self.height // self.patch_height, self.width // self.patch_width

    @property
    def num_tokens(self) -> int:
        rows, columns = self.grid
        return rows * columns

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


@dataclass(frozen=True)
class ModelGeometry:
    """Size of the encoder and decoder.

    Roughly ViT-tiny. Chosen for the CPU budget rather than for accuracy: see the
    latency table above.
    """

    embed_dim: int = 192
    encoder_depth: int = 6
    encoder_heads: int = 3
    decoder_depth: int = 1
    decoder_heads: int = 3
    mlp_ratio: int = 4
    dropout: float = 0.1

    #: One output position per character. A TD3 line is always exactly this long,
    #: which is why the decoder does not need to predict a length or a stop token.
    max_label_length: int = F.LINE_LENGTH
    #: A-Z, 0-9 and the filler. No case, no punctuation, no symbols.
    num_classes: int = ALPHABET_SIZE


INPUT = InputGeometry()
MODEL = ModelGeometry()
