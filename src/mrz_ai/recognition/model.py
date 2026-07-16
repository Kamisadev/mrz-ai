"""The MRZ line recognizer: a PARSeq-lineage ViT encoder with a purpose-built head.

**This is not stock PARSeq, and the deviation is deliberate.** The encoder is the
same idea — patch embedding into a plain ViT — but the decoder predicts all 44
characters at once instead of autoregressively. Three reasons, in order of
weight:

1. *It emits what Phase 4 consumes.* The candidate decoder's plan is "top-K per
   position, then ICAO-validate". Predicting positions independently gives 44
   per-position marginals directly, which is exactly that input. An
   autoregressive model instead gives a conditional factorization, where the
   top-K at position *t* depends on what was chosen before it — the wrong output
   shape for the pipeline, not merely a slower one.
2. *The context autoregression exists to learn is not there.* PARSeq's permuted
   language modelling buys an implicit language model. MRZ names are arbitrary
   transliterations and document numbers are arbitrary alphanumerics, so there is
   no prior to learn. The one real cross-position structure is the check digits,
   and Phase 4 enforces those exactly rather than approximately.
3. *The CPU budget.* Forty-four sequential decoder passes against one.

The blueprint asks for an "MRZ-aware decoder" with "top-K hypotheses"; a
fixed-length head emitting per-position marginals is a more faithful reading of
that than stock PARSeq would be.

**Output contract, which Phase 4 depends on:** ``forward`` returns logits shaped
``(batch, 44, 37)`` — one distribution over the MRZ alphabet per character
position, in reading order. Never a length, never a stop token: a TD3 line is
always exactly 44 characters.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .geometry import INPUT, MODEL, InputGeometry, ModelGeometry


class PatchEmbed(nn.Module):
    """Cut the image into patches and project each to a vector.

    A strided convolution is the patch grid: kernel and stride both equal the
    patch size, so patches do not overlap.
    """

    def __init__(self, geometry: InputGeometry, embed_dim: int) -> None:
        super().__init__()
        self.geometry = geometry
        self.proj = nn.Conv2d(
            1,
            embed_dim,
            kernel_size=(geometry.patch_height, geometry.patch_width),
            stride=(geometry.patch_height, geometry.patch_width),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # (B, 1, H, W) -> (B, dim, rows, cols) -> (B, rows*cols, dim)
        patches: torch.Tensor = self.proj(images)
        return patches.flatten(2).transpose(1, 2)


class MRZRecognizer(nn.Module):
    """Read one MRZ line.

    Input is a greyscale line crop of exactly ``INPUT.height`` x ``INPUT.width``,
    scaled to [0, 1]. Output is ``(batch, 44, 37)`` logits.
    """

    def __init__(
        self,
        input_geometry: InputGeometry | None = None,
        model_geometry: ModelGeometry | None = None,
    ) -> None:
        super().__init__()
        self.input_geometry = input_geometry or INPUT
        self.model_geometry = model_geometry or MODEL
        dim = self.model_geometry.embed_dim

        self.patch_embed = PatchEmbed(self.input_geometry, dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.input_geometry.num_tokens, dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=self.model_geometry.encoder_heads,
            dim_feedforward=dim * self.model_geometry.mlp_ratio,
            dropout=self.model_geometry.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, self.model_geometry.encoder_depth, enable_nested_tensor=False
        )
        self.encoder_norm = nn.LayerNorm(dim)

        # One learned query per character position. The queries locate their own
        # characters by attending to the encoder's output rather than slicing it
        # at fixed columns: the crop jitter and the squeeze from resizing a
        # variable-aspect crop to a fixed width mean a character does not land at
        # a predictable pixel. Pooling token columns would assume precisely the
        # alignment the data pipeline deliberately breaks.
        self.query_embed = nn.Parameter(
            torch.zeros(1, self.model_geometry.max_label_length, dim)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=self.model_geometry.decoder_heads,
            dim_feedforward=dim * self.model_geometry.mlp_ratio,
            dropout=self.model_geometry.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, self.model_geometry.decoder_depth)
        self.decoder_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, self.model_geometry.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.query_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return ``(batch, 44, 37)`` logits for a batch of line crops."""
        expected = (self.input_geometry.height, self.input_geometry.width)
        if images.shape[-2:] != expected:
            raise ValueError(
                f"expected a {expected[0]}x{expected[1]} crop, got "
                f"{tuple(images.shape[-2:])}. Resize with recognition.preprocess."
            )

        tokens = self.patch_embed(images) + self.pos_embed
        memory = self.encoder_norm(self.encoder(tokens))

        queries = self.query_embed.expand(images.shape[0], -1, -1)
        # No causal mask: every position is predicted from the image alone, not
        # from its neighbours.
        decoded = self.decoder_norm(self.decoder(queries, memory))
        logits: torch.Tensor = self.head(decoded)
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
