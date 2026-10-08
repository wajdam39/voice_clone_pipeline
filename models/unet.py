"""2D U-Net architecture for diffusion-based mel-spectrogram generation.

Designed specifically for inputs of shape ``[batch, 1, n_mels, segment_length]``
(default: ``[B, 1, 80, 512]``, matching ``SegmentedVoiceDataset`` output). The
network predicts the Gaussian noise added to a spectrogram at a given
diffusion timestep ``t`` (DDPM-style epsilon-prediction), conditioned on ``t``
via sinusoidal position embeddings injected into every residual block.

The asymmetry between the mel axis (80) and the time axis (512) is handled by
downsampling both axes together with stride-2 convolutions between each pair
of resolution stages. With the default 3 channel-multiplier stages (2
downsampling steps), 80 and 512 are both evenly divisible by 4
(80 -> 40 -> 20, 512 -> 256 -> 128), so skip-connection shapes line up
exactly with no rounding/cropping needed. ``UNet2D.__init__`` validates this
divisibility for whatever ``channel_multipliers`` length is configured.
"""

from __future__ import annotations

import math
from typing import Sequence

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - import guard
    torch = None
    nn = object
    F = None

DEFAULT_BASE_CHANNELS = 32
DEFAULT_CHANNEL_MULTIPLIERS = (1, 2, 4)
DEFAULT_TIME_EMB_DIM = 128
DEFAULT_NUM_RES_BLOCKS = 2
DEFAULT_NUM_GROUPS = 8


def _resolve_num_groups(preferred_num_groups: int, num_channels: int) -> int:
    """Pick a valid ``num_groups`` for ``nn.GroupNorm`` given ``num_channels``.

    ``nn.GroupNorm`` requires ``num_channels % num_groups == 0``. Simply
    clamping with ``min(preferred_num_groups, num_channels)`` can still be
    invalid (e.g. min(8, 12) = 8, but 12 % 8 != 0), so this searches downward
    from the clamped value for the largest valid divisor, always succeeding
    at worst with 1 (which is always a valid, if degenerate, choice).

    Args:
        preferred_num_groups: Desired number of groups.
        num_channels: Number of channels to normalize.

    Returns:
        A number of groups ``g`` such that ``1 <= g <= num_channels`` and
        ``num_channels % g == 0``.
    """
    candidate = max(1, min(preferred_num_groups, num_channels))
    while num_channels % candidate != 0:
        candidate -= 1
    return candidate


class SinusoidalPositionEmbeddings(nn.Module if nn is not object else object):
    """Sinusoidal embedding of the diffusion timestep ``t`` (as in DDPM/Transformer).

    Args:
        dim: Dimensionality of the output embedding vector.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """Embed a batch of integer timesteps.

        Args:
            t: 1-D tensor of shape (batch,), integer or float diffusion
                timestep indices.

        Returns:
            Tensor of shape (batch, dim).
        """
        device = t.device
        half_dim = self.dim // 2
        freq_exponent = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -freq_exponent)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:  # pragma: no cover - only for odd dims
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module if nn is not object else object):
    """A GroupNorm -> SiLU -> Conv2d residual block with time-embedding injection.

    The diffusion timestep embedding is projected and added (broadcast over
    the spatial dimensions) between the two convolutions, conditioning the
    block's output on how much noise is currently present.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        time_emb_dim: Dimensionality of the incoming time embedding vector.
        num_groups: Number of groups for GroupNorm.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_groups: int = DEFAULT_NUM_GROUPS,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_resolve_num_groups(num_groups, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        self.norm2 = nn.GroupNorm(_resolve_num_groups(num_groups, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual_proj = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.SiLU()

    def forward(self, x, time_emb):
        """
        Args:
            x: Tensor of shape (batch, in_channels, H, W).
            time_emb: Tensor of shape (batch, time_emb_dim).

        Returns:
            Tensor of shape (batch, out_channels, H, W).
        """
        h = self.conv1(self.activation(self.norm1(x)))

        time_bias = self.time_proj(self.activation(time_emb))
        h = h + time_bias[:, :, None, None]

        h = self.conv2(self.activation(self.norm2(h)))

        return h + self.residual_proj(x)


class Downsample(nn.Module if nn is not object else object):
    """Stride-2 convolution that halves both spatial dimensions (mel and time)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module if nn is not object else object):
    """Nearest-neighbor upsample + convolution that doubles both spatial dimensions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet2D(nn.Module if nn is not object else object):
    """2D U-Net predicting the noise added to a mel-spectrogram at timestep ``t``.

    Expects input of shape ``[batch, 1, n_mels, segment_length]`` (default:
    ``[B, 1, 80, 512]``) and returns a tensor of the identical shape: the
    predicted Gaussian noise component (DDPM epsilon-prediction).

    Args:
        n_mels: Height of the input spectrogram (mel channels), default 80.
        segment_length: Width of the input spectrogram (time frames), default 512.
        base_channels: Number of channels in the first conv block; doubled at
            each downsampling stage according to ``channel_multipliers``.
        channel_multipliers: Per-stage channel multiplier, e.g. ``(1, 2, 4)``
            gives 3 resolution stages (2 downsample steps between them).
        time_emb_dim: Dimensionality of the sinusoidal diffusion-timestep embedding.
        num_res_blocks: Number of residual blocks per resolution stage.
    """

    def __init__(
        self,
        n_mels: int = 80,
        segment_length: int = 512,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        channel_multipliers: Sequence[int] = DEFAULT_CHANNEL_MULTIPLIERS,
        time_emb_dim: int = DEFAULT_TIME_EMB_DIM,
        num_res_blocks: int = DEFAULT_NUM_RES_BLOCKS,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError(
                "PyTorch is required to use UNet2D. Install it with `pip install torch`."
            )
        super().__init__()

        num_stages = len(channel_multipliers)
        downsample_factor = 2 ** (num_stages - 1)
        if n_mels % downsample_factor != 0 or segment_length % downsample_factor != 0:
            raise ValueError(
                f"n_mels ({n_mels}) and segment_length ({segment_length}) must both "
                f"be divisible by {downsample_factor} (2^(num_stages-1)) so that "
                "skip-connection shapes line up exactly during upsampling."
            )

        self.n_mels = n_mels
        self.segment_length = segment_length

        # Time embedding MLP: sinusoidal embedding -> 2-layer MLP.
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.input_conv = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)

        channels = [base_channels * mult for mult in channel_multipliers]

        # ---- Encoder (downsampling path) ----
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        in_ch = base_channels
        for stage_idx, out_ch in enumerate(channels):
            stage_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                stage_blocks.append(ResidualBlock(in_ch, out_ch, time_emb_dim))
                in_ch = out_ch
            self.down_blocks.append(stage_blocks)

            is_last_stage = stage_idx == num_stages - 1
            self.downsamplers.append(
                Downsample(out_ch) if not is_last_stage else nn.Identity()
            )

        # ---- Bottleneck ----
        bottleneck_ch = channels[-1]
        self.bottleneck1 = ResidualBlock(bottleneck_ch, bottleneck_ch, time_emb_dim)
        self.bottleneck2 = ResidualBlock(bottleneck_ch, bottleneck_ch, time_emb_dim)

        # ---- Decoder (upsampling path) ----
        self.up_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        for stage_idx, out_ch in enumerate(reversed_channels):
            is_first_stage = stage_idx == 0
            self.upsamplers.append(
                Upsample(reversed_channels[stage_idx - 1]) if not is_first_stage else nn.Identity()
            )

            stage_blocks = nn.ModuleList()
            # Each block consumes the concatenation of the upsampled features
            # and the corresponding encoder skip connection.
            skip_ch = out_ch
            block_in_ch = (reversed_channels[stage_idx - 1] if not is_first_stage else bottleneck_ch) + skip_ch
            for block_idx in range(num_res_blocks):
                stage_blocks.append(ResidualBlock(block_in_ch, out_ch, time_emb_dim))
                block_in_ch = out_ch + skip_ch
            self.up_blocks.append(stage_blocks)

        self.output_norm = nn.GroupNorm(
            _resolve_num_groups(DEFAULT_NUM_GROUPS, channels[0]), channels[0]
        )
        self.output_activation = nn.SiLU()
        self.output_conv = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)

    def forward(self, x, t):
        """
        Args:
            x: Noisy spectrogram tensor of shape (batch, 1, n_mels, segment_length).
            t: Diffusion timestep tensor of shape (batch,).

        Returns:
            Predicted noise tensor of shape (batch, 1, n_mels, segment_length).
        """
        time_emb = self.time_mlp(t)

        h = self.input_conv(x)

        skip_connections_per_stage = []
        for stage_blocks, downsampler in zip(self.down_blocks, self.downsamplers):
            for block in stage_blocks:
                h = block(h, time_emb)
            skip_connections_per_stage.append(h)
            h = downsampler(h)

        h = self.bottleneck1(h, time_emb)
        h = self.bottleneck2(h, time_emb)

        for stage_idx, (stage_blocks, upsampler) in enumerate(zip(self.up_blocks, self.upsamplers)):
            h = upsampler(h)
            skip = skip_connections_per_stage[-(stage_idx + 1)]
            for block in stage_blocks:
                h = block(torch.cat([h, skip], dim=1), time_emb)

        h = self.output_conv(self.output_activation(self.output_norm(h)))
        return h
