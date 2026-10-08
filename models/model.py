"""Acoustic model architectures for voice cloning.

Includes:

* ``VoiceCloneModel``: a minimal LSTM encoder-decoder placeholder, kept for
  backward compatibility / quick experimentation.
* ``DiffusionUNet``: the main acoustic model -- a thin, pre-configured
  wrapper around ``models.unet.UNet2D`` for diffusion-based spectrogram
  generation, matching ``SegmentedVoiceDataset``'s default output shape
  ``[batch, 80, 512]`` (consumed here as ``[batch, 1, 80, 512]``).
"""

from __future__ import annotations

from models.unet import (
    DEFAULT_BASE_CHANNELS,
    DEFAULT_CHANNEL_MULTIPLIERS,
    DEFAULT_NUM_RES_BLOCKS,
    DEFAULT_TIME_EMB_DIM,
    UNet2D,
)

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - import guard
    torch = None
    nn = object


class VoiceCloneModel(nn.Module if nn is not object else object):
    """A minimal encoder-decoder placeholder model.

    Args:
        n_mels: Number of mel-spectrogram channels (input/output feature dim).
        hidden_dim: Size of the hidden representation.
        num_layers: Number of stacked recurrent layers.
        dropout: Dropout probability applied between layers.
    """

    def __init__(
        self,
        n_mels: int = 80,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError(
                "PyTorch is required to use VoiceCloneModel. Install it with "
                "`pip install torch`."
            )
        super().__init__()
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim

        self.encoder = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.decoder_proj = nn.Linear(hidden_dim * 2, n_mels)

    def forward(self, mel_input):
        """Run a forward pass.

        Args:
            mel_input: Tensor of shape (batch, n_frames, n_mels).

        Returns:
            Tensor of shape (batch, n_frames, n_mels), a reconstructed/
            predicted mel-spectrogram.
        """
        encoded, _ = self.encoder(mel_input)
        return self.decoder_proj(encoded)


class DiffusionUNet(nn.Module if nn is not object else object):
    """Noise-predicting 2D U-Net for diffusion-based spectrogram generation.

    A thin, pre-configured wrapper around ``models.unet.UNet2D`` matching the
    project's default spectrogram segment shape. Intended to be driven by
    ``models.diffusion.GaussianDiffusion`` (see that module's
    ``training_loss``/``sample`` methods).

    Args:
        n_mels: Height of the input spectrogram (mel channels), default 80.
        segment_length: Width of the input spectrogram (time frames), default 512.
        base_channels: Number of channels in the first conv block.
        channel_multipliers: Per-stage channel multiplier tuple.
        time_emb_dim: Dimensionality of the diffusion-timestep embedding.
        num_res_blocks: Number of residual blocks per resolution stage.
    """

    def __init__(
        self,
        n_mels: int = 80,
        segment_length: int = 512,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        channel_multipliers=DEFAULT_CHANNEL_MULTIPLIERS,
        time_emb_dim: int = DEFAULT_TIME_EMB_DIM,
        num_res_blocks: int = DEFAULT_NUM_RES_BLOCKS,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError(
                "PyTorch is required to use DiffusionUNet. Install it with "
                "`pip install torch`."
            )
        super().__init__()
        self.unet = UNet2D(
            n_mels=n_mels,
            segment_length=segment_length,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            time_emb_dim=time_emb_dim,
            num_res_blocks=num_res_blocks,
        )

    def forward(self, x, t):
        """
        Args:
            x: Noisy spectrogram tensor of shape (batch, 1, n_mels, segment_length).
            t: Diffusion timestep tensor of shape (batch,).

        Returns:
            Predicted noise tensor of shape (batch, 1, n_mels, segment_length).
        """
        return self.unet(x, t)
