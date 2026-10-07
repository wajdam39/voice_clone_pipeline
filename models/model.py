"""Skeleton acoustic model architecture for voice cloning.

This is a minimal placeholder architecture meant to be replaced/extended
with a real acoustic model (e.g. Tacotron2, FastSpeech2). Keeping it small
and self-contained makes it easy to unit test independently of training
data and the vocoder.
"""

from __future__ import annotations

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
