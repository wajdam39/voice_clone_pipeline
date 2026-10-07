"""PyTorch Dataset for paired text/mel-spectrogram (or audio) training examples."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - import guard for environments without torch
    torch = None
    Dataset = object


class VoiceDataset(Dataset):
    """A simple dataset that loads mel-spectrograms from `.npy` files.

    Args:
        file_paths: List of paths to `.npy` files, each containing a
            precomputed mel-spectrogram of shape (n_mels, n_frames).
        transform: Optional callable applied to each loaded spectrogram.
    """

    def __init__(
        self,
        file_paths: List[str],
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError(
                "PyTorch is required to use VoiceDataset. Install it with "
                "`pip install torch`."
            )
        self.file_paths = [Path(p) for p in file_paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int):
        path = self.file_paths[index]
        mel = np.load(path)

        if self.transform is not None:
            mel = self.transform(mel)

        return torch.from_numpy(mel.astype(np.float32))
