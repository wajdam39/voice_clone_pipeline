"""PyTorch Dataset for paired text/mel-spectrogram (or audio) training examples."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from models.segmentation import (
    DEFAULT_SEGMENT_HOP,
    DEFAULT_SEGMENT_LENGTH,
    build_segment_index,
    pad_to_length,
)

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


@lru_cache(maxsize=32)
def _load_npy_cached(path: str) -> np.ndarray:
    """Load a ``.npy`` file with a small LRU cache.

    Overlapping segments frequently reuse the same source file for several
    consecutive windows; caching the last few loaded files avoids re-reading
    the same spectrogram from disk for every single segment.
    """
    return np.load(path)


class SegmentedVoiceDataset(Dataset):
    """Dataset yielding fixed-size, overlapping mel-spectrogram segments.

    Builds a flat index of ``(file_path, start_frame)`` pairs across all
    given ``.npy`` mel-spectrogram files (see
    ``models.segmentation.build_segment_index``), then lazily extracts and
    zero-pads a fixed-width window (default: 512 frames, hop 256) per
    ``__getitem__`` call. Every returned item has an identical shape, so it
    can be batched by a plain PyTorch ``DataLoader`` without a custom
    ``collate_fn``.

    Args:
        file_paths: List of paths to ``.npy`` mel-spectrogram files, each of
            shape (n_mels, n_frames). Typically the (locally normalized)
            output of ``preprocessing.preprocess``.
        segment_length: Width, in frames, of each fixed-size segment
            (default: 512).
        hop: Stride between consecutive segment start positions (default:
            256, i.e. 50% overlap with the default ``segment_length``).
        transform: Optional callable applied to each extracted segment
            (e.g. additional normalization or augmentation) before it is
            converted to a tensor.
    """

    def __init__(
        self,
        file_paths: List[str],
        segment_length: int = DEFAULT_SEGMENT_LENGTH,
        hop: int = DEFAULT_SEGMENT_HOP,
        transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError(
                "PyTorch is required to use SegmentedVoiceDataset. Install it "
                "with `pip install torch`."
            )
        if not file_paths:
            raise ValueError("file_paths must not be empty.")

        self.file_paths = [str(p) for p in file_paths]
        self.segment_length = segment_length
        self.hop = hop
        self.transform = transform
        self.index: List[Tuple[str, int]] = build_segment_index(
            self.file_paths, segment_length, hop
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int):
        path, start = self.index[index]
        mel = _load_npy_cached(path)

        end = start + self.segment_length
        chunk = np.asarray(mel[:, start:end])
        if chunk.shape[1] < self.segment_length:
            chunk = pad_to_length(chunk, self.segment_length)

        if self.transform is not None:
            chunk = self.transform(chunk)

        return torch.from_numpy(chunk.astype(np.float32))
