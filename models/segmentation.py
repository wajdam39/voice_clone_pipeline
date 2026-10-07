"""Fixed-size windowing (segmentation) of mel-spectrograms for model training.

Splits each mel-spectrogram of shape (n_mels, n_frames) into fixed-width,
overlapping segments of shape (n_mels, segment_length) -- e.g. (80, 512) --
using a configurable hop/stride between consecutive windows. The final
segment of a track, which may be shorter than ``segment_length``, is
zero-padded on the right so every segment has an identical shape. That, in
turn, keeps batching in a PyTorch ``DataLoader`` trivial (no custom
``collate_fn`` needed, since every item already has the same shape).

This module is pure NumPy (no PyTorch dependency), so the core windowing
logic can be developed and unit-tested in isolation.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

DEFAULT_SEGMENT_LENGTH = 512
DEFAULT_SEGMENT_HOP = 256


def pad_to_length(mel: np.ndarray, length: int) -> np.ndarray:
    """Zero-pad a mel-spectrogram on the right (time axis) to ``length`` frames.

    Args:
        mel: 2-D array of shape (n_mels, n_frames).
        length: Target number of frames. If ``mel`` already has at least
            ``length`` frames, it is returned unchanged (never truncated).

    Returns:
        A float32 array of shape (n_mels, max(n_frames, length)).
    """
    n_frames = mel.shape[1]
    if n_frames >= length:
        return mel.astype(np.float32)

    pad_width = length - n_frames
    padded = np.pad(mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)
    return padded.astype(np.float32)


def compute_segment_starts(
    n_frames: int,
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
    hop: int = DEFAULT_SEGMENT_HOP,
) -> List[int]:
    """Compute the starting frame index of each fixed-size, overlapping segment.

    Args:
        n_frames: Total number of time frames in the source spectrogram.
        segment_length: Width (in frames) of each segment.
        hop: Stride between consecutive segment start positions. Use
            ``hop < segment_length`` for overlapping windows (e.g.
            ``segment_length=512, hop=256`` gives 50% overlap).

    Returns:
        A list of starting frame indices, one per segment. Always contains
        at least one entry (``[0]``), even when ``n_frames`` is 0, so very
        short/silent tracks still produce one zero-padded segment.

    Raises:
        ValueError: If ``segment_length`` or ``hop`` is not a positive integer.
    """
    if segment_length <= 0:
        raise ValueError("segment_length must be a positive integer.")
    if hop <= 0:
        raise ValueError("hop must be a positive integer.")

    if n_frames <= 0:
        return [0]

    return list(range(0, n_frames, hop))


def segment_spectrogram(
    mel: np.ndarray,
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
    hop: int = DEFAULT_SEGMENT_HOP,
) -> List[np.ndarray]:
    """Split a mel-spectrogram into fixed-size, overlapping, zero-padded segments.

    Args:
        mel: 2-D mel-spectrogram array of shape (n_mels, n_frames).
        segment_length: Width (in frames) of each output segment (default: 512).
        hop: Stride between consecutive segment start positions (default: 256).
            With ``hop < segment_length``, consecutive segments overlap,
            increasing the number of training samples while preserving
            wider musical/temporal context across the dataset.

    Returns:
        A list of float32 arrays, each of shape (n_mels, segment_length).
        The final segment is zero-padded on the right whenever the
        remaining audio is shorter than ``segment_length``.
    """
    n_frames = mel.shape[1]
    starts = compute_segment_starts(n_frames, segment_length, hop)

    segments = []
    for start in starts:
        end = start + segment_length
        chunk = mel[:, start:end]
        if chunk.shape[1] < segment_length:
            chunk = pad_to_length(chunk, segment_length)
        segments.append(chunk.astype(np.float32))

    return segments


def build_segment_index(
    npy_paths: List[str],
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
    hop: int = DEFAULT_SEGMENT_HOP,
) -> List[Tuple[str, int]]:
    """Build a lightweight (file_path, start_frame) index across many ``.npy`` files.

    Only each file's shape is read (via memory-mapping, without loading the
    full array into RAM) to compute segment boundaries, making this cheap to
    call even over large datasets (e.g. 200 preprocessed tracks).

    Args:
        npy_paths: Paths to ``.npy`` mel-spectrogram files, each of shape
            (n_mels, n_frames).
        segment_length: Width (in frames) of each segment.
        hop: Stride between consecutive segment start positions.

    Returns:
        A flat list of ``(file_path, start_frame)`` tuples, one entry per
        segment across all files combined, in file order.
    """
    index: List[Tuple[str, int]] = []
    for path in npy_paths:
        mel = np.load(path, mmap_mode="r")
        n_frames = mel.shape[1]
        for start in compute_segment_starts(n_frames, segment_length, hop):
            index.append((str(path), start))
    return index


def generate_segments(
    npy_paths: List[str],
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
    hop: int = DEFAULT_SEGMENT_HOP,
):
    """Lazily yield fixed-size, zero-padded segments across many ``.npy`` files.

    A memory-efficient NumPy generator, usable without PyTorch: loads one
    source file at a time and yields all of its segments before moving on,
    never holding more than one full spectrogram in memory at once.

    Args:
        npy_paths: Paths to ``.npy`` mel-spectrogram files.
        segment_length: Width (in frames) of each segment.
        hop: Stride between consecutive segment start positions.

    Yields:
        float32 numpy arrays of shape (n_mels, segment_length).
    """
    for path in npy_paths:
        mel = np.load(path)
        for segment in segment_spectrogram(mel, segment_length, hop):
            yield segment
