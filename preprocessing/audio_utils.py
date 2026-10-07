"""Basic audio I/O and signal preprocessing utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def load_audio(path: str, sample_rate: int | None = None) -> Tuple[np.ndarray, int]:
    """Load an audio file as a mono float32 waveform.

    Args:
        path: Path to an audio file (wav, mp3, flac, ...).
        sample_rate: If provided, resample to this rate while loading.
            If None, the file's native sample rate is kept.

    Returns:
        A tuple of (waveform, sample_rate), where waveform is a 1-D
        numpy array of float32 samples in the range [-1, 1].
    """
    import librosa

    waveform, sr = librosa.load(path, sr=sample_rate, mono=True)
    return waveform.astype(np.float32), sr


def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a waveform from ``orig_sr`` to ``target_sr``.

    Args:
        waveform: 1-D numpy array of audio samples.
        orig_sr: Original sample rate of ``waveform``.
        target_sr: Desired sample rate.

    Returns:
        The resampled waveform as a float32 numpy array.
    """
    if orig_sr == target_sr:
        return waveform.astype(np.float32)

    import librosa

    resampled = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    return resampled.astype(np.float32)


def normalize_audio(waveform: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalize a waveform to ``target_peak``.

    Args:
        waveform: 1-D numpy array of audio samples.
        target_peak: Desired maximum absolute amplitude after normalization.

    Returns:
        The normalized waveform. If the input is silent (all zeros), it is
        returned unchanged to avoid division by zero.
    """
    peak = np.max(np.abs(waveform)) if waveform.size else 0.0
    if peak == 0:
        return waveform.astype(np.float32)

    return (waveform * (target_peak / peak)).astype(np.float32)
