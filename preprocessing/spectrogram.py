"""Mel-spectrogram feature extraction."""

from __future__ import annotations

import numpy as np


def audio_to_melspectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: float | None = 8000.0,
) -> np.ndarray:
    """Convert a waveform into a log-scaled mel-spectrogram.

    Args:
        waveform: 1-D numpy array of audio samples.
        sample_rate: Sample rate of ``waveform``.
        n_fft: FFT window size.
        hop_length: Number of samples between successive frames.
        win_length: Window length for STFT.
        n_mels: Number of mel filterbank channels.
        f_min: Minimum frequency for mel filterbank.
        f_max: Maximum frequency for mel filterbank.

    Returns:
        A 2-D numpy array of shape (n_mels, n_frames) containing the
        log-scaled mel-spectrogram (natural log, with a small epsilon floor).
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
        power=1.0,
    )
    log_mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    return log_mel.astype(np.float32)


def normalize_melspectrogram(
    mel: np.ndarray,
    method: str = "zscore",
    eps: float = 1e-8,
) -> np.ndarray:
    """Locally normalize a single mel-spectrogram using only its own statistics.

    This is *local* (per-spectrogram) normalization: each spectrogram is
    rescaled using statistics computed from itself alone, independent of any
    other file in the dataset. This differs from dataset-wide/global
    normalization, which would instead use statistics aggregated across many
    files (see ``preprocessing.preprocess.compute_npy_statistics``).

    Args:
        mel: A 2-D mel-spectrogram array of shape (n_mels, n_frames), e.g.
            as produced by ``audio_to_melspectrogram``.
        method: Normalization method to use:
            - ``"zscore"`` (default): ``(mel - mean) / (std + eps)``, using
              the spectrogram's own mean and standard deviation. Result has
              approximately zero mean and unit standard deviation.
            - ``"minmax"``: Rescales to the ``[0, 1]`` range using the
              spectrogram's own minimum and maximum values.
        eps: Small constant added to the denominator to avoid division by
            zero on constant/silent spectrograms.

    Returns:
        The normalized mel-spectrogram as a float32 array with the same
        shape as ``mel``.

    Raises:
        ValueError: If ``method`` is not one of ``"zscore"`` or ``"minmax"``.
    """
    mel = np.asarray(mel, dtype=np.float32)

    if method == "zscore":
        mean = np.mean(mel)
        std = np.std(mel)
        normalized = (mel - mean) / (std + eps)
    elif method == "minmax":
        mel_min = np.min(mel)
        mel_max = np.max(mel)
        normalized = (mel - mel_min) / (mel_max - mel_min + eps)
    else:
        raise ValueError(
            f"Unknown normalization method: '{method}'. Use 'zscore' or 'minmax'."
        )

    return normalized.astype(np.float32)
