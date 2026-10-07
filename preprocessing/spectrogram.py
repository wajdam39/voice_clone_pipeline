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
