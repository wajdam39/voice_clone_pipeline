"""Vocoder: reconstructs an audio waveform from a mel-spectrogram.

The default implementation uses Griffin-Lim, a classic signal-processing
algorithm, as a dependency-light placeholder. It is intended to be swapped
out for a learned neural vocoder (e.g. HiFi-GAN) once one is trained and
available, without changing the public function signature.
"""

from __future__ import annotations

import numpy as np


def melspectrogram_to_audio(
    log_mel_spectrogram: np.ndarray,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_iter: int = 32,
) -> np.ndarray:
    """Reconstruct a waveform from a log-scaled mel-spectrogram.

    Args:
        log_mel_spectrogram: Array of shape (n_mels, n_frames), as produced
            by ``preprocessing.spectrogram.audio_to_melspectrogram``.
        sample_rate: Target sample rate of the output waveform.
        n_fft: FFT window size (must match the one used for analysis).
        hop_length: Hop length (must match the one used for analysis).
        win_length: Window length (must match the one used for analysis).
        n_iter: Number of Griffin-Lim iterations.

    Returns:
        A 1-D numpy float32 array containing the reconstructed waveform.
    """
    import librosa

    mel = np.exp(log_mel_spectrogram)

    waveform = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_iter=n_iter,
    )
    return waveform.astype(np.float32)
