"""Unit tests for vocoder.vocoder."""

import numpy as np
import pytest

librosa = pytest.importorskip("librosa")

from preprocessing.spectrogram import audio_to_melspectrogram
from vocoder.vocoder import melspectrogram_to_audio


def test_melspectrogram_to_audio_returns_1d_waveform():
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    waveform = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    mel = audio_to_melspectrogram(
        waveform,
        sample_rate=sr,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
    )

    reconstructed = melspectrogram_to_audio(
        mel,
        sample_rate=sr,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_iter=4,
    )

    assert reconstructed.ndim == 1
    assert reconstructed.dtype == np.float32
    assert reconstructed.shape[0] > 0
