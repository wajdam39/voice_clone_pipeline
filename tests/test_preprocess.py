"""Unit tests for preprocessing.preprocess."""

import numpy as np
import pytest
import soundfile as sf

librosa = pytest.importorskip("librosa")

from preprocessing.preprocess import preprocess_directory, preprocess_file


def _make_sine_wave(duration_s: float = 0.5, sr: int = 22050, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_preprocess_file_creates_npy(tmp_path):
    sr = 22050
    waveform = _make_sine_wave(sr=sr)
    audio_path = tmp_path / "input.wav"
    sf.write(audio_path, waveform, sr)

    output_dir = tmp_path / "processed"
    out_path = preprocess_file(
        str(audio_path), str(output_dir), target_sample_rate=sr, n_mels=80
    )

    assert out_path.endswith("input.npy")

    mel = np.load(out_path)
    assert mel.ndim == 2
    assert mel.shape[0] == 80
    assert mel.dtype == np.float32


def test_preprocess_directory_processes_all_files(tmp_path):
    sr = 22050
    input_dir = tmp_path / "raw"
    input_dir.mkdir()

    waveform = _make_sine_wave(sr=sr)
    sf.write(input_dir / "a.wav", waveform, sr)
    sf.write(input_dir / "b.wav", waveform, sr)

    output_dir = tmp_path / "processed"
    output_paths = preprocess_directory(
        str(input_dir), str(output_dir), target_sample_rate=sr
    )

    assert len(output_paths) == 2
    for path in output_paths:
        assert path.endswith(".npy")
        mel = np.load(path)
        assert mel.shape[0] == 80  # default n_mels


def test_preprocess_directory_raises_when_no_audio_files(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        preprocess_directory(str(empty_dir), str(tmp_path / "processed"))
