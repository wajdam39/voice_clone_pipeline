"""Unit tests for preprocessing.preprocess."""

import json

import numpy as np
import pytest
import soundfile as sf

librosa = pytest.importorskip("librosa")

from preprocessing.preprocess import (
    compute_npy_statistics,
    generate_dataset_statistics,
    preprocess_directory,
    preprocess_file,
    save_statistics,
)


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


def test_compute_npy_statistics_basic_shape(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    mel_a = np.random.randn(80, 50).astype(np.float32)
    mel_b = np.random.randn(80, 30).astype(np.float32)
    np.save(processed_dir / "a.npy", mel_a)
    np.save(processed_dir / "b.npy", mel_b)

    stats = compute_npy_statistics(str(processed_dir))

    assert stats["num_files"] == 2
    assert stats["n_mels"] == 80
    assert stats["total_frames"] == 80  # 50 + 30
    assert stats["frames_per_file"]["min"] == 30
    assert stats["frames_per_file"]["max"] == 50
    assert "mean" in stats["value_stats"]
    assert "std" in stats["value_stats"]
    assert len(stats["files"]) == 2
    assert {f["file"] for f in stats["files"]} == {"a.npy", "b.npy"}


def test_compute_npy_statistics_raises_when_no_npy_files(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        compute_npy_statistics(str(empty_dir))


def test_save_statistics_writes_valid_json(tmp_path):
    stats = {"num_files": 3, "value_stats": {"mean": 1.0}}
    output_path = tmp_path / "stats" / "dataset_stats.json"

    result_path = save_statistics(stats, str(output_path))

    assert result_path == str(output_path)
    with open(output_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == stats


def test_generate_dataset_statistics_end_to_end(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    np.save(processed_dir / "x.npy", np.random.randn(80, 40).astype(np.float32))

    output_path = tmp_path / "dataset_stats.json"
    result_path = generate_dataset_statistics(str(processed_dir), str(output_path))

    assert result_path == str(output_path)
    with open(output_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["num_files"] == 1
    assert loaded["n_mels"] == 80
