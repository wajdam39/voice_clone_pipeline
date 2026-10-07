"""Unit tests for preprocessing.spectrogram and preprocessing.audio_utils."""

import numpy as np
import pytest
import soundfile as sf

librosa = pytest.importorskip("librosa")

from preprocessing.audio_utils import list_audio_files, load_audio, normalize_audio, resample_audio
from preprocessing.spectrogram import audio_to_melspectrogram, normalize_melspectrogram


def _make_sine_wave(duration_s: float = 1.0, sr: int = 22050, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_normalize_audio_peak_is_within_target():
    waveform = _make_sine_wave() * 0.1
    normalized = normalize_audio(waveform, target_peak=0.9)

    assert np.max(np.abs(normalized)) == pytest.approx(0.9, abs=1e-4)


def test_normalize_audio_handles_silence():
    waveform = np.zeros(1000, dtype=np.float32)
    normalized = normalize_audio(waveform)

    assert np.all(normalized == 0.0)


def test_resample_audio_changes_length_proportionally():
    waveform = _make_sine_wave(duration_s=1.0, sr=22050)
    resampled = resample_audio(waveform, orig_sr=22050, target_sr=11025)

    assert resampled.shape[0] == pytest.approx(waveform.shape[0] / 2, rel=0.05)


def test_resample_audio_noop_when_same_rate():
    waveform = _make_sine_wave()
    resampled = resample_audio(waveform, orig_sr=22050, target_sr=22050)

    np.testing.assert_array_equal(resampled, waveform.astype(np.float32))


def test_load_audio_direct_path(tmp_path):
    sr = 22050
    waveform = _make_sine_wave(duration_s=0.5, sr=sr)
    file_path = tmp_path / "direct.wav"
    sf.write(file_path, waveform, sr)

    loaded, loaded_sr = load_audio(str(file_path))

    assert loaded_sr == sr
    assert loaded.dtype == np.float32
    assert loaded.shape[0] > 0


def test_load_audio_falls_back_to_raw_dir(tmp_path):
    sr = 22050
    waveform = _make_sine_wave(duration_s=0.5, sr=sr)
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    file_path = raw_dir / "song.wav"
    sf.write(file_path, waveform, sr)

    # Only the bare filename is given; it should be resolved inside raw_dir.
    loaded, loaded_sr = load_audio("song.wav", raw_dir=str(raw_dir))

    assert loaded_sr == sr
    assert loaded.shape[0] > 0


def test_load_audio_raises_when_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_audio("does_not_exist.wav", raw_dir=str(tmp_path / "data" / "raw"))


def test_list_audio_files_returns_empty_when_dir_missing(tmp_path):
    missing_dir = tmp_path / "no_such_dir"
    assert list_audio_files(str(missing_dir)) == []


def test_list_audio_files_filters_by_extension(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)

    sr = 22050
    waveform = _make_sine_wave(duration_s=0.2, sr=sr)
    sf.write(raw_dir / "a.wav", waveform, sr)
    sf.write(raw_dir / "b.wav", waveform, sr)
    (raw_dir / "notes.txt").write_text("not audio")

    files = list_audio_files(str(raw_dir))

    assert len(files) == 2
    assert all(f.endswith(".wav") for f in files)


def test_audio_to_melspectrogram_output_shape():
    waveform = _make_sine_wave(duration_s=1.0, sr=22050)
    mel = audio_to_melspectrogram(
        waveform,
        sample_rate=22050,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
    )

    assert mel.ndim == 2
    assert mel.shape[0] == 80
    assert mel.dtype == np.float32


def test_normalize_melspectrogram_zscore_has_zero_mean_unit_std():
    mel = (np.random.randn(80, 100) * 5 + 3).astype(np.float32)

    normalized = normalize_melspectrogram(mel, method="zscore")

    assert normalized.shape == mel.shape
    assert normalized.dtype == np.float32
    assert np.mean(normalized) == pytest.approx(0.0, abs=1e-4)
    assert np.std(normalized) == pytest.approx(1.0, abs=1e-4)


def test_normalize_melspectrogram_minmax_is_within_zero_one():
    mel = (np.random.randn(80, 100) * 5 + 3).astype(np.float32)

    normalized = normalize_melspectrogram(mel, method="minmax")

    assert normalized.shape == mel.shape
    assert np.min(normalized) == pytest.approx(0.0, abs=1e-4)
    assert np.max(normalized) == pytest.approx(1.0, abs=1e-4)
    assert np.all(normalized >= 0.0) and np.all(normalized <= 1.0)


def test_normalize_melspectrogram_handles_constant_input():
    mel = np.full((80, 10), 5.0, dtype=np.float32)

    zscore_result = normalize_melspectrogram(mel, method="zscore")
    minmax_result = normalize_melspectrogram(mel, method="minmax")

    assert np.all(np.isfinite(zscore_result))
    assert np.all(np.isfinite(minmax_result))


def test_normalize_melspectrogram_each_spectrogram_normalized_independently():
    # Two spectrograms with very different scales; local normalization
    # should bring both to a comparable (e.g. zero-mean/unit-std) range,
    # independent of each other's statistics.
    mel_quiet = (np.random.randn(80, 50) * 0.1 + 1.0).astype(np.float32)
    mel_loud = (np.random.randn(80, 50) * 50 + 100.0).astype(np.float32)

    norm_quiet = normalize_melspectrogram(mel_quiet, method="zscore")
    norm_loud = normalize_melspectrogram(mel_loud, method="zscore")

    assert np.mean(norm_quiet) == pytest.approx(0.0, abs=1e-4)
    assert np.mean(norm_loud) == pytest.approx(0.0, abs=1e-4)
    assert np.std(norm_quiet) == pytest.approx(1.0, abs=1e-4)
    assert np.std(norm_loud) == pytest.approx(1.0, abs=1e-4)


def test_normalize_melspectrogram_raises_on_unknown_method():
    mel = np.random.randn(80, 10).astype(np.float32)
    with pytest.raises(ValueError):
        normalize_melspectrogram(mel, method="not_a_real_method")
