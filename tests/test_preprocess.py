"""Unit tests for preprocessing.preprocess."""

import json

import numpy as np
import pytest
import soundfile as sf

librosa = pytest.importorskip("librosa")

from preprocessing.preprocess import (
    compute_global_mel_stats,
    compute_npy_statistics,
    generate_dataset_statistics,
    load_normalization_stats,
    normalize_npy_directory,
    preprocess_directory,
    preprocess_file,
    save_normalization_stats,
    save_statistics,
)
from preprocessing.spectrogram import normalize_melspectrogram_with_stats


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


def test_preprocess_file_with_normalization(tmp_path):
    sr = 22050
    waveform = _make_sine_wave(sr=sr)
    audio_path = tmp_path / "input.wav"
    sf.write(audio_path, waveform, sr)

    output_dir = tmp_path / "processed"
    out_path = preprocess_file(
        str(audio_path),
        str(output_dir),
        target_sample_rate=sr,
        n_mels=80,
        normalize_spectrogram=True,
        normalize_method="zscore",
    )

    mel = np.load(out_path)
    assert mel.shape[0] == 80
    assert np.mean(mel) == pytest.approx(0.0, abs=1e-3)
    assert np.std(mel) == pytest.approx(1.0, abs=1e-3)


def test_normalize_npy_directory_local_scope_normalizes_each_file_independently(tmp_path):
    input_dir = tmp_path / "processed"
    input_dir.mkdir()

    mel_a = (np.random.randn(80, 40) * 5 + 2).astype(np.float32)
    mel_b = (np.random.randn(80, 20) * 10 - 3).astype(np.float32)
    np.save(input_dir / "a.npy", mel_a)
    np.save(input_dir / "b.npy", mel_b)

    output_dir = tmp_path / "processed_normalized"
    output_paths = normalize_npy_directory(
        str(input_dir), str(output_dir), method="zscore", scope="local"
    )

    assert len(output_paths) == 2
    for path in output_paths:
        mel = np.load(path)
        assert np.mean(mel) == pytest.approx(0.0, abs=1e-3)
        assert np.std(mel) == pytest.approx(1.0, abs=1e-3)

    # Original files must remain untouched (not overwritten).
    original_a = np.load(input_dir / "a.npy")
    np.testing.assert_array_equal(original_a, mel_a)


def test_normalize_npy_directory_global_scope_uses_one_fixed_stat_for_all_files(tmp_path):
    input_dir = tmp_path / "processed"
    input_dir.mkdir()

    # Two files with very different individual scales.
    mel_a = (np.random.randn(80, 40) * 1 + 2).astype(np.float32)
    mel_b = (np.random.randn(80, 20) * 10 - 3).astype(np.float32)
    np.save(input_dir / "a.npy", mel_a)
    np.save(input_dir / "b.npy", mel_b)

    output_dir = tmp_path / "processed_normalized"
    stats_path = tmp_path / "norm_stats.json"
    output_paths = normalize_npy_directory(
        str(input_dir),
        str(output_dir),
        method="zscore",
        scope="global",
        stats_path=str(stats_path),
    )

    assert len(output_paths) == 2
    # Global stats file must have been written.
    saved_stats = load_normalization_stats(str(stats_path))
    assert saved_stats["method"] == "zscore"

    # Each output file must equal applying the SAME saved stats to the
    # original (un-normalized) array -- i.e. one fixed transform for all files.
    for path, original in [(output_paths[0], mel_a), (output_paths[1], mel_b)]:
        normalized = np.load(path)
        expected = normalize_melspectrogram_with_stats(original, saved_stats)
        np.testing.assert_allclose(normalized, expected, rtol=1e-5)

    # Because the two files have different individual scales, a single
    # global transform must NOT bring both to mean=0/std=1 individually.
    mean_a = np.mean(np.load(output_paths[0]))
    mean_b = np.mean(np.load(output_paths[1]))
    assert not (abs(mean_a) < 1e-3 and abs(mean_b) < 1e-3)


def test_normalize_npy_directory_raises_when_no_npy_files(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        normalize_npy_directory(str(empty_dir), str(tmp_path / "out"))


def test_normalize_npy_directory_raises_on_invalid_scope(tmp_path):
    input_dir = tmp_path / "processed"
    input_dir.mkdir()
    np.save(input_dir / "a.npy", np.random.randn(80, 10).astype(np.float32))

    with pytest.raises(ValueError):
        normalize_npy_directory(str(input_dir), str(tmp_path / "out"), scope="bogus")


def test_compute_global_mel_stats_matches_compute_npy_statistics(tmp_path):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    np.save(processed_dir / "a.npy", np.random.randn(80, 40).astype(np.float32))
    np.save(processed_dir / "b.npy", np.random.randn(80, 20).astype(np.float32))

    global_stats = compute_global_mel_stats(str(processed_dir), method="zscore")
    reference = compute_npy_statistics(str(processed_dir))["value_stats"]

    assert global_stats["method"] == "zscore"
    assert global_stats["mean"] == pytest.approx(reference["mean"])
    assert global_stats["std"] == pytest.approx(reference["std"])


def test_save_and_load_normalization_stats_roundtrip(tmp_path):
    stats = {"method": "zscore", "mean": 1.5, "std": 2.5}
    stats_path = tmp_path / "norm_stats.json"

    saved_path = save_normalization_stats(stats, str(stats_path))
    loaded = load_normalization_stats(saved_path)

    assert loaded == stats


def test_global_normalization_commutes_with_segmentation(tmp_path):
    """The key property motivating global normalization: normalizing a full
    track and then slicing a fixed-size segment out of it must be identical
    to slicing first and normalizing the slice with the same fixed stats --
    on the actual (non-padded) data. Zero-padding is intentionally applied
    only AFTER normalization in the real pipeline (SegmentedVoiceDataset cuts
    windows from already-normalized files), so padded regions are excluded
    from this comparison: padding raw zeros and then normalizing them would
    map to a different value than the normalized-space zero used as padding
    in the real pipeline -- that asymmetry is expected, not a bug.
    """
    from models.segmentation import compute_segment_starts

    input_dir = tmp_path / "processed"
    input_dir.mkdir()

    n_frames = 600
    segment_length, hop = 512, 256
    full_mel = (np.random.randn(80, n_frames) * 4 + 1).astype(np.float32)
    np.save(input_dir / "track.npy", full_mel)

    output_dir = tmp_path / "processed_normalized"
    stats_path = tmp_path / "norm_stats.json"
    normalize_npy_directory(
        str(input_dir), str(output_dir), scope="global", stats_path=str(stats_path)
    )

    normalized_full = np.load(output_dir / "track.npy")
    stats = load_normalization_stats(str(stats_path))

    starts = compute_segment_starts(n_frames, segment_length, hop)
    for start in starts:
        end = min(start + segment_length, n_frames)
        valid_len = end - start

        # Path 1: slice the ALREADY-normalized full track.
        seg_from_normalized_full = normalized_full[:, start:end]

        # Path 2: slice the ORIGINAL (un-normalized) track, then normalize
        # that same valid (non-padded) slice with the same saved global stats.
        seg_from_raw_then_normalized = normalize_melspectrogram_with_stats(
            full_mel[:, start:end], stats
        )

        assert seg_from_normalized_full.shape[1] == valid_len
        np.testing.assert_allclose(
            seg_from_normalized_full, seg_from_raw_then_normalized, rtol=1e-4, atol=1e-5
        )


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
