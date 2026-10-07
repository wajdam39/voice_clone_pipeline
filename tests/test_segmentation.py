"""Unit tests for models.segmentation."""

import numpy as np
import pytest

from models.segmentation import (
    build_segment_index,
    compute_segment_starts,
    generate_segments,
    pad_to_length,
    segment_spectrogram,
)


def test_pad_to_length_pads_short_spectrogram():
    mel = np.ones((80, 100), dtype=np.float32)
    padded = pad_to_length(mel, 512)

    assert padded.shape == (80, 512)
    assert np.all(padded[:, :100] == 1.0)
    assert np.all(padded[:, 100:] == 0.0)


def test_pad_to_length_leaves_long_spectrogram_unchanged():
    mel = np.ones((80, 600), dtype=np.float32)
    result = pad_to_length(mel, 512)

    assert result.shape == (80, 600)
    np.testing.assert_array_equal(result, mel)


def test_compute_segment_starts_basic_overlap():
    starts = compute_segment_starts(n_frames=1200, segment_length=512, hop=256)

    assert starts == [0, 256, 512, 768, 1024]


def test_compute_segment_starts_handles_zero_frames():
    starts = compute_segment_starts(n_frames=0, segment_length=512, hop=256)
    assert starts == [0]


def test_compute_segment_starts_raises_on_invalid_params():
    with pytest.raises(ValueError):
        compute_segment_starts(100, segment_length=0, hop=256)
    with pytest.raises(ValueError):
        compute_segment_starts(100, segment_length=512, hop=0)


def test_segment_spectrogram_shapes_are_all_identical():
    mel = np.random.randn(80, 1200).astype(np.float32)
    segments = segment_spectrogram(mel, segment_length=512, hop=256)

    assert len(segments) == 5
    for segment in segments:
        assert segment.shape == (80, 512)
        assert segment.dtype == np.float32


def test_segment_spectrogram_last_segment_is_zero_padded():
    # 600 frames: starts at 0, 256, 512 -> last segment covers [512:1024],
    # but only frames [512:600] (88 frames) exist -> rest must be zero-padded.
    mel = np.random.randn(80, 600).astype(np.float32)
    segments = segment_spectrogram(mel, segment_length=512, hop=256)

    last_segment = segments[-1]
    assert last_segment.shape == (80, 512)
    original_remaining = mel[:, 512:600]
    np.testing.assert_array_equal(last_segment[:, : original_remaining.shape[1]], original_remaining)
    assert np.all(last_segment[:, original_remaining.shape[1]:] == 0.0)


def test_segment_spectrogram_short_track_produces_one_padded_segment():
    mel = np.random.randn(80, 100).astype(np.float32)
    segments = segment_spectrogram(mel, segment_length=512, hop=256)

    assert len(segments) == 1
    assert segments[0].shape == (80, 512)
    np.testing.assert_array_equal(segments[0][:, :100], mel)
    assert np.all(segments[0][:, 100:] == 0.0)


def test_build_segment_index_across_multiple_files(tmp_path):
    path_a = tmp_path / "a.npy"
    path_b = tmp_path / "b.npy"
    np.save(path_a, np.random.randn(80, 1200).astype(np.float32))  # starts: 0,256,512,768,1024 -> 5 segments
    np.save(path_b, np.random.randn(80, 300).astype(np.float32))  # starts: 0,256 -> 2 segments

    index = build_segment_index([str(path_a), str(path_b)], segment_length=512, hop=256)

    assert len(index) == 7
    a_entries = [entry for entry in index if entry[0] == str(path_a)]
    b_entries = [entry for entry in index if entry[0] == str(path_b)]
    assert len(a_entries) == 5
    assert len(b_entries) == 2
    assert [start for _, start in a_entries] == [0, 256, 512, 768, 1024]
    assert [start for _, start in b_entries] == [0, 256]


def test_generate_segments_yields_correct_total_count(tmp_path):
    path_a = tmp_path / "a.npy"
    path_b = tmp_path / "b.npy"
    np.save(path_a, np.random.randn(80, 1200).astype(np.float32))  # 5 segments
    np.save(path_b, np.random.randn(80, 300).astype(np.float32))  # 2 segments

    segments = list(generate_segments([str(path_a), str(path_b)], segment_length=512, hop=256))

    assert len(segments) == 7
    for segment in segments:
        assert segment.shape == (80, 512)
        assert segment.dtype == np.float32
