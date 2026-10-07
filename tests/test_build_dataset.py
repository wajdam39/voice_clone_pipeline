"""Unit tests for scripts.build_dataset (mocks both downloader and preprocessing)."""

from unittest.mock import patch

import pytest

librosa = pytest.importorskip("librosa")

import soundfile as sf
import numpy as np

from scripts.build_dataset import build_dataset


def _make_sine_wave(duration_s: float = 0.3, sr: int = 22050, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_build_dataset_downloads_then_preprocesses(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    sr = 22050

    def fake_search_and_download_many(query, output_dir, count, audio_format, skip_errors=True):
        out = []
        raw_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            path = raw_dir / f"track_{i}.{audio_format}"
            sf.write(path, _make_sine_wave(sr=sr), sr)
            out.append(str(path))
        return out

    with patch(
        "scripts.build_dataset.search_and_download_many",
        side_effect=fake_search_and_download_many,
    ):
        processed_paths = build_dataset(
            query="test query",
            count=3,
            raw_dir=str(raw_dir),
            processed_dir=str(processed_dir),
            config_path=str(tmp_path / "nonexistent_config.yaml"),
        )

    assert len(processed_paths) == 3
    for path in processed_paths:
        mel = np.load(path)
        assert mel.ndim == 2
