"""Unit tests for scripts.run_pipeline (mocks every stage: download, preprocess,
train, generate)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import yaml

librosa = pytest.importorskip("librosa")

from scripts.run_pipeline import run_pipeline


def _make_sine_wave(duration_s: float = 0.3, sr: int = 22050, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_run_pipeline_executes_all_stages_in_order(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    normalized_dir = tmp_path / "processed_normalized"
    norm_stats_path = tmp_path / "norm_stats.json"
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "output" / "generated.wav"

    config_data = {
        "preprocessing": {
            "target_sample_rate": 22050, "n_fft": 1024, "hop_length": 256,
            "win_length": 1024, "n_mels": 80, "f_min": 0, "f_max": 8000,
        },
        "segmentation": {"segment_length": 512, "hop": 256},
        "model": {"base_channels": 4, "channel_multipliers": [1, 2, 4], "time_emb_dim": 16, "num_res_blocks": 1},
        "diffusion": {"num_timesteps": 5, "beta_start": 1e-4, "beta_end": 0.02},
        "train": {"batch_size": 2, "learning_rate": 1e-3, "num_epochs": 1,
                   "checkpoint_dir": str(tmp_path / "checkpoints")},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f)

    call_order = []

    def fake_search_and_download_many(query, output_dir, count, audio_format, skip_errors=True):
        call_order.append("download")
        raw_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(count):
            path = raw_dir / f"track_{i}.wav"
            sf.write(path, _make_sine_wave(), 22050)
            paths.append(str(path))
        return paths

    def fake_train(data_dir, config):
        call_order.append("train")
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "model_final.pt").touch()
        return MagicMock()

    def fake_generate_audio(**kwargs):
        call_order.append("generate")
        out_path = kwargs["output_path"]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).touch()
        return [out_path]

    with patch(
        "scripts.run_pipeline.search_and_download_many", side_effect=fake_search_and_download_many
    ), patch(
        "scripts.run_pipeline.train", side_effect=fake_train
    ), patch(
        "scripts.run_pipeline.generate_audio", side_effect=fake_generate_audio
    ):
        result = run_pipeline(
            query="test query",
            count=2,
            raw_dir=str(raw_dir),
            processed_dir=str(processed_dir),
            normalized_dir=str(normalized_dir),
            norm_stats_path=str(norm_stats_path),
            config_path=str(config_path),
            output_path=str(output_path),
        )

    assert call_order == ["download", "train", "generate"]
    assert result == [str(output_path)]
    assert normalized_dir.exists()
    assert norm_stats_path.exists()


def test_run_pipeline_skip_download_and_preprocess(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    normalized_dir = tmp_path / "processed_normalized"
    norm_stats_path = tmp_path / "norm_stats.json"
    config_path = tmp_path / "config.yaml"
    output_path = tmp_path / "output" / "generated.wav"

    config_data = {"train": {"checkpoint_dir": str(tmp_path / "checkpoints")}}
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f)

    # Pre-populate normalized_dir/stats as if stage 2 already ran.
    normalized_dir.mkdir(parents=True)
    norm_stats_path.write_text('{"method": "zscore", "mean": 0.0, "std": 1.0}')

    call_order = []

    def fake_search_and_download_many(*args, **kwargs):
        call_order.append("download")
        return []

    def fake_train(data_dir, config):
        call_order.append("train")
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "model_final.pt").touch()

    def fake_generate_audio(**kwargs):
        call_order.append("generate")
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).touch()
        return [kwargs["output_path"]]

    with patch(
        "scripts.run_pipeline.search_and_download_many", side_effect=fake_search_and_download_many
    ), patch(
        "scripts.run_pipeline.train", side_effect=fake_train
    ), patch(
        "scripts.run_pipeline.generate_audio", side_effect=fake_generate_audio
    ):
        run_pipeline(
            query="test query",
            raw_dir=str(raw_dir),
            processed_dir=str(processed_dir),
            normalized_dir=str(normalized_dir),
            norm_stats_path=str(norm_stats_path),
            config_path=str(config_path),
            output_path=str(output_path),
            skip_download=True,
            skip_preprocess=True,
        )

    assert call_order == ["train", "generate"]


def test_run_pipeline_skip_generate_returns_none(tmp_path):
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"train": {"checkpoint_dir": str(tmp_path / "checkpoints")}}, f)

    with patch("scripts.run_pipeline.search_and_download_many", return_value=[]), patch(
        "scripts.run_pipeline.preprocess_directory", return_value=[]
    ), patch(
        "scripts.run_pipeline.normalize_npy_directory", return_value=[]
    ), patch(
        "scripts.run_pipeline.train"
    ) as mock_train, patch(
        "scripts.run_pipeline.generate_audio"
    ) as mock_generate:
        result = run_pipeline(
            query="test query",
            config_path=str(config_path),
            skip_generate=True,
        )

    assert result is None
    mock_train.assert_called_once()
    mock_generate.assert_not_called()
