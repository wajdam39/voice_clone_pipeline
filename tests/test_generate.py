"""Unit tests for scripts.generate (mocks the trained model and vocoder)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.generate import generate_audio


def _write_config(tmp_path, n_mels=16, segment_length=32):
    import yaml

    config = {
        "preprocessing": {"n_mels": n_mels},
        "segmentation": {"segment_length": segment_length, "hop": segment_length // 2},
        "model": {
            "base_channels": 4,
            "channel_multipliers": [1, 2, 4],
            "time_emb_dim": 16,
            "num_res_blocks": 1,
        },
        "diffusion": {"num_timesteps": 5, "beta_start": 1e-4, "beta_end": 0.02},
        "train": {"checkpoint_dir": str(tmp_path / "checkpoints")},
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return str(config_path)


def test_generate_audio_runs_full_chain(tmp_path):
    n_mels, segment_length = 16, 32
    config_path = _write_config(tmp_path, n_mels, segment_length)

    stats = {"method": "zscore", "mean": -9.0, "std": 3.0}
    stats_path = tmp_path / "norm_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f)

    fake_model = MagicMock()
    fake_model.parameters.return_value = iter([torch.zeros(1)])
    fake_diffusion = MagicMock()
    fake_diffusion.sample.return_value = torch.randn(1, 1, n_mels, segment_length)

    fake_vocoder = MagicMock()

    def fake_mel_to_audio(vocoder, mel):
        return np.zeros(1000, dtype=np.float32)

    checkpoint_path = tmp_path / "checkpoints" / "model_final.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.touch()

    output_path = tmp_path / "output" / "generated.wav"

    with patch(
        "scripts.generate.load_trained_model", return_value=(fake_model, fake_diffusion)
    ), patch(
        "scripts.generate.load_pretrained_vocoder", return_value=fake_vocoder
    ), patch(
        "scripts.generate.mel_to_audio", side_effect=fake_mel_to_audio
    ):
        result_paths = generate_audio(
            checkpoint_path=str(checkpoint_path),
            config_path=config_path,
            norm_stats_path=str(stats_path),
            output_path=str(output_path),
            num_samples=1,
        )

    assert len(result_paths) == 1
    assert output_path.exists()
    fake_diffusion.sample.assert_called_once()


def test_generate_audio_multiple_samples_get_suffixed_filenames(tmp_path):
    n_mels, segment_length = 16, 32
    config_path = _write_config(tmp_path, n_mels, segment_length)

    stats = {"method": "zscore", "mean": -9.0, "std": 3.0}
    stats_path = tmp_path / "norm_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f)

    fake_model = MagicMock()
    fake_diffusion = MagicMock()
    fake_diffusion.sample.return_value = torch.randn(1, 1, n_mels, segment_length)
    fake_vocoder = MagicMock()

    checkpoint_path = tmp_path / "checkpoints" / "model_final.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.touch()

    output_path = tmp_path / "output" / "generated.wav"

    with patch(
        "scripts.generate.load_trained_model", return_value=(fake_model, fake_diffusion)
    ), patch(
        "scripts.generate.load_pretrained_vocoder", return_value=fake_vocoder
    ), patch(
        "scripts.generate.mel_to_audio", return_value=np.zeros(500, dtype=np.float32)
    ):
        result_paths = generate_audio(
            checkpoint_path=str(checkpoint_path),
            config_path=config_path,
            norm_stats_path=str(stats_path),
            output_path=str(output_path),
            num_samples=3,
        )

    assert len(result_paths) == 3
    assert result_paths[0].endswith("generated_0.wav")
    assert result_paths[1].endswith("generated_1.wav")
    assert result_paths[2].endswith("generated_2.wav")
    for path in result_paths:
        assert Path(path).exists()
