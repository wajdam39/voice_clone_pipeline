"""Unit tests for models.train (trains a tiny model on synthetic .npy data)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# Pre-import torch's lazily-loaded internals (triggered by the first
# optimizer.step() call) *before* any test patches sys.modules below.
# patch.dict("sys.modules", ...) restores the dict to its pre-patch snapshot
# on exit, which would otherwise evict these lazily-imported submodules and
# make a *second* optimizer.step() call try to re-register torch's C++
# TORCH_LIBRARY bindings (which fails: they can only be registered once per
# process).
import torch._dynamo  # noqa: F401,E402

from models.train import train  # noqa: E402


def _write_config(tmp_path, n_mels=16, segment_length=32, wandb_cfg=None):
    train_cfg = {
        "batch_size": 2,
        "learning_rate": 1e-3,
        "num_epochs": 2,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        # Redirect metrics output into tmp_path too, so tests don't write
        # metrics/train_metrics.json / metrics/train_loss.csv into the real
        # repo working tree (those are DVC-tracked pipeline outputs).
        "metrics_path": str(tmp_path / "metrics" / "train_metrics.json"),
        "loss_plot_path": str(tmp_path / "metrics" / "train_loss.csv"),
    }
    if wandb_cfg is not None:
        train_cfg["wandb"] = wandb_cfg

    return {
        "preprocessing": {"n_mels": n_mels},
        "segmentation": {"segment_length": segment_length, "hop": segment_length // 2},
        "model": {
            "base_channels": 4,
            "channel_multipliers": [1, 2, 4],
            "time_emb_dim": 16,
            "num_res_blocks": 1,
        },
        "diffusion": {"num_timesteps": 5, "beta_start": 1e-4, "beta_end": 0.02},
        "train": train_cfg,
    }


def _write_npy_files(tmp_path, n_mels, n_frames_list):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for i, n_frames in enumerate(n_frames_list):
        arr = np.random.randn(n_mels, n_frames).astype(np.float32)
        np.save(data_dir / f"track_{i}.npy", arr)
    return str(data_dir)


def test_train_logs_metrics_to_wandb_when_enabled(tmp_path):
    n_mels, segment_length = 16, 32
    config = _write_config(tmp_path, n_mels, segment_length, wandb_cfg={"enabled": True})
    data_dir = _write_npy_files(tmp_path, n_mels, [64, 64])

    fake_run = MagicMock()
    fake_run.project = "voice-clone-diffusion"
    fake_wandb = MagicMock()
    fake_wandb.init.return_value = fake_run

    # `wandb` is imported lazily inside train(), so patch it at the module
    # import level (sys.modules) rather than as a models.train attribute.
    with patch.dict("sys.modules", {"wandb": fake_wandb}):
        model = train(data_dir, config)

    fake_wandb.init.assert_called_once()
    _, init_kwargs = fake_wandb.init.call_args
    assert init_kwargs["project"] == "voice-clone-diffusion"
    assert init_kwargs["config"] == config

    assert fake_run.log.called
    epoch_logs = [
        call.args[0] for call in fake_run.log.call_args_list if "train_loss" in call.args[0]
    ]
    assert len(epoch_logs) == config["train"]["num_epochs"]
    fake_run.finish.assert_called_once()

    assert model.training is False  # train() returns the model in eval mode


def test_train_skips_wandb_when_disabled(tmp_path):
    n_mels, segment_length = 16, 32
    config = _write_config(tmp_path, n_mels, segment_length, wandb_cfg={"enabled": False})
    data_dir = _write_npy_files(tmp_path, n_mels, [64, 64])

    fake_wandb = MagicMock()

    with patch.dict("sys.modules", {"wandb": fake_wandb}):
        train(data_dir, config)

    fake_wandb.init.assert_not_called()
    fake_wandb.finish.assert_not_called()


def test_train_skips_wandb_gracefully_when_package_missing(tmp_path):
    n_mels, segment_length = 16, 32
    config = _write_config(tmp_path, n_mels, segment_length, wandb_cfg={"enabled": True})
    data_dir = _write_npy_files(tmp_path, n_mels, [64, 64])

    # Simulate the optional `wandb` dependency not being installed: the
    # import inside train() should fail and fall back to wandb=None,
    # disabling logging without raising.
    with patch.dict("sys.modules", {"wandb": None}):
        model = train(data_dir, config)

    assert model is not None
