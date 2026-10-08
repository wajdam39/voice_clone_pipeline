"""Training entry point for the voice clone diffusion acoustic model.

Trains a ``DiffusionUNet`` (2D U-Net) to predict the Gaussian noise added to
mel-spectrogram segments at a randomly sampled diffusion timestep, using
``GaussianDiffusion`` to drive the forward noising process and the training
loss (see models/diffusion.py).
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Optional

import yaml

from models.dataset import SegmentedVoiceDataset
from models.diffusion import GaussianDiffusion
from models.model import DiffusionUNet


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the voice clone acoustic model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing preprocessed mel-spectrogram .npy files.",
    )
    return parser.parse_args(argv)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(data_dir: str, config: dict) -> None:
    """Run the training loop.

    Args:
        data_dir: Directory containing `.npy` mel-spectrogram files.
        config: Parsed configuration dictionary (see configs/config.yaml).
    """
    import torch
    from torch.utils.data import DataLoader

    file_paths = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    if not file_paths:
        raise FileNotFoundError(f"No .npy files found in '{data_dir}'.")

    segmentation_cfg = config.get("segmentation", {})
    segment_length = segmentation_cfg.get("segment_length", 512)
    dataset = SegmentedVoiceDataset(
        file_paths,
        segment_length=segment_length,
        hop=segmentation_cfg.get("hop", 256),
    )
    train_cfg = config.get("train", {})
    batch_size = train_cfg.get("batch_size", 16)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model_cfg = config.get("model", {})
    n_mels = model_cfg.get("n_mels") or config.get("preprocessing", {}).get("n_mels", 80)
    model = DiffusionUNet(
        n_mels=n_mels,
        segment_length=segment_length,
        base_channels=model_cfg.get("base_channels", 32),
        channel_multipliers=tuple(model_cfg.get("channel_multipliers", [1, 2, 4])),
        time_emb_dim=model_cfg.get("time_emb_dim", 128),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
    )

    diffusion_cfg = config.get("diffusion", {})
    diffusion = GaussianDiffusion(
        model,
        num_timesteps=diffusion_cfg.get("num_timesteps", 1000),
        beta_start=diffusion_cfg.get("beta_start", 1e-4),
        beta_end=diffusion_cfg.get("beta_end", 0.02),
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.get("learning_rate", 1e-4)
    )

    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = train_cfg.get("num_epochs", 100)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            # batch: (batch_size, n_mels, segment_length) -> add channel dim
            # expected by DiffusionUNet: (batch_size, 1, n_mels, segment_length).
            x_start = batch.unsqueeze(1)

            optimizer.zero_grad()
            loss = diffusion.training_loss(x_start)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(loader), 1)
        print(f"Epoch {epoch + 1}/{num_epochs} - loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), checkpoint_dir / f"model_epoch{epoch + 1}.pt")


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    train(args.data_dir, config)


if __name__ == "__main__":
    main()
