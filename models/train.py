"""Training entry point for the voice clone acoustic model."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Optional

import yaml

from models.dataset import SegmentedVoiceDataset
from models.model import VoiceCloneModel


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
    dataset = SegmentedVoiceDataset(
        file_paths,
        segment_length=segmentation_cfg.get("segment_length", 512),
        hop=segmentation_cfg.get("hop", 256),
    )
    train_cfg = config.get("train", {})
    batch_size = train_cfg.get("batch_size", 16)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model_cfg = config.get("model", {})
    model = VoiceCloneModel(
        hidden_dim=model_cfg.get("hidden_dim", 256),
        num_layers=model_cfg.get("num_layers", 4),
        dropout=model_cfg.get("dropout", 0.1),
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.get("learning_rate", 1e-4)
    )
    criterion = torch.nn.MSELoss()

    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = train_cfg.get("num_epochs", 100)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            mel = batch.transpose(1, 2)  # (batch, n_frames, n_mels)
            optimizer.zero_grad()
            output = model(mel)
            loss = criterion(output, mel)
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
