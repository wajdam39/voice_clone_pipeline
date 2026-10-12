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
from preprocessing.preprocess import safe_print


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


def get_model_dims(config: dict) -> tuple:
    """Resolve ``(n_mels, segment_length)`` from a parsed config dict.

    Args:
        config: Parsed configuration dictionary (see configs/config.yaml).

    Returns:
        A tuple ``(n_mels, segment_length)``.
    """
    model_cfg = config.get("model", {})
    segmentation_cfg = config.get("segmentation", {})
    n_mels = model_cfg.get("n_mels") or config.get("preprocessing", {}).get("n_mels", 80)
    segment_length = segmentation_cfg.get("segment_length", 512)
    return n_mels, segment_length


def build_diffusion_model(config: dict):
    """Build a ``(DiffusionUNet, GaussianDiffusion)`` pair from a config dict.

    Centralizes model/diffusion construction so it can be reused both for
    training (this module) and for standalone generation/inference (e.g.
    ``scripts.generate``) without duplicating hyperparameter wiring.

    Args:
        config: Parsed configuration dictionary (see configs/config.yaml),
            using its ``model``, ``diffusion``, ``segmentation`` and
            ``preprocessing`` sections.

    Returns:
        A tuple ``(model, diffusion)``: an untrained (randomly initialized)
        ``DiffusionUNet`` and a ``GaussianDiffusion`` wrapping it.
    """
    model_cfg = config.get("model", {})
    diffusion_cfg = config.get("diffusion", {})
    n_mels, segment_length = get_model_dims(config)

    model = DiffusionUNet(
        n_mels=n_mels,
        segment_length=segment_length,
        base_channels=model_cfg.get("base_channels", 32),
        channel_multipliers=tuple(model_cfg.get("channel_multipliers", [1, 2, 4])),
        time_emb_dim=model_cfg.get("time_emb_dim", 128),
        num_res_blocks=model_cfg.get("num_res_blocks", 2),
    )

    diffusion = GaussianDiffusion(
        model,
        num_timesteps=diffusion_cfg.get("num_timesteps", 1000),
        beta_start=diffusion_cfg.get("beta_start", 1e-4),
        beta_end=diffusion_cfg.get("beta_end", 0.02),
    )

    return model, diffusion


def load_trained_model(checkpoint_path: str, config: dict, device: Optional[str] = None):
    """Build a ``(DiffusionUNet, GaussianDiffusion)`` pair and load trained weights.

    Args:
        checkpoint_path: Path to a ``.pt`` file saved by ``train()``
            (``model.state_dict()``).
        config: Parsed configuration dictionary matching the one used for
            training (architecture hyperparameters must match).
        device: Device to load the model onto, e.g. "cpu" or "cuda". Defaults
            to "cuda" if available, else "cpu".

    Returns:
        A tuple ``(model, diffusion)`` with trained weights loaded, in eval mode.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, diffusion = build_diffusion_model(config)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, diffusion


def train(data_dir: str, config: dict):
    """Run the training loop.

    Args:
        data_dir: Directory containing `.npy` mel-spectrogram files.
        config: Parsed configuration dictionary (see configs/config.yaml).

    Returns:
        The trained ``DiffusionUNet`` model (in eval mode), so callers (e.g.
        ``scripts.run_pipeline``) can immediately use it for generation
        without reloading from disk.
    """
    import torch
    from torch.utils.data import DataLoader

    safe_print(f"Rozpoczynam trening modelu z danymi w: {data_dir}", flush=True)
    file_paths = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
    if not file_paths:
        raise FileNotFoundError(f"No .npy files found in '{data_dir}'.")

    safe_print(f"Znaleziono {len(file_paths)} plików .npy w '{data_dir}'.", flush=True)
    segmentation_cfg = config.get("segmentation", {})
    _, segment_length = get_model_dims(config)
    dataset = SegmentedVoiceDataset(
        file_paths,
        segment_length=segment_length,
        hop=segmentation_cfg.get("hop", 256),
    )
    train_cfg = config.get("train", {})
    batch_size = train_cfg.get("batch_size", 16)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_print(f"Używane urządzenie: {device}", flush=True)
    safe_print(
        f"Zainicjalizowano DataLoader z batch_size={batch_size} i {len(loader)} batchami.",
        flush=True,
    )
    model, diffusion = build_diffusion_model(config)
    model.to(device)
    safe_print("Zbudowano model DiffusionUNet i GaussianDiffusion z configu.", flush=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.get("learning_rate", 1e-4)
    )

    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = train_cfg.get("num_epochs", 100)
    # Log progress every `log_every` batches so large datasets (many/long
    # audio files -> many segments -> many batches per epoch) don't make
    # training look stuck for minutes/hours with no output between epochs.
    log_every = max(1, train_cfg.get("log_every", 50))
    num_batches = len(loader)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(loader, start=1):
            # batch: (batch_size, n_mels, segment_length) -> add channel dim
            # expected by DiffusionUNet: (batch_size, 1, n_mels, segment_length).
            x_start = batch.unsqueeze(1).to(device)

            optimizer.zero_grad()
            loss = diffusion.training_loss(x_start)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % log_every == 0 or batch_idx == num_batches:
                safe_print(
                    f"  Epoch {epoch + 1}/{num_epochs} - batch {batch_idx}/{num_batches} "
                    f"- loss: {loss.item():.4f}",
                    flush=True,
                )

        avg_loss = epoch_loss / max(len(loader), 1)
        safe_print(f"Epoch {epoch + 1}/{num_epochs} - loss: {avg_loss:.4f}", flush=True)

        torch.save(model.state_dict(), checkpoint_dir / f"model_epoch{epoch + 1}.pt")

    final_checkpoint_path = checkpoint_dir / "model_final.pt"
    torch.save(model.state_dict(), final_checkpoint_path)
    safe_print(f"Zapisano finalny checkpoint: {final_checkpoint_path}", flush=True)

    model.eval()
    return model


def main(argv: Optional[list] = None) -> None:
    safe_print(
        "Uruchomiono trening modelu akustycznego (DiffusionUNet) dla klonowania głosu.",
        flush=True,
    )
    args = parse_args(argv)
    safe_print(
        f"Uruchomiono trening z configiem: {args.config} i danymi w: {args.data_dir}",
        flush=True,
    )
    config = load_config(args.config)
    safe_print(f"Wczytano config treningu: {config}", flush=True)
    train(args.data_dir, config)
    safe_print("Trening zakończony.", flush=True)


if __name__ == "__main__":
    main()
