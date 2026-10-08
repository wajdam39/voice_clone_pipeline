"""Generation (sampling) + vocoder synthesis: trained model -> new .wav file.

Loads a trained ``DiffusionUNet`` checkpoint, generates a new (normalized)
mel-spectrogram via ``GaussianDiffusion.sample``, reverses the normalization
using the saved dataset statistics, and synthesizes the final audio waveform
with a pretrained HiFi-GAN vocoder.

Usage:
    python -m scripts.generate --checkpoint checkpoints/model_final.pt --output output.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import yaml

from models.train import load_trained_model
from preprocessing.preprocess import DEFAULT_NORM_STATS_PATH, load_normalization_stats
from preprocessing.spectrogram import denormalize_melspectrogram_with_stats
from vocoder.hifigan import DEFAULT_SAVEDIR, DEFAULT_SOURCE, load_pretrained_vocoder, mel_to_audio

DEFAULT_CONFIG_PATH = "configs/config.yaml"
DEFAULT_CHECKPOINT_PATH = "checkpoints/model_final.pt"
DEFAULT_OUTPUT_PATH = "output/generated.wav"


def generate_audio(
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    config_path: str = DEFAULT_CONFIG_PATH,
    norm_stats_path: str = DEFAULT_NORM_STATS_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    num_samples: int = 1,
    vocoder_source: str = DEFAULT_SOURCE,
    vocoder_savedir: str = DEFAULT_SAVEDIR,
    sample_rate: int = 22050,
) -> list:
    """Generate new spectrogram(s) with a trained model and synthesize audio.

    Pipeline: load trained ``DiffusionUNet`` -> ``GaussianDiffusion.sample``
    (pure noise -> denoised, normalized spectrogram) -> denormalize using the
    saved dataset statistics -> pretrained HiFi-GAN vocoder -> ``.wav`` file(s).

    Args:
        checkpoint_path: Path to a trained model ``.pt`` checkpoint (as saved
            by ``models.train.train``).
        config_path: Path to the YAML config used for training (must match
            the checkpoint's architecture hyperparameters).
        norm_stats_path: Path to the global normalization statistics JSON
            saved during preprocessing (see
            ``preprocessing.preprocess.save_normalization_stats``), used to
            denormalize the generated spectrogram before vocoding.
        output_path: Path to save the generated ``.wav`` file into. If
            ``num_samples`` > 1, a numeric suffix is appended to each file
            (e.g. ``output_0.wav``, ``output_1.wav``).
        num_samples: How many independent spectrograms/audio clips to generate.
        vocoder_source: Hugging Face Hub repo id of the pretrained vocoder.
        vocoder_savedir: Local cache directory for the downloaded vocoder.
        sample_rate: Sample rate to write the output ``.wav`` file(s) at.

    Returns:
        A list of paths to the generated ``.wav`` file(s).
    """
    import soundfile as sf

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Wczytywanie wytrenowanego modelu z: {checkpoint_path}")
    model, diffusion = load_trained_model(checkpoint_path, config)

    stats = load_normalization_stats(norm_stats_path)

    print("Wczytywanie pretrenowanego wokodera HiFi-GAN...")
    vocoder = load_pretrained_vocoder(source=vocoder_source, savedir=vocoder_savedir)

    model_cfg = config.get("model", {})
    segmentation_cfg = config.get("segmentation", {})
    n_mels = model_cfg.get("n_mels") or config.get("preprocessing", {}).get("n_mels", 80)
    segment_length = segmentation_cfg.get("segment_length", 512)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    generated_paths = []
    for i in range(num_samples):
        print(f"Generowanie spektrogramu {i + 1}/{num_samples} (odszumianie od czystego szumu)...")
        generated_normalized = diffusion.sample(shape=(1, 1, n_mels, segment_length))
        mel_np = generated_normalized.squeeze(0).squeeze(0).cpu().numpy()

        denormalized_mel = denormalize_melspectrogram_with_stats(mel_np, stats)

        audio = mel_to_audio(vocoder, denormalized_mel)

        if num_samples == 1:
            sample_path = output_path_obj
        else:
            sample_path = output_path_obj.with_name(
                f"{output_path_obj.stem}_{i}{output_path_obj.suffix}"
            )

        sf.write(str(sample_path), audio, samplerate=sample_rate)
        print(f"Zapisano wygenerowane audio do: {sample_path}")
        generated_paths.append(str(sample_path))

    return generated_paths


def main(argv: Optional[list] = None) -> None:
    """CLI entry point: generate new audio from a trained model."""
    parser = argparse.ArgumentParser(
        description="Generate new audio by sampling a trained diffusion model and "
        "synthesizing it with a pretrained HiFi-GAN vocoder."
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Path to a trained model checkpoint (default: {DEFAULT_CHECKPOINT_PATH}).",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the YAML config used for training (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--norm-stats-path",
        default=DEFAULT_NORM_STATS_PATH,
        help=f"Path to the saved global normalization statistics JSON "
        f"(default: {DEFAULT_NORM_STATS_PATH}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to save the generated .wav file (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of independent audio clips to generate (default: 1).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Sample rate for the output .wav file(s) (default: 22050).",
    )
    args = parser.parse_args(argv)

    generate_audio(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        norm_stats_path=args.norm_stats_path,
        output_path=args.output,
        num_samples=args.num_samples,
        sample_rate=args.sample_rate,
    )


if __name__ == "__main__":
    main()
