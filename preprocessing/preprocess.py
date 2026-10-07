"""CLI entry point for batch audio preprocessing.

Scans a directory of raw audio files (e.g. downloaded via
``downloader.youtube_downloader``), converts each one into a log-mel
spectrogram, and saves the result as a ``.npy`` file ready to be consumed by
``models.dataset.VoiceDataset`` / ``models.train``.

Run it as a script, e.g.::

    python -m preprocessing.preprocess
    python -m preprocessing.preprocess --input-dir data/raw --output-dir data/processed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import yaml

from preprocessing.audio_utils import (
    DEFAULT_RAW_AUDIO_DIR,
    list_audio_files,
    load_audio,
    normalize_audio,
    resample_audio,
)
from preprocessing.spectrogram import audio_to_melspectrogram

DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_CONFIG_PATH = "configs/config.yaml"
DEFAULT_STATS_PATH = "data/dataset_stats.json"


def safe_print(message: str) -> None:
    """Print ``message``, falling back to a safe encoding on narrow consoles.

    Some Windows terminals use a legacy codepage (e.g. cp1252) that cannot
    encode every Unicode character that may appear in a downloaded video
    title (emoji, full-width punctuation, etc.). Rather than crashing the
    whole CLI over a cosmetic print, fall back to replacing unprintable
    characters.
    """
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(message.encode(encoding, errors="replace").decode(encoding))


def load_preprocessing_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load the ``preprocessing`` section of the YAML config, if present."""
    path = Path(config_path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f) or {}

    return full_config.get("preprocessing", {})


def preprocess_file(
    audio_path: str,
    output_dir: str,
    target_sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: Optional[float] = 8000.0,
) -> str:
    """Preprocess a single audio file into a saved mel-spectrogram ``.npy`` file.

    Args:
        audio_path: Path to the source audio file (or a bare filename
            resolvable inside ``data/raw``, see ``preprocessing.audio_utils``).
        output_dir: Directory to save the resulting ``.npy`` file into.
        target_sample_rate: Sample rate to resample audio to before analysis.
        n_fft, hop_length, win_length, n_mels, f_min, f_max: Mel-spectrogram
            parameters, forwarded to ``audio_to_melspectrogram``.

    Returns:
        The path to the saved ``.npy`` file.
    """
    waveform, sr = load_audio(audio_path)
    waveform = normalize_audio(waveform)
    waveform = resample_audio(waveform, orig_sr=sr, target_sr=target_sample_rate)

    mel = audio_to_melspectrogram(
        waveform,
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stem = Path(audio_path).stem
    out_path = Path(output_dir) / f"{stem}.npy"
    np.save(out_path, mel)
    return str(out_path)


def preprocess_directory(
    input_dir: str = DEFAULT_RAW_AUDIO_DIR,
    output_dir: str = DEFAULT_PROCESSED_DIR,
    **mel_kwargs,
) -> List[str]:
    """Preprocess every audio file found in ``input_dir``.

    Args:
        input_dir: Directory containing raw audio files (default: ``data/raw``).
        output_dir: Directory to save resulting ``.npy`` mel-spectrograms into
            (default: ``data/processed``).
        **mel_kwargs: Extra keyword arguments forwarded to ``preprocess_file``
            (e.g. ``target_sample_rate``, ``n_fft``, ``n_mels``, ...).

    Returns:
        A list of paths to the saved ``.npy`` files, one per input audio file.

    Raises:
        FileNotFoundError: If no audio files are found in ``input_dir``.
    """
    audio_files = list_audio_files(input_dir)
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in '{input_dir}'.")

    return [
        preprocess_file(audio_path, output_dir, **mel_kwargs)
        for audio_path in audio_files
    ]


def compute_npy_statistics(processed_dir: str = DEFAULT_PROCESSED_DIR) -> Dict:
    """Compute dataset-wide statistics over mel-spectrogram ``.npy`` files.

    Scans every ``.npy`` file in ``processed_dir`` (as produced by
    ``preprocess_file``/``preprocess_directory``) and computes both per-file
    and aggregate statistics describing the dataset: number of files, frame
    (time-step) counts, and the distribution of mel-spectrogram values.

    Args:
        processed_dir: Directory containing ``.npy`` mel-spectrogram files
            (default: ``data/processed``).

    Returns:
        A dictionary with the following structure::

            {
                "processed_dir": str,
                "num_files": int,
                "n_mels": int | None,
                "total_frames": int,
                "frames_per_file": {"mean": float, "min": int, "max": int, "std": float},
                "value_stats": {"mean": float, "std": float, "min": float, "max": float},
                "files": [
                    {"file": str, "shape": [n_mels, n_frames], "mean": float,
                     "std": float, "min": float, "max": float},
                    ...
                ],
            }

    Raises:
        FileNotFoundError: If ``processed_dir`` contains no ``.npy`` files.
    """
    dir_path = Path(processed_dir)
    npy_paths = sorted(dir_path.glob("*.npy")) if dir_path.exists() else []

    if not npy_paths:
        raise FileNotFoundError(f"No .npy files found in '{processed_dir}'.")

    per_file_stats = []
    frame_counts = []
    n_mels_values = set()
    all_values_sum = 0.0
    all_values_sq_sum = 0.0
    all_values_count = 0
    global_min = float("inf")
    global_max = float("-inf")

    for npy_path in npy_paths:
        mel = np.load(npy_path)
        n_mels_values.add(mel.shape[0])
        n_frames = mel.shape[1] if mel.ndim > 1 else 0
        frame_counts.append(n_frames)

        file_mean = float(np.mean(mel))
        file_std = float(np.std(mel))
        file_min = float(np.min(mel))
        file_max = float(np.max(mel))

        per_file_stats.append({
            "file": npy_path.name,
            "shape": list(mel.shape),
            "mean": file_mean,
            "std": file_std,
            "min": file_min,
            "max": file_max,
        })

        all_values_sum += float(np.sum(mel))
        all_values_sq_sum += float(np.sum(mel.astype(np.float64) ** 2))
        all_values_count += mel.size
        global_min = min(global_min, file_min)
        global_max = max(global_max, file_max)

    overall_mean = all_values_sum / all_values_count if all_values_count else 0.0
    overall_var = (
        (all_values_sq_sum / all_values_count) - (overall_mean ** 2)
        if all_values_count
        else 0.0
    )
    overall_std = float(np.sqrt(max(overall_var, 0.0)))

    return {
        "processed_dir": str(processed_dir),
        "num_files": len(npy_paths),
        "n_mels": sorted(n_mels_values)[0] if len(n_mels_values) == 1 else sorted(n_mels_values),
        "total_frames": int(sum(frame_counts)),
        "frames_per_file": {
            "mean": float(np.mean(frame_counts)) if frame_counts else 0.0,
            "min": int(min(frame_counts)) if frame_counts else 0,
            "max": int(max(frame_counts)) if frame_counts else 0,
            "std": float(np.std(frame_counts)) if frame_counts else 0.0,
        },
        "value_stats": {
            "mean": overall_mean,
            "std": overall_std,
            "min": global_min,
            "max": global_max,
        },
        "files": per_file_stats,
    }


def save_statistics(stats: Dict, output_path: str = DEFAULT_STATS_PATH) -> str:
    """Save a statistics dictionary as a JSON file.

    Args:
        stats: Statistics dictionary, typically from ``compute_npy_statistics``.
        output_path: Path to the JSON file to write (parent directories are
            created automatically). Default: ``data/dataset_stats.json``.

    Returns:
        The path the statistics were written to (as a string).
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return str(out_path)


def generate_dataset_statistics(
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    output_path: str = DEFAULT_STATS_PATH,
) -> str:
    """Compute statistics over ``.npy`` files in ``processed_dir`` and save them.

    Convenience wrapper combining ``compute_npy_statistics`` and
    ``save_statistics`` into a single call.

    Args:
        processed_dir: Directory containing ``.npy`` mel-spectrogram files
            (default: ``data/processed``).
        output_path: Path to the JSON file to save statistics into
            (default: ``data/dataset_stats.json``).

    Returns:
        The path the statistics JSON file was written to.
    """
    stats = compute_npy_statistics(processed_dir)
    return save_statistics(stats, output_path)


def main(argv: Optional[list] = None) -> None:
    """CLI entry point: preprocess all audio in a directory into mel-spectrograms."""
    parser = argparse.ArgumentParser(
        description="Preprocess raw audio files into mel-spectrogram .npy files."
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_RAW_AUDIO_DIR,
        help=f"Directory with raw audio files (default: {DEFAULT_RAW_AUDIO_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_PROCESSED_DIR,
        help=f"Directory to save processed .npy files into (default: {DEFAULT_PROCESSED_DIR}).",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML config providing default mel-spectrogram settings "
        f"(default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--stats-output",
        default=DEFAULT_STATS_PATH,
        help=f"Path to save dataset statistics JSON into (default: {DEFAULT_STATS_PATH}).",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Skip generating the dataset statistics JSON file after preprocessing.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Skip preprocessing and only (re)generate statistics for existing "
        "--output-dir .npy files.",
    )
    args = parser.parse_args(argv)

    if not args.stats_only:
        config = load_preprocessing_config(args.config)
        mel_kwargs = {
            "target_sample_rate": config.get("target_sample_rate", 22050),
            "n_fft": config.get("n_fft", 1024),
            "hop_length": config.get("hop_length", 256),
            "win_length": config.get("win_length", 1024),
            "n_mels": config.get("n_mels", 80),
            "f_min": config.get("f_min", 0.0),
            "f_max": config.get("f_max", 8000.0),
        }

        output_paths = preprocess_directory(args.input_dir, args.output_dir, **mel_kwargs)

        safe_print(f"Przetworzono {len(output_paths)} plik(ów) audio z '{args.input_dir}':")
        for path in output_paths:
            safe_print(f"  -> {path}")

    if not args.no_stats:
        stats_path = generate_dataset_statistics(args.output_dir, args.stats_output)
        safe_print(f"Zapisano statystyki zbioru danych do: {stats_path}")


if __name__ == "__main__":
    main()
