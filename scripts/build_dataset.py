"""End-to-end dataset builder: search YouTube by keyword, download many
tracks, then preprocess all of them into mel-spectrograms in one command.

IMPORTANT / LEGAL NOTE:
Most music on YouTube is copyrighted. Downloading and reusing it (e.g. to
train a voice-cloning model) without permission may violate copyright law
and YouTube's Terms of Service. Only use this on content you own the rights
to, that is explicitly licensed for reuse (e.g. Creative Commons), or on
your own uploaded recordings. You are responsible for how you use this tool.

Usage:
    python -m scripts.build_dataset "słowo kluczowe" --count 200
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from downloader.youtube_downloader import search_and_download_many
from preprocessing.preprocess import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_PROCESSED_DIR,
    load_preprocessing_config,
    preprocess_directory,
    safe_print,
)
from preprocessing.audio_utils import DEFAULT_RAW_AUDIO_DIR


def build_dataset(
    query: str,
    count: int = 200,
    raw_dir: str = DEFAULT_RAW_AUDIO_DIR,
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    audio_format: str = "wav",
    config_path: str = DEFAULT_CONFIG_PATH,
    skip_errors: bool = True,
) -> List[str]:
    """Download ``count`` tracks for ``query`` and preprocess all of them.

    Args:
        query: Search phrase used to find tracks on YouTube.
        count: Number of tracks to attempt downloading.
        raw_dir: Directory to download raw audio into (default: data/raw).
        processed_dir: Directory to save resulting .npy spectrograms into
            (default: data/processed).
        audio_format: Target audio codec/container for downloads.
        config_path: Path to the YAML config providing mel-spectrogram
            parameters (see configs/config.yaml -> preprocessing).
        skip_errors: Whether to skip individual download failures instead of
            aborting the whole batch (see
            ``downloader.youtube_downloader.search_and_download_many``).

    Returns:
        A list of paths to the saved ``.npy`` mel-spectrogram files.
    """
    safe_print(f"Wyszukiwanie i pobieranie do {count} utwor\u00f3w dla zapytania: '{query}'...")
    downloaded_paths = search_and_download_many(
        query,
        raw_dir,
        count=count,
        audio_format=audio_format,
        skip_errors=skip_errors,
    )
    safe_print(f"Pobrano {len(downloaded_paths)}/{count} plik(ów) do '{raw_dir}'.")

    safe_print(f"Preprocessing wszystkich plik\u00f3w audio w '{raw_dir}'...")
    mel_config = load_preprocessing_config(config_path)
    mel_kwargs = {
        "target_sample_rate": mel_config.get("target_sample_rate", 22050),
        "n_fft": mel_config.get("n_fft", 1024),
        "hop_length": mel_config.get("hop_length", 256),
        "win_length": mel_config.get("win_length", 1024),
        "n_mels": mel_config.get("n_mels", 80),
        "f_min": mel_config.get("f_min", 0.0),
        "f_max": mel_config.get("f_max", 8000.0),
    }
    processed_paths = preprocess_directory(raw_dir, processed_dir, **mel_kwargs)
    safe_print(f"Zapisano {len(processed_paths)} plik(ów) .npy do '{processed_dir}'.")

    return processed_paths


def main(argv: Optional[list] = None) -> None:
    """CLI entry point: download N tracks for a keyword and preprocess all of them."""
    parser = argparse.ArgumentParser(
        description=(
            "Search YouTube by keyword, download up to N tracks, and preprocess "
            "all of them into mel-spectrogram .npy files in one command. "
            "Only use on content you have the rights to use."
        )
    )
    parser.add_argument("query", help="Search phrase, e.g. artist name or genre/topic.")
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Number of tracks to attempt downloading (default: 200).",
    )
    parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_AUDIO_DIR,
        help=f"Directory to download raw audio into (default: {DEFAULT_RAW_AUDIO_DIR}).",
    )
    parser.add_argument(
        "--processed-dir",
        default=DEFAULT_PROCESSED_DIR,
        help=f"Directory to save processed .npy files into (default: {DEFAULT_PROCESSED_DIR}).",
    )
    parser.add_argument(
        "--format",
        dest="audio_format",
        default="wav",
        help="Target audio format for downloads, e.g. wav, mp3 (default: wav).",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML config with mel-spectrogram settings (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort the whole batch on the first download error instead of skipping it.",
    )
    args = parser.parse_args(argv)

    build_dataset(
        query=args.query,
        count=args.count,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        audio_format=args.audio_format,
        config_path=args.config,
        skip_errors=not args.fail_fast,
    )


if __name__ == "__main__":
    main()
