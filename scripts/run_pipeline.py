"""End-to-end orchestrator: download -> preprocess -> train -> generate -> synthesize.

Runs the entire voice-clone pipeline with a single command:

1. **Pobieranie (download)**: search YouTube by keyword and download up to
   ``--count`` tracks (see ``downloader.youtube_downloader``).
2. **Przygotowanie (preprocess)**: convert raw audio into mel-spectrograms,
   then normalize them globally so normalization commutes with the later
   fixed-window segmentation (see ``preprocessing.preprocess``).
3. **Trenowanie (train)**: train a ``DiffusionUNet`` acoustic model on
   fixed-size, overlapping spectrogram segments (see ``models.train``).
4. **Generowanie (generate)**: sample a brand-new, normalized spectrogram
   from pure Gaussian noise using the trained model (see
   ``models.diffusion.GaussianDiffusion.sample``).
5. **Przywracanie (restore/synthesize)**: denormalize the generated
   spectrogram and synthesize the final audio waveform with a pretrained
   HiFi-GAN vocoder (see ``vocoder.hifigan``).

IMPORTANT / LEGAL NOTE: most music on YouTube is copyrighted. Only run the
download stage on content you own the rights to, that is explicitly
licensed for reuse (e.g. Creative Commons), or on your own recordings.

Usage:
    python -m scripts.run_pipeline "słowo kluczowe" --count 50 --epochs 20

Each stage can be skipped (e.g. to resume after a crash, or to re-run only
generation with an already-trained model) via ``--skip-download``,
``--skip-preprocess``, ``--skip-train``, ``--skip-generate``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import yaml

from downloader.youtube_downloader import (
    DEFAULT_COOKIES_FILE,
    DEFAULT_JS_RUNTIME,
    DEFAULT_MAX_VIDEO_DURATION_SECONDS,
    DEFAULT_REMOTE_COMPONENTS,
    search_and_download_many,
)
from models.train import load_config, train
from preprocessing.audio_utils import DEFAULT_RAW_AUDIO_DIR
from preprocessing.preprocess import (
    DEFAULT_NORM_STATS_PATH,
    DEFAULT_NORMALIZED_DIR,
    DEFAULT_PROCESSED_DIR,
    load_preprocessing_config,
    normalize_npy_directory,
    preprocess_directory,
    safe_print,
)
from scripts.generate import generate_audio

DEFAULT_CONFIG_PATH = "configs/config.yaml"
DEFAULT_CHECKPOINT_PATH = "checkpoints/model_final.pt"
DEFAULT_OUTPUT_PATH = "output/generated.wav"


def run_pipeline(
    query: str,
    count: int = 50,
    raw_dir: str = DEFAULT_RAW_AUDIO_DIR,
    processed_dir: str = DEFAULT_PROCESSED_DIR,
    normalized_dir: str = DEFAULT_NORMALIZED_DIR,
    norm_stats_path: str = DEFAULT_NORM_STATS_PATH,
    config_path: str = DEFAULT_CONFIG_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    audio_format: str = "wav",
    skip_download: bool = False,
    skip_preprocess: bool = False,
    skip_train: bool = False,
    skip_generate: bool = False,
    skip_errors: bool = True,
    num_epochs_override: Optional[int] = None,
    max_duration_seconds: Optional[int] = DEFAULT_MAX_VIDEO_DURATION_SECONDS,
    cookies_file: Optional[str] = DEFAULT_COOKIES_FILE,
    js_runtime: Optional[str] = DEFAULT_JS_RUNTIME,
    remote_components=DEFAULT_REMOTE_COMPONENTS,
) -> Optional[list]:
    """Run the full pipeline: download -> preprocess -> train -> generate -> restore.

    Each stage can be skipped independently (e.g. to resume after a crash,
    iterate on training without re-downloading, or only (re-)generate audio
    from an already-trained model).

    Args:
        query: Search phrase used to find tracks on YouTube (download stage).
        count: Number of tracks to attempt downloading.
        raw_dir: Directory for raw downloaded audio (default: data/raw).
        processed_dir: Directory for preprocessed (un-normalized) mel-spectrogram
            ``.npy`` files (default: data/processed).
        normalized_dir: Directory for globally normalized ``.npy`` files, used
            for training (default: data/processed_normalized).
        norm_stats_path: Path to save/load the global normalization statistics
            JSON (default: data/normalization_stats.json).
        config_path: Path to the YAML config with model/diffusion/training
            hyperparameters (default: configs/config.yaml).
        output_path: Path to save the final generated ``.wav`` file.
        audio_format: Target audio codec/container for downloads.
        skip_download: Skip stage 1 (download) -- use existing files in ``raw_dir``.
        skip_preprocess: Skip stage 2 (preprocess + normalize) -- use existing
            files in ``normalized_dir``/``norm_stats_path``.
        skip_train: Skip stage 3 (train) -- use the existing checkpoint at
            ``<train.checkpoint_dir>/model_final.pt`` from ``config_path``.
        skip_generate: Skip stages 4-5 (generate + restore/synthesize)
            entirely, stopping right after training.
        skip_errors: Whether to skip individual download failures instead of
            aborting the whole batch (download stage only).
        max_duration_seconds: Skip videos longer than this many seconds
            (default: 2 hours). Pass ``None`` to disable the limit.
        cookies_file: Path to a Netscape-format cookies file passed to
            yt-dlp (download stage only), used to authenticate requests as
            a logged-in browser session (default: ``youtube.com_cookies.txt``,
            silently ignored if it doesn't exist).
        js_runtime: External JS runtime yt-dlp uses to solve YouTube's JS
            challenge (download stage only, default: ``"node"``). Without a
            working JS runtime, downloads from datacenter/cloud IPs (e.g.
            EC2) are very likely to be blocked.
        remote_components: Remote component(s) yt-dlp may fetch on demand
            (download stage only, default: ``"ejs:github"``).

    Returns:
        A list with the path to the generated ``.wav`` file (or multiple, if
        configured), or None if ``skip_generate`` is True.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ---- Stage 1: Pobieranie (download) ----
    if not skip_download:
        safe_print(f"[1/5] Wyszukiwanie i pobieranie do {count} utwor\u00f3w dla: '{query}'...")
        downloaded_paths = search_and_download_many(
            query,
            raw_dir,
            count=count,
            audio_format=audio_format,
            skip_errors=skip_errors,
            max_duration_seconds=max_duration_seconds,
            cookies_file=cookies_file,
            js_runtime=js_runtime,
            remote_components=remote_components,
        )
        safe_print(f"[1/5] Pobrano {len(downloaded_paths)}/{count} plik(ów) do '{raw_dir}'.")
    else:
        safe_print("[1/5] Pomini\u0119to pobieranie (--skip-download).")

    # ---- Stage 2: Przygotowanie (preprocess + global normalization) ----
    if not skip_preprocess:
        safe_print(f"[2/5] Preprocessing audio z '{raw_dir}'...")
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
        safe_print(f"[2/5] Zapisano {len(processed_paths)} plik(ów) .npy do '{processed_dir}'.")

        safe_print(f"[2/5] Normalizacja globalna ('{processed_dir}' -> '{normalized_dir}')...")
        normalized_paths = normalize_npy_directory(
            processed_dir, normalized_dir, scope="global", stats_path=norm_stats_path
        )
        safe_print(f"[2/5] Znormalizowano {len(normalized_paths)} plik(ów). Statystyki: {norm_stats_path}")
    else:
        safe_print("[2/5] Pomini\u0119to preprocessing (--skip-preprocess).")

    # ---- Stage 3: Trenowanie (train) ----
    train_cfg = config.get("train", {})
    safe_print(f"[3/5] Trenowanie modelu z configu '{config_path}'...")
    checkpoint_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    safe_print(f"[3/5] Checkpointy modelu zostaną zapisane w: {checkpoint_dir}")
    final_checkpoint_path = checkpoint_dir / "model_final.pt"
    safe_print(f"[3/5] Finalny checkpoint modelu zostanie zapisany w: {final_checkpoint_path}")

    if not skip_train:
        safe_print(f"[3/5] Trenowanie modelu na '{normalized_dir}'...")
        train_config = load_config(config_path)
        safe_print(f"[3/5] Wczytano config treningu: {train_config}")
        if num_epochs_override is not None:
            train_config.setdefault("train", {})["num_epochs"] = num_epochs_override
        train(normalized_dir, train_config)
        safe_print(f"[3/5] Zako\u0144czono trening. Checkpoint: {final_checkpoint_path}")
    else:
        safe_print("[3/5] Pomini\u0119to trening (--skip-train).")

    if skip_generate:
        safe_print("[4-5/5] Pomini\u0119to generowanie i synteza (--skip-generate).")
        return None

    # ---- Stage 4-5: Generowanie + Przywracanie (generate + restore/synthesize) ----
    safe_print("[4/5] Generowanie nowego spektrogramu (odszumianie od czystego szumu)...")
    safe_print("[5/5] Denormalizacja i synteza audio pretrenowanym wokoderem HiFi-GAN...")
    generated_paths = generate_audio(
        checkpoint_path=str(final_checkpoint_path),
        config_path=config_path,
        norm_stats_path=norm_stats_path,
        output_path=output_path,
    )
    safe_print(f"[5/5] Gotowe! Wygenerowane audio: {generated_paths}")

    return generated_paths


def main(argv: Optional[list] = None) -> None:
    """CLI entry point: run the entire pipeline with a single command."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the full voice-clone pipeline in one command: "
            "pobieranie (download) -> przygotowanie (preprocess) -> "
            "trenowanie (train) -> generowanie (generate) -> "
            "przywracanie (restore/synthesize). "
            "Only use the download stage on content you have the rights to use."
        )
    )
    parser.add_argument("query", help="Search phrase for the download stage, e.g. artist/genre.")
    parser.add_argument(
        "--count", type=int, default=50, help="Number of tracks to download (default: 50)."
    )
    parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW_AUDIO_DIR,
        help=f"Raw audio directory (default: {DEFAULT_RAW_AUDIO_DIR}).",
    )
    parser.add_argument(
        "--processed-dir",
        default=DEFAULT_PROCESSED_DIR,
        help=f"Preprocessed (un-normalized) .npy directory (default: {DEFAULT_PROCESSED_DIR}).",
    )
    parser.add_argument(
        "--normalized-dir",
        default=DEFAULT_NORMALIZED_DIR,
        help=f"Normalized .npy directory used for training (default: {DEFAULT_NORMALIZED_DIR}).",
    )
    parser.add_argument(
        "--norm-stats-path",
        default=DEFAULT_NORM_STATS_PATH,
        help=f"Global normalization statistics JSON path (default: {DEFAULT_NORM_STATS_PATH}).",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config with model/diffusion/train hyperparameters (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to save the final generated .wav file (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--format",
        dest="audio_format",
        default="wav",
        help="Target audio format for downloads (default: wav).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train.num_epochs from the config for this run.",
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip stage 1 (download)."
    )
    parser.add_argument(
        "--skip-preprocess", action="store_true", help="Skip stage 2 (preprocess + normalize)."
    )
    parser.add_argument("--skip-train", action="store_true", help="Skip stage 3 (train).")
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip stages 4-5 (generate + restore/synthesize); stop after training.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort the whole download batch on the first error instead of skipping it.",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=DEFAULT_MAX_VIDEO_DURATION_SECONDS,
        help="Skip videos longer than this many seconds (default: 7200 = 2 hours). "
        "Use 0 or a negative value to disable the limit.",
    )
    parser.add_argument(
        "--cookies-file",
        default=DEFAULT_COOKIES_FILE,
        help="Path to a Netscape-format cookies file passed to yt-dlp for the "
        f"download stage (default: {DEFAULT_COOKIES_FILE}, silently ignored if "
        "it doesn't exist). Pass an empty string to disable cookies.",
    )
    parser.add_argument(
        "--js-runtime",
        default=DEFAULT_JS_RUNTIME,
        help="External JavaScript runtime used by yt-dlp to solve YouTube's JS "
        f"challenge (default: {DEFAULT_JS_RUNTIME!r}, e.g. 'node', 'deno'). "
        "Without a working JS runtime, downloads from datacenter/cloud IPs "
        "(e.g. EC2) are very likely to be blocked. Pass an empty string to "
        "fall back to yt-dlp's own default (deno).",
    )
    parser.add_argument(
        "--remote-components",
        default=",".join(DEFAULT_REMOTE_COMPONENTS),
        help="Comma-separated remote component(s) yt-dlp may fetch on demand "
        f"(default: {','.join(DEFAULT_REMOTE_COMPONENTS)!r}) so the JS runtime "
        "above can actually solve YouTube's JS challenge. Pass an empty string "
        "to disable.",
    )
    args = parser.parse_args(argv)

    max_duration_seconds = args.max_duration if args.max_duration and args.max_duration > 0 else None
    remote_components = (
        [c.strip() for c in args.remote_components.split(",") if c.strip()]
        if args.remote_components
        else None
    )

    run_pipeline(
        query=args.query,
        count=args.count,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        normalized_dir=args.normalized_dir,
        norm_stats_path=args.norm_stats_path,
        config_path=args.config,
        output_path=args.output,
        audio_format=args.audio_format,
        skip_download=args.skip_download,
        skip_preprocess=args.skip_preprocess,
        skip_train=args.skip_train,
        skip_generate=args.skip_generate,
        skip_errors=not args.fail_fast,
        num_epochs_override=args.epochs,
        max_duration_seconds=max_duration_seconds,
        cookies_file=args.cookies_file,
        js_runtime=args.js_runtime,
        remote_components=remote_components,
    )


if __name__ == "__main__":
    main()
