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

from downloader.youtube_downloader import (
    DEFAULT_COOKIES_FILE,
    DEFAULT_JS_RUNTIME,
    DEFAULT_MAX_VIDEO_DURATION_SECONDS,
    DEFAULT_REMOTE_COMPONENTS,
    search_and_download_many,
)
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
    max_duration_seconds: Optional[int] = DEFAULT_MAX_VIDEO_DURATION_SECONDS,
    cookies_file: Optional[str] = DEFAULT_COOKIES_FILE,
    js_runtime: Optional[str] = DEFAULT_JS_RUNTIME,
    remote_components=DEFAULT_REMOTE_COMPONENTS,
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
        max_duration_seconds: Skip videos longer than this many seconds
            (default: 2 hours). Pass ``None`` to disable the limit.
        cookies_file: Path to a Netscape-format cookies file passed to
            yt-dlp, used to authenticate requests as a logged-in browser
            session (default: ``youtube.com_cookies.txt``, silently ignored
            if it doesn't exist).
        js_runtime: External JS runtime yt-dlp uses to solve YouTube's JS
            challenge (default: ``"node"``). Without a working JS runtime,
            downloads from datacenter/cloud IPs (e.g. EC2) are very likely
            to be blocked.
        remote_components: Remote component(s) yt-dlp may fetch on demand
            (default: ``"ejs:github"``).

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
        max_duration_seconds=max_duration_seconds,
        cookies_file=cookies_file,
        js_runtime=js_runtime,
        remote_components=remote_components,
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
        help="Path to a Netscape-format cookies file passed to yt-dlp "
        f"(default: {DEFAULT_COOKIES_FILE}, silently ignored if it doesn't "
        "exist). Pass an empty string to disable cookies.",
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

    build_dataset(
        query=args.query,
        count=args.count,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        audio_format=args.audio_format,
        config_path=args.config,
        skip_errors=not args.fail_fast,
        max_duration_seconds=max_duration_seconds,
        cookies_file=args.cookies_file,
        js_runtime=args.js_runtime,
        remote_components=remote_components,
    )


if __name__ == "__main__":
    main()
