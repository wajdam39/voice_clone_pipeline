"""Download audio tracks from YouTube using yt-dlp.

This module is intentionally kept small and focused so it can be developed
and tested in isolation from the rest of the pipeline. It supports two ways
of obtaining audio:

* ``download_audio``: download from a known YouTube URL.
* ``search_and_download_audio``: search YouTube by a free-text query (e.g.
  a song or artist name) and download the audio of the best match.

Running this file directly also exposes a small interactive CLI, so a user
can simply type a search phrase and get an audio file back.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional


def _import_yt_dlp():
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "yt-dlp is required for downloading audio. "
            "Install it with `pip install yt-dlp`."
        ) from exc
    return yt_dlp


def _build_ydl_opts(
    output_dir: str,
    audio_format: str,
    filename: Optional[str] = None,
) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    out_template = (
        os.path.join(output_dir, f"{filename}.%(ext)s")
        if filename
        else os.path.join(output_dir, "%(title)s.%(ext)s")
    )

    return {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": audio_format,
            }
        ],
        "quiet": True,
        "noplaylist": True,
    }


def _resolve_final_path(ydl: Any, info: dict, audio_format: str) -> str:
    base_path = ydl.prepare_filename(info)
    final_path = str(Path(base_path).with_suffix(f".{audio_format}"))

    if not os.path.exists(final_path):
        raise RuntimeError(
            f"Expected downloaded audio file at '{final_path}' but it was not found."
        )

    return final_path


def download_audio(
    url: str,
    output_dir: str,
    audio_format: str = "wav",
    filename: Optional[str] = None,
) -> str:
    """Download the audio track of a YouTube video and save it to disk.

    Args:
        url: The YouTube video URL.
        output_dir: Directory where the audio file will be saved. Created if
            it does not already exist.
        audio_format: Target audio codec/container (e.g. "wav", "mp3").
        filename: Optional base filename (without extension). If not
            provided, yt-dlp's video title/id template is used.

    Returns:
        The absolute path to the downloaded audio file.

    Raises:
        ImportError: If the optional ``yt-dlp`` dependency is not installed.
        RuntimeError: If the download finished but no output file was found.
    """
    yt_dlp = _import_yt_dlp()
    ydl_opts = _build_ydl_opts(output_dir, audio_format, filename)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return _resolve_final_path(ydl, info, audio_format)


def search_and_download_audio(
    query: str,
    output_dir: str,
    audio_format: str = "wav",
    filename: Optional[str] = None,
    max_results: int = 1,
) -> str:
    """Search YouTube for ``query`` and download the audio of the top match.

    This lets a user provide a free-text phrase (e.g. a song title and/or
    artist name) instead of a direct video URL.

    Args:
        query: Free-text search phrase, e.g. "Artist - Song Title".
        output_dir: Directory where the audio file will be saved. Created if
            it does not already exist.
        audio_format: Target audio codec/container (e.g. "wav", "mp3").
        filename: Optional base filename (without extension) for the
            downloaded result. If not provided, the video's title is used.
        max_results: How many search results yt-dlp should fetch before
            picking the first one. Kept at 1 by default to only download a
            single, best-ranked match.

    Returns:
        The absolute path to the downloaded audio file.

    Raises:
        ImportError: If the optional ``yt-dlp`` dependency is not installed.
        ValueError: If ``query`` is empty or blank.
        RuntimeError: If no search results were found, or the download
            finished but no output file was found.
    """
    if not query or not query.strip():
        raise ValueError("Search query must not be empty.")

    yt_dlp = _import_yt_dlp()
    ydl_opts = _build_ydl_opts(output_dir, audio_format, filename)
    search_term = f"ytsearch{max_results}:{query.strip()}"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_term, download=True)
        entries = list(info.get("entries") or [])

        if not entries:
            raise RuntimeError(f"No YouTube results found for query: '{query}'")

        return _resolve_final_path(ydl, entries[0], audio_format)


def main(argv: Optional[list] = None) -> None:
    """Small interactive CLI: type a search phrase, get an audio file back."""
    parser = argparse.ArgumentParser(
        description="Search YouTube by keyword and download the audio of the top match."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search phrase (e.g. song or artist name). If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory to save the downloaded audio file (default: data/raw).",
    )
    parser.add_argument(
        "--format",
        dest="audio_format",
        default="wav",
        help="Target audio format, e.g. wav, mp3 (default: wav).",
    )
    args = parser.parse_args(argv)

    query = args.query or input("Wpisz słowo/frazę do wyszukania na YouTube: ").strip()
    if not query:
        raise SystemExit("Nie podano frazy wyszukiwania.")

    path = search_and_download_audio(query, args.output_dir, args.audio_format)
    print(f"Pobrano: {path}")


if __name__ == "__main__":
    main()
