"""Download audio tracks from YouTube using yt-dlp.

This module is intentionally kept small and focused so it can be developed
and tested in isolation from the rest of the pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


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
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "yt-dlp is required for downloading audio. "
            "Install it with `pip install yt-dlp`."
        ) from exc

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    out_template = (
        os.path.join(output_dir, f"{filename}.%(ext)s")
        if filename
        else os.path.join(output_dir, "%(title)s.%(ext)s")
    )

    ydl_opts = {
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

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        base_path = ydl.prepare_filename(info)

    final_path = str(Path(base_path).with_suffix(f".{audio_format}"))

    if not os.path.exists(final_path):
        raise RuntimeError(
            f"Expected downloaded audio file at '{final_path}' but it was not found."
        )

    return final_path
