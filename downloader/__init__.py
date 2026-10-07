"""Downloader package: utilities for acquiring source audio (e.g. from YouTube)."""

from .youtube_downloader import (
    download_audio,
    search_and_download_audio,
    search_and_download_many,
)

__all__ = ["download_audio", "search_and_download_audio", "search_and_download_many"]
