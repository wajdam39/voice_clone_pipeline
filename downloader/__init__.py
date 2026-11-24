"""Downloader package: utilities for acquiring source audio (e.g. from YouTube, Internet Archive)."""

from .youtube_downloader import (
    download_audio,
    search_and_download_audio,
    search_and_download_many,
)
from .internet_archive_downloader import (
    download_audio as ia_download_audio,
    search_and_download_audio as ia_search_and_download_audio,
    search_and_download_many as ia_search_and_download_many,
)

__all__ = [
    "download_audio",
    "search_and_download_audio",
    "search_and_download_many",
    "ia_download_audio",
    "ia_search_and_download_audio",
    "ia_search_and_download_many",
]
