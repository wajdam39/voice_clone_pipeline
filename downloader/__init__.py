"""Downloader package: utilities for acquiring source audio (e.g. from YouTube)."""

from .youtube_downloader import download_audio

__all__ = ["download_audio"]
