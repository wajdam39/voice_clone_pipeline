"""Basic audio I/O and signal preprocessing utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Default directory where downloaded source audio lives (see
# downloader.youtube_downloader and configs/config.yaml -> downloader.output_dir).
# Keeping this in sync lets preprocessing code refer to files by name alone,
# without needing to know/repeat the download directory everywhere.
DEFAULT_RAW_AUDIO_DIR = "data/raw"

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg")


def _resolve_audio_path(path: str, raw_dir: str = DEFAULT_RAW_AUDIO_DIR) -> str:
    """Resolve ``path`` to an existing audio file.

    If ``path`` already points to an existing file (absolute or relative to
    the current working directory), it is returned unchanged. Otherwise, it
    is looked up inside ``raw_dir`` (``data/raw`` by default) so callers can
    pass just a filename, e.g. ``"song.wav"``, instead of a full path.

    Args:
        path: A file path or bare filename.
        raw_dir: Directory to search in when ``path`` is not directly found.

    Returns:
        The resolved, existing path as a string.

    Raises:
        FileNotFoundError: If the file cannot be found either directly or
            inside ``raw_dir``.
    """
    if os.path.exists(path):
        return path

    candidate = os.path.join(raw_dir, path)
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        f"Audio file '{path}' not found directly or inside '{raw_dir}'."
    )


def list_audio_files(directory: str = DEFAULT_RAW_AUDIO_DIR) -> List[str]:
    """List all audio files in ``directory`` (default: ``data/raw``).

    Args:
        directory: Directory to scan for audio files. Defaults to the same
            directory the downloader saves audio into.

    Returns:
        A sorted list of absolute/relative file paths (as strings) for every
        file in ``directory`` whose extension matches a known audio format.
        Returns an empty list if the directory does not exist.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    return sorted(
        str(p) for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def load_audio(
    path: str,
    sample_rate: int | None = None,
    raw_dir: str = DEFAULT_RAW_AUDIO_DIR,
) -> Tuple[np.ndarray, int]:
    """Load an audio file as a mono float32 waveform.

    Args:
        path: Path to an audio file (wav, mp3, flac, ...). If ``path`` does
            not exist as given (e.g. you pass just a filename like
            ``"song.wav"``), it is also looked up inside ``raw_dir``
            (``data/raw`` by default, the directory the downloader saves to).
        sample_rate: If provided, resample to this rate while loading.
            If None, the file's native sample rate is kept.
        raw_dir: Fallback directory to search for ``path`` in when it is not
            found directly. Defaults to ``data/raw``.

    Returns:
        A tuple of (waveform, sample_rate), where waveform is a 1-D
        numpy array of float32 samples in the range [-1, 1].

    Raises:
        FileNotFoundError: If the audio file cannot be found either directly
            or inside ``raw_dir``.
    """
    import librosa

    resolved_path = _resolve_audio_path(path, raw_dir)
    waveform, sr = librosa.load(resolved_path, sr=sample_rate, mono=True)
    return waveform.astype(np.float32), sr


def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a waveform from ``orig_sr`` to ``target_sr``.

    Args:
        waveform: 1-D numpy array of audio samples.
        orig_sr: Original sample rate of ``waveform``.
        target_sr: Desired sample rate.

    Returns:
        The resampled waveform as a float32 numpy array.
    """
    if orig_sr == target_sr:
        return waveform.astype(np.float32)

    import librosa

    resampled = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    return resampled.astype(np.float32)


def normalize_audio(waveform: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalize a waveform to ``target_peak``.

    Args:
        waveform: 1-D numpy array of audio samples.
        target_peak: Desired maximum absolute amplitude after normalization.

    Returns:
        The normalized waveform. If the input is silent (all zeros), it is
        returned unchanged to avoid division by zero.
    """
    peak = np.max(np.abs(waveform)) if waveform.size else 0.0
    if peak == 0:
        return waveform.astype(np.float32)

    return (waveform * (target_peak / peak)).astype(np.float32)
