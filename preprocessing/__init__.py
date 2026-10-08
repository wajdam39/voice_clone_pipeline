"""Preprocessing package: audio loading, normalization and feature extraction."""

from .audio_utils import (
    DEFAULT_RAW_AUDIO_DIR,
    list_audio_files,
    load_audio,
    normalize_audio,
    resample_audio,
)
from .preprocess import (
    DEFAULT_NORM_STATS_PATH,
    DEFAULT_NORMALIZED_DIR,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_STATS_PATH,
    compute_global_mel_stats,
    compute_npy_statistics,
    generate_dataset_statistics,
    load_normalization_stats,
    normalize_npy_directory,
    preprocess_directory,
    preprocess_file,
    save_normalization_stats,
    save_statistics,
)
from .spectrogram import (
    audio_to_melspectrogram,
    denormalize_melspectrogram_with_stats,
    normalize_melspectrogram,
    normalize_melspectrogram_with_stats,
)

__all__ = [
    "DEFAULT_RAW_AUDIO_DIR",
    "DEFAULT_PROCESSED_DIR",
    "DEFAULT_NORMALIZED_DIR",
    "DEFAULT_STATS_PATH",
    "DEFAULT_NORM_STATS_PATH",
    "list_audio_files",
    "load_audio",
    "normalize_audio",
    "resample_audio",
    "audio_to_melspectrogram",
    "normalize_melspectrogram",
    "normalize_melspectrogram_with_stats",
    "denormalize_melspectrogram_with_stats",
    "preprocess_file",
    "preprocess_directory",
    "normalize_npy_directory",
    "compute_global_mel_stats",
    "save_normalization_stats",
    "load_normalization_stats",
    "compute_npy_statistics",
    "save_statistics",
    "generate_dataset_statistics",
]
