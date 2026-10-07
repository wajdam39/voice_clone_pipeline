"""Models package: dataset, model architecture and training loop."""

from .dataset import SegmentedVoiceDataset, VoiceDataset
from .model import VoiceCloneModel
from .segmentation import (
    DEFAULT_SEGMENT_HOP,
    DEFAULT_SEGMENT_LENGTH,
    build_segment_index,
    compute_segment_starts,
    generate_segments,
    pad_to_length,
    segment_spectrogram,
)

__all__ = [
    "VoiceDataset",
    "SegmentedVoiceDataset",
    "VoiceCloneModel",
    "DEFAULT_SEGMENT_LENGTH",
    "DEFAULT_SEGMENT_HOP",
    "pad_to_length",
    "compute_segment_starts",
    "segment_spectrogram",
    "build_segment_index",
    "generate_segments",
]
