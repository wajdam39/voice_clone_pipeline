"""Unit tests for models.model and models.dataset."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.dataset import SegmentedVoiceDataset, VoiceDataset
from models.model import DiffusionUNet, VoiceCloneModel


def test_voice_clone_model_forward_shape():
    n_mels = 80
    model = VoiceCloneModel(n_mels=n_mels, hidden_dim=32, num_layers=2, dropout=0.0)

    batch_size, n_frames = 2, 10
    dummy_input = torch.randn(batch_size, n_frames, n_mels)

    output = model(dummy_input)

    assert output.shape == (batch_size, n_frames, n_mels)


def test_voice_dataset_loads_npy_files(tmp_path):
    n_mels, n_frames = 80, 20
    file_paths = []
    for i in range(3):
        arr = np.random.randn(n_mels, n_frames).astype(np.float32)
        path = tmp_path / f"sample_{i}.npy"
        np.save(path, arr)
        file_paths.append(str(path))

    dataset = VoiceDataset(file_paths)

    assert len(dataset) == 3
    sample = dataset[0]
    assert sample.shape == (n_mels, n_frames)
    assert sample.dtype == torch.float32


def test_segmented_voice_dataset_fixed_shape_segments(tmp_path):
    n_mels = 80
    file_paths = []
    for n_frames in (1200, 300, 600):
        arr = np.random.randn(n_mels, n_frames).astype(np.float32)
        path = tmp_path / f"track_{n_frames}.npy"
        np.save(path, arr)
        file_paths.append(str(path))

    dataset = SegmentedVoiceDataset(file_paths, segment_length=512, hop=256)

    # starts = range(0, n_frames, 256):
    # 1200 frames -> starts [0,256,512,768,1024] -> 5 segments
    # 300 frames  -> starts [0,256]              -> 2 segments
    # 600 frames  -> starts [0,256,512]          -> 3 segments
    assert len(dataset) == 5 + 2 + 3

    for i in range(len(dataset)):
        segment = dataset[i]
        assert segment.shape == (n_mels, 512)
        assert segment.dtype == torch.float32


def test_segmented_voice_dataset_is_batchable_without_custom_collate(tmp_path):
    n_mels = 80
    file_paths = []
    for n_frames in (1200, 300, 600):
        arr = np.random.randn(n_mels, n_frames).astype(np.float32)
        path = tmp_path / f"track_{n_frames}.npy"
        np.save(path, arr)
        file_paths.append(str(path))

    from torch.utils.data import DataLoader

    dataset = SegmentedVoiceDataset(file_paths, segment_length=512, hop=256)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch = next(iter(loader))
    assert batch.shape[1:] == (n_mels, 512)
    assert batch.shape[0] <= 4


def test_segmented_voice_dataset_raises_on_empty_file_list():
    with pytest.raises(ValueError):
        SegmentedVoiceDataset([], segment_length=512, hop=256)


def test_diffusion_unet_forward_shape():
    model = DiffusionUNet(
        n_mels=16,
        segment_length=32,
        base_channels=4,
        channel_multipliers=(1, 2, 4),
        time_emb_dim=16,
        num_res_blocks=1,
    )

    batch_size = 2
    x = torch.randn(batch_size, 1, 16, 32)
    t = torch.randint(0, 1000, (batch_size,))

    output = model(x, t)

    assert output.shape == x.shape
