"""Unit tests for models.model and models.dataset."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from models.dataset import VoiceDataset
from models.model import VoiceCloneModel


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
