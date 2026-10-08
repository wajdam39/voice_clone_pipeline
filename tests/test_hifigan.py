"""Unit tests for vocoder.hifigan.

Includes fast, fully-mocked tests for the shape-handling/denormalization
logic, plus a real integration test that downloads the actual pretrained
HiFi-GAN checkpoint from Hugging Face Hub (skipped automatically if
``speechbrain`` is not installed or the checkpoint cannot be fetched, e.g.
no network access).
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vocoder.hifigan import _to_batched_mel_tensor, mel_to_audio, normalized_mel_to_audio


def _make_fake_vocoder(num_samples: int = 2048):
    """A fake vocoder mimicking SpeechBrain's HIFIGAN.decode_batch signature."""
    fake = MagicMock()

    def fake_decode_batch(mel_batch):
        batch_size = mel_batch.shape[0]
        return torch.zeros(batch_size, 1, num_samples)

    fake.decode_batch.side_effect = fake_decode_batch
    return fake


def test_to_batched_mel_tensor_accepts_2d():
    mel = np.random.randn(80, 512).astype(np.float32)
    batched = _to_batched_mel_tensor(mel)
    assert batched.shape == (1, 80, 512)


def test_to_batched_mel_tensor_accepts_3d():
    mel = np.random.randn(4, 80, 512).astype(np.float32)
    batched = _to_batched_mel_tensor(mel)
    assert batched.shape == (4, 80, 512)


def test_to_batched_mel_tensor_accepts_4d_with_channel_dim():
    mel = np.random.randn(4, 1, 80, 512).astype(np.float32)
    batched = _to_batched_mel_tensor(mel)
    assert batched.shape == (4, 80, 512)


def test_to_batched_mel_tensor_rejects_bad_channel_dim():
    mel = np.random.randn(4, 2, 80, 512).astype(np.float32)
    with pytest.raises(ValueError):
        _to_batched_mel_tensor(mel)


def test_to_batched_mel_tensor_rejects_wrong_ndim():
    mel = np.random.randn(80).astype(np.float32)
    with pytest.raises(ValueError):
        _to_batched_mel_tensor(mel)


def test_mel_to_audio_single_spectrogram_returns_1d(tmp_path):
    vocoder = _make_fake_vocoder(num_samples=2048)
    mel = np.random.randn(80, 512).astype(np.float32)

    audio = mel_to_audio(vocoder, mel)

    assert isinstance(audio, np.ndarray)
    assert audio.shape == (2048,)
    assert audio.dtype == np.float32


def test_mel_to_audio_batch_returns_2d(tmp_path):
    vocoder = _make_fake_vocoder(num_samples=2048)
    mel = np.random.randn(4, 80, 512).astype(np.float32)

    audio = mel_to_audio(vocoder, mel)

    assert audio.shape == (4, 2048)


def test_mel_to_audio_accepts_channel_dim_batch():
    vocoder = _make_fake_vocoder(num_samples=1000)
    mel = np.random.randn(2, 1, 80, 512).astype(np.float32)

    audio = mel_to_audio(vocoder, mel)

    assert audio.shape == (2, 1000)


def test_mel_to_audio_can_return_torch_tensor():
    vocoder = _make_fake_vocoder(num_samples=500)
    mel = np.random.randn(80, 512).astype(np.float32)

    audio = mel_to_audio(vocoder, mel, return_numpy=False)

    assert isinstance(audio, torch.Tensor)
    assert audio.shape == (500,)


def test_normalized_mel_to_audio_denormalizes_before_synthesis():
    vocoder = _make_fake_vocoder(num_samples=512)
    # A normalized spectrogram with mean=0, std=1.
    normalized = np.random.randn(80, 512).astype(np.float32)
    stats = {"method": "zscore", "mean": -9.0, "std": 3.0}

    captured = {}

    def capturing_decode_batch(mel_batch):
        captured["mel_batch"] = mel_batch.clone()
        return torch.zeros(mel_batch.shape[0], 1, 512)

    vocoder.decode_batch.side_effect = capturing_decode_batch

    normalized_mel_to_audio(vocoder, normalized, stats)

    # The mel batch actually passed to the vocoder must be on the
    # denormalized (natural-log) scale, not the original normalized values.
    passed_mel = captured["mel_batch"].numpy()[0]
    expected = normalized * stats["std"] + stats["mean"]
    np.testing.assert_allclose(passed_mel, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.integration
def test_real_pretrained_hifigan_end_to_end(tmp_path):
    """Loads the real pretrained HiFi-GAN and synthesizes real audio from a
    real mel-spectrogram. Requires network access (first run downloads the
    checkpoint from Hugging Face Hub) and the optional `speechbrain`
    dependency; skipped automatically otherwise.
    """
    pytest.importorskip("speechbrain")
    import soundfile as sf

    from preprocessing.spectrogram import audio_to_melspectrogram
    from vocoder.hifigan import load_pretrained_vocoder

    try:
        vocoder = load_pretrained_vocoder(savedir=str(tmp_path / "hifigan_cache"))
    except Exception as exc:  # pragma: no cover - network-dependent
        pytest.skip(f"Could not load pretrained HiFi-GAN (likely no network): {exc}")

    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    waveform = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    mel = audio_to_melspectrogram(
        waveform, sample_rate=sr, n_fft=1024, hop_length=256, win_length=1024, n_mels=80
    )
    # Trim/pad to exactly 512 frames to match the project's segment shape.
    mel = mel[:, :512] if mel.shape[1] >= 512 else np.pad(mel, ((0, 0), (0, 512 - mel.shape[1])))

    audio = mel_to_audio(vocoder, mel)

    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert audio.shape[0] > 0
    assert np.all(np.isfinite(audio))
    # Real synthesized speech-vocoder output should not be silence.
    assert np.max(np.abs(audio)) > 1e-4
