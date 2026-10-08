"""Pretrained neural vocoder (HiFi-GAN) for mel-spectrogram-to-audio synthesis.

Wraps a pretrained HiFi-GAN checkpoint (via SpeechBrain, loaded from the
Hugging Face Hub) so it can be used as a drop-in, high-quality replacement
for the Griffin-Lim-based ``vocoder.vocoder.melspectrogram_to_audio``. No
training is required: the checkpoint is downloaded once and cached locally.

Why HiFi-GAN / why this specific checkpoint:
``speechbrain/tts-hifigan-ljspeech`` was trained on mel-spectrograms computed
with the exact same recipe this project uses in
``preprocessing.spectrogram.audio_to_melspectrogram``: sample_rate=22050,
n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmin=0, fmax=8000,
natural-log scaling with a 1e-5 floor. Matching the analysis recipe used at
training time is required for a neural vocoder to produce clean audio --
feeding it mel-spectrograms computed with different STFT/mel parameters (or
a different log base / dB scale) will cause pitch, timbre, or loudness
artifacts even if the tensor shapes line up.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from preprocessing.spectrogram import denormalize_melspectrogram_with_stats

try:
    import torch
except ImportError:  # pragma: no cover - import guard
    torch = None

DEFAULT_SOURCE = "speechbrain/tts-hifigan-ljspeech"
DEFAULT_SAVEDIR = "pretrained_models/tts-hifigan-ljspeech"


def load_pretrained_vocoder(
    source: str = DEFAULT_SOURCE,
    savedir: str = DEFAULT_SAVEDIR,
    device: Optional[str] = None,
):
    """Download (if needed) and load a pretrained HiFi-GAN vocoder.

    Args:
        source: Hugging Face Hub repo id (or local path) of a SpeechBrain
            HiFi-GAN checkpoint. Defaults to a HiFi-GAN trained on LJSpeech
            with a mel-spectrogram recipe matching this project's
            ``preprocessing.spectrogram.audio_to_melspectrogram``.
        savedir: Local directory to cache the downloaded checkpoint into.
        device: Device to load the model onto, e.g. "cpu" or "cuda". Defaults
            to "cuda" if available, else "cpu".

    Returns:
        A ``speechbrain.inference.vocoders.HIFIGAN`` instance ready for
        inference (see ``mel_to_audio``).

    Raises:
        ImportError: If the optional ``speechbrain`` dependency is not installed.
    """
    try:
        from speechbrain.inference.vocoders import HIFIGAN
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "speechbrain is required to load the pretrained HiFi-GAN vocoder. "
            "Install it with `pip install speechbrain`."
        ) from exc

    if device is None:
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    return HIFIGAN.from_hparams(
        source=source,
        savedir=savedir,
        run_opts={"device": device},
    )


def _to_batched_mel_tensor(mel: Union[np.ndarray, "torch.Tensor"]) -> "torch.Tensor":
    """Normalize a mel-spectrogram input to shape (batch, n_mels, n_frames).

    Accepts, in order:
        - (n_mels, n_frames)            -> a single spectrogram
        - (batch, n_mels, n_frames)     -> a batch of spectrograms
        - (batch, 1, n_mels, n_frames)  -> a batch with an explicit channel
          dimension (e.g. as produced for ``models.model.DiffusionUNet``)

    Args:
        mel: A numpy array or torch tensor in one of the shapes above.

    Returns:
        A float32 torch tensor of shape (batch, n_mels, n_frames), the shape
        expected by SpeechBrain's ``HIFIGAN.decode_batch``.

    Raises:
        ValueError: If ``mel`` does not have 2, 3, or 4 dimensions, or if a
            4-D input does not have a singleton channel dimension.
    """
    if isinstance(mel, np.ndarray):
        mel = torch.from_numpy(mel.astype(np.float32))
    else:
        mel = mel.float()

    if mel.dim() == 2:
        mel = mel.unsqueeze(0)  # (n_mels, n_frames) -> (1, n_mels, n_frames)
    elif mel.dim() == 4:
        if mel.shape[1] != 1:
            raise ValueError(
                f"Expected a singleton channel dimension for 4-D input, got shape {tuple(mel.shape)}."
            )
        mel = mel.squeeze(1)  # (B, 1, n_mels, T) -> (B, n_mels, T)
    elif mel.dim() != 3:
        raise ValueError(
            f"Expected mel of shape (n_mels, T), (B, n_mels, T) or (B, 1, n_mels, T); "
            f"got shape {tuple(mel.shape)}."
        )

    return mel


def mel_to_audio(
    vocoder,
    mel: Union[np.ndarray, "torch.Tensor"],
    return_numpy: bool = True,
):
    """Run inference: convert a mel-spectrogram (batch) into audio waveform(s).

    Args:
        vocoder: A loaded vocoder, as returned by ``load_pretrained_vocoder``.
        mel: The spectrogram to synthesize, in one of these shapes:
            - ``(80, 512)``: a single spectrogram segment.
            - ``(B, 80, T)``: a batch of spectrograms.
            - ``(B, 1, 80, T)``: a batch with an explicit channel dimension
              (matching ``models.model.DiffusionUNet`` I/O).
            Values must be on the SAME scale the vocoder was trained on
            (natural-log mel-spectrogram, NOT normalized/standardized --
            see ``preprocessing.spectrogram.denormalize_melspectrogram_with_stats``
            to undo normalization first if needed).
        return_numpy: If True (default), return a numpy array. If False,
            return a torch tensor instead.

    Returns:
        The generated waveform(s):
            - If the input was a single spectrogram ``(80, 512)``: a 1-D
              array/tensor of shape ``(num_samples,)``.
            - If the input was batched: a 2-D array/tensor of shape
              ``(B, num_samples)``.
        Type is ``numpy.ndarray`` if ``return_numpy=True``, else ``torch.Tensor``.
    """
    if torch is None:  # pragma: no cover
        raise ImportError(
            "PyTorch is required to run vocoder inference. Install it with "
            "`pip install torch`."
        )

    was_single = isinstance(mel, np.ndarray) and mel.ndim == 2
    was_single = was_single or (hasattr(mel, "dim") and mel.dim() == 2)

    mel_batch = _to_batched_mel_tensor(mel)

    with torch.no_grad():
        waveforms = vocoder.decode_batch(mel_batch)

    # SpeechBrain returns (B, 1, num_samples); drop the channel dim.
    waveforms = waveforms.squeeze(1)

    if was_single:
        waveforms = waveforms.squeeze(0)

    if return_numpy:
        return waveforms.cpu().numpy().astype(np.float32)
    return waveforms


def normalized_mel_to_audio(
    vocoder,
    normalized_mel: Union[np.ndarray, "torch.Tensor"],
    normalization_stats: dict,
    return_numpy: bool = True,
):
    """Denormalize a spectrogram, then run it through the vocoder in one call.

    Convenience wrapper for the common case: a spectrogram was normalized
    before training/generation (see ``preprocessing.preprocess``'s global
    normalization) and now needs to be converted back to the vocoder's
    expected natural-log mel scale before synthesis.

    Args:
        vocoder: A loaded vocoder, as returned by ``load_pretrained_vocoder``.
        normalized_mel: Spectrogram still on the NORMALIZED scale (e.g. the
            direct output of ``GaussianDiffusion.sample``), in shape
            ``(80, 512)``, ``(B, 80, T)``, or ``(B, 1, 80, T)``.
        normalization_stats: The exact stats dict used to normalize this
            data, e.g. loaded via
            ``preprocessing.preprocess.load_normalization_stats``.
        return_numpy: If True (default), return a numpy array; else a torch tensor.

    Returns:
        The generated waveform(s), see ``mel_to_audio``.
    """
    if isinstance(normalized_mel, np.ndarray):
        denormalized = denormalize_melspectrogram_with_stats(normalized_mel, normalization_stats)
    else:
        denormalized = denormalize_melspectrogram_with_stats(
            normalized_mel.detach().cpu().numpy(), normalization_stats
        )

    return mel_to_audio(vocoder, denormalized, return_numpy=return_numpy)
