"""Unit tests for models.unet (2D U-Net for diffusion)."""

import pytest

torch = pytest.importorskip("torch")

from models.unet import SinusoidalPositionEmbeddings, UNet2D


def test_sinusoidal_position_embeddings_shape():
    embed = SinusoidalPositionEmbeddings(dim=16)
    t = torch.randint(0, 1000, (4,))

    out = embed(t)

    assert out.shape == (4, 16)


def test_unet2d_forward_shape_small():
    # Small but valid config: n_mels=16, segment_length=32, both divisible by 4
    # (2 downsampling steps for the default 3 channel-multiplier stages).
    model = UNet2D(
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


def test_unet2d_forward_shape_default_dimensions():
    # Matches the project's default spectrogram segment shape [B, 1, 80, 512],
    # but with a smaller base_channels/time_emb_dim to keep the test fast.
    model = UNet2D(
        n_mels=80,
        segment_length=512,
        base_channels=4,
        channel_multipliers=(1, 2, 4),
        time_emb_dim=16,
        num_res_blocks=1,
    )

    batch_size = 1
    x = torch.randn(batch_size, 1, 80, 512)
    t = torch.randint(0, 1000, (batch_size,))

    output = model(x, t)

    assert output.shape == (1, 1, 80, 512)


def test_unet2d_rejects_indivisible_dimensions():
    with pytest.raises(ValueError):
        UNet2D(n_mels=15, segment_length=32, channel_multipliers=(1, 2, 4))


def test_unet2d_output_depends_on_timestep():
    model = UNet2D(
        n_mels=16,
        segment_length=32,
        base_channels=4,
        channel_multipliers=(1, 2, 4),
        time_emb_dim=16,
        num_res_blocks=1,
    )
    model.eval()

    x = torch.randn(1, 1, 16, 32)
    t0 = torch.tensor([0])
    t1 = torch.tensor([999])

    with torch.no_grad():
        out0 = model(x, t0)
        out1 = model(x, t1)

    assert not torch.allclose(out0, out1)
