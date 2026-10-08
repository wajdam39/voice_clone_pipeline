"""Unit tests for models.diffusion (Gaussian diffusion / DDPM process)."""

import pytest

torch = pytest.importorskip("torch")

from models.diffusion import GaussianDiffusion, linear_beta_schedule
from models.unet import UNet2D


def _make_small_model():
    return UNet2D(
        n_mels=16,
        segment_length=32,
        base_channels=4,
        channel_multipliers=(1, 2, 4),
        time_emb_dim=16,
        num_res_blocks=1,
    )


def test_linear_beta_schedule_shape_and_bounds():
    betas = linear_beta_schedule(num_timesteps=10, beta_start=1e-4, beta_end=0.02)

    assert betas.shape == (10,)
    assert betas[0] == pytest.approx(1e-4)
    assert betas[-1] == pytest.approx(0.02)
    assert torch.all(betas[1:] >= betas[:-1])  # monotonically non-decreasing


def test_q_sample_adds_noise_and_matches_shape():
    model = _make_small_model()
    diffusion = GaussianDiffusion(model, num_timesteps=50)

    x_start = torch.randn(4, 1, 16, 32)
    t = torch.randint(0, 50, (4,))

    x_t, noise = diffusion.q_sample(x_start, t)

    assert x_t.shape == x_start.shape
    assert noise.shape == x_start.shape
    # At non-trivial timesteps, the noisy version should differ from the input.
    assert not torch.allclose(x_t, x_start)


def test_q_sample_at_timestep_zero_is_close_to_original():
    model = _make_small_model()
    # With beta_start very small, t=0 should barely perturb x_start.
    diffusion = GaussianDiffusion(model, num_timesteps=50, beta_start=1e-6, beta_end=0.02)

    x_start = torch.randn(2, 1, 16, 32)
    t = torch.zeros(2, dtype=torch.long)

    x_t, _ = diffusion.q_sample(x_start, t)

    assert torch.allclose(x_t, x_start, atol=0.05)


def test_training_loss_is_finite_scalar_and_backprops():
    model = _make_small_model()
    diffusion = GaussianDiffusion(model, num_timesteps=50)

    x_start = torch.randn(4, 1, 16, 32)
    loss = diffusion.training_loss(x_start)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    # At least one parameter should have received a gradient.
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_p_sample_preserves_shape():
    model = _make_small_model()
    diffusion = GaussianDiffusion(model, num_timesteps=50)

    x_t = torch.randn(2, 1, 16, 32)
    x_prev = diffusion.p_sample(x_t, t=49)

    assert x_prev.shape == x_t.shape
    assert torch.all(torch.isfinite(x_prev))


def test_sample_generates_correct_shape_and_is_deterministic_free_of_nans():
    model = _make_small_model()
    model.eval()
    # Small num_timesteps keeps the full reverse loop fast for a unit test.
    diffusion = GaussianDiffusion(model, num_timesteps=5)

    generated = diffusion.sample(shape=(1, 1, 16, 32), device=torch.device("cpu"))

    assert generated.shape == (1, 1, 16, 32)
    assert torch.all(torch.isfinite(generated))
