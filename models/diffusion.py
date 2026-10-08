"""Gaussian diffusion (DDPM) process for mel-spectrogram generation.

Implements the three core pieces of a denoising diffusion probabilistic
model (Ho et al., 2020) around a noise-predicting network such as
``models.unet.UNet2D``:

1. A noise schedule (``betas``/``alphas``) precomputed for a fixed number of
   diffusion timesteps.
2. The forward process ``q_sample``: adding Gaussian noise to a clean
   spectrogram at a randomly chosen timestep ``t`` in closed form.
3. The reverse process ``p_sample`` / ``sample``: iteratively denoising
   starting from pure Gaussian noise to produce a new spectrogram.

This module only depends on the noise-predicting network's call signature
(``model(x, t) -> predicted_noise``), so it works with ``UNet2D`` or any
compatible network.
"""

from __future__ import annotations

from typing import Optional, Tuple

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - import guard
    torch = None
    F = None

DEFAULT_NUM_TIMESTEPS = 1000
DEFAULT_BETA_START = 1e-4
DEFAULT_BETA_END = 0.02


def linear_beta_schedule(
    num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
    beta_start: float = DEFAULT_BETA_START,
    beta_end: float = DEFAULT_BETA_END,
):
    """Create a linearly increasing noise-variance (beta) schedule.

    Args:
        num_timesteps: Total number of diffusion steps ``T``.
        beta_start: Beta value at timestep 0 (small amount of noise added).
        beta_end: Beta value at the final timestep (largest noise step).

    Returns:
        A 1-D float32 tensor of shape (num_timesteps,).
    """
    if torch is None:  # pragma: no cover
        raise ImportError(
            "PyTorch is required to use the diffusion module. Install it with "
            "`pip install torch`."
        )
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)


def _extract(values, t, broadcast_shape: Tuple[int, ...]):
    """Gather per-timestep scalar values for a batch and reshape for broadcasting.

    Args:
        values: 1-D tensor of shape (num_timesteps,) indexed by timestep.
        t: 1-D integer tensor of shape (batch,) with the timestep for each item.
        broadcast_shape: The shape of the tensor the result will be broadcast
            against, e.g. (batch, 1, n_mels, segment_length).

    Returns:
        A tensor of shape (batch, 1, 1, ...) ready to broadcast-multiply
        against a tensor of ``broadcast_shape``.
    """
    batch_size = t.shape[0]
    out = values.gather(0, t)
    return out.reshape(batch_size, *([1] * (len(broadcast_shape) - 1)))


class GaussianDiffusion:
    """DDPM-style Gaussian diffusion process wrapping a noise-predicting model.

    Args:
        model: A noise-predicting network callable as ``model(x, t)`` where
            ``x`` has shape (batch, 1, n_mels, segment_length) and ``t`` has
            shape (batch,); returns a tensor of the same shape as ``x``
            (e.g. ``models.unet.UNet2D``).
        num_timesteps: Total number of diffusion steps ``T``.
        beta_start: Noise-schedule beta at timestep 0.
        beta_end: Noise-schedule beta at the final timestep.
    """

    def __init__(
        self,
        model,
        num_timesteps: int = DEFAULT_NUM_TIMESTEPS,
        beta_start: float = DEFAULT_BETA_START,
        beta_end: float = DEFAULT_BETA_END,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError(
                "PyTorch is required to use GaussianDiffusion. Install it with "
                "`pip install torch`."
            )
        self.model = model
        self.num_timesteps = num_timesteps

        betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # Posterior variance for the reverse process, beta_tilde_t (DDPM eq. 7).
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def _to(self, device):
        """Move all precomputed schedule buffers to ``device`` in place."""
        for name in (
            "betas",
            "alphas",
            "alphas_cumprod",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas",
            "posterior_variance",
        ):
            setattr(self, name, getattr(self, name).to(device))

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process: add Gaussian noise to ``x_start`` at step ``t``.

        Uses the closed-form expression for sampling ``x_t`` directly from
        ``x_0`` (DDPM eq. 4): ``x_t = sqrt(alpha_cumprod_t) * x_0 +
        sqrt(1 - alpha_cumprod_t) * noise``.

        Args:
            x_start: Clean spectrogram batch, shape (batch, 1, n_mels, segment_length).
            t: Integer timestep per batch item, shape (batch,).
            noise: Optional pre-sampled Gaussian noise of the same shape as
                ``x_start``. If None, it is sampled internally (and returned).

        Returns:
            A tuple ``(x_t, noise)``: the noisy spectrogram at step ``t``, and
            the exact noise tensor that was added (needed as the training
            target).
        """
        self._to(x_start.device)
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = _extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def training_loss(self, x_start, t: Optional[object] = None):
        """Compute the DDPM training loss (MSE between true and predicted noise).

        Args:
            x_start: Clean spectrogram batch, shape (batch, 1, n_mels, segment_length).
            t: Optional pre-sampled integer timestep tensor of shape (batch,).
                If None, timesteps are sampled uniformly at random per item.

        Returns:
            A scalar MSE loss tensor between the true noise and the noise
            predicted by ``self.model``.
        """
        batch_size = x_start.shape[0]
        if t is None:
            t = torch.randint(
                0, self.num_timesteps, (batch_size,), device=x_start.device, dtype=torch.long
            )

        x_t, noise = self.q_sample(x_start, t)
        predicted_noise = self.model(x_t, t)

        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, x_t, t: int):
        """Single reverse-diffusion step: denoise ``x_t`` from step ``t`` to ``t-1``.

        Args:
            x_t: Current noisy spectrogram batch, shape (batch, 1, n_mels, segment_length).
            t: The current (scalar) timestep shared by the whole batch.

        Returns:
            The spectrogram batch one step less noisy (at timestep ``t - 1``).
        """
        self._to(x_t.device)
        batch_size = x_t.shape[0]
        t_batch = torch.full((batch_size,), t, device=x_t.device, dtype=torch.long)

        predicted_noise = self.model(x_t, t_batch)

        sqrt_recip_alphas_t = _extract(self.sqrt_recip_alphas, t_batch, x_t.shape)
        betas_t = _extract(self.betas, t_batch, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract(
            self.sqrt_one_minus_alphas_cumprod, t_batch, x_t.shape
        )

        model_mean = sqrt_recip_alphas_t * (
            x_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t == 0:
            return model_mean

        posterior_variance_t = _extract(self.posterior_variance, t_batch, x_t.shape)
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, int, int, int] = (1, 1, 80, 512),
        device: Optional[object] = None,
    ):
        """Generate a new spectrogram by iterative denoising from pure noise.

        Starts from ``x_T ~ N(0, I)`` of the requested ``shape`` and applies
        ``p_sample`` for every timestep from ``T-1`` down to ``0``, returning
        the final, fully denoised spectrogram.

        Args:
            shape: Output tensor shape, default ``(1, 1, 80, 512)`` -- a
                single generated mel-spectrogram segment matching
                ``SegmentedVoiceDataset``'s default segment shape.
            device: Device to run generation on. Defaults to the device of
                the wrapped model's parameters.

        Returns:
            The generated spectrogram tensor of shape ``shape``.
        """
        if device is None:
            device = next(self.model.parameters()).device

        self._to(device)
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, t)

        return x
