import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

# isort: off

from nerd.utils import pairwise_d2

# isort: on


@dataclass
class NERDConfig:
    """
    Configuration for the NERD decoder, training loop, and replay buffer.

    Attributes
    ----------
    dz : int
        Latent dimension of decoder.
    hidden : int
        Width of the decoder MLP hidden layers.
    beta : float
        Lagrange multiplier used in the BA/NERD objective.
    sigma_init : float
        Initial value for the global Gaussian noise scale.
    learn_sigma : bool
        Whether `sigma` is learned or kept fixed at ``sigma_init``.
    Kz : int
        Number of decoder latent samples used per training step.
    batch_u : int
        Batch size of encoder latents pulled from the replay buffer.
    lr : float
        Learning rate for Adam in the decoder training loop.
    offline_steps : int
        Number of decoder optimization steps in the offline phase.
    online_steps : int
        Number of decoder optimization steps in each online refresh.
    buffer_size : int
        Maximum number of encoder latents stored in the replay buffer.
    buffer_dtype : torch.dtype
        Storage dtype for the replay buffer to save memory.
    """

    # generator
    dz: int = 8
    hidden: int = 256

    # objective / smoothing
    beta: float = 10.0
    sigma_init: float = 0.12
    learn_sigma: bool = True

    # training
    Kz: int = 256
    batch_u: int = 512
    lr: float = 2e-3
    enable_online: bool = True
    offline_steps: int = 600
    online_steps: int = 100
    latents_per_step: int = 1024

    # buffer
    buffer_size: int = 200_000
    buffer_dtype: torch.dtype = torch.float16

    # RD estimation
    rd_Kz: int = 1024
    rd_bisect_iters: int = 12


class NERDDecoder(nn.Module):
    """
    Returns mu_theta(z) in R^d, plus (optionally learned) global sigma.

    Parameters
    ----------
    d : int
        Dimensionality of the output space.
    cfg : NERDConfig
        Configuration controlling hidden width and sigma learning.

    Attributes
    ----------
    d : int
        Output dimensionality.
    dz : int
        Latent dimensionality.
    net : nn.Sequential
        MLP mapping latent `z` to mean vectors ``mu_theta(z)``.
    log_sigma : torch.Tensor
        Logarithm of the global Gaussian standard deviation.
    """

    def __init__(self, d: int, cfg: NERDConfig):
        super().__init__()
        self.d = d
        self.dz = cfg.dz
        self.net = nn.Sequential(
            nn.Linear(cfg.dz, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.SiLU(),
            nn.Linear(cfg.hidden, d),
        )
        if cfg.learn_sigma:
            self.log_sigma = nn.Parameter(
                torch.tensor(math.log(cfg.sigma_init), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "log_sigma", torch.tensor(math.log(cfg.sigma_init), dtype=torch.float32)
            )

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate mean vectors from latent inputs.

        Parameters
        ----------
        z : torch.Tensor
            Latent samples of shape ``(B, dz)``.

        Returns
        -------
        torch.Tensor
            Mean vectors ``mu_theta(z)`` with shape ``(B, d)``.
        """
        return self.net(z)


class ReplayBuffer:
    """
    CPU-side circular buffer for encoder latents.

    Parameters
    ----------
    max_size : int
        Maximum number of vectors retained.
    dim : int
        Dimensionality of each stored vector.
    dtype : torch.dtype, optional
        Storage dtype used on CPU, by default ``torch.float16``.

    Attributes
    ----------
    data : torch.Tensor
        Backing tensor of shape ``(max_size, dim)`` on CPU.
    ptr : int
        Write pointer into the circular buffer.
    full : bool
        Whether the buffer has been completely filled at least once.
    """

    def __init__(self, max_size: int, dim: int, dtype=torch.float16):
        self.max_size = int(max_size)
        self.dim = int(dim)
        self.dtype = dtype
        self.data = torch.empty((self.max_size, self.dim), dtype=dtype, device="cpu")
        self.ptr = 0
        self.full = False

    def __len__(self) -> int:
        return self.max_size if self.full else self.ptr

    @torch.no_grad()
    def add(self, x_cpu: torch.Tensor) -> None:
        """
        Insert a batch of vectors into the buffer, overwriting oldest entries.

        Parameters
        ----------
        x_cpu : torch.Tensor
            Tensor of shape ``(N, dim)`` on any device; moved to CPU/dtype.
        """
        x_cpu = x_cpu.to("cpu", non_blocking=True).to(self.dtype)
        n = int(x_cpu.shape[0])
        if n <= 0:
            return

        if n >= self.max_size:
            self.data.copy_(x_cpu[-self.max_size :])
            self.ptr = 0
            self.full = True
            return

        end = self.ptr + n
        if end <= self.max_size:
            self.data[self.ptr : end].copy_(x_cpu)
            self.ptr = end
            if self.ptr == self.max_size:
                self.ptr = 0
                self.full = True
        else:
            first = self.max_size - self.ptr
            self.data[self.ptr :].copy_(x_cpu[:first])
            rest = n - first
            self.data[:rest].copy_(x_cpu[first:])
            self.ptr = rest
            self.full = True

    @torch.no_grad()
    def sample(self, batch: int, device: torch.device) -> torch.Tensor:
        """
        Randomly sample stored vectors.

        Parameters
        ----------
        batch : int
            Number of vectors to draw.
        device : torch.device
            Device to return the batch on.

        Returns
        -------
        torch.Tensor
            Batch of shape ``(batch, dim)`` on the requested device/dtype float32.
        """
        n = len(self)
        if n <= 0:
            raise RuntimeError("ReplayBuffer is empty")
        idx = torch.randint(0, n, (batch,), device=device)
        return self.data[idx.cpu()].to(
            device=device, dtype=torch.float32, non_blocking=True
        )


def nerd_log_g(
    u: torch.Tensor, mu_k: torch.Tensor, beta: float, sigma: torch.Tensor
) -> torch.Tensor:
    """
    Compute ``log g_beta(u) = log E_{Y~Q_Y} exp(-beta ||u - Y||^2)``.

    The reproduction distribution QY is a Gaussian mixture:
    ``Z ~ N(0, I)``, ``Y|Z ~ N(mu_theta(Z), sigma^2 I)``.

    Parameters
    ----------
    u : torch.Tensor
        Encoder latents of shape ``(B, D)``.
    mu_k : torch.Tensor
        Decoder means ``mu_theta(z_k)`` of shape ``(K, D)`` drawn from ``z_k ~ N(0, I)``.
    beta : float
        Blahut–Arimoto Lagrange multiplier.
    sigma : torch.Tensor
        Global standard deviation of the Gaussian components (scalar tensor).

    Returns
    -------
    torch.Tensor
        Log expectations ``log g_beta(u)`` with shape ``(B,)``.
    """
    D = u.shape[1]
    s2 = sigma**2
    c = -0.5 * D * torch.log1p(2 * beta * s2)  # scalar
    gamma = beta / (1 + 2 * beta * s2)  # scalar
    dist2 = pairwise_d2(u, mu_k)  # [B,K]
    lse = torch.logsumexp(-gamma * dist2, dim=1) - math.log(mu_k.shape[0])
    return c + lse


class NERDSampler(nn.Module):
    """
    Practical NERD object combining decoder, replay buffer, and training loops.
      - decoder mu_theta(z) (+ sigma)
      - replay buffer of encoder latents u
      - offline/online training on E_u [ log g_beta(u) ]
      - samples for init/respawn

    Parameters
    ----------
    d : int
        Dimensionality of the encoder latent space.
    cfg : NERDConfig
        Configuration for the decoder and optimization.
    device : torch.device
        Device where the decoder is hosted.

    Attributes
    ----------
    dec : NERDDecoder
        Neural decoder representing the reproduction distribution ``Q_Y``.
    buf : ReplayBuffer
        Storage for encoder latents used to fit ``Q_Y``.
    """

    def __init__(self, d: int, cfg: NERDConfig, device: torch.device):
        super().__init__()
        self.d = d
        self.cfg = cfg
        self.dec = NERDDecoder(d=d, cfg=cfg)#.to(device)
        self.buf = ReplayBuffer(max_size=cfg.buffer_size, dim=d, dtype=cfg.buffer_dtype)

    @torch.no_grad()
    def add_latents(self, u: torch.Tensor) -> None:
        """
        Add a batch of encoder latents to the replay buffer.

        Parameters
        ----------
        u : torch.Tensor
            Latents of shape ``(B, d)`` on any device.
        """
        self.buf.add(u.detach().cpu())

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        """
        Samples means mu_theta(z) from the learned decoder.

        For codeword init/respawn, returning means is usually what we want.

        Parameters
        ----------
        n : int
            Number of samples to generate.

        Returns
        -------
        torch.Tensor
            Mean vectors ``mu_theta(z)`` with shape ``(n, d)``.
        """
        self.dec.eval()
        dec_device = next(self.dec.parameters()).device
        z = torch.randn(n, self.cfg.dz, device=dec_device)
        return self.dec(z)

    def _train_step(self) -> torch.Tensor:
        """
        Compute loss on buffered latents for a single step.

        Returns
        -------
        torch.Tensor
            Training loss value.
        """
        # self.dec.train()
        dec_device = next(self.dec.parameters()).device
        u = self.buf.sample(self.cfg.batch_u, device=dec_device)
        z = torch.randn(self.cfg.Kz, self.cfg.dz, device=dec_device)
        mu_k = self.dec(z)
        logg = nerd_log_g(u, mu_k, beta=self.cfg.beta, sigma=self.dec.sigma).mean()
        loss = -logg
        # opt = torch.optim.Adam(self.dec.parameters(), lr=self.cfg.lr)
        # opt.zero_grad(set_to_none=True)
        # loss.backward()
        # opt.step()
        return loss

    def _train_steps(self, steps: int) -> None:
        """
        Optimize the decoder on buffered latents for a fixed number of steps.

        Parameters
        ----------
        steps : int
            Number of gradient steps to perform.
        """
        self.dec.train()
        opt = torch.optim.Adam(self.dec.parameters(), lr=self.cfg.lr)
        losses = []
        for _ in range(int(steps)):
            dec_device = next(self.dec.parameters()).device
            u = self.buf.sample(self.cfg.batch_u, device=dec_device)
            z = torch.randn(self.cfg.Kz, self.cfg.dz, device=dec_device)
            mu_k = self.dec(z)
            logg = nerd_log_g(u, mu_k, beta=self.cfg.beta, sigma=self.dec.sigma).mean()
            loss = -logg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        return losses

    def fit_offline(self) -> None:
        """
        Run an offline training phase on the decoder using the replay buffer.

        Raises
        ------
        AssertionError
            If the buffer does not contain enough samples.
        """
        assert len(self.buf) >= max(
            2048, self.cfg.batch_u
        ), "NERD buffer too small for offline fit."
        losses = self._train_steps(self.cfg.offline_steps)
        return losses

    def update_online(self) -> None:
        """
        Refresh the decoder with a shorter online training burst if buffer is ready.
        """
        if len(self.buf) < max(2048, self.cfg.batch_u):
            return
        losses = self._train_steps(self.cfg.online_steps)
        return losses


@dataclass
class RDEstimatorConfig:
    """
    Configuration for RD curve estimation.

    Attributes
    ----------
    Kz : int
        Number of decoder samples used to approximate ``Q_Y``.
    beta_bisect_iters : int
        Number of bisection iterations when solving for a target rate/distortion.
    beta_lo : float
        Lower bound for the bisection search.
    beta_hi : float
        Upper bound for the bisection search.
    beta_expand : float
        Multiplicative factor to enlarge ``beta_hi`` if targets are unmet.
    """

    Kz: int = 256
    beta_bisect_iters: int = 12
    beta_lo: float = 1e-4
    beta_hi: float = 200.0
    beta_expand: float = 2.0


class NERDRDEstimator:
    """
    Estimate the rate–distortion curve implied by a trained NERD decoder.

    With QY fixed (represented by the trained NERD decoder), we can produce
    BA-style points (D_beta, R_beta) and invert either:
      - D(beta) ~= D_target  -> rate needed for that distortion
      - R(beta) ~= R_target  -> frontier distortion at that rate

    Key stability trick:
      Use common random numbers in bisection by sampling mu_k once per solve call.

    Parameters
    ----------
    nerd : NERDSampler
        Trained sampler providing the decoder and sigma.
    cfg : RDEstimatorConfig
        Configuration for sampling and bisection.
    """

    def __init__(self, nerd: NERDSampler, cfg: RDEstimatorConfig):
        self.nerd = nerd
        self.cfg = cfg

    @torch.no_grad()
    def _sample_mu_k(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Draw decoder means and retrieve sigma for RD estimation.

        Parameters
        ----------
        device : torch.device
            Device on which to generate the samples.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of ``(mu_k, sigma)`` where ``mu_k`` has shape ``(Kz, d)`` and
            ``sigma`` is a scalar tensor.
        """
        dec = self.nerd.dec
        dec.eval()
        sigma = dec.sigma
        z = torch.randn(self.cfg.Kz, self.nerd.cfg.dz, device=device)
        mu_k = dec(z)  # [K,D]
        return mu_k, sigma

    @torch.no_grad()
    def estimate_DR(
        self,
        u: torch.Tensor,
        beta: float,
        mu_k: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float, float]:
        """
        Compute distortion and rate for a fixed ``beta``.

        Parameters
        ----------
        u : torch.Tensor
            Encoder latents of shape ``(B, D)`` in float32.
        beta : float
            Lagrange multiplier controlling the trade-off.
        mu_k : torch.Tensor, optional
            Pre-sampled decoder means of shape ``(K, D)``; if None, resampled.
        sigma : torch.Tensor, optional
            Sigma tensor matching the decoder; if None, fetched from decoder.

        Returnsm
        -------
        tuple[float, float, float]
            Distortion ``D_beta``, rate in bits ``R_beta_bits``, and mean ``log g_beta``.
        """
        B, D = u.shape
        if mu_k is None or sigma is None:
            mu_k, sigma = self._sample_mu_k(device=u.device)

        s2 = sigma**2
        c = -0.5 * D * torch.log1p(2 * beta * s2)
        gamma = beta / (1 + 2 * beta * s2)

        dist2 = pairwise_d2(u, mu_k)  # [B,K]
        logw = -gamma * dist2
        logZ = torch.logsumexp(logw, dim=1, keepdim=True)  # [B,1]
        w = torch.exp(logw - logZ)  # [B,K]
        mean_logg = (c + (logZ.squeeze(1) - math.log(mu_k.shape[0]))).mean()

        inv_s2 = 1.0 / (s2 + 1e-30)
        den = inv_s2 + 2.0 * beta
        v = 1.0 / den
        m_k = (
            mu_k[None, :, :] * inv_s2 + (2.0 * beta) * u[:, None, :]
        ) / den  # [B,K,D]
        diff2 = ((u[:, None, :] - m_k) ** 2).sum(dim=2) + D * v  # [B,K]
        D_beta = (w * diff2).sum(dim=1).mean()

        R_bits = float(((-beta * D_beta - mean_logg) / math.log(2.0)).detach().cpu())
        return float(D_beta.detach().cpu()), R_bits, float(mean_logg.detach().cpu())

    @torch.no_grad()
    def solve_beta_for_D(
        self, u: torch.Tensor, D_target: float
    ) -> Tuple[float, float, float]:
        """
        Solve for the ``beta`` that matches a target distortion.

        Parameters
        ----------
        u : torch.Tensor
            Encoder latents of shape ``(B, D)``.
        D_target : float
            Desired distortion level.

        Returns
        -------
        tuple[float, float, float]
            Optimal ``beta``, achieved distortion (``D_at_beta_star``), and corresponding rate in bits (``R_bits_at_beta_star``).
        """
        mu_k, sigma = self._sample_mu_k(device=u.device)

        lo, hi = self.cfg.beta_lo, self.cfg.beta_hi
        D_lo, R_lo, _ = self.estimate_DR(u, lo, mu_k=mu_k, sigma=sigma)
        D_hi, R_hi, _ = self.estimate_DR(u, hi, mu_k=mu_k, sigma=sigma)

        expand_tries = 0
        while D_hi > D_target and expand_tries < 8:
            hi *= self.cfg.beta_expand
            D_hi, R_hi, _ = self.estimate_DR(u, hi, mu_k=mu_k, sigma=sigma)
            expand_tries += 1

        if D_hi > D_target:
            return float(hi), float(D_hi), float(R_hi)

        if D_lo <= D_target:
            return float(lo), float(D_lo), float(R_lo)

        beta_star, D_star, R_star = hi, D_hi, R_hi
        for _ in range(self.cfg.beta_bisect_iters):
            mid = 0.5 * (lo + hi)
            D_mid, R_mid, _ = self.estimate_DR(u, mid, mu_k=mu_k, sigma=sigma)
            if D_mid > D_target:
                lo = mid
            else:
                hi = mid
                beta_star, D_star, R_star = mid, D_mid, R_mid

        return float(beta_star), float(D_star), float(R_star)

    @torch.no_grad()
    def solve_beta_for_R(
        self, u: torch.Tensor, R_target_bits: float
    ) -> Tuple[float, float, float]:
        """
        Solve for the ``beta`` that matches a target rate.

        Returns (beta_star, D_at_beta_star, R_at_beta_star_bits) where R ~= R_target_bits.

        Parameters
        ----------
        u : torch.Tensor
            Encoder latents of shape ``(B, D)``.
        R_target_bits : float
            Desired rate in bits.

        Returns
        -------
        tuple[float, float, float]
            Optimal ``beta``, achieved distortion, and achieved rate in bits.
        """
        mu_k, sigma = self._sample_mu_k(device=u.device)

        lo, hi = self.cfg.beta_lo, self.cfg.beta_hi
        D_lo, R_lo, _ = self.estimate_DR(u, lo, mu_k=mu_k, sigma=sigma)
        D_hi, R_hi, _ = self.estimate_DR(u, hi, mu_k=mu_k, sigma=sigma)

        expand_tries = 0
        while R_hi < R_target_bits and expand_tries < 8:
            hi *= self.cfg.beta_expand
            D_hi, R_hi, _ = self.estimate_DR(u, hi, mu_k=mu_k, sigma=sigma)
            expand_tries += 1

        if R_hi < R_target_bits:
            return float(hi), float(D_hi), float(R_hi)

        if R_lo >= R_target_bits:
            return float(lo), float(D_lo), float(R_lo)

        beta_star, D_star, R_star = hi, D_hi, R_hi
        for _ in range(self.cfg.beta_bisect_iters):
            mid = 0.5 * (lo + hi)
            D_mid, R_mid, _ = self.estimate_DR(u, mid, mu_k=mu_k, sigma=sigma)
            if R_mid < R_target_bits:
                lo = mid
            else:
                hi = mid
                beta_star, D_star, R_star = mid, D_mid, R_mid

        return float(beta_star), float(D_star), float(R_star)
