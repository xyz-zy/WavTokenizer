# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Core vector quantization implementation."""

import math
import time
import typing as tp
import warnings

from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F

from .. import distrib


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(
    samples,
    num_clusters: int,
    num_iters: int = 10,
    tol: float = 1e-4,
    return_history: bool = False,
):
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    history = []

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(
            means, "c d -> () c d"
        )
        dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        inertia = (-dists.gather(1, buckets[:, None])).sum().item()
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        new_means = torch.where(zero_mask[..., None], means, new_means)

        deltas = (new_means - means).norm(dim=-1)
        moved = (deltas > tol).sum().item()
        history.append(
            {
                "mean_delta": deltas.mean().item(),
                "max_delta": deltas.max().item(),
                "moved_centers": moved,
                "inertia": inertia,
            }
        )

        means = new_means

    if return_history:
        return means, bins, history
    return means, bins


_BA_LN2 = math.log(2.0)


@torch.no_grad()
def _ba_kmeanspp_init(Z: torch.Tensor, M: int, seed: int = 0, chunk: int = 8192) -> torch.Tensor:
    """KMeans++ seeding. Z: (N, k) → Y: (M, k)."""
    N, k = Z.shape
    g = torch.Generator(device=Z.device).manual_seed(seed)
    Y = torch.empty(M, k, device=Z.device)
    Y[0] = Z[torch.randint(0, N, (1,), generator=g, device=Z.device)]
    dmin = torch.full((N,), float("inf"), device=Z.device)
    for m in range(1, M):
        c = Y[m - 1:m]
        for s in range(0, N, chunk):
            d = ((Z[s:s + chunk] - c) ** 2).sum(-1)
            dmin[s:s + chunk] = torch.minimum(dmin[s:s + chunk], d)
        probs = dmin.clamp(min=1e-12)
        Y[m] = Z[torch.multinomial(probs / probs.sum(), 1, generator=g)]
        # if m % 512 == 0:
        #     print(f"  kmeans++ init: {m}/{M}")
    return Y


@torch.no_grad()
def _ba_pass(
    Z: torch.Tensor, Y: torch.Tensor, logq: torch.Tensor, beta: float, chunk_N: int = 2048
):
    """One BA E-step. Returns (q_new, D, R_bits, num, den)."""
    N, k = Z.shape
    M = Y.shape[0]
    y2 = (Y * Y).sum(-1)

    q_acc = torch.zeros(M, device=Z.device)
    num = torch.zeros(M, k, device=Z.device)
    den = torch.zeros(M, device=Z.device)
    D_sum = 0.0
    I_sum = 0.0

    for s in range(0, N, chunk_N):
        z = Z[s:s + chunk_N]
        d = ((z * z).sum(-1, keepdim=True) + y2 - 2 * z @ Y.t()).clamp(min=0)
        logits = logq - beta * d
        lse = torch.logsumexp(logits, dim=1, keepdim=True)
        logp = logits - lse
        p = torch.exp(logp)

        q_acc += p.sum(0)
        num += p.t() @ z
        den += p.sum(0)
        D_sum += (p * d).sum().item()
        I_sum += (p * (logp - logq)).sum().item()

    q_new = (q_acc / N).clamp(min=1e-30)
    return q_new, D_sum / N, (I_sum / N) / _BA_LN2, num, den


@torch.no_grad()
def _ba_solve(
    Z: torch.Tensor,
    beta: float,
    M: int,
    Y_init: torch.Tensor,
    outer_iters: int = 25,
    inner_iters: int = 5,
    tol_q: float = 1e-6,
    chunk_N: int = 2048,
    verbose: bool = False,
) -> dict:
    """Blahut-Arimoto solver. Returns dict: Y, q, logq, D, R_bits, elapsed_s, converged, n_outer_iters, final_delta."""
    t_start = time.time()
    logq = torch.full((M,), -math.log(M), device=Z.device)
    Y = Y_init.clone()
    delta = float("inf")

    for t in range(outer_iters):
        for _ in range(inner_iters):
            q_new, D, R_bits, num, den = _ba_pass(Z, Y, logq, beta, chunk_N)
            delta = (q_new - torch.exp(logq)).abs().max().item()
            logq = torch.log(q_new)
            if delta < tol_q:
                break
        Y = (num / (den.unsqueeze(1) + 1e-12)).contiguous()
        if verbose:
            print(f"  [outer {t:02d}] D={D:.6f}  R={R_bits:.4f} bits  |Δq|={delta:.2e}")
        if delta < 1e-6:
            break

    converged = delta < 1e-6
    n_outer_iters = t + 1  # t is 0-indexed; +1 gives number of outer iters completed

    _, D, R_bits, _, _ = _ba_pass(Z, Y, logq, beta, chunk_N)
    return {"Y": Y, "q": q_new, "logq": logq, "D": float(D), "R_bits": float(R_bits),
            "elapsed_s": time.time() - t_start,
            "converged": converged, "n_outer_iters": n_outer_iters, "final_delta": float(delta)}


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())
        self.expired_codes = -1

        self.kmeans_history = None
        self.use_ba_respawn = False  # set externally by experiment once global_step threshold is reached
        self.ba_out = None

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return

        embed, cluster_size, kmeans_history = kmeans(
            data, self.codebook_size, self.kmeans_iters, return_history=True
        ) #data不变
        self.kmeans_history = kmeans_history
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        distrib.broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return
        self.expired_codes = expired_codes.sum().item()
        self.expired_codes_mask = expired_codes
        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        distrib.broadcast_tensors(self.buffers())

    @torch.no_grad()
    def expire_codes_ba_(self, x):
        self.ba_time = 0
        if self.threshold_ema_dead_code == 0:
            return
        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            self.expired_codes = 0
            return
        self.expired_codes = expired_codes.sum().item()
        self.expired_codes_mask = expired_codes
        ba_out = self.run_ba(x, update_codebook=False)
        self.ba_out = ba_out
        ba_centroids = ba_out["embed_new"]               # [M, dim] pool to sample replacements from
        self.replace_(ba_centroids, mask=expired_codes)  # samples from BA centroid pool
        distrib.broadcast_tensors(self.buffers())

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    @torch.no_grad()
    def run_ba(
        self,
        x: torch.Tensor,
        beta: float = 10.0,
        var_threshold: float = 0.99,
        outer_iters: int = 25,
        inner_iters: int = 5,
        chunk_N: int = 2048,
        seed: int = 0,
        verbose: bool = False,
        update_codebook: bool = False,
    ) -> dict:
        """Re-initialize the codebook using Blahut-Arimoto in a PCA subspace.

        Runs PCA on x, projects to the subspace capturing var_threshold of the
        variance, runs BA to find optimal centroids in that subspace, then
        back-projects to the full latent dimension and stores the result in
        self.embed / self.embed_avg.

        Args:
            x: Latent vectors of arbitrary leading shape with final dim matching
               the codebook dimension, e.g. (B, T, dim) or (N, dim).
            beta: Inverse temperature for BA (higher = lower distortion / higher rate).
            var_threshold: Fraction of variance to retain in PCA subspace (0 < t <= 1).
            outer_iters: Max outer BA iterations.
            inner_iters: Max inner q-update iterations per outer step.
            chunk_N: Chunk size for chunked distance computation.
            seed: RNG seed for KMeans++ init.
            verbose: Print per-iteration diagnostics.
            update_codebook: Whether to update the codebook buffers.
        Returns:
            dict with keys: D, R_bits, k_pca, var_explained, elapsed_s.
        """
        x = self.preprocess(x).float()  # [N, dim]

        # PCA
        mu = x.mean(0)                          # [dim]
        x_c = x - mu                            # centered
        q_rank = min(x_c.shape[0], x_c.shape[1])
        _, S, V = torch.pca_lowrank(x_c, q=q_rank, center=False)
        var_ratio = (S ** 2).cumsum(0) / (S ** 2).sum()
        k = int((var_ratio >= var_threshold).nonzero(as_tuple=False)[0, 0].item()) + 1
        B = V[:, :k].contiguous()               # [dim, k]

        if verbose:
            print(f"  PCA: retaining k={k} components, var_explained={var_ratio[k-1]:.4f}")

        Z = x_c @ B                             # [N, k]

        # BA
        Y_init = _ba_kmeanspp_init(Z, self.codebook_size, seed=seed)
        out = _ba_solve(Z, beta, M=self.codebook_size, Y_init=Y_init,
                        outer_iters=outer_iters, inner_iters=inner_iters,
                        tol_q=1e-6, chunk_N=chunk_N, verbose=verbose)

        # Back-project centroids to full dim
        embed_new = out["Y"] @ B.t() + mu      # [M, dim]

        if update_codebook:
            # Update codebook buffers
            self.embed.copy_(embed_new)
            self.embed_avg.copy_(embed_new * self.threshold_ema_dead_code)
            self.cluster_size.fill_(self.threshold_ema_dead_code)
            self.inited.fill_(1.0)

        return {
            "embed_new": embed_new,
            "D": out["D"],
            "R_bits": out["R_bits"],
            "k_pca": k,
            "var_explained": float(var_ratio[k - 1].item()),
            "elapsed_s": out["elapsed_s"],
            "final_delta": out["final_delta"],
        }

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)

        self.init_embed_(x)

        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            if not self.use_ba_respawn:
                self.expire_codes_(x)
            self.embed_onehot_sum = embed_onehot.sum(0)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            # print(embed_sum.cpu().detach().numpy().tolist())
            # print(embed_sum.shape)
            # embed_sum_summed = embed_sum.abs().sum(0)
            # print(embed_sum_summed.shape)
            # print(embed_sum_summed)
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
            if self.use_ba_respawn:
                self.expire_codes_ba_(x)

        return quantize, embed_ind


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(dim=_codebook_dim, codebook_size=codebook_size,
                                           kmeans_init=kmeans_init, kmeans_iters=kmeans_iters,
                                           decay=decay, epsilon=epsilon,
                                           threshold_ema_dead_code=threshold_ema_dead_code)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x):

        # breakpoint()
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        quantize, embed_ind = self._codebook(x)
        if self.training:
            quantize = x + (quantize - x).detach()
        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            # warnings.warn('When using RVQ in training model, first check '
            #               'https://github.com/facebookresearch/encodec/issues/25 . '
            #               'The bug wasn\'t fixed here for reproducibility.')
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(self, x, n_q: tp.Optional[int] = None):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_losses.append(loss)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            all_indices.append(indices)
            quantized = layer.decode(indices)
            residual = residual - quantized.detach()
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out


class LanguageVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )
        # print("core_vq.py:self.layers",self.layers)

    def forward(self, x, n_q: tp.Optional[int] = None):
        # breakpoint()  x[b,t,c] #[64,75,128]  
        quantized_out = 0.0
        residual = x


        all_losses = []
        all_indices = []

        # breakpoint()

        n_q = n_q or len(self.layers)
          
        for layer in self.layers[:n_q]:
            quantized_out, indices, loss = layer(residual)  #得到该层的表征，该层的indices,该层的loss  [64,75]
            # residual = residual - quantized.detach()
            # quantized_out = quantized_out + quantized
            all_indices.append(indices)
            all_losses.append(loss)
        # breakpoint()
        # breakpoint()

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            all_indices.append(indices)
            quantized = layer.decode(indices)
            residual = residual - quantized.detach()
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out
