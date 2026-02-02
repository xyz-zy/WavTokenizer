"""
Generate a t-SNE visualization of encoder latent vectors (pre-quantization).

Usage example:
python latent_space_viz.py \
  --input_path ./test-clean_filelist.txt \
  --config_path ../WavTokenizer_models/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
  --model_path ../WavTokenizer_models/WavTokenizer_small_600_24k_4096.ckpt \
  --out_folder ./result/latent_viz --max_vectors_per_file 10 --max_total_vectors 20000 --tsne_with_codebook

python latent_space_viz.py \
  --input_path ./test_filelist.txt \
  --config_path ../WavTokenizer_models/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
  --model_path ../WavTokenizer_models/WavTokenizer_small_600_24k_4096.ckpt \
  --out_folder ./result/latent_viz --max_vectors_per_file 100 --max_total_vectors 100000 --tsne_with_codebook \
  --stop_at_global_cap

python latent_space_viz.py \
  --input_path ./test-clean_filelist.txt \
  --config_path ../WavTokenizer_models/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
  --model_path ../WavTokenizer_models/WavTokenizer_small_320_24k_4096.ckpt \
  --out_folder ./result/latent_viz --max_vectors_per_file 100 --max_total_vectors 20000 \
  --tsne_with_codebook --stop_at_global_cap

python latent_space_viz.py \
  --input_path ./test_filelist.txt \
  --config_path ../WavTokenizer_models/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
  --model_path ../WavTokenizer_models/WavTokenizer_small_320_24k_4096.ckpt \
  --out_folder ./result/latent_viz --max_vectors_per_file 100 --max_total_vectors 100000 --tsne_with_codebook \
  --stop_at_global_cap

python latent_space_viz.py \
  --input_path ./test-clean_filelist.txt \
  --config_path configs/WavTokenizer_small_600_24k_4096_nerdonly.yaml \
  --model_path result/train/WavTokenizer_small_600_24k_4096_nerdonly/lightning_logs/version_0/checkpoints/last.ckpt \
  --out_folder ./result/latent_viz --max_vectors_per_file 100 --max_total_vectors 20000 \
  --tsne_with_codebook --stop_at_global_cap

python latent_space_viz.py \
  --input_path ./test_filelist.txt \
  --config_path configs/WavTokenizer_small_320_24k_4096_nerdonly.yaml \
  --model_path result/train/WavTokenizer_small_320_24k_4096_nerdonly/lightning_logs/version_3/checkpoints/last.ckpt \
  --out_folder ./result/latent_viz --max_vectors_per_file 100 --max_total_vectors 20000 \
  --tsne_with_codebook --stop_at_global_cap

python latent_space_viz.py \
  --input_path ./test-clean_filelist.txt \
  --config_path configs/WavTokenizer_small_320_24k_4096_nerdonly.yaml \
  --model_path result/train/WavTokenizer_small_320_24k_4096_nerdonly/lightning_logs/version_0/checkpoints/last.ckpt \
  --out_folder ./result/latent_viz --max_vectors_per_file 100 --max_total_vectors 20000 --tsne_with_codebook \
  --stop_at_global_cap

python latent_space_viz.py \
  --input_path ./test_filelist.txt \
  --config_path configs/WavTokenizer_small_600_24k_4096_nerdonly_beta0.yaml \
  --model_path result/train/WavTokenizer_small_600_24k_4096_nerdonly_beta0/lightning_logs/version_0/checkpoints/last.ckpt \
  --out_folder ./result/latent_viz --max_vectors_per_file 100 --max_total_vectors 20000 --tsne_with_codebook \
  --stop_at_global_cap

python latent_space_viz.py \
  --input_path ./test_filelist.txt \
  --config_path configs/WavTokenizer_small_600_24k_4096_nerdonly_beta20.yaml \
  --model_path result/train/WavTokenizer_small_600_24k_4096_nerdonly_beta20/lightning_logs/version_0/checkpoints/last.ckpt \
  --out_folder ./result/latent_viz --max_vectors_per_file 100 --max_total_vectors 20000 --tsne_with_codebook \
  --stop_at_global_cap
  
python latent_space_viz.py \
  --input_path ./test_filelist.txt \
  --config_path configs/WavTokenizer_small_600_24k_4096_nerdonly_beta10.yaml \
  --model_path result/train/WavTokenizer_small_600_24k_4096_nerdonly_beta10/lightning_logs/version_0/checkpoints/last.ckpt \
  --out_folder ./result/latent_viz --max_vectors_per_file 100 --max_total_vectors 20000 --tsne_with_codebook \
  --stop_at_global_cap

python latent_space_viz.py \
  --input_path ./test_filelist.txt \
  --config_path configs/WavTokenizer_small_600_24k_4096_nerdonly_beta5.yaml \
  --model_path result/train/WavTokenizer_small_600_24k_4096_nerdonly_beta5/lightning_logs/version_0/checkpoints/last.ckpt \
  --out_folder ./result/latent_viz --max_vectors_per_file 100 --max_total_vectors 20000 --tsne_with_codebook \
  --stop_at_global_cap

python latent_space_viz.py \
  --input_path ./test_filelist.txt \
  --config_path configs/WavTokenizer_small_600_24k_4096_nerdonly_beta5_nobuf.yaml \
  --model_path result/train/WavTokenizer_small_600_24k_4096_nerdonly_beta5_nobuf/lightning_logs/version_0/checkpoints/last.ckpt \
  --out_folder ./result/latent_viz --max_vectors_per_file 100 --max_total_vectors 100000 --tsne_with_codebook \
  --stop_at_global_cap

python latent_space_viz.py \
  --input_path ./test_filelist.txt \
  --config_path configs/WavTokenizer_small_600_24k_4096_nerdonly_beta20_nobuf.yaml \
  --model_path result/train/WavTokenizer_small_600_24k_4096_nerdonly_beta20_nobuf/lightning_logs/version_0/checkpoints/last.ckpt \
  --out_folder ./result/latent_viz --max_vectors_per_file 100 --max_total_vectors 100000 --tsne_with_codebook \
  --stop_at_global_cap

python latent_space_viz.py \
  --input_path ./test_filelist.txt \
  --config_path configs/WavTokenizer_small_600_24k_4096_nerdonly_beta100_nobuf.yaml \
  --model_path result/train/WavTokenizer_small_600_24k_4096_nerdonly_beta100_nobuf/lightning_logs/version_0/checkpoints/last.ckpt \
  --out_folder ./result/latent_viz --max_vectors_per_file 100 --max_total_vectors 100000 --tsne_with_codebook \
  --stop_at_global_cap
  
python latent_space_viz.py \
  --input_path ./test_filelist.txt \
  --config_path configs/WavTokenizer_small_600_24k_4096_nerdonly_beta10_nobuf.yaml \
  --model_path result/train/WavTokenizer_small_600_24k_4096_nerdonly_beta10_nobuf/lightning_logs/version_0/checkpoints/last.ckpt \
  --out_folder ./result/latent_viz --max_vectors_per_file 100 --max_total_vectors 100000 --tsne_with_codebook \
  --stop_at_global_cap


python latent_space_viz.py \
  --input_path ./test_filelist.txt \
  --config_path configs/WavTokenizer_small_320_24k_4096_nerdonly_beta10_nobuf.yaml \
  --model_path result/train/WavTokenizer_small_320_24k_4096_nerdonly_beta10_nobuf/lightning_logs/version_0/checkpoints/last.ckpt \
  --out_folder ./result/latent_viz --max_vectors_per_file 100 --max_total_vectors 100000 --tsne_with_codebook \
  --stop_at_global_cap

python latent_space_viz.py \
  --input_path ./test_filelist.txt \
  --config_path configs/WavTokenizer_small_320_24k_4096_nerdonly_beta100_nobuf.yaml \
  --model_path result/train/WavTokenizer_small_320_24k_4096_nerdonly_beta100_nobuf/lightning_logs/version_0/checkpoints/last.ckpt \
  --out_folder ./result/latent_viz --max_vectors_per_file 100 --max_total_vectors 100000 --tsne_with_codebook \
  --stop_at_global_cap

python latent_space_viz.py \
    --input_path ./test_filelist.txt \
    --config_path configs/WavTokenizer_small_600_24k_4096_nerdonly_beta10_nobuf_continue_bs40.yaml \
    --model_path result/train/WavTokenizer_small_600_24k_4096_nerdonly_beta10_nobuf_continue_bs40/lightning_logs/version_0/checkpoints/last.ckpt \
    --out_folder ./result/latent_viz --max_vectors_per_file 100 --max_total_vectors 100000 --tsne_with_codebook \
    --stop_at_global_cap

python latent_space_viz.py \
    --input_path ./test_filelist.txt \
    --config_path configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn_train860.yaml \
    --model_path result/train/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn_train860/lightning_logs/version_0/checkpoints/last.ckpt \
    --out_folder ./result/latent_viz --max_vectors_per_file 100 --max_total_vectors 100000 --tsne_with_codebook \
    --stop_at_global_cap
"""

import argparse
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from decoder.pretrained import WavTokenizer
from einops import rearrange
from encoder.quantization.core_vq import EuclideanCodebook
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from nerd.utils import pairwise_d2


def collect_vectors(
    wavtokenizer,
    filepaths,
    device,
    max_vectors_per_file=200,
    max_total_vectors=20000,
    stop_at_global_cap: bool = False,
):
    vectors = []
    labels = []
    for idx, p in enumerate(tqdm(filepaths, desc="Extracting latents")):
        try:
            wav, sr = torchaudio.load(p)
        except Exception as e:
            print("Failed to load", p, e)
            continue

        wav = wav.to(device)
        # encoder expects (B,1,T)
        audio = wav.unsqueeze(1) if wav.dim() == 2 else wav
        with torch.inference_mode():
            emb = wavtokenizer.feature_extractor.encodec.encoder(audio)
            emb = rearrange(emb, "b d n -> b n d")
            emb = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[
                0
            ].project_in(emb)
        # print(f"{emb.shape=}")
        # emb: (B, L, C)
        emb_np = emb.detach().cpu().numpy()
        B, L, C = emb_np.shape
        frames = emb_np.reshape(-1, C)  # (B*L, C)
        # print(f"{frames.shape=}")

        # subsample frames for this file
        n = frames.shape[0]
        if n > max_vectors_per_file:
            idxs = np.random.choice(n, max_vectors_per_file, replace=False)
            sel = frames[idxs]
        else:
            sel = frames

        # print(f"sel shape: {sel.shape}")
        vectors.append(sel)
        labels.extend([idx] * sel.shape[0])

        # enforce global cap
        if stop_at_global_cap:
            total = sum([v.shape[0] for v in vectors])
            if total >= max_total_vectors:
                print(f"Reached global cap of {max_total_vectors} vectors; stopping.")
                break

    if len(vectors) == 0:
        return None, None

    X = np.concatenate(vectors, axis=0)
    y = np.array(labels, dtype=np.int32)
    # if exceeded max_total, subsample rows
    if X.shape[0] > max_total_vectors:
        print("Subsampling to max total vectors:", max_total_vectors)
        sel = np.random.choice(X.shape[0], max_total_vectors, replace=False)
        X = X[sel]
        y = y[sel]
    return X, y


def collect_codebook_vectors(wavtokenizer, pca=None):
    vq_layers = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers
    cb_list = []
    for vq in vq_layers:
        cb = vq.codebook.detach().cpu()
        # if hasattr(vq, "project_out"):
        #     with torch.no_grad():
        #         cb_proj = vq.project_out(cb)
        # else:
        #     cb_proj = cb
        # cb_list.append(cb_proj.numpy())
        # print(f"Codebook shape: {cb.shape}")
        cb_list.append(cb.numpy())
    codebook_vectors = np.concatenate(cb_list, axis=0)
    if pca is not None:
        codebook_proc = pca.transform(codebook_vectors)
    else:
        codebook_proc = codebook_vectors
    return codebook_vectors, codebook_proc


def run_pca(X, n_components, random_seed):
    if not n_components or n_components <= 0 or X.shape[1] <= n_components:
        return X, None
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_components, random_state=random_seed)
        X_proc = pca.fit_transform(X)
        print(f"PCA reduced {X.shape} -> {X_proc.shape}")
        return X_proc, pca
    except Exception as e:
        print("PCA failed, continuing without PCA:", e)
        return X, None


def run_tsne(
    X_proc, codebook_proc, perplexity, n_iter, random_seed, use_codebook: bool
):
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_seed,
    )
    if codebook_proc is None or not use_codebook:
        X_tsne = tsne.fit_transform(X_proc)
        codebook_tsne = None
    else:
        combined = np.concatenate([X_proc, codebook_proc], axis=0)
        combined_tsne = tsne.fit_transform(combined)
        X_tsne = combined_tsne[: X_proc.shape[0]]
        codebook_tsne = combined_tsne[X_proc.shape[0] :]
    return X_tsne, codebook_tsne


def plot_latent_tsne(
    out_folder,
    X_tsne,
    codebook_tsne,
    codebook_vectors,
    y,
    plot_codebook: bool,
    suffix: str = "",
):
    suf = f"_{suffix}" if suffix else ""
    if plot_codebook:
        output_file = out_folder / f"latent_tsne{suf}_codebook.png"
    else:
        output_file = out_folder / f"latent_tsne{suf}.png"

    n_bins = 100

    # joint scatter with marginal histograms
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4), hspace=0.05, wspace=0.05
    )
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_histy = fig.add_subplot(gs[1, 1])

    # main scatter
    ax_main.scatter(
        X_tsne[:, 0], X_tsne[:, 1], c="tab:blue", s=4, alpha=0.6, label="latents"
    )
    if codebook_tsne is not None and plot_codebook:
        ax_main.scatter(
            codebook_tsne[:, 0],
            codebook_tsne[:, 1],
            c="tab:orange",
            s=20,
            alpha=0.9,
            marker="o",
            edgecolors="k",
            label="codebook",
        )

    ax_main.set_xlabel("t-SNE dim 1")
    ax_main.set_ylabel("t-SNE dim 2")
    ax_main.legend()

    # top marginal (x)
    ax_histx.hist(X_tsne[:, 0], bins=n_bins, color="tab:blue", alpha=0.6, density=True)
    if codebook_tsne is not None and plot_codebook:
        ax_histx.hist(
            codebook_tsne[:, 0],
            bins=n_bins,
            color="tab:orange",
            alpha=0.6,
            density=True,
        )
    ax_histx.axis("off")

    # right marginal (y)
    ax_histy.hist(
        X_tsne[:, 1],
        bins=n_bins,
        orientation="horizontal",
        color="tab:blue",
        alpha=0.6,
        density=True,
    )
    if codebook_tsne is not None and plot_codebook:
        ax_histy.hist(
            codebook_tsne[:, 1],
            bins=n_bins,
            orientation="horizontal",
            color="tab:orange",
            alpha=0.6,
            density=True,
        )
    ax_histy.axis("off")

    plt.suptitle("t-SNE of encoder latents (pre-quant) with codebook overlay")
    fig.savefig(output_file, dpi=200)
    plt.close(fig)
    print("Wrote t-SNE plot to", output_file)


def plot_nerd_metrics(
    codebook: EuclideanCodebook, latents: torch.Tensor, out_folder: Path
):
    """Plot NERD rate-distortion metrics from the given codebook's nerd_sampler."""

    if not hasattr(codebook, "nerd_sampler"):
        print("Codebook has no nerd_sampler; skipping NERD metrics plot.")
        return
    if not hasattr(codebook, "nerd_rd_estimator"):
        print("Codebook has no nerd_rd_estimator; skipping NERD metrics plot.")
        return

    nerd_sampler = codebook.nerd_sampler
    nerd_rd_estimator = codebook.nerd_rd_estimator

    beta_train = (
        getattr(nerd_sampler, "cfg", None).beta
        if hasattr(nerd_sampler, "cfg")
        else None
    )

    if latents is None or latents.numel() == 0:
        print("No latents provided; skipping NERD metrics plot.")
        return

    latents = latents.reshape(-1, latents.shape[-1]).to(torch.float32)
    n_latents = min(4096, latents.shape[0])
    if n_latents <= 0:
        print("No latents available after reshape; skipping NERD metrics plot.")
        return

    device = codebook.embed.device
    if latents.device.type == "cpu" and device.type != "cpu":
        idx = torch.randperm(latents.shape[0])[:n_latents]
        u = latents[idx].to(device=device, dtype=torch.float32)
    else:
        idx = torch.randperm(latents.shape[0], device=latents.device)[:n_latents]
        u = latents[idx].to(device=device, dtype=torch.float32)

    # Compute empirical VQ distortion and H_bits from assignments
    with torch.no_grad():
        embed_ind = codebook.quantize(u).reshape(-1)
        u_q = codebook.dequantize(embed_ind)
        dist2 = pairwise_d2(u, u_q)  # [B, B]
        codebook_distortion = float(dist2.diagonal().mean().detach().cpu())
        # codebook_distortion = float(((u - u_q) ** 2).sum(dim=1).mean().detach().cpu())
        counts = torch.bincount(embed_ind, minlength=codebook.codebook_size).to(
            torch.float32
        )
        p = (counts / counts.sum()).clamp_min(1e-12)
        codebook_H_bits = float((-p * torch.log2(p)).sum().detach().cpu())

    # Generate plots of rate-distortion curves
    beta_min, beta_max = 1e-3, 1e3
    beta_eval_list = torch.logspace(
        math.log10(beta_min), math.log10(beta_max), steps=32, device=device
    )

    mu_k, sigma = nerd_rd_estimator._sample_mu_k(device=device)
    D_list = []
    R_list = []
    beta_list = []
    for beta in beta_eval_list:
        # Compute rate-distortion metrics on latents
        with torch.no_grad():
            D, R, _ = nerd_rd_estimator.estimate_DR(
                u, beta=float(beta.item()), mu_k=mu_k, sigma=sigma
            )
        D_list.append(D)
        R_list.append(R)
        beta_list.append(float(beta.item()))
    if not D_list or not R_list:
        print("NERD RD estimation produced no points; skipping plot.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.plot(R_list, D_list, marker="o", ms=3, lw=1.2, label="NERD frontier")
    ax.scatter(
        [codebook_H_bits],
        [codebook_distortion],
        c="tab:red",
        s=40,
        label="VQ (H_bits, D_vq)",
        zorder=3,
    )
    ax.annotate(f"beta={beta_list[-1]:.4g}", (R_list[-1], D_list[-1]))
    ax.annotate(f"beta={beta_list[0]:.4g}", (R_list[0], D_list[0]))
    ax.set_xlabel("Rate (bits)")
    ax.set_ylabel("Distortion")
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    sigma_val = float(sigma.detach().cpu())
    title_parts = [
        f"sigma={sigma_val:.4g}",
        f"D_vq={codebook_distortion:.4g}",
        f"H_bits={codebook_H_bits:.4g}",
    ]
    if beta_train is not None:
        title_parts.insert(1, f"beta_train={beta_train:.4g}")
    ax.set_title("NERD rate-distortion curve\n" + " | ".join(title_parts))
    ax.legend(loc="best")
    out_file = out_folder / "nerd_rd_curve.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print("Wrote NERD RD plot to", out_file)


def _joint_with_marginals(fig, gs, n_bins, X2, cb2=None, dead=None, title=""):
    sub = gs.subgridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4), hspace=0.05, wspace=0.05
    )
    ax_histx = fig.add_subplot(sub[0, 0])
    ax_main = fig.add_subplot(sub[1, 0])
    ax_histy = fig.add_subplot(sub[1, 1])

    ax_main.scatter(X2[:, 0], X2[:, 1], c="tab:blue", s=6, alpha=0.6, label="latents")
    if cb2 is not None and dead is None:
        ax_main.scatter(
            cb2[:, 0],
            cb2[:, 1],
            c="tab:orange",
            s=24,
            alpha=0.95,
            edgecolors="k",
            label="codebook",
        )
    if cb2 is not None and dead is not None:
        ax_main.scatter(
            cb2[dead, 0],
            cb2[dead, 1],
            c="tab:red",
            s=26,
            alpha=0.95,
            edgecolors="k",
            label="dead codewords",
        )
    ax_main.set_xlabel("PC 1")
    ax_main.set_ylabel("PC 2")
    ax_main.legend(loc="best")

    ax_histx.set_title(title)

    ax_histx.hist(X2[:, 0], bins=n_bins, color="tab:blue", alpha=0.6, density=True)
    if cb2 is not None and dead is None:
        ax_histx.hist(
            cb2[:, 0], bins=n_bins, color="tab:orange", alpha=0.6, density=True
        )
    if cb2 is not None and dead is not None:
        ax_histx.hist(
            cb2[dead, 0], bins=n_bins, color="tab:red", alpha=0.6, density=True
        )
    ax_histx.axis("off")

    ax_histy.hist(
        X2[:, 1],
        bins=n_bins,
        orientation="horizontal",
        color="tab:blue",
        alpha=0.6,
        density=True,
    )
    if cb2 is not None and dead is None:
        ax_histy.hist(
            cb2[:, 1],
            bins=n_bins,
            orientation="horizontal",
            color="tab:orange",
            alpha=0.6,
            density=True,
        )
    if cb2 is not None and dead is not None:
        ax_histy.hist(
            cb2[dead, 1],
            bins=n_bins,
            orientation="horizontal",
            color="tab:red",
            alpha=0.6,
            density=True,
        )
    ax_histy.axis("off")


def _plot_pca_component_pair(
    out_folder: Path,
    X_pca_full: np.ndarray,
    cb_pca_full: np.ndarray,
    usage_counts,
    pc_x: int,
    pc_y: int,
    filestem: str,
    codebook_name: str,
    title_suffix: str,
):
    """Plot the given PCA component pair (pc_x, pc_y) from the full PCA projections."""

    X_pca2 = X_pca_full[:, [pc_x, pc_y]]
    cb_pca2 = None if cb_pca_full is None else cb_pca_full[:, [pc_x, pc_y]]

    np.save(out_folder / f"{filestem}.npy", X_pca2)
    if cb_pca2 is not None and codebook_name is not None:
        np.save(out_folder / f"{codebook_name}.npy", cb_pca2)

    n_latents = int(X_pca2.shape[0])
    n_codebook = int(cb_pca2.shape[0]) if cb_pca2 is not None else 0

    dead_mask = None
    dead_count = None
    dead_pct = None
    usage_arr = None
    if usage_counts is not None and cb_pca2 is not None:
        usage_arr = np.asarray(usage_counts)
        if usage_arr.shape[0] == cb_pca2.shape[0]:
            dead_mask = usage_arr == 0
            dead_count = int(dead_mask.sum())
            dead_pct = 100.0 * dead_count / max(1, usage_arr.shape[0])
        else:
            print(
                f"Usage counts size mismatch: usage_counts={usage_arr.shape[0]} "
                f"codebook_vectors={cb_pca2.shape[0]}; skipping dead codewords plot."
            )

    n_bins = 100
    fig = plt.figure(figsize=(24, 6))
    outer = fig.add_gridspec(1, 4, wspace=0.25)

    _joint_with_marginals(
        fig, outer[0], n_bins, X_pca2, cb2=None, dead=None, title="Latents only"
    )
    _joint_with_marginals(
        fig,
        outer[1],
        n_bins,
        X_pca2,
        cb2=cb_pca2,
        dead=None,
        title="Latents + codebook",
    )
    if cb_pca2 is not None and dead_mask is not None:
        alive_mask = ~dead_mask
        alive_count = int(alive_mask.sum())
        alive_pct = 100.0 * alive_count / max(1, usage_arr.shape[0])
        alive_title = f"Alive codewords: {alive_count} ({alive_pct:.2f}%)"
        _joint_with_marginals(
            fig,
            outer[2],
            n_bins,
            X_pca2,
            cb2=cb_pca2,
            dead=alive_mask,
            title=alive_title,
        )
        dead_title = f"Dead codewords: {dead_count} ({dead_pct:.2f}%)"
        _joint_with_marginals(
            fig, outer[3], n_bins, X_pca2, cb2=cb_pca2, dead=dead_mask, title=dead_title
        )
    else:
        _joint_with_marginals(
            fig,
            outer[2],
            n_bins,
            X_pca2,
            cb2=None,
            dead=None,
            title="Alive codewords: N/A",
        )
        _joint_with_marginals(
            fig,
            outer[3],
            n_bins,
            X_pca2,
            cb2=None,
            dead=None,
            title="Dead codewords: N/A",
        )

    fig.suptitle(f"PCA ({title_suffix}) | latents={n_latents} | codebook={n_codebook}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_file = out_folder / f"{filestem}.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print("Wrote PCA plot to", out_file)


def plot_pca_components(
    out_folder,
    X,
    codebook_vectors,
    random_seed,
    usage_counts=None,
    suffix: str = "",
):
    """Compute top-2 PCA on X and project codebook_vectors (if given). Save plots & arrays."""
    try:
        from sklearn.decomposition import PCA
    except Exception as e:
        print("Skipping PCA visualization (sklearn not available):", e)
        return

    try:
        n_comp = min(4, X.shape[1])
        pca = PCA(n_components=n_comp, random_state=random_seed)
        X_pca_full = pca.fit_transform(X)
    except Exception as e:
        print("PCA failed:", e)
        return

    suf = f"_{suffix}" if suffix else ""

    cb_pca_full = None
    if codebook_vectors is not None:
        try:
            cb_pca_full = pca.transform(codebook_vectors)
        except Exception as e:
            print("Failed to project codebook into PCA space:", e)

    if codebook_vectors is not None:
        filestem2 = f"latent_pca2_codebook{suf}"
        codebook_name2 = f"codebook_pca2{suf}"
    else:
        filestem2 = f"latent_pca2{suf}"
        codebook_name2 = None

    _plot_pca_component_pair(
        out_folder,
        X_pca_full,
        cb_pca_full,
        usage_counts,
        0,
        1,
        filestem2,
        codebook_name2,
        "PC1 vs PC2",
    )

    # If we have 4 components, prepare and plot PC3-4
    if n_comp >= 4:
        if codebook_vectors is not None:
            filestem34 = f"latent_pca34_codebook{suf}"
            codebook_name34 = f"codebook_pca34{suf}"
        else:
            filestem34 = f"latent_pca34{suf}"
            codebook_name34 = None
        _plot_pca_component_pair(
            out_folder,
            X_pca_full,
            cb_pca_full,
            usage_counts,
            2,
            3,
            filestem34,
            codebook_name34,
            "PC3 vs PC4",
        )


def compute_sampled_codebook_usage(
    latents: np.ndarray,
    sampled_codebook: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
):
    """Compute usage counts if sampled_codebook were used as the codebook."""
    if latents is None or sampled_codebook is None:
        return None
    if latents.size == 0 or sampled_codebook.size == 0:
        return None

    latents_t = torch.from_numpy(latents).to(device=device, dtype=torch.float32)
    codebook_t = torch.from_numpy(sampled_codebook).to(
        device=device, dtype=torch.float32
    )
    ncb = int(codebook_t.shape[0])
    counts = torch.zeros(ncb, dtype=torch.int64, device="cpu")

    with torch.no_grad():
        for start in range(0, latents_t.shape[0], batch_size):
            end = min(start + batch_size, latents_t.shape[0])
            batch = latents_t[start:end]
            d2 = torch.cdist(batch, codebook_t, p=2) ** 2
            idx = torch.argmin(d2, dim=1).cpu()
            counts += torch.bincount(idx, minlength=ncb)

    return counts.numpy()


def plot_codebook_usage(
    out_folder: Path,
    codebook_usage_counts: np.ndarray,
    sampled_codebook_usage_counts: np.ndarray = None,
    suffix: str = "",
):
    """Plot sorted usage counts for current and sampled codebooks on the same histogram."""
    if codebook_usage_counts is None:
        print("No codebook usage counts provided; skipping usage plot.")
        return

    import matplotlib.pyplot as plt

    def _usage_metrics(counts: np.ndarray):
        counts = np.asarray(counts).astype(np.int64)
        total = counts.sum()
        if total <= 0:
            return counts, 0, 0.0, 0.0
        p = counts.astype(np.float64) / total.astype(np.float64)
        p = np.clip(p, 1e-12, 1.0)
        entropy_nats = float(-(p * np.log(p)).sum())
        H_bits = float(-(p * np.log2(p)).sum())
        dead = int((counts == 0).sum())
        return counts, dead, entropy_nats, H_bits

    suf = f"_{suffix}" if suffix else ""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    counts, dead, entropy_nats, H_bits = _usage_metrics(codebook_usage_counts)
    total = counts.sum()
    p = counts.astype(np.float64) / total.astype(np.float64)
    p_sorted = np.sort(p)[::-1]
    ax.bar(range(len(p_sorted)), p_sorted, alpha=0.65, label="current codebook")
    # counts_sorted = np.sort(counts)[::-1]
    # ax.bar(range(len(counts_sorted)), counts_sorted, alpha=0.65, label="current codebook")
    current_title = (
        f"current: dead={dead}, H_bits={H_bits:.4f}"
    )

    if sampled_codebook_usage_counts is not None:
        sampled, s_dead, s_entropy, s_hbits = _usage_metrics(
            sampled_codebook_usage_counts
        )
        sampled_p = sampled.astype(np.float64) / float(sampled.sum())
        sampled_p_sorted = np.sort(sampled_p)[::-1]
        ax.bar(range(len(sampled_p_sorted)), sampled_p_sorted, alpha=0.55, label="sampled codebook")
        # sampled_sorted = np.sort(sampled)[::-1] / np.sum(sampled)
        # ax.bar(range(len(sampled_sorted)), sampled_sorted, alpha=0.55, label="sampled codebook")
        sampled_title = (
            f"sampled: dead={s_dead}, H_bits={s_hbits:.4f}"
        )
    else:
        sampled_title = None

    title_parts = ["Codebook usage (sorted frequencies)", current_title]
    if sampled_title is not None:
        title_parts.append(sampled_title)
    ax.set_title("\n".join(title_parts))
    ax.set_xlabel("N-th Most Used Codeword")
    ax.set_ylabel("Frequency")
    ax.legend(loc="best")
    out_file = out_folder / f"codebook_usage_hist{suf}.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print("Wrote codebook usage plot to", out_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", required=True, type=Path, help="File list, one path per line"
    )
    parser.add_argument("--config_path", required=True, help="WavTokenizer config yaml")
    parser.add_argument(
        "--model_path", required=True, type=Path, help="WavTokenizer checkpoint"
    )
    parser.add_argument(
        "--out_folder", default="./result/latent_viz", type=Path, help="Output folder"
    )
    parser.add_argument("--device", default="cuda:0", help="torch device")
    parser.add_argument("--max_vectors_per_file", type=int, default=200)
    parser.add_argument("--max_total_vectors", type=int, default=20000)
    parser.add_argument(
        "--pca_components",
        type=int,
        default=50,
        help="PCA dim before t-SNE (set 0 to skip)",
    )
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_iter", type=int, default=1000)
    parser.add_argument(
        "--tsne_with_codebook",
        action="store_true",
        help="Use codebook vectors in t-SNE",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--stop_at_global_cap",
        action="store_true",
        help="Stop processing files when global cap is reached",
    )
    args = parser.parse_args()
    random.seed(args.random_seed)

    device = torch.device(
        args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu"
    )
    foldername = "tsne_with_codebook" if args.tsne_with_codebook else "tsne_no_codebook"
    model_name = args.model_path.stem
    if model_name == "last":
        model_name = args.model_path.parent.parent.parent.parent.stem
    out_folder = args.out_folder / model_name / args.input_path.stem / foldername
    os.makedirs(out_folder, exist_ok=True)

    print("Loading model...")
    wavtokenizer = WavTokenizer.from_pretrained0802(args.config_path, args.model_path)
    wavtokenizer = wavtokenizer.to(device)
    wavtokenizer.eval()

    with open(args.input_path, "r") as f:
        files = [l.strip() for l in f if l.strip()]
    random.shuffle(files)

    X, y = collect_vectors(
        wavtokenizer,
        files,
        device,
        args.max_vectors_per_file,
        args.max_total_vectors,
        stop_at_global_cap=args.stop_at_global_cap,
    )
    if X is None:
        print("No vectors collected; exiting")
        return

    np.random.seed(args.random_seed)

    # optional PCA
    X_proc, pca = run_pca(X, args.pca_components, args.random_seed)

    # extract codebook vectors and project to same space
    codebook_vectors, codebook_proc = collect_codebook_vectors(wavtokenizer, pca)

    # compute usage counts for the first codebook (if possible)
    usage_counts = None
    try:
        first_vq_codebook = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[
            0
        ]._codebook
        if codebook_vectors is not None and first_vq_codebook is not None:
            ncb = int(first_vq_codebook.codebook_size)
            if codebook_vectors.shape[0] == ncb:
                counts = torch.zeros(ncb, dtype=torch.int64)
                device = first_vq_codebook.embed.device
                x_tensor = torch.from_numpy(X)
                with torch.no_grad():
                    for start in range(0, x_tensor.shape[0], 8192):
                        end = min(start + 8192, x_tensor.shape[0])
                        batch = x_tensor[start:end].to(
                            device=device, dtype=torch.float32
                        )
                        idx = first_vq_codebook.quantize(batch).reshape(-1).cpu()
                        counts += torch.bincount(idx, minlength=ncb)
                usage_counts = counts.numpy()
            else:
                print(
                    f"Skipping usage counts: codebook_vectors={codebook_vectors.shape[0]} "
                    f"codebook_size={ncb}."
                )
    except Exception as e:
        print("Failed to compute usage counts for codebook:", e)

    # Plot top-2 PCA components for latents and codebook (saves arrays and PNG)
    plot_pca_components(
        out_folder, X, codebook_vectors, args.random_seed, usage_counts=usage_counts
    )
    plot_pca_components(out_folder, X, None, args.random_seed, usage_counts=None)

    # X_tsne, codebook_tsne = run_tsne(
    #     X_proc,
    #     codebook_proc,
    #     args.tsne_perplexity,
    #     args.tsne_iter,
    #     args.random_seed,
    #     use_codebook=args.tsne_with_codebook,
    # )
    # if X_tsne is None:
    #     return

    # # save embeddings
    # np.save(out_folder / "latent_vectors.npy", X)
    # np.save(out_folder / "latent_labels.npy", y)
    # np.save(out_folder / "latent_tsne.npy", X_tsne)
    # if codebook_vectors is not None:
    #     np.save(out_folder / "codebook_vectors.npy", codebook_vectors)
    #     if "codebook_tsne" in locals() and codebook_tsne is not None:
    #         np.save(out_folder / "codebook_tsne.npy", codebook_tsne)

    # plot_latent_tsne(
    #     out_folder, X_tsne, codebook_tsne, codebook_vectors, y, plot_codebook=True
    # )
    # plot_latent_tsne(
    #     out_folder, X_tsne, codebook_tsne, codebook_vectors, y, plot_codebook=False
    # )

    # If first vq._codebook has a nerd_sampler, sample additional codebook vectors
    try:
        first_vq_codebook = wavtokenizer.feature_extractor.encodec.quantizer.vq.layers[
            0
        ]._codebook
    except Exception:
        first_vq_codebook = None

    if first_vq_codebook is not None and hasattr(first_vq_codebook, "nerd_sampler"):
        try:
            print(first_vq_codebook.nerd_config)
            # determine number of codebook vectors from model's codebook
            ncb = first_vq_codebook.codebook_size
            assert (
                ncb == codebook_vectors.shape[0]
            ), f"Mismatch in codebook vector count {ncb=}, {codebook_vectors.shape[0]=}"

            sampled = first_vq_codebook.nerd_sampler.sample(ncb)
            print(f"{first_vq_codebook.nerd_sampler.dec.log_sigma=}")
            if isinstance(sampled, torch.Tensor):
                sampled_np = sampled.detach().cpu().numpy()
            else:
                sampled_np = np.asarray(sampled)

            # save sampled vectors
            np.save(out_folder / f"nerd_sampled_codebook_vectors_{ncb}.npy", sampled_np)

            # compute usage counts if sampled vectors were the codebook
            sampled_usage_counts = compute_sampled_codebook_usage(
                X, sampled_np, device=first_vq_codebook.embed.device
            )
            if sampled_usage_counts is not None:
                np.save(
                    out_folder / f"nerd_sampled_codebook_usage_{ncb}.npy",
                    sampled_usage_counts,
                )

            # PCA plot with sampled codebook (use suffix to avoid overwriting)
            plot_pca_components(
                out_folder,
                X,
                sampled_np,
                args.random_seed,
                usage_counts=sampled_usage_counts,
                suffix=f"nerd{ncb}",
            )

            # plot NERD RD metrics based on sampled latents
            plot_nerd_metrics(first_vq_codebook, torch.from_numpy(X), out_folder)

            plot_codebook_usage(
                out_folder,
                usage_counts,
                sampled_codebook_usage_counts=sampled_usage_counts,
                suffix=f"nerd{ncb}",
            )

            # # prepare proc for t-SNE
            # if pca is not None:
            #     sampled_proc = pca.transform(sampled_np)
            # else:
            #     sampled_proc = sampled_np

            # # run t-SNE including sampled codebook
            # X_tsne_s, codebook_tsne_s = run_tsne(
            #     X_proc,
            #     sampled_proc,
            #     args.tsne_perplexity,
            #     args.tsne_iter,
            #     args.random_seed,
            #     use_codebook=True,
            # )
            # if X_tsne_s is not None:
            #     # save embeddings with suffix
            #     np.save(out_folder / f"latent_tsne_nerd{ncb}.npy", X_tsne_s)
            #     if codebook_tsne_s is not None:
            #         np.save(
            #             out_folder / f"codebook_tsne_nerd{ncb}.npy", codebook_tsne_s
            #         )

            #     plot t-SNE with sampled codebook
            #     plot_latent_tsne(
            #         out_folder,
            #         X_tsne_s,
            #         codebook_tsne_s,
            #         sampled_np,
            #         y,
            #         plot_codebook=True,
            #         suffix=f"nerd{ncb}",
            #     )
            #     plot_latent_tsne(
            #         out_folder,
            #         X_tsne_s,
            #         codebook_tsne_s,
            #         sampled_np,
            #         y,
            #         plot_codebook=False,
            #         suffix=f"nerd{ncb}",
            #     )

        except Exception as e:
            print("Failed to sample or plot nerd_sampler codebook vectors:", e)


if __name__ == "__main__":
    main()
