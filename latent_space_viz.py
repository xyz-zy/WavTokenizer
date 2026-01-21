"""
Generate a t-SNE visualization of encoder latent vectors (pre-quantization).

Usage example:
python latent_space_viz.py \
  --input_path ./test-clean_filelist.txt \
  --config_path ../WavTokenizer_models/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
  --model_path ../WavTokenizer_models/WavTokenizer_small_600_24k_4096.ckpt \
  --out_folder ./result/tsne --max_vectors_per_file 10 --max_total_vectors 20000 --tsne_with_codebook

python latent_space_viz.py \
  --input_path ./test_filelist.txt \
  --config_path ../WavTokenizer_models/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
  --model_path ../WavTokenizer_models/WavTokenizer_small_600_24k_4096.ckpt \
  --out_folder ./result/tsne --max_vectors_per_file 10 --max_total_vectors 20000 --tsne_with_codebook

python latent_space_viz.py \
  --input_path ./test-clean_filelist.txt \
  --config_path ../WavTokenizer_models/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
  --model_path ../WavTokenizer_models/WavTokenizer_small_320_24k_4096.ckpt \
  --out_folder ./result/tsne --max_vectors_per_file 10 --max_total_vectors 20000 --tsne_with_codebook

python latent_space_viz.py \
  --input_path ./test_filelist.txt \
  --config_path ../WavTokenizer_models/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml \
  --model_path ../WavTokenizer_models/WavTokenizer_small_320_24k_4096.ckpt \
  --out_folder ./result/tsne --max_vectors_per_file 10 --max_total_vectors 20000 --tsne_with_codebook
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from decoder.pretrained import WavTokenizer
from sklearn.manifold import TSNE
from tqdm import tqdm


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
        # emb: (B, C, L)
        emb_np = emb.detach().cpu().numpy()
        B, C, L = emb_np.shape
        frames = emb_np.transpose(0, 2, 1).reshape(-1, C)  # (B*L, C)

        # subsample frames for this file
        n = frames.shape[0]
        if n > max_vectors_per_file:
            idxs = np.random.choice(n, max_vectors_per_file, replace=False)
            sel = frames[idxs]
        else:
            sel = frames

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
        if hasattr(vq, "project_out"):
            with torch.no_grad():
                cb_proj = vq.project_out(cb)
        else:
            cb_proj = cb
        cb_list.append(cb_proj.numpy())
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
    out_folder, X_tsne, codebook_tsne, codebook_vectors, y, plot_codebook: bool
):
    if plot_codebook:
        output_file = out_folder / "latent_tsne_codebook.png"
    else:
        output_file = out_folder / "latent_tsne.png"
    import matplotlib.pyplot as plt

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


def plot_pca_components(out_folder, X, codebook_vectors, random_seed):
    """Compute top-2 PCA on X and project codebook_vectors (if given). Save plots & arrays."""
    try:
        from sklearn.decomposition import PCA
    except Exception as e:
        print("Skipping PCA-2 visualization (sklearn not available):", e)
        return

    try:
        pca2 = PCA(n_components=2, random_state=random_seed)
        X_pca = pca2.fit_transform(X)
    except Exception as e:
        print("PCA-2 failed:", e)
        return

    if codebook_vectors is not None:
        output_filestem = "latent_pca2_codebook"
    else:
        output_filestem = "latent_pca2"

    np.save(out_folder / f"{output_filestem}.npy", X_pca)

    import matplotlib.pyplot as plt

    n_bins = 100

    # joint scatter with marginal histograms
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4), hspace=0.05, wspace=0.05
    )
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_histy = fig.add_subplot(gs[1, 1])

    ax_main.scatter(
        X_pca[:, 0], X_pca[:, 1], c="tab:blue", s=6, alpha=0.6, label="latents"
    )

    cb_pca = None
    if codebook_vectors is not None:
        try:
            cb_pca = pca2.transform(codebook_vectors)
            np.save(out_folder / "codebook_pca2.npy", cb_pca)
            ax_main.scatter(
                cb_pca[:, 0],
                cb_pca[:, 1],
                c="tab:orange",
                s=24,
                alpha=0.95,
                edgecolors="k",
                label="codebook",
            )
        except Exception as e:
            print("Failed to project codebook into PCA-2 space:", e)

    ax_main.set_xlabel("PC 1")
    ax_main.set_ylabel("PC 2")
    ax_main.legend()

    # top marginal (x)
    ax_histx.hist(X_pca[:, 0], bins=n_bins, color="tab:blue", alpha=0.6, density=True)
    if cb_pca is not None:
        ax_histx.hist(
            cb_pca[:, 0], bins=n_bins, color="tab:orange", alpha=0.6, density=True
        )
    ax_histx.axis("off")

    # right marginal (y)
    ax_histy.hist(
        X_pca[:, 1],
        bins=n_bins,
        orientation="horizontal",
        color="tab:blue",
        alpha=0.6,
        density=True,
    )
    if cb_pca is not None:
        ax_histy.hist(
            cb_pca[:, 1],
            bins=n_bins,
            orientation="horizontal",
            color="tab:orange",
            alpha=0.6,
            density=True,
        )
    ax_histy.axis("off")

    plt.suptitle("Top-2 PCA of encoder latents (pre-quant) with codebook overlay")
    out_file = out_folder / f"{output_filestem}.png"
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print("Wrote PCA-2 plot to", out_file)


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
        "--out_folder", default="./result/tsne", type=Path, help="Output folder"
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

    device = torch.device(
        args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu"
    )
    foldername = "tsne_with_codebook" if args.tsne_with_codebook else "tsne_no_codebook"
    out_folder = (
        args.out_folder / args.model_path.stem / args.input_path.stem / foldername
    )
    os.makedirs(out_folder, exist_ok=True)

    print("Loading model...")
    wavtokenizer = WavTokenizer.from_pretrained0802(args.config_path, args.model_path)
    wavtokenizer = wavtokenizer.to(device)
    wavtokenizer.eval()

    with open(args.input_path, "r") as f:
        files = [l.strip() for l in f if l.strip()]

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

    # Plot top-2 PCA components for latents and codebook (saves arrays and PNG)
    plot_pca_components(out_folder, X, codebook_vectors, args.random_seed)
    plot_pca_components(out_folder, X, None, args.random_seed)

    X_tsne, codebook_tsne = run_tsne(
        X_proc,
        codebook_proc,
        args.tsne_perplexity,
        args.tsne_iter,
        args.random_seed,
        use_codebook=args.tsne_with_codebook,
    )
    if X_tsne is None:
        return

    # save embeddings
    np.save(out_folder / "latent_vectors.npy", X)
    np.save(out_folder / "latent_labels.npy", y)
    np.save(out_folder / "latent_tsne.npy", X_tsne)
    if codebook_vectors is not None:
        np.save(out_folder / "codebook_vectors.npy", codebook_vectors)
        if "codebook_tsne" in locals() and codebook_tsne is not None:
            np.save(out_folder / "codebook_tsne.npy", codebook_tsne)

    plot_latent_tsne(
        out_folder, X_tsne, codebook_tsne, codebook_vectors, y, plot_codebook=True
    )
    plot_latent_tsne(
        out_folder, X_tsne, codebook_tsne, codebook_vectors, y, plot_codebook=False
    )


if __name__ == "__main__":
    main()
