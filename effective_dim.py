"""Compute effective dimensionality (99% variance) of encoder latents and codebook."""

import random
import numpy as np
import torch
import torchaudio
from pathlib import Path
from decoder.pretrained import WavTokenizer
from encoder.utils import convert_audio

CONFIG = "checkpoints/config_frame75.yaml"
CKPT = "checkpoints/WavTokenizer_small_320_24k_4096.ckpt"
DATA_ROOT = Path("/home/lxz/data/LibriTTS/test-clean")
N_FILES = 200
SEED = 42
THRESHOLD = 0.99

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
print("Loading model...")
model = WavTokenizer.from_pretrained0802(CONFIG, CKPT).to(device)

# --- Encoder latents ---
print(f"Sampling encoder latents from {N_FILES} random files...")
random.seed(SEED)
all_wavs = sorted(DATA_ROOT.rglob("*.wav"))
sampled = random.sample(all_wavs, min(N_FILES, len(all_wavs)))

latents = []
with torch.inference_mode():
    for p in sampled:
        wav, sr = torchaudio.load(p)
        wav = convert_audio(wav, sr, 24000, 1).to(device)
        audio = wav.unsqueeze(1)  # (1, 1, L)
        emb = model.feature_extractor.encodec.encoder(audio)  # (1, 512, T)
        # Collect all time-step vectors
        latents.append(emb.squeeze(0).T.cpu().numpy())  # (T, 512)

latents = np.concatenate(latents, axis=0)  # (total_T, 512)
print(f"Collected {latents.shape[0]} latent vectors, dim={latents.shape[1]}")

# PCA via SVD
latents_centered = latents - latents.mean(axis=0, keepdims=True)
_, S_lat, _ = np.linalg.svd(latents_centered, full_matrices=False)
var_lat = S_lat ** 2
cumvar_lat = np.cumsum(var_lat) / np.sum(var_lat)
eff_dim_lat = int(np.searchsorted(cumvar_lat, THRESHOLD)) + 1

print(f"\n=== Encoder Latent Space ===")
print(f"Total dimension: {latents.shape[1]}")
print(f"Effective dimension ({THRESHOLD*100:.0f}% variance): {eff_dim_lat}")
print(f"Top-10 cumulative variance: {cumvar_lat[:10].round(4)}")

# --- Codebook ---
print(f"\n--- Codebook ---")
with torch.inference_mode():
    codebook = model.feature_extractor.encodec.quantizer.vq.layers[0].codebook.cpu().numpy()

print(f"Codebook shape: {codebook.shape} (num_entries, dim)")

cb_centered = codebook - codebook.mean(axis=0, keepdims=True)
_, S_cb, _ = np.linalg.svd(cb_centered, full_matrices=False)
var_cb = S_cb ** 2
cumvar_cb = np.cumsum(var_cb) / np.sum(var_cb)
eff_dim_cb = int(np.searchsorted(cumvar_cb, THRESHOLD)) + 1

print(f"\n=== Codebook ===")
print(f"Num entries: {codebook.shape[0]}, dimension: {codebook.shape[1]}")
print(f"Effective dimension ({THRESHOLD*100:.0f}% variance): {eff_dim_cb}")
print(f"Top-10 cumulative variance: {cumvar_cb[:10].round(4)}")

# Summary
print(f"\n=== Summary ===")
print(f"Encoder latent effective dim: {eff_dim_lat} / {latents.shape[1]}")
print(f"Codebook effective dim:       {eff_dim_cb} / {codebook.shape[1]}")
