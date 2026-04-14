"""Smoke-test: load WavTokenizer, encode a short audio clip, decode it back."""

import torch
import torchaudio
from encoder.utils import convert_audio
from decoder.pretrained import WavTokenizer

CONFIG = "checkpoints/config_frame75.yaml"
CKPT   = "checkpoints/WavTokenizer_small_320_24k_4096.ckpt"
AUDIO  = "checkpoints/test_tone.wav"
OUT    = "checkpoints/test_tone_reconstructed.wav"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# --- Load model ---
print("Loading model ...")
model = WavTokenizer.from_pretrained0802(CONFIG, CKPT)
model = model.to(device)
print("Model loaded.")

# --- Encode ---
wav, sr = torchaudio.load(AUDIO)
wav = convert_audio(wav, sr, 24000, 1).to(device)
print(f"Input shape: {wav.shape}  (samples={wav.shape[-1]}, duration={wav.shape[-1]/24000:.2f}s)")

bandwidth_id = torch.tensor([0], device=device)
features, codes = model.encode_infer(wav, bandwidth_id=bandwidth_id)
print(f"Encoded  -> features {features.shape}, codes {codes.shape}")

# --- Decode ---
audio_out = model.decode(features, bandwidth_id=bandwidth_id)
print(f"Decoded  -> audio_out {audio_out.shape}")

torchaudio.save(OUT, audio_out.cpu(), 24000)
print(f"Saved reconstructed audio to {OUT}")

# --- Round-trip from codes ---
features2 = model.codes_to_features(codes)
audio_out2 = model.decode(features2, bandwidth_id=bandwidth_id)
print(f"Codes->features->decode -> {audio_out2.shape}")

# --- No-VQ path (encoder -> decoder, bypassing quantization) ---
raw_features = model.encode_infer_no_vq(wav)
print(f"Raw encoder features -> {raw_features.shape}")
audio_no_vq = model.decode(raw_features, bandwidth_id=bandwidth_id)
print(f"No-VQ decode -> {audio_no_vq.shape}")
torchaudio.save("checkpoints/test_tone_no_vq.wav", audio_no_vq.cpu(), 24000)
print(f"Saved no-VQ reconstructed audio to checkpoints/test_tone_no_vq.wav")

print("All checks passed!")
