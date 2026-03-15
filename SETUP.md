# GCP Instance Setup for WavTokenizer

## Overview

This documents everything required to run WavTokenizer training on a GCP instance.
The main complication is GCP's custom NCCL library (Google Internal Backend / gIB)
which conflicts with PyTorch's multi-GPU training.

---

## 1. Miniconda

Installed to `~/miniconda3` (user home, not `/opt` — no root needed).

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3
```

---

## 2. Conda Environment

Created at `./env` (relative to the repo root) with Python 3.9.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create --prefix ./env python=3.9 -y
```

---

## 3. Package Installation

### PyTorch (CUDA 11.8)
```bash
./env/bin/pip install torch==2.0.0 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

### requirements.txt packages
```bash
./env/bin/pip install scipy==1.10.1 einops==0.6.1 pyyaml==6.0 huggingface_hub==0.23.0 \
    encodec==0.1.1 matplotlib==3.7.1 transformers==4.28.1 pytorch-lightning==1.8.6 \
    tensorboardX==2.6 soundfile==0.12.1 numpy==1.23.5 "jsonargparse[signatures]>=4.15.2"
```

### fairseq (PyPI releases are broken — install from GitHub)
```bash
./env/bin/pip install "fairseq @ git+https://github.com/facebookresearch/fairseq.git" --no-deps
```

### fairseq dependencies (worked around omegaconf metadata bug in pip>=24.1)
```bash
./env/bin/pip install omegaconf==2.1.1 --no-deps
./env/bin/pip install antlr4-python3-runtime==4.8
./env/bin/pip install portalocker lxml tabulate colorama
./env/bin/pip install bitarray cython sacrebleu hydra-core==1.0.7 --no-deps
```

### Remaining requirements.txt packages
```bash
./env/bin/pip install torchcrepe librosa pesq
```

### Extra tools
```bash
./env/bin/pip install tensorboard jupyterlab nvitop
```

### nvidia-nccl-cu11 — critical for GCP (see section 5)
```bash
./env/bin/pip install nvidia-nccl-cu11
```

---

## 4. Data Paths

LibriTTS data lives in GCS bucket `gs://nerd-vqvae-libritts` (project `nerd-vqvae`, region `us-east4`).
Mount it on each VM before training (see section 10).

Filelists point to `/mnt/gcs/libritts/LibriTTS/` with splits:
`train-clean-360`, `dev-clean`, `test-clean`, `train-other-500`, etc.

---

## 5. GCP NCCL Issue and Fix

### The problem

GCP instances have Google's Internal Backend (gIB) NCCL installed at `/usr/local/gib/lib64/`.
This is registered in the system ldconfig cache, so `dlopen('libnccl.so.2')` finds gIB's NCCL
(v2.27.5) even with an empty `LD_LIBRARY_PATH`. gIB requires Google's internal RDMA network
fabric, which is not available on standard VM instances, causing:

```
torch.distributed.DistBackendError: NCCL error ... internal error
Last error: Error: network gIB not found.
```

The `NCCL version 2.14.3` in the error message is a red herring — it's PyTorch's
compile-time version string, not the runtime version.

### The fix

1. Install NVIDIA's official NCCL for CUDA 11: `pip install nvidia-nccl-cu11` (v2.21.5)
2. In `activate`, prepend it to `LD_LIBRARY_PATH` (takes precedence over ldconfig cache)
   and strip gIB paths from the rest of `LD_LIBRARY_PATH`
3. Set `NCCL_IB_DISABLE=1` and `NCCL_SOCKET_IFNAME=ens7` so NCCL uses the ethernet
   interface for inter-GPU communication instead of InfiniBand
4. At training time, additionally set `NCCL_NET=Socket` and `NCCL_DEBUG=INFO`

### Verify the fix

After `source activate`:
```bash
python3 -c "
import ctypes
lib = ctypes.CDLL('libnccl.so.2', mode=ctypes.RTLD_GLOBAL)
v = ctypes.c_int()
lib.ncclGetVersion(ctypes.byref(v))
val = v.value
print(f'NCCL {val//10000}.{(val//100)%100}.{val%100}')
"
# Should print NCCL 2.21.5, NOT 2.27.5 (which would be gIB)
```

---

## 6. The `activate` Script

Located at `./activate`. Always `source` it before working:

```bash
source activate
```

Contents:
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/ssd/WavTokenizer/env
export NCCL_SOCKET_IFNAME=ens7
export NCCL_IB_DISABLE=1
# Use NVIDIA's official NCCL and strip GCP's gIB paths from LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/mnt/ssd/WavTokenizer/env/lib/python3.9/site-packages/nvidia/nccl/lib:$(python3 -c "import os; print(':'.join(p for p in os.environ.get('LD_LIBRARY_PATH','').split(':') if p and 'gib' not in p))")
```

**Note:** The network interface name may differ on other GCP instances. Check with `ip link show`.
If the interface is not `ens7`, update `NCCL_SOCKET_IFNAME` accordingly.

---

## 7. Training Command

```bash
source activate
NCCL_NET=Socket NCCL_DEBUG=INFO python3 train.py fit \
    --config configs/wavtokenizer_smalldata_frame75_3s_nq1_novq_dim512_kmeans200_attn_2gpu_code4096_offline_sklearn_40960_test.yaml
```

`NCCL_DEBUG=INFO` can be removed once confirmed working — it produces verbose output.

---

## 8. Notes on DDP Configuration

The config uses `strategy: ddp` (standard NCCL DDP, matching the original SLURM setup).
No `find_unused_parameters` needed — NCCL's DDP is lenient about parameters that don't
receive gradients on every step (expected behavior: discriminator uses `no_grad` on the
generator, and the VQ uses variable `n_q` per step).

---

## 9. GPU Memory

On a 2× A100-80GB setup, batch_size=40 requires the NCCL setup above to work correctly.
If you see CUDA OOM after fixing NCCL, consider:
- Reducing `batch_size` in the config (currently 40)
- Adding `precision: bf16` to the trainer config (saves ~30-40% activation memory)

---

## 10. GCS Data Bucket

LibriTTS (90 GB) is stored in `gs://nerd-vqvae-libritts` (project `nerd-vqvae`, region `us-east4`).
This lets any new VM access the data without copying it.

### Mount on a new VM

```bash
bash mount_data.sh
```

Or manually:

```bash
sudo mkdir -p /mnt/gcs/libritts
sudo chown $USER /mnt/gcs/libritts
gcsfuse --implicit-dirs nerd-vqvae-libritts /mnt/gcs/libritts
```

**Note:** gcsfuse mounts are not persistent across reboots. Re-run `mount_data.sh` after each reboot.

### gcsfuse install (if not already installed)

```bash
export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" \
    | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install -y gcsfuse
```

### Re-uploading data (if needed)

```bash
gsutil -m rsync -r /path/to/LibriTTS gs://nerd-vqvae-libritts/LibriTTS
```
