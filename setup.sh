#!/bin/bash
# Setup script for WavTokenizer on a GCP instance.
# Run from the repo root: bash setup.sh
# See SETUP.md for full details and troubleshooting.

set -e
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="/mnt/gcs/libritts/LibriTTS"
GCS_BUCKET="nerd-vqvae-libritts"
GCS_MOUNT="/mnt/gcs/libritts"
ENV_DIR="$REPO_DIR/env"
MINICONDA_DIR="$HOME/miniconda3"

echo "=== WavTokenizer GCP Setup ==="
echo "Repo: $REPO_DIR"
echo "Env:  $ENV_DIR"
echo ""

# ── 1. Miniconda ──────────────────────────────────────────────────────────────
if [ ! -f "$MINICONDA_DIR/bin/conda" ]; then
    echo "[1/6] Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
    rm /tmp/miniconda.sh
else
    echo "[1/6] Miniconda already installed at $MINICONDA_DIR"
fi

source "$MINICONDA_DIR/etc/profile.d/conda.sh"

# Accept Anaconda ToS (required on first install)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# ── 2. Conda environment ──────────────────────────────────────────────────────
if [ ! -f "$ENV_DIR/bin/python" ]; then
    echo "[2/6] Creating conda environment at $ENV_DIR..."
    conda create --prefix "$ENV_DIR" python=3.9 -y
else
    echo "[2/6] Conda environment already exists at $ENV_DIR"
fi

PIP="$ENV_DIR/bin/pip"

# ── 3. Python packages ────────────────────────────────────────────────────────
echo "[3/6] Installing Python packages..."

# PyTorch with CUDA 11.8
echo "  Installing PyTorch (CUDA 11.8)..."
$PIP install torch==2.0.0 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# requirements.txt (pinned packages, excluding fairseq/torchcrepe/librosa/pesq)
echo "  Installing pinned requirements..."
$PIP install \
    scipy==1.10.1 \
    einops==0.6.1 \
    pyyaml==6.0 \
    huggingface_hub==0.23.0 \
    encodec==0.1.1 \
    matplotlib==3.7.1 \
    transformers==4.28.1 \
    pytorch-lightning==1.8.6 \
    tensorboardX==2.6 \
    soundfile==0.12.1 \
    numpy==1.23.5 \
    "jsonargparse[signatures]>=4.15.2"

# fairseq: PyPI releases are broken (missing version.txt), install from GitHub
echo "  Installing fairseq from GitHub (PyPI releases broken)..."
$PIP install "fairseq @ git+https://github.com/facebookresearch/fairseq.git" --no-deps

# fairseq dependencies — omegaconf 2.0.5-2.0.6 have invalid metadata in pip>=24.1,
# so we install 2.1.1 with --no-deps and handle hydra-core similarly
echo "  Installing fairseq dependencies (omegaconf workaround)..."
$PIP install omegaconf==2.1.1 --no-deps
$PIP install antlr4-python3-runtime==4.8
$PIP install portalocker lxml tabulate colorama
$PIP install bitarray cython sacrebleu "hydra-core==1.0.7" --no-deps

# Remaining unpinned requirements
echo "  Installing torchcrepe, librosa, pesq..."
$PIP install torchcrepe librosa pesq

# Extra tools
echo "  Installing tensorboard, jupyterlab, nvitop..."
$PIP install tensorboard jupyterlab nvitop

# nvidia-nccl-cu11: overrides GCP's gIB NCCL (see SETUP.md section 5)
echo "  Installing nvidia-nccl-cu11 (GCP NCCL fix)..."
$PIP install nvidia-nccl-cu11

# ── 4. gcsfuse + mount GCS bucket ─────────────────────────────────────────────
echo "[4/7] Installing gcsfuse and mounting GCS bucket..."

if ! command -v gcsfuse &>/dev/null; then
    echo "  Installing gcsfuse..."
    export GCSFUSE_REPO="gcsfuse-$(lsb_release -c -s)"
    echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" \
        | sudo tee /etc/apt/sources.list.d/gcsfuse.list
    curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt-get update -q && sudo apt-get install -y gcsfuse
else
    echo "  gcsfuse already installed"
fi

sudo mkdir -p "$GCS_MOUNT"
sudo chown "$USER" "$GCS_MOUNT"
if mountpoint -q "$GCS_MOUNT"; then
    echo "  $GCS_MOUNT already mounted"
else
    gcsfuse --implicit-dirs "$GCS_BUCKET" "$GCS_MOUNT"
    echo "  Mounted gs://$GCS_BUCKET at $GCS_MOUNT"
fi

# ── 5. Data filelists ─────────────────────────────────────────────────────────
echo "[5/7] Updating data filelists to point to $DATA_DIR..."
for f in "$REPO_DIR/train_filelist.txt" "$REPO_DIR/dev_filelist.txt" "$REPO_DIR/test_filelist.txt"; do
    if [ -f "$f" ]; then
        "$ENV_DIR/bin/python3" -c "
import re, sys
path = sys.argv[1]
data_dir = sys.argv[2]
with open(path) as f:
    content = f.read()
# Replace any absolute path prefix before the split name (train-*, dev-*, test-*)
updated = re.sub(r'^/.+?/(train-|dev-|test-)', lambda m: data_dir + '/' + m.group(1), content, flags=re.MULTILINE)
with open(path, 'w') as f:
    f.write(updated)
print(f'  Updated {path}')
" "$f" "$DATA_DIR"
    else
        echo "  Warning: $f not found, skipping"
    fi
done

# ── 6. activate script ────────────────────────────────────────────────────────
echo "[6/7] Writing activate script..."

# Detect the primary non-loopback network interface
IFACE=$(ip link show | awk '/^[0-9]+: /{iface=$2} /link\/ether/{print iface}' | head -1 | tr -d ':')
if [ -z "$IFACE" ]; then
    IFACE="ens7"
    echo "  Warning: could not detect network interface, defaulting to ens7"
else
    echo "  Detected network interface: $IFACE"
fi

NCCL_LIB="$ENV_DIR/lib/python3.9/site-packages/nvidia/nccl/lib"

cat > "$REPO_DIR/activate" << EOF
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $ENV_DIR
export NCCL_SOCKET_IFNAME=$IFACE
export NCCL_IB_DISABLE=1
# Use NVIDIA's official NCCL and strip GCP's gIB paths from LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_LIB:\$(python3 -c "import os; print(':'.join(p for p in os.environ.get('LD_LIBRARY_PATH','').split(':') if p and 'gib' not in p))")
EOF

echo "  Written to $REPO_DIR/activate"

# ── 7. Verify ─────────────────────────────────────────────────────────────────
echo "[7/7] Verifying NCCL version..."
source "$REPO_DIR/activate"
"$ENV_DIR/bin/python3" -c "
import ctypes
lib = ctypes.CDLL('libnccl.so.2', mode=ctypes.RTLD_GLOBAL)
v = ctypes.c_int()
lib.ncclGetVersion(ctypes.byref(v))
val = v.value
version = f'{val//10000}.{(val//100)%100}.{val%100}'
print(f'  NCCL version: {version}')
if val == 22705:
    print('  WARNING: gIB NCCL (2.27.5) is still loading! Check LD_LIBRARY_PATH.')
else:
    print('  OK: NVIDIA official NCCL is loaded (not gIB).')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the environment:"
echo "  source activate"
echo ""
echo "To run training:"
echo "  NCCL_NET=Socket NCCL_DEBUG=INFO python3 train.py fit --config configs/<your_config>.yaml"
echo ""
echo "See SETUP.md for full documentation."
