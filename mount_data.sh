#!/bin/bash
# Mount the LibriTTS GCS bucket at /mnt/gcs/libritts.
# Run this on each new VM before training.

set -e
BUCKET="nerd-vqvae-libritts"
MOUNT_DIR="/mnt/gcs/libritts"

sudo mkdir -p "$MOUNT_DIR"
sudo chown "$USER" "$MOUNT_DIR"
gcsfuse --implicit-dirs "$BUCKET" "$MOUNT_DIR"
echo "Mounted gs://$BUCKET at $MOUNT_DIR"
