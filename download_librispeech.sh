#!/bin/bash
set -e

# Where to download and extract
DEST="/mnt/ssd/data/librispeech"
mkdir -p "$DEST"
cd "$DEST"

BASE_URL="https://www.openslr.org/resources/60"

# Files and their expected md5sums
declare -A FILES=(
  ["dev-clean.tar.gz"]="0c3076c1e5245bb3f0af7d82087ee207"
  ["dev-other.tar.gz"]="815555d8d75995782ac3ccd7f047213d"
  ["test-clean.tar.gz"]="7bed3bdb047c4c197f1ad3bc412db59f"
  ["test-other.tar.gz"]="ae3258249472a13b5abef2a816f733e4"
  ["train-clean-360.tar.gz"]="a84ef10ddade5fd25df69596a2767b2d"
  ["train-other-500.tar.gz"]="7b181dd5ace343a5f38427999684aa6f"
)

for FILE in "${!FILES[@]}"; do
  EXPECTED_MD5="${FILES[$FILE]}"

  # Skip if already downloaded and checksum matches
  if [ -f "$FILE" ]; then
    ACTUAL_MD5=$(md5sum "$FILE" | awk '{print $1}')
    if [ "$ACTUAL_MD5" == "$EXPECTED_MD5" ]; then
      echo "✓ $FILE already downloaded, skipping"
    else
      echo "✗ $FILE checksum mismatch, re-downloading"
      rm "$FILE"
      wget "$BASE_URL/$FILE"
    fi
  else
    echo "Downloading $FILE..."
    wget "$BASE_URL/$FILE"
  fi

  # Verify checksum
  ACTUAL_MD5=$(md5sum "$FILE" | awk '{print $1}')
  if [ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]; then
    echo "ERROR: $FILE checksum failed! Expected $EXPECTED_MD5, got $ACTUAL_MD5"
    exit 1
  fi

  # Extract
  echo "Extracting $FILE..."
  tar xzf "$FILE"

  # Remove tar.gz to save space
  rm "$FILE"
  echo "✓ $FILE done"
  echo ""
done

echo "All downloads complete! Data is in $DEST"
