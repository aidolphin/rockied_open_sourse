#!/bin/bash
# Download Kaggle rock dataset, prepare it and start training
# Usage: ./download_and_train.sh [kaggle_dataset_slug] [epochs]
# Example: ./download_and_train.sh salmaneunus/rock-classification 8

set -euo pipefail

DATASET_SLUG=${1:-salmaneunus/rock-classification}
EPOCHS=${2:-8}
WORKDIR=$(dirname "$0")/../data
DEST_PREPARED=$WORKDIR/rock-classification-prepared

echo "Dataset slug: $DATASET_SLUG"
echo "Working dir: $WORKDIR"

mkdir -p "$WORKDIR"

echo "Downloading dataset via kaggle CLI (requires ~/.kaggle/kaggle.json) ..."
kaggle datasets download -d "$DATASET_SLUG" -p "$WORKDIR" --unzip

# The downloaded folder name may vary; try to find a sensible folder
echo "Searching for downloaded dataset folder..."
DS_FOLDER=$(find "$WORKDIR" -maxdepth 1 -type d -iname "*rock*" -print -quit || true)
if [ -z "$DS_FOLDER" ]; then
  echo "Could not locate the unzipped dataset folder under $WORKDIR"
  echo "Please inspect $WORKDIR and rerun the script with the dataset present."
  exit 1
fi

echo "Found dataset folder: $DS_FOLDER"

echo "Preparing dataset into ImageFolder format (train/ and val/) ..."
python3 "$WORKDIR/../prepare_dataset.py" --source "$DS_FOLDER" --dest "$DEST_PREPARED" --val-split 0.2

echo "Starting training (this may be slow on CPU). Epochs=$EPOCHS"
python3 "$WORKDIR/../train.py" --dataset "$DEST_PREPARED" --out ../models/rock_cnn.pth --epochs $EPOCHS --batch 16

echo "Done. Trained model saved to backend/models/rock_cnn.pth"
