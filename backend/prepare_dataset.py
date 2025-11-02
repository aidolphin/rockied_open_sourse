#!/usr/bin/env python3
"""
Prepare a dataset folder for training in PyTorch ImageFolder format.

This script expects the source directory to contain one subdirectory per class
with images inside each class folder. It will create `dest/train/<class>` and
`dest/val/<class>` with a randomized split.

If your Kaggle dataset has a different structure (CSV labels or single folder),
adapt the script or preprocess the data manually.
"""
import argparse
import os
import random
import shutil
from pathlib import Path


def is_image_file(p: Path):
    return p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']


def prepare_from_class_folders(src: Path, dest: Path, val_split: float, seed: int):
    random.seed(seed)
    classes = [d for d in src.iterdir() if d.is_dir()]
    if not classes:
        raise RuntimeError(f"No class subfolders found in {src}. Expected each class to be a folder of images.")

    for c in classes:
        images = [p for p in c.iterdir() if p.is_file() and is_image_file(p)]
        random.shuffle(images)
        n_val = int(len(images) * val_split)
        val = images[:n_val]
        train = images[n_val:]

        train_dir = dest / 'train' / c.name
        val_dir = dest / 'val' / c.name
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for p in train:
            shutil.copy(p, train_dir / p.name)
        for p in val:
            shutil.copy(p, val_dir / p.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', required=True, help='Path to unzipped dataset (contains class subfolders)')
    parser.add_argument('--dest', '-d', required=True, help='Destination path for prepared dataset (ImageFolder format)')
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    src = Path(args.source).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()

    if not src.exists():
        raise SystemExit(f"Source path does not exist: {src}")

    # If source already looks like ImageFolder with train/val, just copy
    if (src / 'train').exists() and (src / 'val').exists():
        print("Source already contains train/ and val/. Copying to dest...")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        print("Copied existing train/val structure.")
        return

    print(f"Preparing dataset from {src} into {dest} with val_split={args.val_split}")
    dest.mkdir(parents=True, exist_ok=True)

    try:
        prepare_from_class_folders(src, dest, args.val_split, args.seed)
    except RuntimeError as e:
        raise SystemExit(str(e))

    print("Dataset prepared successfully.")


if __name__ == '__main__':
    main()
