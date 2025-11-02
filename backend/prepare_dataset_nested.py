#!/usr/bin/env python3
"""
Prepare dataset when source has a two-level taxonomy (group/type) like:
  Dataset/Igneous/Granite/*.jpg
  Dataset/Igneous/Basalt/*.jpg

This script finds the deepest directories that contain images and treats each
as a class. It creates dest/train/<class> and dest/val/<class> with a randomized split.
"""
import argparse
import random
import shutil
from pathlib import Path


def is_image_file(p: Path):
    return p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']


def find_leaf_image_dirs(root: Path):
    """Return a list of directories under root that contain image files (leaf classes)."""
    leaf_dirs = []
    for p in root.rglob('*'):
        if p.is_dir():
            # if this dir contains image files directly, consider it a leaf class
            imgs = [f for f in p.iterdir() if f.is_file() and is_image_file(f)]
            if imgs:
                leaf_dirs.append(p)
    return sorted(leaf_dirs)


def prepare(src: Path, dest: Path, val_split: float, seed: int):
    random.seed(seed)
    leaf_dirs = find_leaf_image_dirs(src)
    if not leaf_dirs:
        raise RuntimeError(f'No image-containing leaf directories found under {src}')

    for d in leaf_dirs:
        class_name = d.name
        images = [p for p in d.iterdir() if p.is_file() and is_image_file(p)]
        random.shuffle(images)
        n_val = int(len(images) * val_split)
        val = images[:n_val]
        train = images[n_val:]

        train_dir = dest / 'train' / class_name
        val_dir = dest / 'val' / class_name
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for p in train:
            shutil.copy(p, train_dir / p.name)
        for p in val:
            shutil.copy(p, val_dir / p.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', '-s', required=True)
    parser.add_argument('--dest', '-d', required=True)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    src = Path(args.source).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f'Source not found: {src}')

    print(f'Preparing nested dataset from {src} into {dest} (val_split={args.val_split})')
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    prepare(src, dest, args.val_split, args.seed)
    print('Prepared nested dataset successfully')


if __name__ == '__main__':
    main()
