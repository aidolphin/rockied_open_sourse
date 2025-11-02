import os
import random
import shutil
from pathlib import Path

base_dir = Path(__file__).parent.parent / 'data' / 'rock-classification'
train_dir = base_dir / 'train'
val_dir = base_dir / 'val'

# Make sure directories exist
for dir in [train_dir, val_dir]:
    dir.mkdir(exist_ok=True, parents=True)

# Classes
classes = ['Quartzite', 'Coal', 'Limestone', 'Sandstone']

# Create class directories if they don't exist
for cls in classes:
    (train_dir / cls).mkdir(exist_ok=True, parents=True)
    (val_dir / cls).mkdir(exist_ok=True, parents=True)

# Split ratio (e.g., 80% train, 20% validation)
split_ratio = 0.8

# Copy files from dataset to train and validation
for cls in classes:
    if cls == 'Quartzite':
        src_dir = base_dir / 'Dataset' / 'Metamorphic' / cls
    else:
        src_dir = base_dir / 'Dataset' / 'Sedimentary' / cls
        
    files = list(src_dir.glob('*.*'))
    random.shuffle(files)
    
    split_point = int(len(files) * split_ratio)
    train_files = files[:split_point]
    val_files = files[split_point:]
    
    # Copy to train
    for file in train_files:
        shutil.copy2(file, train_dir / cls / file.name)
        
    # Copy to validation
    for file in val_files:
        shutil.copy2(file, val_dir / cls / file.name)
    
    print(f"{cls}:")
    print(f"  Train: {len(train_files)}")
    print(f"  Validation: {len(val_files)}")
