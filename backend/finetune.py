import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', required=True)
    parser.add_argument('--checkpoint', '-c', default='models/rock_cnn.pth')
    parser.add_argument('--out', '-o', default='models/rock_cnn_finetuned.pth')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=8)
    args = parser.parse_args()

    # Import torch lazily so script can fail with a clear message if not installed
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torchvision import datasets, transforms, models
        from torch.utils.data import DataLoader
    except Exception as e:
        print('PyTorch is required to run finetune.py:', e)
        raise

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Transforms (same as train.py)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_root = Path(args.dataset).expanduser().resolve()
    train_dir = dataset_root / 'train'
    val_dir = dataset_root / 'val'

    if not train_dir.exists() or not val_dir.exists():
        print(f"Dataset train/val not found at: {dataset_root}")
        print("Expected structure:\n  <dataset>/train/<class>/*.jpg\n  <dataset>/val/<class>/*.jpg")
        print("If you ran this from the 'backend' directory, pass '-d dataset' (not 'backend/dataset').")
        raise SystemExit(1)

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    num_classes = len(train_ds.classes)
    print('Classes:', train_ds.classes)

    # Build model (MobileNet-like as in train.py)
    class RockCNN(nn.Module):
        def __init__(self, num_classes):
            super(RockCNN, self).__init__()
            self.features = models.mobilenet_v2(pretrained=True).features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(1280, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    model = RockCNN(num_classes=num_classes).to(device)

    start_epoch = 0
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # If checkpoint exists, try to load weights and optimizer state
    if os.path.exists(args.checkpoint):
        try:
            ckpt = torch.load(args.checkpoint, map_location=device)
            if 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            else:
                # old checkpoint format may store state dict directly
                try:
                    model.load_state_dict(ckpt)
                except Exception:
                    print('Checkpoint found but could not load model_state_dict automatically.')
            if 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception:
                    print('Checkpoint optimizer state was incompatible, continuing with fresh optimizer')
            start_epoch = ckpt.get('epoch', 0) + 1
            print(f'Loaded checkpoint {args.checkpoint}, starting from epoch {start_epoch}')
        except Exception as e:
            print('Failed to load checkpoint:', e)

    best_acc = 0.0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total if total > 0 else 0
        print(f'Epoch {epoch+1}: train_acc={train_acc:.4f}, loss={running_loss/len(train_loader):.4f}')

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f'Validation: acc={val_acc:.4f}, loss={val_loss/len(val_loader):.4f}')

        # Save checkpoint
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': max(best_acc, val_acc),
            'classes': train_ds.classes
        }, args.out)
        print(f'Saved checkpoint to {args.out}')

    print('Finetune finished')


if __name__ == '__main__':
    main()
