import os
import ssl
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Temporarily disable SSL verification for downloading pretrained model
ssl._create_default_https_context = ssl._create_unverified_context


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


def train(dataset_dir, output_path, epochs=3, batch_size=16, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Validation transform without augmentation
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dir = Path(dataset_dir) / 'train'
    val_dir = Path(dataset_dir) / 'val'

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    num_classes = len(train_ds.classes)
    print(f"Found classes: {train_ds.classes}")

    model = RockCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)

    best_acc = 0.0
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        train_acc = correct_train / total_train
        avg_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"Training - Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        class_correct = [0] * len(train_ds.classes)
        class_total = [0] * len(train_ds.classes)
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                
                # Per-class accuracy
                correct = (preds == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1

        val_acc = correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.4f}")
        
        # Print per-class accuracy
        print("\nPer-class Validation Accuracy:")
        for i in range(len(train_ds.classes)):
            acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            print(f'{train_ds.classes[i]}: {acc:.4f}')
            
        # Update learning rate scheduler
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'classes': train_ds.classes
            }, output_path)
            print(f"\nSaved best model to {output_path} (accuracy: {best_acc:.4f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', required=True, help='Path to dataset directory (must contain train/ and val/ folders)')
    parser.add_argument('--out', '-o', default='models/rock_cnn.pth', help='Output path for saved model')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=16)
    args = parser.parse_args()

    train(args.dataset, args.out, epochs=args.epochs, batch_size=args.batch)
