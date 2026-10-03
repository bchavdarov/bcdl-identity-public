"""
Custom CNN for CIFAR-10, sized for Jetson Nano (JetPack 4.6, PyTorch 1.10).

Run inside your jetson-inference container:
    python3 train_cifar10.py --epochs 15 --batch-size 32

First run will download CIFAR-10 (~170MB) to ./data automatically.
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

CLASSES = ('airplane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


class SimpleCNN(nn.Module):
    """Small CNN for 32x32 input. ~550K params, light enough to train on-device."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 8 -> 4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_dataloaders(batch_size, data_dir='./data'):
    # Light augmentation on train, plain normalization on test
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_tf)

    # num_workers=2 keeps CPU preprocessing from starving the GPU without
    # overloading the Nano's 4 CPU cores / 4GB RAM
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--out', type=str, default='cifar10_cnn.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_loader, test_loader = get_dataloaders(args.batch_size, args.data_dir)

    model = SimpleCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        acc = evaluate(model, test_loader, device)
        elapsed = time.time() - t0

        print(f'Epoch {epoch:2d}/{args.epochs} | loss: {avg_loss:.4f} | '
              f'test acc: {acc:.2f}% | time: {elapsed:.1f}s')

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.out)
            print(f'  -> saved new best model ({acc:.2f}%) to {args.out}')

    print(f'\nTraining done. Best test accuracy: {best_acc:.2f}%')
    print(f'Weights saved to: {args.out}')


if __name__ == '__main__':
    main()