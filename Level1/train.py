'''Level 1 centralized training.'''

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import LeNetWide32, LeNet5


def load_data(batch_size=50):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)

    print("The number of training data:", len(train_dataset))
    print("The number of testing data:", len(test_dataset))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    print(f"training has: {len(train_loader)} batch of data!")
    print(f"testing has: {len(test_loader)} batch of data!")
    return train_loader, test_loader


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model choice: "wide" (LeNetWide32, primary) or "lenet5" (baseline)
    model_type = "wide"
    learning_rate = 0.0005
    epochs = 5
    batch_size = 50
    checkpoint_path = "best_model.pth"

    train_loader, test_loader = load_data(batch_size=batch_size)

    if model_type == "lenet5":
        model = LeNet5().to(device)
    else:
        model = LeNetWide32().to(device)

    print(f"Training model: {model.__class__.__name__}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        for i, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            running_acc += (predicted == labels).sum().item() / labels.size(0)

            if i % 100 == 99:
                print(f"epoch: {epoch + 1}, batch: {i + 1}, loss: {(running_loss / 100):.6f}, acc: {(running_acc / 100):.2%}")
                running_loss = 0.0
                running_acc = 0.0

        # Evaluate on test set after epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for test_features, test_labels in test_loader:
                test_features = test_features.to(device)
                test_labels = test_labels.to(device)
                preds = model(test_features)
                _, predicted = torch.max(preds, 1)
                correct += (predicted == test_labels).sum().item()
                total += test_labels.size(0)

        epoch_acc = correct / total
        print(f"Epoch {epoch + 1} test accuracy: {epoch_acc:.2%}")

        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model with accuracy: {best_accuracy:.2%} to {checkpoint_path}")

    print(f"\nTraining completed. Best accuracy: {best_accuracy:.2%}")


if __name__ == "__main__":
    main()
