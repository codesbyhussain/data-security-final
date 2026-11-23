'''Level 3 evaluation script.'''

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import ConvNet


def load_test_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
    print("The number of testing data:", len(test_dataset))
    return test_loader


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = "best_model.pth"

    if not os.path.exists(checkpoint_path):
        checkpoint_path = "robust_federated_model.pth"

    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found. Please run train.py first.")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")
    test_loader = load_test_data()

    model = ConvNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
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

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2%} ({correct}/{total})")


if __name__ == "__main__":
    main()
