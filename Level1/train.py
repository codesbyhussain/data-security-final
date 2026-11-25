import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from models import LeNetWide32, LeNet5, SimpleMLP, ConvNet


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dataloader(train_dataset, test_dataset):
    # TODO: Set the length of the batch (number of samples per batch)
    batch_size = 50

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g
    )
   
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=len(test_dataset),
        shuffle=False
    )
    print(f'training has: {len(train_loader)} batch of data!')
    print(f'testing has: {len(test_loader)} batch of data!')
    return train_loader, test_loader


def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)

    print("The number of training data:", len(train_dataset))
    print("The number of testing data:", len(test_dataset))

    return dataloader(train_dataset, test_dataset)


def main():
    set_seed(42)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("Using device:", device)

    # Model choice: "wide" (LeNetWide32, primary), "lenet5" (baseline), or "mlp" (SimpleMLP)
    model_type = "wide"
    learning_rate = 0.0005
    epoches = 5
    best_model_path = 'best_model.pth'

    train_loader, test_loader = load_data()

    if model_type == "lenet5":
        model = LeNet5().to(device)
    elif model_type == "mlp":
        model = SimpleMLP().to(device)
    else:
        model = ConvNet().to(device)

    print(f"--- Running Level 1 (model={model.__class__.__name__}, epochs={epoches}, lr={learning_rate}) ---")

    lossFun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0

    for epoch in range(epoches):
        running_loss = 0.0
        running_acc = 0.0

        model.train()  # The model starts the training step.

        for i, data in enumerate(train_loader):
            features = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()
            preds = model(features)
            loss = lossFun(preds, labels)

            loss.backward()  # Backpropagation
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(preds, 1)
            total = labels.size(0)  # the length of labels

            # accuracy of prediction
            correct = (predicted == labels).sum().item()
            accuracy = correct / total
            running_acc += accuracy

            if i % 100 == 99:
                print(f'epoch: {epoch+1}, batch: {i+1}, loss: {(running_loss/100):.6f}, acc: {(running_acc/100):.2%}')
                running_loss = 0.0
                running_acc = 0.0

        # Evaluate on test set after epoch
        model.eval()
        val_accuracy = 0.0
        num_correct = 0
        num_samples = 0
        with torch.no_grad():
            for val_features, val_labels in test_loader:
                val_features = val_features.to(device)
                val_labels = val_labels.to(device)
                valiprediction = model(val_features)
                _, val_predicted = torch.max(valiprediction, 1)
                num_correct += (val_predicted == val_labels).sum().item()
                num_samples += len(val_labels)

        val_accuracy = num_correct / num_samples
        print(f"Epoch {epoch + 1} test accuracy: {val_accuracy:.2%}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model with accuracy:", f"{best_accuracy:.2%}")


if __name__ == '__main__':
    main()
