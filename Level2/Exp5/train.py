import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import copy

from models import LeNetWide32, LeNet5, SimpleMLP, LeNetWide32Avg, ConvNet


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Load data (each client will load its own data in a real FL scenario)
def load_data(transform, datasets='MNIST'):
    if datasets == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(
            root="./data/mnist", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(
            root="./data/mnist", train=False, download=True, transform=transform)
    else:
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data/cifar-10-python", train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data/cifar-10-python", train=False, download=True, transform=transform)

    return train_dataset, test_dataset


# Split the dataset into 'n_clients' partitions
def partition_dataset(dataset, n_clients=10, seed=42):
    split_size = len(dataset) // n_clients
    g = torch.Generator()
    g.manual_seed(seed)
    return random_split(dataset, [split_size] * n_clients, generator=g)


# Client-side local training
def client_update(client_model, optimizer, train_loader, device, epochs=1):
    client_model.train()
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            output = client_model(data)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()


# Server-side aggregation of client models
def server_aggregate(global_model, client_models):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [client_models[i].state_dict()[key].float() for i in range(len(client_models))], dim=0
        ).mean(dim=0)
    global_model.load_state_dict(global_dict)

    for client in client_models:
        client.load_state_dict(global_model.state_dict())


# Test model on test dataset
def test_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# Federated Learning process
def federated_learning(n_clients=10, global_epochs=10, local_epochs=1, model_type="wide"):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # Set up the data transformation and load dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset, test_dataset = load_data(transform)

    # Partition the dataset for each client
    client_datasets = partition_dataset(train_dataset, n_clients, seed=42)
    g = torch.Generator()
    g.manual_seed(42)
    client_loaders = [DataLoader(dataset, batch_size=50, shuffle=True, generator=g) for dataset in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Initialize global model and n_clients client models
    if model_type == "lenet5":
        global_model = LeNet5().to(device)
    elif model_type == "mlp":
        global_model = SimpleMLP().to(device)
    elif model_type == "wide_avg":
        global_model = LeNetWide32Avg().to(device)
    else:
        global_model = ConvNet().to(device)

    print(f"--- Running Level 2 (model={global_model.__class__.__name__}, clients={n_clients}, rounds={global_epochs}) ---")

    client_models = [copy.deepcopy(global_model) for _ in range(n_clients)]

    # Optimizers for client models
    optimizers = [torch.optim.Adam(model.parameters(), lr=0.0005) for model in client_models]

    best_accuracy = 0.0

    # Federated Learning process
    for global_epoch in range(global_epochs):
        print(f'Global Round {global_epoch + 1}/{global_epochs}')

        # Each client trains locally
        for client_idx in range(n_clients):
            client_update(client_models[client_idx], optimizers[client_idx], client_loaders[client_idx], device, local_epochs)

        # Server aggregates the models
        server_aggregate(global_model, client_models)

        # Evaluate global model on test dataset
        test_accuracy = test_model(global_model, test_loader, device)
        print(f'Round {global_epoch + 1} Global Test Accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})')

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(global_model.state_dict(), 'best_model.pth')
            print(f'Saved best model with accuracy {best_accuracy:.2%} to best_model.pth')

    print(f"\nFederated training finished. Best Test Accuracy: {best_accuracy:.2%}")


if __name__ == '__main__':
    federated_learning(n_clients=10, global_epochs=10, local_epochs=1, model_type="lenet5")
