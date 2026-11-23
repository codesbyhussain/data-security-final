'''Level 2 federated learning training.'''

import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from models import LeNetWide32, LeNet5


def load_data(transform):
    train_dataset = torchvision.datasets.MNIST(
        root="./data/mnist", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data/mnist", train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


def partition_dataset(dataset, n_clients=10):
    split_size = len(dataset) // n_clients
    return random_split(dataset, [split_size] * n_clients)


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


def server_aggregate(global_model, client_models):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack(
            [client_models[i].state_dict()[key].float() for i in range(len(client_models))], dim=0
        ).mean(dim=0)
    global_model.load_state_dict(global_dict)

    for client in client_models:
        client.load_state_dict(global_model.state_dict())


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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model choice: "wide" (LeNetWide32, primary) or "lenet5" (baseline)
    model_type = "wide"
    n_clients = 10
    global_epochs = 10
    local_epochs = 1
    learning_rate = 0.0005
    checkpoint_path = "best_model.pth"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset, test_dataset = load_data(transform)

    client_datasets = partition_dataset(train_dataset, n_clients)
    client_loaders = [DataLoader(ds, batch_size=50, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    if model_type == "lenet5":
        global_model = LeNet5().to(device)
    else:
        global_model = LeNetWide32().to(device)

    print(f"Federated training with: {global_model.__class__.__name__}")
    client_models = [copy.deepcopy(global_model) for _ in range(n_clients)]
    optimizers = [torch.optim.Adam(m.parameters(), lr=learning_rate) for m in client_models]

    best_accuracy = 0.0

    print("Starting Federated Learning...")
    for round_num in range(global_epochs):
        print(f"Global Round {round_num + 1}/{global_epochs}")

        for idx in range(n_clients):
            client_update(client_models[idx], optimizers[idx], client_loaders[idx], device, local_epochs)

        server_aggregate(global_model, client_models)

        accuracy = test_model(global_model, test_loader, device)
        print(f"Round {round_num + 1} Global Test Accuracy: {accuracy:.4f} ({accuracy:.2%})")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(global_model.state_dict(), checkpoint_path)
            print(f"Saved best model with accuracy {best_accuracy:.2%} to {checkpoint_path}")

    print(f"\nFederated training finished. Best Test Accuracy: {best_accuracy:.2%}")


if __name__ == '__main__':
    main()
