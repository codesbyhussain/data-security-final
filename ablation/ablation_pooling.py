'''Ablation 1: Max Pooling vs. Average Pooling in Federated Learning.'''

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


class LeNetMaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 48, 5)
        self.fc1 = nn.Linear(48 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.max_pool2d(F.relu(self.conv1(xb)), 2)
        xb = F.max_pool2d(F.relu(self.conv2(xb)), 2)
        xb = xb.flatten(1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(xb)))))


class LeNetAvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 48, 5)
        self.fc1 = nn.Linear(48 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.avg_pool2d(F.relu(self.conv1(xb)), 2)
        xb = F.avg_pool2d(F.relu(self.conv2(xb)), 2)
        xb = xb.flatten(1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(xb)))))


def run_experiment(pool_type="max"):
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n==========================================")
    print(f"Running Ablation: Pooling Type = {pool_type.upper()}")
    print(f"==========================================")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)

    client_datasets = random_split(train_dataset, [6000] * 10)
    client_loaders = [DataLoader(ds, batch_size=50, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    if pool_type == "max":
        global_model = LeNetMaxPool().to(device)
    else:
        global_model = LeNetAvgPool().to(device)

    client_models = [copy.deepcopy(global_model) for _ in range(10)]
    optimizers = [torch.optim.Adam(m.parameters(), lr=0.0005) for m in client_models]

    for round_num in range(1, 11):
        for idx in range(10):
            client_models[idx].train()
            for x, y in client_loaders[idx]:
                x, y = x.to(device), y.to(device)
                optimizers[idx].zero_grad()
                loss = F.cross_entropy(client_models[idx](x), y)
                loss.backward()
                optimizers[idx].step()

        # FedAvg
        global_dict = global_model.state_dict()
        for k in global_dict:
            global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(10)], dim=0).mean(dim=0)
        global_model.load_state_dict(global_dict)
        for client in client_models:
            client.load_state_dict(global_model.state_dict())

        # Evaluate
        global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (global_model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        print(f"Round {round_num:2d} | Test Accuracy: {correct / total:.2%}")

    return correct / total


if __name__ == '__main__':
    acc_max = run_experiment("max")
    acc_avg = run_experiment("avg")
    print(f"\n--- Summary: Pooling Ablation ---")
    print(f"Max Pooling Final Accuracy:  {acc_max:.2%}")
    print(f"Avg Pooling Final Accuracy:  {acc_avg:.2%}")
