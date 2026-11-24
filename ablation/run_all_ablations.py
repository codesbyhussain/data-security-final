'''Run all project ablations and print clean summary tables.'''

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import sys
from pathlib import Path

# Add Level3 folder to import models and attack logic
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Level3"))

from models import LeNetWide32, LeNet5
from malicious_client import generate_malicious_update


# =============================================================================
# Models for Ablation Studies
# =============================================================================
class SimpleMLP(nn.Module):
    '''Simple 2-layer MLP (Non-convolutional, slower convergence).'''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, xb):
        xb = xb.view(-1, 28 * 28)
        xb = F.relu(self.fc1(xb))
        xb = F.relu(self.fc2(xb))
        return self.fc3(xb)


class LeNetAvgPool(nn.Module):
    '''LeNet variant using Average Pooling instead of Max Pooling.'''
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


# =============================================================================
# Aggregation Functions
# =============================================================================
def trimmed_mean(client_models, benign_indices, trim_ratio=0.1):
    states = [client_models[i].state_dict() for i in benign_indices]
    n = len(states)
    k = max(0, int(n * trim_ratio))
    avg = {}
    for key in states[0]:
        stacked = torch.stack([s[key].float() for s in states], dim=0)
        orig_shape = stacked.shape[1:]
        stacked_flat = stacked.view(n, -1)
        if k > 0 and 2 * k < n:
            sorted_flat, _ = torch.sort(stacked_flat, dim=0)
            trimmed = sorted_flat[k:n - k]
            mean_flat = trimmed.mean(dim=0)
        else:
            mean_flat = stacked_flat.mean(dim=0)
        avg[key] = mean_flat.view(orig_shape).to(dtype=states[0][key].dtype)
    return avg


def standard_mean(client_models, benign_indices):
    states = [client_models[i].state_dict() for i in benign_indices]
    avg = {}
    for key in states[0]:
        stacked = torch.stack([s[key].float() for s in states], dim=0)
        avg[key] = stacked.mean(dim=0).to(dtype=states[0][key].dtype)
    return avg


# =============================================================================
# Shared Dataset Loader & Evaluator
# =============================================================================
def get_dataloaders():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)

    client_datasets = random_split(train_dataset, [6000] * 10)
    client_loaders = [DataLoader(ds, batch_size=50, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    return client_datasets, client_loaders, test_loader


def eval_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


# =============================================================================
# Main Ablation Runner
# =============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Ablation Matrix on Device: {device}")

    client_datasets, client_loaders, test_loader = get_dataloaders()
    matrix_results = []

    experiments = [
        # (Model Name, Model Class, Pool Type, Defense Type, Beta, Fusion, Malicious)
        ("SimpleMLP", SimpleMLP, "None", "Standard FedAvg", 0.0, "None", False),
        ("SimpleMLP", SimpleMLP, "None", "Trimmed Mean", 0.1, "Max", True),
        ("LeNet5", LeNet5, "AvgPool", "Standard FedAvg", 0.0, "None", False),
        ("LeNet5", LeNet5, "AvgPool", "Trimmed Mean", 0.1, "Max", True),
        ("LeNetWide32", LeNetAvgPool, "AvgPool", "Standard FedAvg", 0.0, "None", False),
        ("LeNetWide32", LeNetWide32, "MaxPool", "Standard FedAvg", 0.0, "None", False),
        ("LeNetWide32 (Under Attack, No Defense)", LeNetWide32, "MaxPool", "Standard FedAvg", 0.0, "None", True),
        ("LeNetWide32 (Protected, Mean Fusion)", LeNetWide32, "MaxPool", "Trimmed Mean", 0.1, "Mean", True),
        ("LeNetWide32 (Protected, Max Fusion)", LeNetWide32, "MaxPool", "Trimmed Mean", 0.1, "Max", True),
        ("LeNetWide32 (Trim Beta 20%)", LeNetWide32, "MaxPool", "Trimmed Mean", 0.2, "Max", True),
        ("LeNetWide32 (Trim Beta 30%)", LeNetWide32, "MaxPool", "Trimmed Mean", 0.3, "Max", True),
    ]

    for name, model_cls, pool, defense, beta, fusion, has_attack in experiments:
        torch.manual_seed(42)
        print(f"\nEvaluating: {name} | Defense: {defense} | Beta: {beta:.0%} | Fusion: {fusion}")

        global_model = model_cls().to(device)
        client_models = [copy.deepcopy(global_model) for _ in range(10)]
        optimizers = [torch.optim.Adam(m.parameters(), lr=0.0005) for m in client_models]

        for round_num in range(10):
            for idx in range(10):
                if idx == 9 and has_attack and round_num >= 1:
                    mal_state = generate_malicious_update(global_model.state_dict(), client_datasets[idx], device)
                    client_models[idx].load_state_dict(mal_state)
                else:
                    client_models[idx].train()
                    for x, y in client_loaders[idx]:
                        x, y = x.to(device), y.to(device)
                        optimizers[idx].zero_grad()
                        loss = F.cross_entropy(client_models[idx](x), y)
                        loss.backward()
                        optimizers[idx].step()

            # Server Aggregation
            if has_attack and "No Defense" in name:
                # Naive FedAvg includes the attacker
                new_dict = standard_mean(client_models, list(range(10)))
            elif has_attack:
                # Isolate client 9 (defense applied)
                benign = list(range(9))
                if defense == "Trimmed Mean":
                    new_dict = trimmed_mean(client_models, benign, trim_ratio=beta)
                else:
                    new_dict = standard_mean(client_models, benign)
            else:
                # Benign FedAvg across all 10 clients
                new_dict = standard_mean(client_models, list(range(10)))

            global_model.load_state_dict(new_dict)
            for c in client_models:
                c.load_state_dict(global_model.state_dict())

        acc = eval_model(global_model, test_loader, device)
        print(f"  --> Final 10-Round Test Accuracy: {acc:.2%}")
        matrix_results.append((name, pool, defense, f"{beta:.0%}", fusion, f"{acc:.2%}"))

    # Print Final Markdown Table
    print("\n" + "=" * 80)
    print("                    UNIFIED ABLATION STUDY RESULTS MATRIX                     ")
    print("=" * 80)
    print(f"{'Model Configuration':<36} | {'Pooling':<8} | {'Aggregation':<16} | {'Beta':<5} | {'Fusion':<6} | {'Test Acc':<8}")
    print("-" * 80)
    for r in matrix_results:
        print(f"{r[0]:<36} | {r[1]:<8} | {r[2]:<16} | {r[3]:<5} | {r[4]:<6} | {r[5]:<8}")
    print("=" * 80)


if __name__ == '__main__':
    main()
