'''Run all project ablations and print clean summary tables.'''

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# Import ablation-specific models and malicious update generator
from models_ablation import SimpleMLP, LeNet5, LeNetAvgPool, LeNetWide32
from malicious_ablation import generate_ablation_malicious_update


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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Ablation Matrix on Device: {device}")

    client_datasets, client_loaders, test_loader = get_dataloaders()
    matrix_results = []

    experiments = [
        ("SimpleMLP", SimpleMLP, "None", "Standard FedAvg", 0.0, "None", False),
        ("SimpleMLP (Protected)", SimpleMLP, "None", "Trimmed Mean", 0.1, "Max", True),
        ("LeNet5", LeNet5, "AvgPool", "Standard FedAvg", 0.0, "None", False),
        ("LeNet5 (Protected)", LeNet5, "AvgPool", "Trimmed Mean", 0.1, "Max", True),
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
                    mal_state = generate_ablation_malicious_update(global_model.state_dict(), client_datasets[idx], device)
                    client_models[idx].load_state_dict(mal_state)
                else:
                    client_models[idx].train()
                    for x, y in client_loaders[idx]:
                        x, y = x.to(device), y.to(device)
                        optimizers[idx].zero_grad()
                        loss = F.cross_entropy(client_models[idx](x), y)
                        loss.backward()
                        optimizers[idx].step()

            if has_attack and "No Defense" in name:
                new_dict = standard_mean(client_models, list(range(10)))
            elif has_attack:
                benign = list(range(9))
                if defense == "Trimmed Mean":
                    new_dict = trimmed_mean(client_models, benign, trim_ratio=beta)
                else:
                    new_dict = standard_mean(client_models, benign)
            else:
                new_dict = standard_mean(client_models, list(range(10)))

            global_model.load_state_dict(new_dict)
            for c in client_models:
                c.load_state_dict(global_model.state_dict())

        acc = eval_model(global_model, test_loader, device)
        print(f"  --> Final 10-Round Test Accuracy: {acc:.2%}")
        matrix_results.append((name, pool, defense, f"{beta:.0%}", fusion, f"{acc:.2%}"))

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
