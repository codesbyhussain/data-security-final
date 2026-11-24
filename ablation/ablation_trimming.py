'''Ablation 2: Trimming vs. No Trimming in Level 3 Robust Aggregation.'''

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Level3"))

from models import LeNetWide32
from malicious_client import generate_malicious_update


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


def run_experiment(use_trimming=True):
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name = "Trimmed Mean (beta=0.1)" if use_trimming else "Standard Mean (No Trimming)"
    print(f"\n==========================================")
    print(f"Running Ablation: Aggregation = {name}")
    print(f"==========================================")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)

    client_datasets = random_split(train_dataset, [6000] * 10)
    client_loaders = [DataLoader(ds, batch_size=50, shuffle=True) for ds in client_datasets]
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    global_model = LeNetWide32().to(device)
    client_models = [copy.deepcopy(global_model) for _ in range(10)]
    optimizers = [torch.optim.Adam(m.parameters(), lr=0.0005) for m in client_models]

    for round_num in range(1, 11):
        for idx in range(10):
            if idx == 9 and round_num >= 2:
                # Malicious client generates update
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

        # Isolate client 9 (simulated defense filter)
        benign = list(range(9))

        if use_trimming:
            new_global = trimmed_mean(client_models, benign, trim_ratio=0.1)
        else:
            new_global = standard_mean(client_models, benign)

        global_model.load_state_dict(new_global)
        for c in client_models:
            c.load_state_dict(global_model.state_dict())

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
    acc_trimmed = run_experiment(use_trimming=True)
    acc_untrimmed = run_experiment(use_trimming=False)
    print(f"\n--- Summary: Trimming Ablation ---")
    print(f"With Trimmed Mean (10%):  {acc_trimmed:.2%}")
    print(f"Without Trimming (Mean):  {acc_untrimmed:.2%}")
