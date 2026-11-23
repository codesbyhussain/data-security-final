'''Level 3 robust federated learning training.'''

import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from models import LeNetWide32, LeNet5
from malicious_client import generate_malicious_update


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


def trimmed_mean(client_models, benign_indices, trim_ratio=0.1):
    states = [client_models[i].state_dict() for i in benign_indices]
    n = len(states)
    if n <= 2:
        avg = {}
        for key in states[0]:
            stacked = torch.stack([s[key].float() for s in states], dim=0)
            avg[key] = stacked.mean(dim=0).to(dtype=states[0][key].dtype)
        return avg

    # k = max(0, int(n * trim_ratio))
    k = max(0, round(n * trim_ratio))
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


def detect_and_aggregate(global_model, client_models, current_round):
    global_dict = global_model.state_dict()
    num_clients = len(client_models)

    # 0. DEFENSE A: Check for NaN / Inf injection attacks
    corrupted_clients = []
    for i, client in enumerate(client_models):
        for param in client.state_dict().values():
            if not torch.isfinite(param).all():
                corrupted_clients.append(i)
                break

    if corrupted_clients and current_round >= 2:
        flagged_client = corrupted_clients[0]
        print(f"  [Defense] Flagged malicious client: Client {flagged_client} (detected NaN/Inf values)")
        benign_indices = [i for i in range(num_clients) if i != flagged_client]
        new_global_dict = trimmed_mean(client_models, benign_indices, trim_ratio=0.1)
        global_model.load_state_dict(new_global_dict)
        for client in client_models:
            client.load_state_dict(global_model.state_dict())
        return flagged_client

    # 1. GET DELTA VECTORS FOR DEFENSES: Update delta vectors (w_i - w_global)
    deltas = []
    for client in client_models:
        c_dict = client.state_dict()
        diffs = []
        for k in global_dict.keys():
            diffs.append((c_dict[k].float() - global_dict[k].float().to(c_dict[k].device)).reshape(-1))
        deltas.append(torch.cat(diffs))

    # 2. DEFENSE B: L2 norms and median 
    norms = [torch.norm(d).item() for d in deltas]
    # sorted_norms = sorted(norms)
    # med_norm = sorted_norms[len(sorted_norms) // 2]
    med_norm = torch.tensor(norms).median().item()

    # 3. DEFENSE C: Pairwise cosine consensus scores
    cos_scores = []
    for i in range(num_clients):
        sims = []
        for j in range(num_clients):
            if i != j: # Only compare to other clients
                sim = torch.cosine_similarity(deltas[i].unsqueeze(0), deltas[j].unsqueeze(0)).item() # Unsqueeze needed for pytrch's cosine_similarity.
                sims.append(sim)
        cos_scores.append(sum(sims) / len(sims))

    # 4. DEFENSE D: Multi-Krum scores
    f = 1 # maximum number of attackers.
    keep_neighbors = max(1, num_clients - f - 2) # Removes more distant clients, reducing outlier impact.
    krum_scores = []
    for i in range(num_clients):
        # Calculate distances between each client, take norm, and  sort smallest to largest.
        dists = sorted([(deltas[i] - deltas[j]).norm().item() for j in range(num_clients) if i != j])
        krum_scores.append(sum(dists[:keep_neighbors])) # Sum largest for scores.

    def minmax(vals):
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-12: # if values are too close, return 0
            return [0.0] * len(vals)
        return [(v - lo) / (hi - lo) for v in vals]

    cos_anom = minmax([1.0 - c for c in cos_scores]) # High is good, so invert.
    l2_anom = minmax([abs(nr - med_norm) / max(med_norm, 1e-12) for nr in norms])
    krum_anom = minmax(krum_scores)

    combined_scores = [max(c, l, k) for c, l, k in zip(cos_anom, l2_anom, krum_anom)]
    # combined_scores = [mean(c, l, k) for c, l, k in zip(cos_anom, l2_anom, krum_anom)] # ablation test

    flagged_client = -1
    if current_round >= 2:
        hard_outliers = []
        for i in range(num_clients):  
            if norms[i] > 3.0 * max(med_norm, 1e-12) or cos_scores[i] < 0.0: # Checks for enormous or opposite direction updates.
                hard_outliers.append(i)

        if hard_outliers:
            hard_outliers.sort(key=lambda idx: combined_scores[idx], reverse=True)
            flagged_client = hard_outliers[0]\
            flag_type = "hard"

        else: # Otherwise, flags the client with the highest anonamly score.
            flagged_client = int(torch.tensor(combined_scores).argmax().item())
            flag_type = "soft"

        print(f"  [Defense] Flagged malicious client: Client {flagged_client} (norm={norms[flagged_client]:.2f}, cos={cos_scores[flagged_client]:.4f}, type={flag_type})")

    benign_indices = [i for i in range(num_clients) if i != flagged_client]
    print(f"  [Server] Robust aggregation (Trimmed Mean) over {len(benign_indices)} benign clients...")

    new_global_dict = trimmed_mean(client_models, benign_indices, trim_ratio=0.1)
    global_model.load_state_dict(new_global_dict)

    for client in client_models:
        client.load_state_dict(global_model.state_dict())

    return flagged_client


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
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model choice: "wide" (LeNetWide32, primary) or "lenet5" (baseline)
    model_type = "wide"
    n_clients = 10
    malicious_client_id = 9
    attack_start_round = 2
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

    print(f"Level 3 training with model: {global_model.__class__.__name__}")
    client_models = [copy.deepcopy(global_model) for _ in range(n_clients)]
    optimizers = [torch.optim.Adam(m.parameters(), lr=learning_rate) for m in client_models]

    best_accuracy = 0.0
    caught_rounds = []

    print("Starting Level 3 Robust Federated Learning...")
    for round_num in range(1, global_epochs + 1):
        print(f"\nGlobal Round {round_num}/{global_epochs}")

        for idx in range(n_clients):
            if idx == malicious_client_id and round_num >= attack_start_round:
                # === MALICIOUS CLIENT LOGIC HERE ===
                malicious_state = generate_malicious_update(
                    global_model.state_dict(),
                    client_datasets[idx],
                    device
                )
                client_models[idx].load_state_dict(malicious_state)
            else:
                client_update(client_models[idx], optimizers[idx], client_loaders[idx], device, local_epochs)

        detected_id = detect_and_aggregate(global_model, client_models, round_num)

        if round_num >= attack_start_round:
            if detected_id == malicious_client_id:
                caught_rounds.append(round_num)
                print(f"  [Result] Success: Correctly identified malicious client {malicious_client_id}!")
            else:
                print(f"  [Result] Warning: Injected client was {malicious_client_id}, but flagged client was {detected_id}")

        accuracy = test_model(global_model, test_loader, device)
        print(f"Global Model Test Accuracy after round {round_num}: {accuracy:.4f} ({accuracy:.2%})")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(global_model.state_dict(), checkpoint_path)
            print(f"Saved best model with accuracy {best_accuracy:.2%} to {checkpoint_path}")

    torch.save(global_model.state_dict(), 'robust_federated_model.pth')
    print(f"\nTraining completed. Malicious client caught in rounds: {caught_rounds}")
    print(f"Best Test Accuracy: {best_accuracy:.2%}")


if __name__ == '__main__':
    main()
