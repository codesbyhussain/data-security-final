'''Level 3 robust federated learning training.'''

import copy
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from models import ConvNet
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


def detect_and_aggregate(global_model, client_models, current_round):
    # Detect outlier client updates and aggregate benign ones
    global_dict = global_model.state_dict()
    num_clients = len(client_models)

    # Calculate update vectors (delta from global model)
    deltas = []
    for client in client_models:
        c_dict = client.state_dict()
        diffs = []
        for k in global_dict.keys():
            diffs.append((c_dict[k].float() - global_dict[k].float().to(c_dict[k].device)).reshape(-1))
        deltas.append(torch.cat(diffs))

    # Calculate L2 norms of client updates
    norms = [torch.norm(d).item() for d in deltas]
    sorted_norms = sorted(norms)
    median_norm = sorted_norms[len(sorted_norms) // 2]

    # Calculate pairwise cosine similarity to measure consensus
    cos_scores = []
    for i in range(num_clients):
        sims = []
        for j in range(num_clients):
            if i != j:
                sim = torch.cosine_similarity(deltas[i].unsqueeze(0), deltas[j].unsqueeze(0)).item()
                sims.append(sim)
        cos_scores.append(sum(sims) / len(sims))

    # Anomaly scoring: large norm difference or low/negative cosine consensus
    flagged_client = -1
    if current_round >= 2:  # Attack only begins from round 2 onward per specs
        # Flag extreme norm outliers (> 3x median) or inverted direction
        for i in range(num_clients):
            if norms[i] > 3.0 * max(median_norm, 1e-6) or cos_scores[i] < 0.0:
                flagged_client = i
                break
        # If no hard threshold hit, pick lowest cosine similarity
        if flagged_client == -1:
            flagged_client = int(torch.tensor(cos_scores).argmin().item())

        print(f"  [Defense] Detected and flagged malicious client: Client {flagged_client}")

    # Keep benign clients
    benign_indices = [i for i in range(num_clients) if i != flagged_client]
    print(f"  [Server] Aggregating {len(benign_indices)} benign client updates...")

    # FedAvg on benign clients
    new_global_dict = {}
    for key in global_dict.keys():
        stacked = torch.stack(
            [client_models[i].state_dict()[key].float() for i in benign_indices], dim=0
        )
        new_global_dict[key] = stacked.mean(dim=0).to(dtype=global_dict[key].dtype)

    global_model.load_state_dict(new_global_dict)

    # Distribute updated weights to all clients
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Level 3 rules: 10 clients (9 benign, 1 malicious), 10 global rounds, 1 local epoch
    n_clients = 10
    malicious_client_id = 9  # Designated malicious client
    attack_start_round = 2   # Malicious client behaves normally in round 1, attacks from round 2
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

    global_model = ConvNet().to(device)
    client_models = [copy.deepcopy(global_model) for _ in range(n_clients)]
    optimizers = [torch.optim.Adam(m.parameters(), lr=learning_rate) for m in client_models]

    best_accuracy = 0.0

    print("Starting Level 3 Robust Federated Learning...")
    for round_num in range(1, global_epochs + 1):
        print(f"\nGlobal Round {round_num}/{global_epochs}")

        # Train benign clients locally
        for idx in range(n_clients):
            if idx == malicious_client_id and round_num >= attack_start_round:
                # === MALICIOUS CLIENT LOGIC HERE ===
                # During Demo Day, TA can replace or inspect this logic
                malicious_state = generate_malicious_update(
                    global_model.state_dict(),
                    client_datasets[idx],
                    device
                )
                client_models[idx].load_state_dict(malicious_state)
            else:
                client_update(client_models[idx], optimizers[idx], client_loaders[idx], device, local_epochs)

        # Defense mechanism: detect malicious updates & aggregate benign updates
        detected_id = detect_and_aggregate(global_model, client_models, round_num)
        if round_num >= attack_start_round:
            if detected_id == malicious_client_id:
                print(f"  [Result] Success! Correctly identified malicious client {malicious_client_id}.")
            else:
                print(f"  [Result] Warning: Detected client {detected_id}, but injected was {malicious_client_id}.")

        # Evaluate global model
        accuracy = test_model(global_model, test_loader, device)
        print(f"Global Model Test Accuracy after round {round_num}: {accuracy:.4f} ({accuracy:.2%})")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(global_model.state_dict(), checkpoint_path)
            print(f"Saved best model with accuracy {best_accuracy:.2%} to {checkpoint_path}")

    # Save final model parameters (required for evaluation)
    torch.save(global_model.state_dict(), 'robust_federated_model.pth')
    print(f"\nTraining completed. Best Test Accuracy: {best_accuracy:.2%}")


if __name__ == '__main__':
    main()
