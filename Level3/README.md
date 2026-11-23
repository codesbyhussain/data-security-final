# Level 3 - Robust Federated Learning with Attack Detection

Implements federated learning with 10 clients (9 benign, 1 malicious) and defense against poisoning attacks.

## Settings
- Number of clients: 10 (Client 0-8 benign, Client 9 malicious)
- Global rounds: 10
- Local epochs per round: 1
- Attack schedule: Client 9 behaves normally in round 1, begins submitting malicious updates in round 2.
- Defense: Server uses update norm difference and pairwise cosine similarity to detect and exclude the attacker before aggregating benign updates.

## Files
- `models.py`: ConvNet model architecture.
- `malicious_client.py`: Isolated malicious logic with `generate_malicious_update(...)`. Marked with `# === MALICIOUS CLIENT LOGIC HERE ===` for TA replacement during Demo Day.
- `train.py`: Runs 10 FL rounds with attack injection and server anomaly detection. Saves `best_model.pth` and `robust_federated_model.pth`.
- `test.py`: Loads the trained robust model and evaluates test accuracy.

## How to Run

Live training & attack detection:
```bash
python train.py
```

Run test evaluation:
```bash
python test.py
```
