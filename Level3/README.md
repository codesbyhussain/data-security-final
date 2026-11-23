# Level 3 - Robust Federated Learning with Attack Detection

Implements federated learning with 10 clients (9 benign, 1 malicious) and defense against poisoning attacks.

## Models
- `LeNetWide32` (Primary model, default)
- `LeNet5` (Baseline comparison model, LeCun et al., 1998)

## Settings
- Number of clients: 10 (Client 0-8 benign, Client 9 malicious)
- Global rounds: 10
- Local epochs per round: 1
- Defense: Multi-metric detection (L2 norm vs median, pairwise cosine consensus, Multi-Krum) with coordinate-wise Trimmed Mean aggregation.

## Malicious Client & Attack Details
Configured in `malicious_client.py`:
- **Silent Period**: Client 9 behaves normally in Round 1. `SILENT_PERIOD = None` randomly waits 0 to 3 additional rounds after Round 2 (lying low) before initiating attacks. Can also be set to a fixed integer (e.g., `0` starts immediately at Round 2).
- **Attack Modes (`ATTACK_TYPE`)**:
  - `random` (default): Dynamically selects an attack or lying low (`benign`) each round.
  - `sign_flip`: Inverts update direction ($2 \times \text{global} - \text{trained}$).
  - `gaussian`: Injects scaled additive Gaussian noise into weights.
  - `scale`: Multiplies model parameters by `ATTACK_SCALE` (10x).
  - `label_flip`: Locally trains on inverted labels ($y \leftarrow 9 - y$).
  - `benign`: Submits clean, legitimate local updates.

## Files
- `models.py`: Model architectures (`LeNetWide32`, `LeNet5`).
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
