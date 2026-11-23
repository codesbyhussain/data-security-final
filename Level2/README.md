# Level 2 - Federated Learning

Implements federated learning with 10 clients and 1 central server using FedAvg.

## Models
- `LeNetWide32` (Primary model, default)
- `LeNet5` (Baseline comparison model, LeCun et al., 1998)

## Settings
- Number of clients: 10
- Global rounds: 10
- Local epochs per round: 1
- Batch size: 50
- Optimizer: Adam (lr=0.0005)

## Files
- `models.py`: Model architectures (`LeNetWide32`, `LeNet5`).
- `train.py`: Partitions MNIST equally across 10 clients, runs 10 global FL rounds, aggregates with FedAvg, and saves `best_model.pth`.
- `test.py`: Loads `best_model.pth` and evaluates accuracy on the test dataset.

## How to Run

Run federated training:
```bash
python train.py
```

Run test evaluation (Inference-only):
```bash
python test.py
```
