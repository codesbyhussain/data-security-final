# CSC8370 Data Security Final Project

MNIST classification across 3 levels:
- **Level 1**: Centralized Learning
- **Level 2**: Federated Learning (10 clients, FedAvg)
- **Level 3**: Robust Federated Learning with Attack Detection

## Models
Defined in `models.py` across each level:
- **`LeNetWide32`** (Primary model): Widened LeNet architecture evaluated on a 32x32 padded canvas with max-pooling.
- **`LeNet5`** (Baseline comparison model): Classic LeNet-5 architecture (LeCun et al., 1998, *"Gradient-based learning applied to document recognition"*, Proc. IEEE).

## Environment Setup

Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:
```bash
pip install -r requirements.txt
```

Tested Environment (As per server provided to Group 04 [student@csc8370-g04]):
- **System**: Linux, NVIDIA GeForce GTX 1080 Ti (Driver 580.178.04, CUDA 13.0)
- **Python**: 3.12.3
- **Packages**: `torch==2.7.0+cu126`, `torchvision==0.22.0+cu126`, `numpy==2.5.4`

---

## Instructions to Run

To run inference only (without retraining), run `python test.py` directly inside `Level1`, `Level2`, or `Level3` using the pre-trained `.pth` checkpoint.


### Level 1 - Centralized Learning
Navigate to `Level1`:
```bash
cd Level1
python train.py
python test.py
```
- `train.py`: Trains the model on MNIST and saves `best_model.pth`.
- `test.py`: Loads `best_model.pth` and evaluates accuracy on the test set.

### Level 2 - Federated Learning
Navigate to `Level2`:
```bash
cd Level2
python train.py
python test.py
```
- `train.py`: Runs 10 global rounds across 10 clients (1 local epoch each) and saves `best_model.pth`.
- `test.py`: Evaluates the pre-trained saved model on the test dataset.

### Level 3 - Robust Federated Learning (Demo Day)
Navigate to `Level3`:
```bash
cd Level3
python train.py
python test.py
```
- `malicious_client.py`: Contains `generate_malicious_update(...)` isolated for Demo Day evaluation. Marked with `# === MALICIOUS CLIENT LOGIC HERE ===`.
- `train.py`: Runs 10 rounds with 1 malicious client (Client 9 attacks starting from round 2). Anomaly detection filters the attacker, Trimmed Mean aggregates benign updates, and saves `best_model.pth` and `robust_federated_model.pth`.
- `test.py`: Evaluates the robust global model on the test dataset.
