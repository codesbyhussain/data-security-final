### Level 3 Code Structure Requirement (For Demo Day Testing)
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import LeNetWide32, LeNet5

# Attack selection:
# 'random' | 'sign_flip' | 'gaussian' | 'scale' | 'label_flip' | 'benign'
ATTACK_TYPE = "random"
ATTACK_SCALE = 10.0

# Silent period: number of rounds to wait after round 2 before starting attack.
# Set to an integer (0 = starts at round 2, 2 = starts at round 4), or None for random (0 to 3 rounds).
SILENT_PERIOD = None
call_count = 0
silent_rounds = None


def _get_local_trained_state(global_state, local_dataset, device, label_flip=False):
    # Detect whether global model is SimpleMLP, LeNet5, or LeNetWide32 from parameter shapes
    if "conv1.weight" not in global_state:
        # Non-convolutional MLP
        from ablation.run_all_ablations import SimpleMLP
        model = SimpleMLP().to(device)
    elif "fc1.weight" in global_state and global_state["fc1.weight"].shape[0] == 120:
        model = LeNet5().to(device)
    else:
        model = LeNetWide32().to(device)

    model.load_state_dict(global_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loader = DataLoader(local_dataset, batch_size=50, shuffle=True)

    model.train()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if label_flip:
            yb = 9 - yb

        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return model.state_dict()


def generate_malicious_update(client_model_state_dict, local_dataset, device):
    """
    Args:
        client_model_state_dict (dict): A copy of the current global model's state_dict().
        local_dataset (Dataset): The local dataset assigned to this client.
        device (torch.device): CPU or CUDA device.
    Returns:
        malicious_state_dict (dict): The modified model parameters (state_dict) to be uploaded to the server.
    """
    # === STUDENTS: IMPLEMENT YOUR SIMULATED ATTACK BELOW ===
    # === MALICIOUS CLIENT LOGIC HERE ===
    global call_count, silent_rounds
    call_count += 1

    # Determine silent period on first invocation
    if silent_rounds is None:
        if SILENT_PERIOD is None:
            silent_rounds = random.randint(0, 3)
        else:
            silent_rounds = SILENT_PERIOD
        attack_start_round = 2 + silent_rounds
        print(f"  [Simulated Malicious Client] Configured silent period: {silent_rounds} round(s). Attack will start in Round {attack_start_round}.")

    global_state = {k: v.detach().clone() for k, v in client_model_state_dict.items()}

    # If within silent period, behave benignly (lying low)
    if call_count <= silent_rounds:
        print(f"  [Simulated Malicious Client] In silent period (call {call_count}/{silent_rounds}); behaving benignly.")
        trained = _get_local_trained_state(global_state, local_dataset, device)
        return {k: v.detach().cpu() for k, v in trained.items()}

    attack = ATTACK_TYPE.lower()

    # Dynamic selection across baseline attacks and benign
    if attack == "random":
        attack_pool = ["sign_flip", "gaussian", "scale", "label_flip", "benign"]
        attack = random.choice(attack_pool)
        print(f"  [Simulated Malicious Client] Executing attack: {attack}")

    # 1. Benign mode (no attack this round)
    if attack == "benign":
        trained = _get_local_trained_state(global_state, local_dataset, device)
        return {k: v.detach().cpu() for k, v in trained.items()}

    # 2. Gaussian noise attack
    if attack == "gaussian":
        out = {}
        for key, param in global_state.items():
            p = param.to(device).float()
            scale = p.std().clamp(min=1e-3) * ATTACK_SCALE
            out[key] = (p + torch.randn_like(p) * scale).cpu()
        return out

    # 3. Scaling attack
    if attack == "scale":
        return {
            k: (v.to(device).float() * ATTACK_SCALE).detach().cpu()
            for k, v in global_state.items()
        }

    # 4. Sign-flipping attack
    if attack == "sign_flip":
        trained = _get_local_trained_state(global_state, local_dataset, device)
        return {
            k: (2 * global_state[k].to(device) - trained[k].to(device)).detach().cpu()
            for k in global_state
        }

    # 5. Label-flipping attack
    if attack == "label_flip":
        trained = _get_local_trained_state(global_state, local_dataset, device, label_flip=True)
        return {k: v.detach().cpu() for k, v in trained.items()}

    raise ValueError(f"Unknown ATTACK_TYPE={attack!r}")
