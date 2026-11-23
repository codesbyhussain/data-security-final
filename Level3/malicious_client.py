### Level 3 Code Structure Requirement (For Demo Day Testing)
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import LeNetWide32

# =============================================================================
# Attack Options:
#
# Category A (Cumulative Drift - best executed in consecutive rounds):
#   - 'alie': A Little Is Enough (Baruch et al., NeurIPS 2019)
#   - 'ipm': Inner Product Manipulation (Xie et al., ICML 2020)
#   - 'label_flip': All labels inverted (0 <-> 9)
#   - 'targeted_flip': Specific label flipped (1 -> 7)
#   - 'zero': Weights zeroed / frozen
#
# Category B (Instant Disruption - best executed in random / isolated rounds):
#   - 'sign_flip': Gradient direction flipped (2 * global - trained)
#   - 'gaussian': Scaled Gaussian noise added to weights
#   - 'scale': Weights scaled by large constant factor
#   - 'min_max': Min-Max boundary evasion (Fang et al., USENIX 2020)
#
# Dynamic Modes:
#   - 'random': Randomly selects an attack from Category B or 'benign' (no attack)
#   - 'benign': Acts normally, submits legitimate local training
# =============================================================================

ATTACK_TYPE = "random"
ATTACK_SCALE = 10.0


def _get_local_trained_state(global_state, local_dataset, device, label_flip=False, targeted_flip=False):
    # Train 1 local epoch on benign or flipped labels to get realistic gradient updates
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
        elif targeted_flip:
            yb = torch.where(yb == 1, torch.tensor(7, device=device), yb)

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
    attack = ATTACK_TYPE.lower()

    # Dynamic selection across Category B (Instant Disruption) and Benign
    if attack == "random":
        attack_pool = [
            "sign_flip",
            "gaussian",
            "scale",
            "min_max",
            "benign",
        ]
        attack = random.choice(attack_pool)
        print(f"  [Simulated Malicious Client] Executing attack: {attack}")

    global_state = {k: v.detach().clone() for k, v in client_model_state_dict.items()}

    # -------------------------------------------------------------------------
    # Dormant / Benign Mode (No attack in this round)
    # -------------------------------------------------------------------------
    if attack == "benign":
        trained = _get_local_trained_state(global_state, local_dataset, device)
        return {k: v.detach().cpu() for k, v in trained.items()}

    # -------------------------------------------------------------------------
    # Category B: Instant Disruption Attacks
    # -------------------------------------------------------------------------

    # 1. Gaussian noise attack
    if attack == "gaussian":
        out = {}
        for key, param in global_state.items():
            p = param.to(device).float()
            scale = p.std().clamp(min=1e-3) * ATTACK_SCALE
            out[key] = (p + torch.randn_like(p) * scale).cpu()
        return out

    # 2. Scaling attack
    if attack == "scale":
        return {
            k: (v.to(device).float() * ATTACK_SCALE).detach().cpu()
            for k, v in global_state.items()
        }

    # 3. Sign-flipping attack
    if attack == "sign_flip":
        trained = _get_local_trained_state(global_state, local_dataset, device)
        return {
            k: (2 * global_state[k].to(device) - trained[k].to(device)).detach().cpu()
            for k in global_state
        }

    # 4. Min-Max boundary attack (Fang et al., USENIX Security 2020)
    if attack == "min_max":
        trained = _get_local_trained_state(global_state, local_dataset, device)
        out = {}
        for k in global_state:
            g = global_state[k].to(device).float()
            t = trained[k].to(device).float()
            delta = t - g
            norm = delta.norm().clamp(min=1e-5)
            unit_reverse = -delta / norm
            out[k] = (g + unit_reverse * (norm * 0.9)).detach().cpu()
        return out

    # -------------------------------------------------------------------------
    # Category A: Cumulative Drift Attacks
    # -------------------------------------------------------------------------

    # 5. ALIE: A Little Is Enough (Baruch et al., NeurIPS 2019)
    if attack == "alie":
        trained = _get_local_trained_state(global_state, local_dataset, device)
        z_max = 1.0
        out = {}
        for k in global_state:
            g = global_state[k].to(device).float()
            t = trained[k].to(device).float()
            delta = t - g
            std = delta.std().clamp(min=1e-5)
            out[k] = (g - z_max * std * torch.sign(delta)).detach().cpu()
        return out

    # 6. IPM: Inner Product Manipulation (Xie et al., ICML 2020)
    if attack == "ipm":
        trained = _get_local_trained_state(global_state, local_dataset, device)
        epsilon = 0.5
        out = {}
        for k in global_state:
            g = global_state[k].to(device).float()
            t = trained[k].to(device).float()
            delta = t - g
            out[k] = (g - epsilon * delta).detach().cpu()
        return out

    # 7. Zeroing / Freezing attack
    if attack == "zero":
        return {k: torch.zeros_like(v).cpu() for k, v in global_state.items()}

    # 8. Label-flipping attacks (untargeted and targeted)
    if attack in ("label_flip", "targeted_flip"):
        trained = _get_local_trained_state(
            global_state, local_dataset, device,
            label_flip=(attack == "label_flip"),
            targeted_flip=(attack == "targeted_flip")
        )
        return {k: v.detach().cpu() for k, v in trained.items()}

    raise ValueError(f"Unknown ATTACK_TYPE={attack!r}")
