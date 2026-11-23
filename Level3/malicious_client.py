### Level 3 Code Structure Requirement (For Demo Day Testing)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import ConvNet

# Attack selection for self-testing:
# 'sign_flip' | 'gaussian' | 'scale' | 'label_flip' | 'targeted_flip' | 'zero'
ATTACK_TYPE = "sign_flip"
ATTACK_SCALE = 10.0


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
    global_state = {k: v.detach().clone() for k, v in client_model_state_dict.items()}

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

    # 3. Zeroing / Freeze attack
    if attack == "zero":
        return {
            k: torch.zeros_like(v).cpu()
            for k, v in global_state.items()
        }

    # For training-based attacks: train locally for 1 epoch, then modify
    model = ConvNet().to(device)
    model.load_state_dict(global_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    loader = DataLoader(local_dataset, batch_size=50, shuffle=True)

    model.train()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if attack == "label_flip":
            yb = 9 - yb  # Invert all labels (0->9, 1->8, etc.)
        elif attack == "targeted_flip":
            yb = torch.where(yb == 1, torch.tensor(7, device=device), yb)  # Target 1 -> 7

        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    trained = model.state_dict()

    # 4. Label-flipping attacks
    if attack in ("label_flip", "targeted_flip"):
        return {k: v.detach().cpu() for k, v in trained.items()}

    # 5. Sign-flipping attack: reverse update direction (2 * global - trained)
    if attack == "sign_flip":
        return {
            k: (2 * global_state[k].to(device) - trained[k].to(device)).detach().cpu()
            for k in global_state
        }

    raise ValueError(f"Unknown ATTACK_TYPE={ATTACK_TYPE!r}")
