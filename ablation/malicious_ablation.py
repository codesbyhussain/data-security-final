'''Ablation malicious client generator supporting all ablation model architectures.'''

import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models_ablation import SimpleMLP, LeNet5, LeNetWide32

ATTACK_TYPE = "random"
ATTACK_SCALE = 10.0


def _get_local_trained_state(global_state, local_dataset, device, label_flip=False):
    # Detect model based on architecture shapes
    if "conv1.weight" not in global_state:
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


def generate_ablation_malicious_update(client_model_state_dict, local_dataset, device):
    global_state = {k: v.detach().clone() for k, v in client_model_state_dict.items()}
    attack_pool = ["sign_flip", "gaussian", "scale", "label_flip", "benign"]
    attack = random.choice(attack_pool)

    if attack == "benign":
        trained = _get_local_trained_state(global_state, local_dataset, device)
        return {k: v.detach().cpu() for k, v in trained.items()}

    if attack == "gaussian":
        out = {}
        for key, param in global_state.items():
            p = param.to(device).float()
            scale = p.std().clamp(min=1e-3) * ATTACK_SCALE
            out[key] = (p + torch.randn_like(p) * scale).cpu()
        return out

    if attack == "scale":
        return {k: (v.to(device).float() * ATTACK_SCALE).detach().cpu() for k, v in global_state.items()}

    if attack == "sign_flip":
        trained = _get_local_trained_state(global_state, local_dataset, device)
        return {k: (2 * global_state[k].to(device) - trained[k].to(device)).detach().cpu() for k in global_state}

    if attack == "label_flip":
        trained = _get_local_trained_state(global_state, local_dataset, device, label_flip=True)
        return {k: v.detach().cpu() for k, v in trained.items()}

    return global_state
