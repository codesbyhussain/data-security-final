### Level 3 Code Structure Requirement (For Demo Day Testing)
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import LeNetWide32, LeNet5, SimpleMLP

# Attack selection:
# 'random' | 'sign_flip' | 'gaussian' | 'scale' | 'label_flip' | 'benign'
ATTACK_TYPE = "random"
ATTACK_SCALE = 10.0

# Silent period: number of rounds to wait after round 2 before starting attack.
# Set to an integer (0 = starts at round 2, 2 = starts at round 4), or None for random (0 to 3 rounds).
SILENT_PERIOD = None
callCount = 0
silentRounds = None


def getLocalTrainedState(gState, localDs, dev, labelFlip=False):
    # Detect whether global model is SimpleMLP, LeNet5, or LeNetWide32
    if "conv1.weight" not in gState:
        net = SimpleMLP().to(dev)
    elif "fc1.weight" in gState and gState["fc1.weight"].shape[0] == 120:
        net = LeNet5().to(dev)
    else:
        net = LeNetWide32().to(dev)

    net.load_state_dict(gState)
    opt = torch.optim.Adam(net.parameters(), lr=0.0005)
    ldr = DataLoader(localDs, batch_size=50, shuffle=True)

    net.train()
    for batch in ldr:
        xb = batch[0].to(dev)
        yb = batch[1].to(dev)
        if labelFlip:
            yb = 9 - yb

        out = net(xb)
        loss = F.cross_entropy(out, yb)
        loss.backward()
        opt.step()
        opt.zero_grad()

    trainedState = net.state_dict()
    return trainedState


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
    global callCount, silentRounds
    callCount += 1

    # Determine silent period on first invocation
    if silentRounds is None:
        if SILENT_PERIOD is None:
            silentRounds = random.randint(0, 3)
        else:
            silentRounds = SILENT_PERIOD
        atkStartRound = 2 + silentRounds
        print(f"  [Simulated Malicious Client] Configured silent period: {silentRounds} round(s). Attack will start in Round {atkStartRound}.")

    gState = {}
    for k, v in client_model_state_dict.items():
        gState[k] = v.detach().clone()

    # If within silent period, behave benignly (lying low)
    if callCount <= silentRounds:
        print(f"  [Simulated Malicious Client] In silent period (call {callCount}/{silentRounds}); behaving benignly.")
        trained = getLocalTrainedState(gState, local_dataset, device)
        res = {}
        for k, v in trained.items():
            res[k] = v.detach().cpu()
        return res

    atk = ATTACK_TYPE.lower()

    # Dynamic selection across baseline attacks and benign
    if atk == "random":
        atkPool = ["sign_flip", "gaussian", "scale", "label_flip", "benign"]
        atk = random.choice(atkPool)
        print(f"  [Simulated Malicious Client] Executing attack: {atk}")

    # 1. Benign mode (no attack this round)
    if atk == "benign":
        trained = getLocalTrainedState(gState, local_dataset, device)
        res = {}
        for k, v in trained.items():
            res[k] = v.detach().cpu()
        return res

    # 2. Gaussian noise attack
    if atk == "gaussian":
        out = {}
        for k, param in gState.items():
            p = param.to(device).float()
            scale = p.std().clamp(min=1e-3) * ATTACK_SCALE
            noise = torch.randn_like(p) * scale
            out[k] = (p + noise).cpu()
        return out

    # 3. Scaling attack
    if atk == "scale":
        out = {}
        for k, v in gState.items():
            scaled = (v.to(device).float() * ATTACK_SCALE).detach().cpu()
            out[k] = scaled
        return out

    # 4. Sign-flipping attack
    if atk == "sign_flip":
        trained = getLocalTrainedState(gState, local_dataset, device)
        out = {}
        for k in gState.keys():
            flipped = (2 * gState[k].to(device) - trained[k].to(device)).detach().cpu()
            out[k] = flipped
        return out

    # 5. Label-flipping attack
    if atk == "label_flip":
        trained = getLocalTrainedState(gState, local_dataset, device, labelFlip=True)
        res = {}
        for k, v in trained.items():
            res[k] = v.detach().cpu()
        return res

    raise ValueError(f"Unknown ATTACK_TYPE={atk!r}")
