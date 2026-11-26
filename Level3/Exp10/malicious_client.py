### Level 3 Code Structure Requirement (For Demo Day Testing)
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from models import LeNetWide32, LeNet5

# Attack selection:
# 'random', 'sign_flip', 'gaussian', 'scale', 'label_flip',  'benign'
ATTACK_TYPE = "random"
ATTACK_SCALE = 10.0

# Silent period: number of rounds to wait after round 2 before starting attack.
# Set to an integer 0 starts at round 2, 1 starts at round 3 etc
SILENT_PERIOD = None
callCount = 0
silentRounds = None


def getLocalTrainedState(gState, localDs, device, labelFlip=False):
    '''This is needed for label/sign flipping attacks because they require local training. Our gaussian and scaling attacks do not retrain, but instead just attack the global parameters directly.
    '''
    # Detect whether global model is LeNet5 or LeNetWide32 from parameter shapes
    if "fc1.weight" in gState and gState["fc1.weight"].shape[0] == 120:
        client_model = LeNet5().to(device)
    else:
        client_model = LeNetWide32().to(device)

    client_model.load_state_dict(gState)
    optimizer = torch.optim.Adam(client_model.parameters(), lr=0.0005)
    train_loader = DataLoader(localDs, batch_size=50, shuffle=True)

    # Client-side local training
    client_model.train()
    loss_fn = nn.CrossEntropyLoss()

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        if labelFlip:
            labels = 9 - labels
        optimizer.zero_grad()
        output = client_model(data)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

    return client_model.state_dict()


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
        res = getLocalTrainedState(gState, local_dataset, device)
        return res

    atk = ATTACK_TYPE.lower()

    # Dynamic selection across baseline attacks and benign
    if atk == "random":
        atkPool = ["sign_flip", "gaussian", "scale", "label_flip", "benign"]
        atk = random.choice(atkPool)
        print(f"  [Simulated Malicious Client] Executing attack: {atk}")

    # 1. Benign mode (no attack this round)
    if atk == "benign":
        res = getLocalTrainedState(gState, local_dataset, device)
        return res

    # 2. Gaussian noise attack
    if atk == "gaussian":
        out = {}
        for k, params in gState.items():
            scale = params.std().clamp(min=1e-3) * ATTACK_SCALE # min >0 here ensures that some attack is applied under all cases
            noise = torch.randn_like(params) * scale
            out[k] = params + noise
        return out

    # 3. Scaling attack
    if atk == "scale":
        out = {}
        for k, params in gState.items():
            out[k] = params * ATTACK_SCALE
        return out

    # 4. Sign-flipping attack
    if atk == "sign_flip":
        trained = getLocalTrainedState(gState, local_dataset, device)
        out = {}
        for k in gState:
            delta = trained[k] - gState[k]
            out[k] = gState[k] - delta
        return out

    # 5. Label-flipping attack
    if atk == "label_flip":
        trained = getLocalTrainedState(gState, local_dataset, device, labelFlip=True)
        return res

    raise ValueError(f"Unknown ATTACK_TYPE={atk!r}")
