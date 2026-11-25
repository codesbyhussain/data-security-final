'''Level 1 evaluation script.'''

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import LeNetWide32, LeNet5, SimpleMLP


def loadTestData():
    tx = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    testDs = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=tx)
    testLdr = DataLoader(dataset=testDs, batch_size=len(testDs), shuffle=False)
    print("The number of testing data:", len(testDs))
    return testLdr


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model choice: "wide" (LeNetWide32, primary), "lenet5" (baseline), or "mlp" (SimpleMLP)
    mType = "wide"
    ckptPath = "best_model.pth"

    if not os.path.exists(ckptPath):
        print(f"Checkpoint {ckptPath} not found. Please run train.py first.")
        return

    testLdr = loadTestData()

    if mType == "lenet5":
        net = LeNet5().to(dev)
    elif mType == "mlp":
        net = SimpleMLP().to(dev)
    else:
        net = LeNetWide32().to(dev)

    loadedWeights = torch.load(ckptPath, map_location=dev, weights_only=True)
    net.load_state_dict(loadedWeights)
    net.eval()

    cor = 0
    tot = 0

    with torch.no_grad():
        for batch in testLdr:
            x = batch[0].to(dev)
            y = batch[1].to(dev)
            out = net(x)
            _, pred = torch.max(out, 1)
            cor += (pred == y).sum().item()
            tot += y.size(0)

    acc = cor / tot
    print(f"Test Accuracy ({net.__class__.__name__}): {acc:.2%} ({cor}/{tot})")


if __name__ == "__main__":
    main()
