'''Level 1 centralized training.'''

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models import LeNetWide32, LeNet5, SimpleMLP


def setSeed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loadData(bs=50, seed=42):
    tx = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainDs = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=tx)
    testDs = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=tx)

    print("The number of training data:", len(trainDs))
    print("The number of testing data:", len(testDs))

    g = torch.Generator()
    g.manual_seed(seed)

    trainLdr = DataLoader(dataset=trainDs, batch_size=bs, shuffle=True, generator=g)
    testLdr = DataLoader(dataset=testDs, batch_size=len(testDs), shuffle=False)

    print(f"training has: {len(trainLdr)} batch of data!")
    print(f"testing has: {len(testLdr)} batch of data!")
    res = (trainLdr, testLdr)
    return res


def main():
    setSeed(42)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {dev}")

    # Model choice: "wide" (LeNetWide32, primary), "lenet5" (baseline), or "mlp" (SimpleMLP)
    mType = "wide"
    lr = 0.0005
    nEpochs = 5
    bs = 50
    ckptPath = "best_model.pth"

    dataLdrs = loadData(bs=bs)
    trainLdr = dataLdrs[0]
    testLdr = dataLdrs[1]

    if mType == "lenet5":
        net = LeNet5().to(dev)
    elif mType == "mlp":
        net = SimpleMLP().to(dev)
    else:
        net = LeNetWide32().to(dev)

    print(f"--- Running Level 1 (model={net.__class__.__name__}, epochs={nEpochs}, bs={bs}, lr={lr}) ---")

    print(f"Training model: {net.__class__.__name__}")
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    bestAcc = 0.0

    for ep in range(nEpochs):
        net.train()
        runLoss = 0.0
        runAcc = 0.0

        for i, batch in enumerate(trainLdr):
            x = batch[0].to(dev)
            y = batch[1].to(dev)

            opt.zero_grad()
            out = net(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()

            runLoss += loss.item()
            _, pred = torch.max(out, 1)
            eq = (pred == y).sum().item()
            cnt = y.size(0)
            accVal = eq / cnt
            runAcc += accVal

            if i % 100 == 99:
                avgL = runLoss / 100
                avgA = runAcc / 100
                print(f"epoch: {ep + 1}, batch: {i + 1}, loss: {avgL:.6f}, acc: {avgA:.2%}")
                runLoss = 0.0
                runAcc = 0.0

        # Evaluate on test set after epoch
        net.eval()
        cor = 0
        tot = 0
        with torch.no_grad():
            for tBatch in testLdr:
                tx = tBatch[0].to(dev)
                ty = tBatch[1].to(dev)
                tOut = net(tx)
                _, tPred = torch.max(tOut, 1)
                cor += (tPred == ty).sum().item()
                tot += ty.size(0)

        epAcc = cor / tot
        print(f"Epoch {ep + 1} test accuracy: {epAcc:.2%}")

        if epAcc > bestAcc:
            bestAcc = epAcc
            torch.save(net.state_dict(), ckptPath)
            print(f"Saved best model with accuracy: {bestAcc:.2%} to {ckptPath}")


if __name__ == "__main__":
    main()
