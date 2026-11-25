'''Level 2 federated learning training.'''

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from models import LeNetWide32, LeNet5, SimpleMLP, LeNetWide32Avg


def setSeed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loadData(tx):
    trainDs = torchvision.datasets.MNIST(
        root="./data/mnist", train=True, download=True, transform=tx
    )
    testDs = torchvision.datasets.MNIST(
        root="./data/mnist", train=False, download=True, transform=tx
    )
    res = (trainDs, testDs)
    return res


def partitionDataset(ds, numClients=10, seed=42):
    sz = len(ds) // numClients
    splits = []
    for _ in range(numClients):
        splits.append(sz)
    g = torch.Generator()
    g.manual_seed(seed)
    subsets = random_split(ds, splits, generator=g)
    return subsets


def clientUpdate(cNet, opt, cLdr, dev, ep=1):
    cNet.train()
    lossFn = nn.CrossEntropyLoss()
    for _ in range(ep):
        for data, lbls in cLdr:
            x = data.to(dev)
            y = lbls.to(dev)
            opt.zero_grad()
            out = cNet(x)
            loss = lossFn(out, y)
            loss.backward()
            opt.step()


def serverAggregate(gNet, cNets):
    gDict = gNet.state_dict()
    for k in gDict.keys():
        tList = []
        for i in range(len(cNets)):
            cDict = cNets[i].state_dict()
            t = cDict[k].float()
            tList.append(t)
        stacked = torch.stack(tList, dim=0)
        meanT = stacked.mean(dim=0)
        gDict[k] = meanT
    gNet.load_state_dict(gDict)

    gState = gNet.state_dict()
    for c in cNets:
        c.load_state_dict(gState)


def testModel(net, tLdr, dev):
    net.eval()
    cor = 0
    tot = 0
    with torch.no_grad():
        for data, lbls in tLdr:
            x = data.to(dev)
            y = lbls.to(dev)
            out = net(x)
            _, pred = torch.max(out, 1)
            tot += y.size(0)
            cor += (pred == y).sum().item()
    acc = cor / tot
    return acc


def main():
    setSeed(42)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {dev}")

    # Model choice: "wide" (LeNetWide32, primary), "lenet5" (baseline), "mlp" (SimpleMLP), "wide_avg" (LeNetWide32Avg)
    mType = "wide_avg"
    numClients = 10
    gEpochs = 10
    locEpochs = 1
    lr = 0.0005
    ckptPath = "best_model.pth"

    tx = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    datasets = loadData(tx)
    trainDs = datasets[0]
    testDs = datasets[1]

    clientDsList = partitionDataset(trainDs, numClients, seed=42)
    cLdrs = []
    g = torch.Generator()
    g.manual_seed(42)
    for ds in clientDsList:
        cLdr = DataLoader(ds, batch_size=50, shuffle=True, generator=g)
        cLdrs.append(cLdr)

    testLdr = DataLoader(testDs, batch_size=1000, shuffle=False)

    if mType == "lenet5":
        gNet = LeNet5().to(dev)
    elif mType == "mlp":
        gNet = SimpleMLP().to(dev)
    elif mType == "wide_avg":
        gNet = LeNetWide32Avg().to(dev)
    else:
        gNet = LeNetWide32().to(dev)

    print(f"--- Running Level 2 Exp 6 (model=LeNetWide32, pool=avg, clients={numClients}, rounds={gEpochs}) ---")
    cNets = []
    opts = []
    for _ in range(numClients):
        copiedNet = copy.deepcopy(gNet)
        opt = torch.optim.Adam(copiedNet.parameters(), lr=lr)
        cNets.append(copiedNet)
        opts.append(opt)

    bestAcc = 0.0

    print("Starting Federated Learning...")
    for r in range(gEpochs):
        print(f"Global Round {r + 1}/{gEpochs}")

        for idx in range(numClients):
            clientUpdate(cNets[idx], opts[idx], cLdrs[idx], dev, locEpochs)

        serverAggregate(gNet, cNets)

        acc = testModel(gNet, testLdr, dev)
        print(f"Round {r + 1} Global Test Accuracy: {acc:.4f} ({acc:.2%})")

        if acc > bestAcc:
            bestAcc = acc
            torch.save(gNet.state_dict(), ckptPath)
            print(f"Saved best model with accuracy {bestAcc:.2%} to {ckptPath}")

    print(f"\nFederated training finished. Best Test Accuracy: {bestAcc:.2%}")


if __name__ == '__main__':
    main()
