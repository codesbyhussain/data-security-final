'''Level 3 robust federated learning training.'''

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from models import LeNetWide32, LeNet5
from malicious_client import generate_malicious_update


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


def trimmedMean(cNets, bIdxs, trimRatio=0.1):
    states = []
    for i in bIdxs:
        s = cNets[i].state_dict()
        states.append(s)

    n = len(states)
    if n <= 2:
        avg = {}
        for k in states[0].keys():
            tList = []
            for s in states:
                tList.append(s[k].float())
            stacked = torch.stack(tList, dim=0)
            avg[k] = stacked.mean(dim=0).to(dtype=states[0][k].dtype)
        return avg

    # k = max(1, int(n * trimRatio))
    k = max(0, round(n * trimRatio))
    avg = {}
    for key in states[0].keys():
        tList = []
        for s in states:
            tList.append(s[key].float())
        stacked = torch.stack(tList, dim=0)
        origShape = stacked.shape[1:]
        stackedFlat = stacked.view(n, -1)
        if k > 0 and 2 * k < n:
            sortedFlat, _ = torch.sort(stackedFlat, dim=0)
            trimmed = sortedFlat[k:n - k]
            meanFlat = trimmed.mean(dim=0)
        else:
            meanFlat = stackedFlat.mean(dim=0)
        avg[key] = meanFlat.view(origShape).to(dtype=states[0][key].dtype)
    return avg


def detectAndAggregate(gNet, cNets, curRound):
    gDict = gNet.state_dict()
    numClients = len(cNets)

    # 0. DEFENSE A: Check for NaN / Inf injection attacks
    corruptedClients = []
    for i, c in enumerate(cNets):
        cDict = c.state_dict()
        for param in cDict.values():
            if not torch.isfinite(param).all():
                corruptedClients.append(i)
                break

    if corruptedClients and curRound >= 2:
        flaggedClient = corruptedClients[0]
        print(f"  [Defense] Flagged malicious client: Client {flaggedClient} (detected NaN/Inf values)")
        bIdxs = []
        for i in range(numClients):
            if i != flaggedClient:
                bIdxs.append(i)
        newGDict = trimmedMean(cNets, bIdxs, trimRatio=0.1)
        gNet.load_state_dict(newGDict)
        gState = gNet.state_dict()
        for c in cNets:
            c.load_state_dict(gState)
        return flaggedClient

    # 1. GET DELTA VECTORS FOR DEFENSES: Update delta vectors (w_i - w_global)
    deltas = []
    for c in cNets:
        cDict = c.state_dict()
        diffs = []
        for k in gDict.keys():
            diff = (cDict[k].float() - gDict[k].float().to(cDict[k].device)).reshape(-1)
            diffs.append(diff)
        catDiffs = torch.cat(diffs)
        deltas.append(catDiffs)

    # 2. DEFENSE B: L2 norms and median 
    norms = []
    for d in deltas:
        nVal = torch.norm(d).item()
        norms.append(nVal)
    # sorted_norms = sorted(norms)
    # med_norm = sorted_norms[len(sorted_norms) // 2]
    medNorm = torch.tensor(norms).median().item()

    # 3. DEFENSE C: Pairwise cosine consensus scores
    cosScores = []
    for i in range(numClients):
        sims = []
        for j in range(numClients):
            if i != j:  # Only compare to other clients
                sim = torch.cosine_similarity(deltas[i].unsqueeze(0), deltas[j].unsqueeze(0)).item()  # Unsqueeze needed for pytorch's cosine_similarity.
                sims.append(sim)
        simAvg = sum(sims) / len(sims)
        cosScores.append(simAvg)

    # 4. DEFENSE D: Multi-Krum scores
    f = 1  # maximum number of attackers.
    keepNeighbors = max(1, numClients - f - 2)  # Removes more distant clients, reducing outlier impact.
    krumScores = []
    for i in range(numClients):
        # Calculate distances between each client, take norm, and sort smallest to largest.
        dists = []
        for j in range(numClients):
            if i != j:
                distVal = (deltas[i] - deltas[j]).norm().item()
                dists.append(distVal)
        dists.sort()
        sumNeighbors = sum(dists[:keepNeighbors])  # Sum largest for scores.
        krumScores.append(sumNeighbors)

    def minmax(vals):
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-12:  # if values are too close, return 0
            res = []
            for _ in range(len(vals)):
                res.append(0.0)
            return res
        res = []
        for v in vals:
            normed = (v - lo) / (hi - lo)
            res.append(normed)
        return res

    invCos = []
    for c in cosScores:
        invCos.append(1.0 - c)  # High is good, so invert.
    cosAnom = minmax(invCos)

    l2Diffs = []
    for nr in norms:
        diffVal = abs(nr - medNorm) / max(medNorm, 1e-12)
        l2Diffs.append(diffVal)
    l2Anom = minmax(l2Diffs)

    krumAnom = minmax(krumScores)

    combinedScores = []
    for c, l, k in zip(cosAnom, l2Anom, krumAnom):
        mScore = max(c, l, k)
        combinedScores.append(mScore)
    # combinedScores = [mean(c, l, k) for c, l, k in zip(cosAnom, l2Anom, krumAnom)] # ablation test

    flaggedClient = -1
    if curRound >= 2:
        hardOutliers = []
        for i in range(numClients):  
            if norms[i] > 3.0 * max(medNorm, 1e-12) or cosScores[i] < 0.0:  # Checks for enormous or opposite direction updates.
                hardOutliers.append(i)

        if hardOutliers:
            hardOutliers.sort(key=lambda idx: combinedScores[idx], reverse=True)
            flaggedClient = hardOutliers[0]
            flagType = "hard"
        else:  # Otherwise, flags the client with the highest anomaly score.
            flaggedClient = int(torch.tensor(combinedScores).argmax().item())
            flagType = "soft"

        print(f"  [Defense] Flagged malicious client: Client {flaggedClient} (norm={norms[flaggedClient]:.2f}, cos={cosScores[flaggedClient]:.4f}, type={flagType})")

    bIdxs = []
    for i in range(numClients):
        if i != flaggedClient:
            bIdxs.append(i)
    print(f"  [Server] Robust aggregation (Trimmed Mean) over {len(bIdxs)} benign clients...")

    newGDict = trimmedMean(cNets, bIdxs, trimRatio=0.1)
    gNet.load_state_dict(newGDict)

    gState = gNet.state_dict()
    for c in cNets:
        c.load_state_dict(gState)

    return flaggedClient


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

    # Model choice: "wide" (LeNetWide32, primary) or "lenet5" (baseline)
    mType = "wide"
    numClients = 10
    malClientId = 9
    atkStartRound = 2
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
    else:
        gNet = LeNetWide32().to(dev)

    print(f"--- Running Level 3 (model={gNet.__class__.__name__}, trim_ratio=0.1, fusion=max, rounds={gEpochs}) ---")
    cNets = []
    opts = []
    for _ in range(numClients):
        copiedNet = copy.deepcopy(gNet)
        opt = torch.optim.Adam(copiedNet.parameters(), lr=lr)
        cNets.append(copiedNet)
        opts.append(opt)

    bestAcc = 0.0
    caughtRounds = []

    print("Starting Level 3 Robust Federated Learning...")
    for r in range(1, gEpochs + 1):
        print(f"\nGlobal Round {r}/{gEpochs}")

        for idx in range(numClients):
            if idx == malClientId and r >= atkStartRound:
                # === MALICIOUS CLIENT LOGIC HERE ===
                gState = gNet.state_dict()
                malState = generate_malicious_update(
                    gState,
                    clientDsList[idx],
                    dev
                )
                cNets[idx].load_state_dict(malState)
            else:
                clientUpdate(cNets[idx], opts[idx], cLdrs[idx], dev, locEpochs)

        detectedId = detectAndAggregate(gNet, cNets, r)

        if r >= atkStartRound:
            if detectedId == malClientId:
                caughtRounds.append(r)
                print(f"  [Result] Success: Correctly identified malicious client {malClientId}!")
            else:
                print(f"  [Result] Warning: Injected client was {malClientId}, but flagged client was {detectedId}")

        acc = testModel(gNet, testLdr, dev)
        print(f"Global Model Test Accuracy after round {r}: {acc:.4f} ({acc:.2%})")

        if acc > bestAcc:
            bestAcc = acc
            torch.save(gNet.state_dict(), ckptPath)
            print(f"Saved best model with accuracy {bestAcc:.2%} to {ckptPath}")

    torch.save(gNet.state_dict(), 'robust_federated_model.pth')
    print(f"\nTraining completed. Malicious client caught in rounds: {caughtRounds}")
    print(f"Best Test Accuracy: {bestAcc:.2%}")


if __name__ == '__main__':
    main()
