'''Models specifically defined for ablation studies.'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    '''Simple 2-layer MLP baseline (non-convolutional).'''
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, xb):
        xb = xb.view(-1, 28 * 28)
        xb = F.relu(self.fc1(xb))
        xb = F.relu(self.fc2(xb))
        return self.fc3(xb)


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.avg_pool2d(F.relu(self.conv1(xb)), 2)
        xb = F.avg_pool2d(F.relu(self.conv2(xb)), 2)
        xb = xb.flatten(1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(xb)))))


class LeNetAvgPool(nn.Module):
    '''LeNetWide with Average Pooling for ablation.'''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 48, 5)
        self.fc1 = nn.Linear(48 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.avg_pool2d(F.relu(self.conv1(xb)), 2)
        xb = F.avg_pool2d(F.relu(self.conv2(xb)), 2)
        xb = xb.flatten(1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(xb)))))


class LeNetWide32(nn.Module):
    '''LeNetWide with Max Pooling.'''
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 48, 5)
        self.fc1 = nn.Linear(48 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.max_pool2d(F.relu(self.conv1(xb)), 2)
        xb = F.max_pool2d(F.relu(self.conv2(xb)), 2)
        xb = xb.flatten(1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(xb)))))
