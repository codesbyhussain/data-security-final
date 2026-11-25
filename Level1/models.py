'''Shared neural network models.'''

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Based on the tutorial:
https://github.com/pytorch/tutorials/blob/main/beginner_source/introyt/modelsyt_tutorial.py
'''


class SimpleMLP(nn.Module):
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
        xb = F.avg_pool2d(F.relu(self.conv1(xb)), 2)  # Avg pool more similar to LeCun's 1998 paper
        xb = F.avg_pool2d(F.relu(self.conv2(xb)), 2)
        xb = xb.flatten(1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(xb)))))


class LeNetWide32(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 48, 5)
        self.fc1 = nn.Linear(48 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.max_pool2d(F.relu(self.conv1(xb)), 2)  # Max pool as per the pytorch tutorial
        xb = F.max_pool2d(F.relu(self.conv2(xb)), 2)
        xb = xb.flatten(1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(xb)))))


# Default model alias
ConvNet = LeNetWide32
