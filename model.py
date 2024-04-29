import torch
import torch.nn as nn

class PhysicalNN(nn.Module):
    def __init__(self):
        super(PhysicalNN, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.aconv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.tconv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(32 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.block(x)
        x2 = self.aconv(x)
        t = torch.cat((x2, x), 1)
        t = self.tconv(t)
        t = t.view(t.size(0), -1)
        t = torch.relu(self.fc1(t))
        t = self.fc2(t)
        return self.softmax(t)
