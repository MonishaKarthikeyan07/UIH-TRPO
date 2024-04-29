import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)  # Use non-inplace ReLU
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.norm(out)
        return out

class PhysicalNN(nn.Module):
    def __init__(self):
        super(PhysicalNN, self).__init__()
        self.tNet = Block(6, 64)
        self.block = Block(64, 64)

    def forward(self, x):
        t = self.tNet(torch.cat((x * 0 + A, x), 1))
        return self.block(t)
