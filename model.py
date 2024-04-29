import torch
import torch.nn as nn

class AConvBlock(nn.Module):
    def __init__(self):
        super(AConvBlock, self).__init__()

        block = [nn.Conv2d(3, 3, 3, padding=1)]
        block += [nn.PReLU()]

        block += [nn.Conv2d(3, 3, 3, padding=1)]
        block += [nn.PReLU()]

        block += [nn.AdaptiveAvgPool2d((1, 1))]
        block += [nn.Conv2d(3, 3, 1)]
        block += [nn.PReLU()]
        block += [nn.Conv2d(3, 3, 1)]
        block += [nn.PReLU()]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)

class tConvBlock(nn.Module):
    def __init__(self):
        super(tConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(6, 8, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(8, 8, 3, padding=5, dilation=5)
        self.conv4 = nn.Conv2d(8, 3, 3, padding=1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.prelu(x1)

        x2 = self.conv2(x1)
        x2 = self.prelu(x2)

        x3 = self.conv3(x2)
        x3 = self.prelu(x3)

        x = torch.cat((x * 0 + x3, x), 1)

        x = self.conv4(x)
        x = self.prelu(x)

        return x

class PhysicalNN(nn.Module):
    def __init__(self):
        super(PhysicalNN, self).__init__()

        self.ANet = AConvBlock()
        self.tNet = tConvBlock()

    def forward(self, x):
        A = self.ANet(x)
        t = self.tNet(torch.cat((x * 0 + A, x), 1))
        out = ((x - A) * t + A)
        return torch.clamp(out, 0., 1.)
