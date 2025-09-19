import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepMod(nn.Module):
    def __init__(self, c_in, k, l, mp=True) -> None:
        super().__init__()

        c_out = k * 2 ** l

        self.conv1 = nn.Conv2d(c_in, c_out, 5, padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(c_out)

        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(c_out)

        self.mp = nn.MaxPool2d((2, 1), ceil_mode=True) if mp else nn.Identity()

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))

        x = self.mp(x)

        return x


class DeepSquare(nn.Module):
    def __init__(self, in_channels=8, num_channels=8, dropout=0.3) -> None:
        super().__init__()
        channel = num_channels

        self.layer1 = DeepMod(in_channels, channel, 0)  # 8 * 2 ** 0 = 8
        self.layer2 = DeepMod(channel, channel, 1)  # 8 * 2 ** 1 = 16
        self.layer3 = DeepMod(channel * 2, channel, 2, mp=False)  # 8 * 2 ** 2 = 32
        self.layer4 = DeepMod(channel * 4, channel, 2, mp=False)  # 8 * 2 ** 2 = 32
        self.layer5 = DeepMod(channel * 4, channel, 3, mp=False)  # 8 * 2 ** 3 = 64
        self.layer6 = DeepMod(channel * 8, channel, 3, mp=False)  # 8 * 2 ** 3 = 64

        self.dropout = nn.Dropout(dropout)

        self.output_dim = channel * 8  # Output dimension is the number of channels in the last layer

    def forward(self, x):
        x = self.dropout(self.layer1(x))
        x = self.dropout(self.layer2(x))
        x = self.dropout(self.layer3(x))
        x = self.dropout(self.layer4(x))
        x = self.dropout(self.layer5(x))
        x = self.dropout(self.layer6(x))

        return x.mean(dim=2).transpose(-1, -2)


