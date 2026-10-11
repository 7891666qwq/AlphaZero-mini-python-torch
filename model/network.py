# model/network.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // r)
        self.fc2 = nn.Linear(ch // r, ch)

    def forward(self, x):
        B, C, _, _ = x.shape
        s = x.mean(dim=(2, 3))
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s)).view(B, C, 1, 1)
        return x * s


class ResidualBlock(nn.Module):
    def __init__(self, ch, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.se = SEBlock(ch) if use_se else None

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.se is not None:
            y = self.se(y)
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, board_size=9, in_ch=8, ch=128, blocks=10, use_se=True):
        super().__init__()
        self.board_size = board_size
        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(ch)
        self.blocks = nn.ModuleList([ResidualBlock(ch, use_se) for _ in range(blocks)])

        self.p_conv = nn.Conv2d(ch, 2, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * board_size * board_size,
                              board_size * board_size + 1)

        self.v_conv = nn.Conv2d(ch, 1, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(board_size * board_size, 256)
        self.v_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        for blk in self.blocks:
            x = blk(x)

        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.p_fc(p)

        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))
        return p, v

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        logits, v = self.forward(x)
        return F.softmax(logits, dim=1), v.squeeze(-1)
