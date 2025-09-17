from __future__ import annotations
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(x + h)

class PolicyValueNet(nn.Module):
    def __init__(self, board_size: int = 15, in_ch: int = 3, ch: int = 128, blocks: int = 12):
        super().__init__()
        self.board_size = board_size
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[ResidualBlock(ch) for _ in range(blocks)])
        # policy head
        self.p_head = nn.Sequential(
            nn.Conv2d(ch, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.p_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        # value head
        self.v_head = nn.Sequential(
            nn.Conv2d(ch, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.v_fc = nn.Sequential(
            nn.Linear(board_size * board_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    @torch.no_grad()
    def initial_infer(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 3, N, N) -> logits (B, N*N), value (B, 1)
        return self.forward(x)

    def forward(self, x):
        b = x.shape[0]
        h = self.stem(x)
        h = self.res(h)
        # policy
        p = self.p_head(h)
        p = p.reshape(b, -1)
        p = self.p_fc(p)  # logits
        # value
        v = self.v_head(h).reshape(b, -1)
        v = self.v_fc(v)  # [-1, 1]
        return p, v.squeeze(-1)

def numerical_params(net:nn.Module):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

if __name__ == "__main__":
    net = PolicyValueNet()
    print(numerical_params(net))
    print(net.initial_infer(torch.randn(1, 3, 15, 15)))