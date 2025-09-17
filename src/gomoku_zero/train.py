from __future__ import annotations
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from .replay import ReplayBuffer
from .net import PolicyValueNet
from .utils import device, ensure_dir

class LossFn(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, policy_logits, value_pred, target_pi, target_z):
        # policy: CE with π (convert to targets by soft targets -> logit CE with probabilities)
        # 这里用 KL 等价实现：-π·log softmax(logits)
        logp = torch.log_softmax(policy_logits, dim=-1)
        policy_loss = -(target_pi * logp).sum(dim=-1).mean()
        value_loss = self.mse(value_pred, target_z)
        loss = value_loss + policy_loss
        return loss, policy_loss.detach(), value_loss.detach()

def train_loop(cfg, replay: ReplayBuffer, board_size: int, net: PolicyValueNet, steps: int, logdir: str):
    dev = device()
    net = net.to(dev)
    opt = optim.AdamW(net.parameters(), lr=cfg.opt.lr, weight_decay=cfg.opt.weight_decay)
    loss_fn = LossFn()
    ensure_dir(logdir)
    writer = SummaryWriter(logdir)

    global_step = 0
    while global_step < steps:
        s, p, z = replay.sample(cfg.replay.batch_size)
        s = torch.from_numpy(s).to(dev)                # (B,3,N,N)
        p = torch.from_numpy(p).to(dev)                # (B,N*N)
        z = torch.from_numpy(z).to(dev)                # (B,)

        net.train()
        logits, v = net(s)
        loss, pl, vl = loss_fn(logits, v, p, z)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()

        if global_step % 50 == 0:
            writer.add_scalar("loss/total", loss.item(), global_step)
            writer.add_scalar("loss/policy", pl.item(), global_step)
            writer.add_scalar("loss/value", vl.item(), global_step)

        global_step += 1

    writer.flush(); writer.close()
