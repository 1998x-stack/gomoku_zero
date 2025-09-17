from __future__ import annotations
import argparse, os, time, numpy as np, torch, yaml
from loguru import logger
from .config import Cfg, default_cfg_path
from .utils import set_seed, device, ensure_dir
from .replay import ReplayBuffer
from .net import PolicyValueNet
from .selfplay import selfplay_many
from .train import train_loop
from .evaluate import evaluate_models
from .serialization import save_checkpoint, load_checkpoint

def load_cfg(cfg_path: str | None) -> Cfg:
    if cfg_path and os.path.exists(cfg_path):
        return Cfg.load(cfg_path)
    # fallback: 项目自带默认
    return Cfg.load(default_cfg_path())

def cmd_selfplay(args):
    cfg = load_cfg(args.cfg)
    set_seed(cfg.env.seed)
    dev = device()
    net = PolicyValueNet(cfg.env.board_size, ch=cfg.net.channels, blocks=cfg.net.blocks).to(dev)

    # 如存在 best.ckpt 则加载
    os.makedirs("checkpoints", exist_ok=True)
    best_ckpt = "checkpoints/best.pt"
    if os.path.exists(best_ckpt):
        load_checkpoint(best_ckpt, net)
        logger.info(f"Loaded best model: {best_ckpt}")

    rb = ReplayBuffer(cfg.replay.capacity_steps, cfg.env.board_size)
    total = 0
    for s, p, z, w in selfplay_many(net, cfg.mcts, cfg.env.board_size, dev, games=args.games):
        # 将整局样本按每步 z_t 已就位（在 selfplay 里）
        # 这里按对称增强策略，我们聚合每步同一个 z_t：胜方=+1，负方=-1，和=0
        # 注意：V1 简化——同一局所有步用相同 z（按当步落子方视角已处理）
        # 若你更偏向 AlphaZero 原义（每步 z 为最终结果从该步“当前玩家”视角），当前实现已满足。
        uniq_z = set([float(x) for x in z])
        assert all(v in (-1.0, 0.0, 1.0) for v in uniq_z)
        # 写入经验池
        for t in range(len(s)):
            rb.add_game([s[t]], [p[t]], z[t])
            total += 1
    logger.info(f"[SelfPlay] collected steps = {total}")

def cmd_train(args):
    cfg = load_cfg(args.cfg)
    set_seed(cfg.env.seed)
    dev = device()
    net = PolicyValueNet(cfg.env.board_size, ch=cfg.net.channels, blocks=cfg.net.blocks).to(dev)

    # warm-start
    best_ckpt = "checkpoints/best.pt"
    if os.path.exists(best_ckpt):
        load_checkpoint(best_ckpt, net)
        logger.info(f"Loaded best model: {best_ckpt}")

    # 这里用一个简单的“预热随机数据”避免经验池为空（实战请先跑 selfplay）
    rb = ReplayBuffer(cfg.replay.capacity_steps, cfg.env.board_size)
    n = cfg.env.board_size * cfg.env.board_size
    for _ in range(max(1024, cfg.replay.batch_size * 4)):
        s = np.zeros((3, cfg.env.board_size, cfg.env.board_size), dtype=np.float32)
        pi = np.ones((n,), dtype=np.float32) / n
        z = float(np.random.choice([-1, 0, 1]))
        rb.add_game([s], [pi], z)

    steps = args.steps or cfg.train.steps
    train_loop(cfg, rb, cfg.env.board_size, net, steps, cfg.train.logdir)

    # 存 candidate
    ensure_dir("checkpoints")
    meta = dict(time=time.time(), steps=steps)
    save_checkpoint("checkpoints/candidate.pt", net, meta)
    logger.info("Saved candidate checkpoint.")

def cmd_evaluate(args):
    cfg = load_cfg(args.cfg)
    set_seed(cfg.env.seed)
    dev = device()
    # 加载 best & candidate
    best = PolicyValueNet(cfg.env.board_size, ch=cfg.net.channels, blocks=cfg.net.blocks).to(dev)
    cand = PolicyValueNet(cfg.env.board_size, ch=cfg.net.channels, blocks=cfg.net.blocks).to(dev)
    load_checkpoint("checkpoints/best.pt", best) if os.path.exists("checkpoints/best.pt") else None
    load_checkpoint("checkpoints/candidate.pt", cand)

    wr = evaluate_models(best, cand, cfg.mcts, cfg.env.board_size, dev, games=args.games or cfg.eval.games)
    thr = args.threshold or cfg.eval.threshold
    if wr >= thr:
        logger.info(f"[Gate] Candidate promoted to best (wr={wr:.3f} >= {thr:.2f})")
        save_checkpoint("checkpoints/best.pt", cand, dict(win_rate=wr))
    else:
        logger.info(f"[Gate] Candidate rejected (wr={wr:.3f} < {thr:.2f})")

def main():
    p = argparse.ArgumentParser("gomoku-zero")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("selfplay")
    sp.add_argument("--cfg", type=str, default=None)
    sp.add_argument("--games", type=int, default=50)
    sp.set_defaults(func=cmd_selfplay)

    tr = sub.add_parser("train")
    tr.add_argument("--cfg", type=str, default=None)
    tr.add_argument("--steps", type=int, default=None)
    tr.set_defaults(func=cmd_train)

    ev = sub.add_parser("evaluate")
    ev.add_argument("--cfg", type=str, default=None)
    ev.add_argument("--games", type=int, default=None)
    ev.add_argument("--threshold", type=float, default=None)
    ev.set_defaults(func=cmd_evaluate)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
