from __future__ import annotations
import argparse, os, time, numpy as np, torch, yaml
from loguru import logger
from .config import Cfg, default_cfg_path
from .utils import set_seed, device, ensure_dir
from .replay import ReplayBuffer
from .dataio import save_game, load_playouts
from .net import PolicyValueNet
from .selfplay import selfplay_many
from .train import train_loop
from .evaluate import evaluate_models, gate_and_promote
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

    data_dir = args.data_dir or cfg.data.selfplay_dir
    ensure_dir(data_dir)

    total = games_saved = 0
    for s, p, z, w in selfplay_many(net, cfg.mcts, cfg.env.board_size, dev, games=args.games):
        # 每局落盘（未增强；D4 增强在训练加载进经验池时再算，省 8 倍空间）
        save_game(data_dir, s, p, z)
        games_saved += 1
        total += len(s)
    logger.info(f"[SelfPlay] saved {games_saved} games / {total} steps -> {data_dir}")

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

    rb = ReplayBuffer(cfg.replay.capacity_steps, cfg.env.board_size)
    data_dir = args.data_dir or cfg.data.selfplay_dir

    # 从 self-play 落盘数据真实恢复经验池（正确的数据流）
    loaded_games = loaded_steps = 0
    for s, p, z in load_playouts(data_dir):
        rb.add_games(s, p, z)
        loaded_games += 1
        loaded_steps += len(z)

    if loaded_games == 0:
        # 兜底：无自对弈数据时给出警告，用随机数据热身（训练无意义，先跑 selfplay）
        logger.warning(
            f"no self-play data in '{data_dir}'! Seeding dummy random data. "
            f"Run `python -m gomoku_zero.cli selfplay` first."
        )
        n = cfg.env.board_size * cfg.env.board_size
        for _ in range(max(1024, cfg.replay.batch_size * 4)):
            s = np.zeros((3, cfg.env.board_size, cfg.env.board_size), dtype=np.float32)
            pi = np.ones((n,), dtype=np.float32) / n
            z = float(np.random.choice([-1, 0, 1]))
            rb.add_game([s], [pi], z)
    else:
        logger.info(f"[Train] loaded {loaded_games} games / {loaded_steps} steps from '{data_dir}'")

    steps = args.steps or cfg.train.steps
    train_loop(cfg, rb, cfg.env.board_size, net, steps, cfg.train.logdir)

    # 存 candidate
    ensure_dir("checkpoints")
    meta = dict(time=time.time(), steps=steps, data_games=loaded_games, data_steps=loaded_steps)
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
    gate_and_promote(
        "checkpoints/best.pt", "checkpoints/candidate.pt", cand, wr, thr
    )

def main():
    p = argparse.ArgumentParser("gomoku-zero")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("selfplay")
    sp.add_argument("--cfg", type=str, default=None)
    sp.add_argument("--games", type=int, default=50)
    sp.add_argument("--data-dir", type=str, default=None, help="自对弈数据落盘目录")
    sp.set_defaults(func=cmd_selfplay)

    tr = sub.add_parser("train")
    tr.add_argument("--cfg", type=str, default=None)
    tr.add_argument("--steps", type=int, default=None)
    tr.add_argument("--data-dir", type=str, default=None, help="从该目录恢复经验池")
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
