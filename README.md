# gomoku-zero

AlphaGo Zero–style training framework for **Exact-5 Gomoku**, designed to work out-of-the-box with `GomokuExact5-v0`. This repository provides a minimal yet complete V1 loop: **self-play → replay buffer → policy-value training → evaluation/gating → best model promotion**.

---

## 1) Overview

`gomoku-zero` follows the AlphaGo Zero paradigm adapted to the **Exact-5** rule: a side only wins with **exactly five in a row**; longer runs (≥6) are **not** a win. The framework generates training targets from MCTS search, learns a shared convolutional backbone with **policy** and **value** heads, and iteratively promotes stronger models via a simple **gating** match. V1 focuses on correctness and clarity: single-machine, single-process components with clean abstractions that you can scale later.

---

## 2) Key Features

* **Exact-5 compliant** self-play and terminal checks (≥6 is *not* a win).
* **MCTS (PUCT)** with root Dirichlet noise, temperature schedule, and action masking from legality.
* **Policy-Value CNN** (ResNet style) with configurable channels/blocks.
* **Replay buffer** with D4 board symmetries (8 transforms) and sliding window.
* **Training loop** with joint policy CE + value MSE loss (PyTorch/AMP ready).
* **Evaluation / Gating**: candidate vs best, threshold-based promotion.
* **Config-driven** (YAML) and TensorBoard logging.

---

## 3) Prerequisites

1. Python 3.9+
2. PyTorch 2.2+ (CUDA optional but recommended)
3. Gymnasium 0.29+
4. The **Exact-5 Gomoku environment** you built earlier (installable as a package exposing `GomokuExact5-v0`).

> Ensure you can run:
>
> ```python
> import gymnasium as gym
> env = gym.make("GomokuExact5-v0")
> ```

---

## 4) Installation

```bash
# In the project root
pip install -e .
```

Optional dev extras:

```bash
pip install -e ".[dev]"
```

---

## 5) Project Structure

```
gomoku-zero/
├─ pyproject.toml
├─ setup.cfg
├─ README.md
├─ LICENSE
├─ configs/
│  └─ default.yaml              # All tunables: env/net/mcts/replay/opt/train/eval
├─ src/
│  └─ gomoku_zero/
│     ├─ __init__.py
│     ├─ config.py              # dataclasses + YAML loader
│     ├─ utils.py               # seed, device, fs helpers
│     ├─ game.py                # lightweight GameState + state encoding
│     ├─ net.py                 # ResNet-style Policy-Value network
│     ├─ mcts.py                # PUCT search with masking & noise
│     ├─ replay.py              # D4 augmentation + sliding-window buffer
│     ├─ selfplay.py            # self-play episodes, (s, π, z) generation
│     ├─ train.py               # training loop, TB logging
│     ├─ evaluate.py            # gating (V1: single-net approximation)
│     ├─ serialization.py       # save/load checkpoints + meta
│     └─ cli.py                 # CLI entrypoints (selfplay/train/evaluate)
└─ scripts/
   ├─ run_selfplay.sh
   ├─ run_train.sh
   └─ run_eval.sh
```

---

## 6) Quickstart

Self-play 收集数据并**落盘**（未增强的 (s, π, z) 每局一个 npz）：

```bash
python -m gomoku_zero.cli selfplay --games 50
# => 写入 data/selfplay/game_*.npz（路径可用 --data-dir 或 configs 里 data.selfplay_dir 覆盖）
```

Train 从落盘数据**真实恢复经验池**（D4 增强在加载时展开）并在其上训练：

```bash
python -m gomoku_zero.cli train --steps 2000
# => 从 data/selfplay 加载真实数据训练；若目录为空会警告并退回随机热身（无意义，先跑 selfplay）
```

评估并门禁（候选 vs 最强 **双网络对弈**，wr ≥ threshold 才晋升 best）：

```bash
python -m gomoku_zero.cli evaluate --games 200 --threshold 0.55
```

> Checkpoints 写入 `checkpoints/` 作为 `best.pt` / `candidate.pt`。
> TensorBoard 日志在 `runs/gomoku_zero`（可用 configs 覆盖）。
> 数据也可用 `--data-dir <dir>` 显式指定，便于多轮 selfplay 增量追加。

---

## 7) Configuration (configs/default.yaml)

Main knobs you’ll likely touch:

* `env`: `id`, `board_size`, `seed`
* `net`: `channels`, `blocks`  (e.g., 128 channels × 12 residual blocks)
* `mcts`: `sims`, `c_puct`, `root_dirichlet_eps/alpha`, `temperature_moves`, `temperature_eval`
* `replay`: `capacity_steps`, `batch_size`
* `opt`: `lr`, `weight_decay`
* `train`: `steps`, `save_every`, `logdir`
* `eval`: `games`, `threshold`

Override with `--cfg path/to/your.yaml` on any CLI command.

---

## 8) Data & Targets

Each self-play move produces a training tuple **(s, π, z)**:

* **s**: state planes `(3, N, N)`

  * current player stones, opponent stones, side-to-move tag plane
* **π**: MCTS visit distribution over `N*N` moves (masked to legal)
* **z**: final game outcome from the *current mover’s* perspective at that step
  `+1` win, `0` draw, `-1` loss

The replay buffer applies **D4 augmentations** to both boards and move distributions.

---

## 9) Network & Loss

* **Backbone**: ResNet-style CNN (configurable depth/width).
* **Policy head**: logits over `N*N` actions; illegal moves are masked downstream.
* **Value head**: scalar `tanh` in `[-1, 1]`.

**Loss**:

* Policy: `- π · log softmax(logits)` (cross-entropy with soft targets)
* Value: MSE between predicted value and `z`
* (Optional) weight decay; AMP and grad clipping recommended.

---

## 10) MCTS (PUCT) Details

* **Selection**: maximize `Q + U`, where `U = c_puct * P * sqrt(ΣN) / (1 + N_a)`.
* **Expansion**: first visit to a leaf → network inference → child priors after masking illegal actions.
* **Noise**: at root, `P ← (1−ε)P + ε Dir(α)`.
* **Temperature**: early moves use `τ=1` (sampling), then switch to `τ≈0` (greedy).
* **Terminal handling**: uses **Exact-5** rules—exact five is a win; ≥6 is not. Full-board without win is a draw.

---

## 11) Evaluation & Gating

* **V1** uses a simplified **single-net approximation** for candidate evaluation (faster to bootstrap).
* If candidate’s win-rate ≥ `threshold` (default 0.55) over `eval.games`, it becomes the new **best**.
* **V2** (planned) will switch to *two-net, no-noise* MCTS matches with strict color alternation.

---

## 12) Checkpoints & Reproducibility

* Models saved to `checkpoints/` with sidecar JSON meta (config/time/steps).
* Set seeds via config; note MCTS search and Dirichlet noise introduce stochasticity.
* Log with TensorBoard: loss components, learning rate, etc.

---

## 13) Performance Tips

* Start with `mcts.sims = 400–800` and increase as throughput allows.
* Turn on AMP (`torch.cuda.amp.autocast`) when you extend the trainer for large models.
* Batch leaf evaluations across concurrent simulations for big wins (V2 roadmap).
* Keep `replay.capacity_steps` large enough (e.g., 0.5–1.0M) to prevent overfitting.

---

## 14) Roadmap (V2+)

* **Two-net evaluation** with strict fairness and τ=0.
* **Batched inference queue** for MCTS leaves (big throughput boost).
* **Distributed self-play** (Ray / gRPC actors), object storage for checkpoints and datasets.
* **Hydra-based configs**, richer logging/metrics (KL, entropy, Elo).
* **Board encodings**: history planes, last move heatmap, move age, legality plane.
* **Optimizations**: bitboards, incremental pattern checks, resign calibration.

---

## 15) Troubleshooting

* **“Cannot make env”** → ensure your Exact-5 package is installed and exposes `GomokuExact5-v0`.
* **All priors zero after masking** → the node had no legal moves; we fall back to uniform over legal.
* **Value stuck near 0** → increase self-play games or MCTS sims; verify terminal detection.
* **Eval never promotes** → lower threshold a bit at the beginning (e.g., 0.52) or increase `eval.games`.

---

## 16) License

MIT License. See `LICENSE`.

---

## 17) Citation / Inspiration

* Silver et al., *Mastering the game of Go without human knowledge*, Nature (2017).
* AlphaGo Zero / AlphaZero training paradigms adapted for **Exact-5 Gomoku**.