"""E2E: CLI 入口冒烟（子进程级，覆盖 argparse / 配置加载 / 文件副作用）。

由于 cmd_selfplay 目前不落盘（数据只在内存经验池里丢掉），这里验证的是：
- 三个子命令都能成功启动并跑完（returncode == 0）；
- train 会写出 checkpoints/candidate.pt 及其 JSON 边车；
- evaluate 能读取 candidate 完成双网络评估并执行门禁日志。

用临时 cwd + 临时配置，避免污染仓库。
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_SRC = os.path.normpath(os.path.join(ROOT, "..", "gomoku-exact5-env", "src"))


@pytest.fixture
def tiny_cfg_file(tmp_path):
    cfg = tmp_path / "tiny.yaml"
    cfg.write_text(textwrap.dedent("""\
        env: {board_size: 5, seed: 1}
        net: {channels: 4, blocks: 1}
        mcts: {sims: 6, c_puct: 1.5, temperature_moves: 2}
        replay: {capacity_steps: 1000, batch_size: 16}
        opt: {lr: 0.001, weight_decay: 0.0001}
        train: {steps: 3, save_every: 1, logdir: runs}
        eval: {games: 1, threshold: 0.5}
    """))
    return str(cfg)


def _run(cwd, *args):
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [os.path.join(ROOT, "src"), ENV_SRC, existing] if p
    )
    return subprocess.run(
        [sys.executable, "-m", "gomoku_zero.cli", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_cli_selfplay_then_train_real_data(tmp_path, tiny_cfg_file):
    """selfplay 落盘 -> train 从同目录真实恢复经验池（核心数据流打通）。"""
    data_dir = tmp_path / "data"

    # 1) selfplay 落盘
    r = _run(str(tmp_path), "selfplay", "--cfg", tiny_cfg_file, "--games", "2",
             "--data-dir", str(data_dir))
    assert r.returncode == 0, r.stderr + r.stdout
    files = sorted(p.name for p in data_dir.glob("game_*.npz"))
    assert len(files) == 2, f"应落盘 2 局，得到 {files}"

    # 2) train 从该目录加载真实数据训练
    r2 = _run(str(tmp_path), "train", "--cfg", tiny_cfg_file, "--steps", "8",
              "--data-dir", str(data_dir))
    assert r2.returncode == 0, r2.stderr + r2.stdout
    output = r2.stdout + r2.stderr
    assert "loaded 2 games" in output, f"train 应报告加载到落盘数据: {output}"
    assert (tmp_path / "checkpoints" / "candidate.pt").exists()
    assert (tmp_path / "checkpoints" / "candidate.pt.json").exists()

    # 3) 保存的 meta 应记录真实数据量（而非兜底 0）
    import json
    meta = json.loads((tmp_path / "checkpoints" / "candidate.pt.json").read_text())
    assert meta["data_games"] == 2 and meta["data_steps"] > 0


def test_cli_train_warns_without_data(tmp_path, tiny_cfg_file):
    """无自对弈数据时 train 应走随机热身兜底并写出候选。"""
    r = _run(str(tmp_path), "train", "--cfg", tiny_cfg_file, "--steps", "2",
             "--data-dir", str(tmp_path / "nope"))
    assert r.returncode == 0, r.stderr + r.stdout
    assert "no self-play data" in (r.stdout + r.stderr), f"应警告无数据: {r.stdout + r.stderr}"
    assert (tmp_path / "checkpoints" / "candidate.pt").exists()


def test_cli_selfplay_smoke(tmp_path, tiny_cfg_file):
    r = _run(str(tmp_path), "selfplay", "--cfg", tiny_cfg_file, "--games", "1")
    assert r.returncode == 0, r.stderr + r.stdout


def test_cli_train_and_evaluate_smoke(tmp_path, tiny_cfg_file):
    r = _run(str(tmp_path), "train", "--cfg", tiny_cfg_file, "--steps", "3")
    assert r.returncode == 0, r.stderr + r.stdout
    assert (tmp_path / "checkpoints" / "candidate.pt").exists(), "train 应写出 candidate.pt"
    assert (tmp_path / "checkpoints" / "candidate.pt.json").exists()

    r2 = _run(str(tmp_path), "evaluate", "--cfg", tiny_cfg_file, "--games", "1", "--threshold", "0.5")
    assert r2.returncode == 0, r2.stderr + r2.stdout