"""Unit tests for gomoku_zero.config module."""
import pytest
import tempfile
import os
from gomoku_zero.config import Cfg, EnvCfg, NetCfg, MctsCfg, ReplayCfg, OptCfg, TrainCfg, EvalCfg


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_cfg_creation(self):
        cfg = Cfg()
        assert isinstance(cfg.env, EnvCfg)
        assert isinstance(cfg.net, NetCfg)
        assert isinstance(cfg.mcts, MctsCfg)
        assert isinstance(cfg.replay, ReplayCfg)
        assert isinstance(cfg.opt, OptCfg)
        assert isinstance(cfg.train, TrainCfg)
        assert isinstance(cfg.eval, EvalCfg)

    def test_env_defaults(self):
        cfg = Cfg()
        assert cfg.env.id == "GomokuExact5-v0"
        assert cfg.env.board_size == 15
        assert cfg.env.seed == 42

    def test_net_defaults(self):
        cfg = Cfg()
        assert cfg.net.channels == 128
        assert cfg.net.blocks == 12

    def test_mcts_defaults(self):
        cfg = Cfg()
        assert cfg.mcts.sims == 800
        assert cfg.mcts.c_puct == 2.0
        assert cfg.mcts.root_dirichlet_eps == 0.25
        assert cfg.mcts.root_dirichlet_alpha == 0.05
        assert cfg.mcts.temperature_moves == 12
        assert cfg.mcts.temperature_eval == 0.0
        assert cfg.mcts.virtual_loss == 1.0

    def test_replay_defaults(self):
        cfg = Cfg()
        assert cfg.replay.capacity_steps == 800_000
        assert cfg.replay.batch_size == 256

    def test_opt_defaults(self):
        cfg = Cfg()
        assert cfg.opt.lr == 2e-3
        assert cfg.opt.weight_decay == 1e-4

    def test_train_defaults(self):
        cfg = Cfg()
        assert cfg.train.steps == 100_000
        assert cfg.train.save_every == 5_000
        assert cfg.train.logdir == "runs/gomoku_zero"

    def test_eval_defaults(self):
        cfg = Cfg()
        assert cfg.eval.games == 400
        assert cfg.eval.threshold == 0.55


class TestConfigLoad:
    """Test YAML configuration loading."""

    def test_load_empty_yaml(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            f.flush()
            try:
                cfg = Cfg.load(f.name)
                assert isinstance(cfg, Cfg)
                assert cfg.env.board_size == 15
            finally:
                os.unlink(f.name)

    def test_load_partial_yaml(self):
        yaml_content = """
env:
  board_size: 19
  seed: 123
net:
  channels: 64
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                cfg = Cfg.load(f.name)
                assert cfg.env.board_size == 19
                assert cfg.env.seed == 123
                assert cfg.net.channels == 64
                # Unspecified fields keep defaults
                assert cfg.net.blocks == 12
                assert cfg.mcts.sims == 800
            finally:
                os.unlink(f.name)

    def test_load_full_yaml(self):
        yaml_content = """
env:
  id: "CustomEnv-v0"
  board_size: 13
  seed: 7
net:
  channels: 256
  blocks: 20
mcts:
  sims: 1600
  c_puct: 1.5
replay:
  capacity_steps: 1000000
  batch_size: 512
opt:
  lr: 0.001
  weight_decay: 0.0001
train:
  steps: 50000
  save_every: 1000
  logdir: "custom/runs"
eval:
  games: 200
  threshold: 0.6
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                cfg = Cfg.load(f.name)
                assert cfg.env.id == "CustomEnv-v0"
                assert cfg.env.board_size == 13
                assert cfg.env.seed == 7
                assert cfg.net.channels == 256
                assert cfg.net.blocks == 20
                assert cfg.mcts.sims == 1600
                assert cfg.mcts.c_puct == 1.5
                assert cfg.replay.capacity_steps == 1_000_000
                assert cfg.replay.batch_size == 512
                assert cfg.opt.lr == 0.001
                assert cfg.train.steps == 50_000
                assert cfg.eval.games == 200
                assert cfg.eval.threshold == 0.6
            finally:
                os.unlink(f.name)

    def test_load_ignores_unknown_keys(self):
        yaml_content = """
env:
  board_size: 15
  unknown_field: "should_be_ignored"
unknown_section:
  foo: bar
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            try:
                cfg = Cfg.load(f.name)
                assert cfg.env.board_size == 15
            finally:
                os.unlink(f.name)


class TestDefaultCfgPath:
    """Test default configuration path resolution."""

    def test_default_cfg_path_exists_or_not(self):
        path = Cfg.__module__  # just check the function exists
        from gomoku_zero.config import default_cfg_path
        p = default_cfg_path()
        assert isinstance(p, str)
        assert p.endswith("default.yaml")
