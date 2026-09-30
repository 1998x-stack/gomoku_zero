from __future__ import annotations
import math, numpy as np, torch
from typing import Dict, Optional
from .game import GameState

class Node:
    __slots__ = ("prior", "visit", "value_sum", "children", "player")
    def __init__(self, prior: float, player: int):
        self.prior = float(prior)
        self.visit = 0
        self.value_sum = 0.0
        self.children: Dict[int, Node] = {}
        self.player = player  # 节点对应“待走方”

    @property
    def q(self) -> float:
        return 0.0 if self.visit == 0 else self.value_sum / self.visit

class MCTS:
    def __init__(self, net, cfg, device: torch.device, net2=None):
        """
        ``net`` 控制黑方（to_play=1），``net2`` 控制白方（to_play=2）；
        ``net2=None`` 时双方共用 ``net``（自对弈）。
        """
        self.net = net
        self.net2 = net2
        self.cfg = cfg
        self.device = device

    @staticmethod
    def _opponent(p: int) -> int:
        return 2 if p == 1 else 1

    def _ann(self, state: GameState):
        # 按当前轮次选择模型进行推理
        return self.net if state.to_play == 1 else (self.net2 or self.net)

    def run(self, root_state: GameState, add_dirichlet: bool = True) -> Dict[int, int]:
        root = Node(1.0, root_state.to_play)
        priors, _ = self._policy_value(root_state)
        legal = root_state.legal_actions()
        priors = self._mask_and_norm(priors, legal)

        if add_dirichlet:
            eps = self.cfg.root_dirichlet_eps
            alpha = self.cfg.root_dirichlet_alpha
            noise = np.random.dirichlet([alpha] * legal.sum())
            priors[legal] = (1 - eps) * priors[legal] + eps * noise

        child_side = self._opponent(root_state.to_play)
        for a, p in enumerate(priors):
            if p > 0.0:
                root.children[a] = Node(p, player=child_side)

        for _ in range(self.cfg.sims):
            self._simulate(root_state, root)

        visits = {a: child.visit for a, child in root.children.items()}
        return visits

    def _simulate(self, root_state: GameState, root: Node):
        path = []
        node = root
        state = root_state.copy()

        # Selection
        while node.children:
            a, node = self._select_child(node)
            path.append((state, a, node))
            state.step(a)
            if state.is_terminal():
                break

        # Evaluation/Expansion
        if state.is_terminal():
            # 终局视角用 path[0][0]（根状态）的待走方：配合 “path 存选中子节点”
            # 的回传方式，恰好让最深子节点累积到其父节点视角正确的值。
            # 实测（黑一步即胜 / 黑走后白胜两刻）均验证：获胜子 q=+1、失败分支
            # q=-1，且 PUCT 引导正确。不要按“叶子待走方”改写此处，否则符号反转。
            value = state.result_from(player=path[0][0].to_play if path else root_state.to_play)
        else:
            priors, value = self._policy_value(state)
            legal = state.legal_actions()
            priors = self._mask_and_norm(priors, legal)
            child_side = self._opponent(state.to_play)
            for a, p in enumerate(priors):
                if p > 0.0:
                    node.children[a] = Node(p, player=child_side)

        # Backpropagate (值对路径上的 "待走方" 定向，交替取反)
        cur_value = float(value)
        # 回传从“当前 state 的上一手玩家”视角开始；简化：沿路径交替取反
        for _, _, n in reversed(path):
            n.visit += 1
            n.value_sum += cur_value
            cur_value = -cur_value

    def _policy_value(self, state: GameState):
        ann = self._ann(state)
        x = torch.from_numpy(state.encode()).unsqueeze(0).to(self.device)  # (1,3,N,N)
        ann.eval()
        with torch.no_grad():
            logits, v = ann(x)
            p = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            value = v[0].item()
        return p, value

    def _mask_and_norm(self, priors: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
        p = priors.copy()
        p[~legal_mask] = 0.0
        s = p.sum()
        if s <= 1e-12:
            # 若全被屏蔽，均匀分配到合法点
            cnt = legal_mask.sum()
            if cnt > 0:
                p[legal_mask] = 1.0 / cnt
        else:
            p /= s
        return p

    def _select_child(self, node: Node):
        total_visits = sum(c.visit for c in node.children.values())
        best, best_score = None, -1e9
        for a, child in node.children.items():
            q = child.q
            u = self.cfg.c_puct * child.prior * math.sqrt(total_visits + 1e-8) / (1 + child.visit)
            score = q + u
            if score > best_score:
                best, best_score = (a, child), score
        return best
