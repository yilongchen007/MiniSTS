from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass

import numpy as np
import torch

from rl.dqn import DQNAgent
from rl.env import MiniSTSEnv


@dataclass
class ActionStats:
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class DQNGuidedMCTSPlanner:
    def __init__(
        self,
        agent: DQNAgent,
        simulations: int = 50,
        rollout_depth: int = 60,
        exploration: float = 1.4,
        gamma: float = 0.99,
        eval_weight: float = 1.0,
        rollout_epsilon: float = 0.0,
        q_scale: float = 1.0,
        q_transform: str = "raw",
    ):
        self.agent = agent
        self.simulations = simulations
        self.rollout_depth = rollout_depth
        self.exploration = exploration
        self.gamma = gamma
        self.eval_weight = eval_weight
        self.rollout_epsilon = rollout_epsilon
        self.q_scale = max(q_scale, 1e-6)
        self.q_transform = q_transform

    def choose_action(self, env: MiniSTSEnv) -> int:
        legal = self._legal_indices(env)
        if len(legal) == 1:
            return legal[0]
        if env.pending_hand_choice is not None:
            return self._dqn_action(env, legal)

        stats = {action: ActionStats() for action in legal}
        for _ in range(self.simulations):
            action = self._select_ucb(stats)
            value = self._simulate(env, action)
            stats[action].visits += 1
            stats[action].value_sum += value

        return max(legal, key=lambda action: (stats[action].mean, stats[action].visits))

    def _select_ucb(self, stats: dict[int, ActionStats]) -> int:
        unvisited = [action for action, action_stats in stats.items() if action_stats.visits == 0]
        if unvisited:
            return random.choice(unvisited)
        total_visits = sum(action_stats.visits for action_stats in stats.values())
        return max(
            stats,
            key=lambda action: stats[action].mean
            + self.exploration * math.sqrt(math.log(total_visits + 1) / stats[action].visits),
        )

    def _simulate(self, root_env: MiniSTSEnv, first_action: int) -> float:
        env: MiniSTSEnv = copy.deepcopy(root_env)
        result = env.step_index(first_action)
        total = result.reward
        discount = self.gamma
        depth = 0

        while not result.done and depth < self.rollout_depth:
            legal = self._legal_indices(env)
            action = self._rollout_action(env, legal)
            result = env.step_index(action)
            total += discount * result.reward
            discount *= self.gamma
            depth += 1

        if not result.done:
            total += discount * self.eval_weight * self._evaluate(env)
        return total

    def _rollout_action(self, env: MiniSTSEnv, legal: list[int]) -> int:
        if self.rollout_epsilon > 0 and random.random() < self.rollout_epsilon:
            return random.choice(legal)
        return self._dqn_action(env, legal)

    def _dqn_action(self, env: MiniSTSEnv, legal: list[int]) -> int:
        q_values = self._q_values(env)
        masked = np.full_like(q_values, -np.inf, dtype=np.float32)
        masked[legal] = q_values[legal]
        return int(np.argmax(masked))

    def _evaluate(self, env: MiniSTSEnv) -> float:
        legal = self._legal_indices(env)
        if len(legal) == 0:
            return 0.0
        value = float(np.max(self._q_values(env)[legal]))
        if self.q_transform == "clip":
            return float(np.clip(value / self.q_scale, -1.0, 1.0))
        if self.q_transform == "tanh":
            return float(np.tanh(value / self.q_scale))
        return value

    def _q_values(self, env: MiniSTSEnv) -> np.ndarray:
        observation = env.observe()
        with torch.no_grad():
            obs = torch.as_tensor(observation, dtype=torch.float32, device=self.agent.device).unsqueeze(0)
            return self.agent.online(obs).squeeze(0).cpu().numpy()

    def _legal_indices(self, env: MiniSTSEnv) -> list[int]:
        return [env.to_action_index(action) for action in env.legal_actions()]
