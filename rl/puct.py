from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from rl.env import MiniSTSEnv


class PUCTEvaluator(Protocol):
    def predict(self, observation: np.ndarray, legal_mask: np.ndarray) -> tuple[np.ndarray, float]:
        ...


@dataclass
class PUCTNode:
    prior: float = 0.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, PUCTNode] = field(default_factory=dict)
    expanded: bool = False

    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class PUCTPlanner:
    def __init__(
        self,
        evaluator: PUCTEvaluator,
        simulations: int = 50,
        c_puct: float = 1.5,
        gamma: float = 0.99,
        temperature: float = 1e-6,
    ):
        self.evaluator = evaluator
        self.simulations = simulations
        self.c_puct = c_puct
        self.gamma = gamma
        self.temperature = temperature

    def search(self, env: MiniSTSEnv) -> tuple[np.ndarray, int]:
        if env.pending_hand_choice is not None:
            return self._policy_action_from_network(env)

        root = PUCTNode(prior=1.0)
        self._expand(root, env)
        for _ in range(self.simulations):
            sim_env: MiniSTSEnv = copy.deepcopy(env)
            self._simulate(root, sim_env)
        policy = self._visit_policy(root, env.action_size)
        return policy, self._policy_action(policy)

    def _policy_action_from_network(self, env: MiniSTSEnv) -> tuple[np.ndarray, int]:
        observation = env.observe()
        mask = env.legal_action_mask()
        policy, _ = self.evaluator.predict(observation, mask)
        legal = np.flatnonzero(mask)
        normalized = np.zeros(env.action_size, dtype=np.float32)
        if len(legal) == 0:
            return normalized, 0
        total = float(policy[legal].sum())
        if total <= 0:
            normalized[legal] = 1.0 / len(legal)
        else:
            normalized[legal] = policy[legal] / total
        return normalized, self._policy_action(normalized)

    def _simulate(self, node: PUCTNode, env: MiniSTSEnv) -> float:
        assert env.battle_state is not None
        if env.battle_state.ended() or env.steps >= env.max_steps:
            return self._terminal_value(env)
        if not node.expanded:
            return self._expand(node, env)

        action = self._select(node)
        result = env.step_index(action)
        child = node.children[action]
        if result.done:
            value = result.reward
        else:
            value = result.reward + self.gamma * self._simulate(child, env)
        child.visits += 1
        child.value_sum += value
        node.visits += 1
        node.value_sum += value
        return value

    def _expand(self, node: PUCTNode, env: MiniSTSEnv) -> float:
        observation = env.observe()
        mask = env.legal_action_mask()
        priors, value = self.evaluator.predict(observation, mask)
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            return 0.0
        prior_sum = float(priors[legal].sum())
        for action in legal:
            prior = float(priors[action] / prior_sum) if prior_sum > 0 else 1.0 / len(legal)
            node.children[int(action)] = PUCTNode(prior=prior)
        node.expanded = True
        node.visits += 1
        node.value_sum += value
        return value

    def _select(self, node: PUCTNode) -> int:
        total = math.sqrt(max(1, node.visits))
        return max(
            node.children,
            key=lambda action: node.children[action].value
            + self.c_puct * node.children[action].prior * total / (1 + node.children[action].visits),
        )

    def _visit_policy(self, root: PUCTNode, action_size: int) -> np.ndarray:
        policy = np.zeros(action_size, dtype=np.float32)
        if not root.children:
            return policy
        if self.temperature <= 1e-6:
            best_action = max(
                root.children,
                key=lambda action: (root.children[action].visits, root.children[action].prior),
            )
            policy[best_action] = 1.0
            return policy
        visits = np.array([child.visits for child in root.children.values()], dtype=np.float32)
        visits = visits ** (1.0 / self.temperature)
        total = float(visits.sum())
        if total <= 0:
            prior_total = sum(child.prior for child in root.children.values())
            for action, child in root.children.items():
                policy[action] = child.prior / prior_total if prior_total > 0 else 1.0 / len(root.children)
            return policy
        for action, count in zip(root.children, visits):
            policy[action] = count / total
        return policy

    def _policy_action(self, policy: np.ndarray) -> int:
        legal = np.flatnonzero(policy > 0)
        if len(legal) == 0:
            return 0
        if self.temperature <= 1e-6:
            return int(legal[np.argmax(policy[legal])])
        probabilities = policy[legal].astype(np.float64)
        probabilities /= probabilities.sum()
        return int(np.random.choice(legal, p=probabilities))

    def _terminal_value(self, env: MiniSTSEnv) -> float:
        assert env.battle_state is not None
        result = env.battle_state.get_end_result()
        if result == 1:
            return 1.0
        if result == -1:
            return -1.0
        return -0.5
