from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from rl.env import MiniSTSEnv


class PolicyValueNetwork(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden_size, action_size)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.body(observations)
        return self.policy(features), torch.tanh(self.value(features)).squeeze(-1)


class PolicyValueAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_size: int = 256,
        learning_rate: float = 1e-3,
        value_coef: float = 1.0,
        entropy_coef: float = 0.0,
        device: str | None = None,
    ):
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = PolicyValueNetwork(observation_size, action_size, hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def predict(self, observation: np.ndarray, legal_mask: np.ndarray) -> tuple[np.ndarray, float]:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(obs)
            logits = logits.masked_fill(~mask, -1e9)
            policy = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return policy.astype(np.float32), float(value.item())

    def greedy_action(self, observation: np.ndarray, legal_mask: np.ndarray) -> int:
        policy, _ = self.predict(observation, legal_mask)
        return int(np.argmax(policy))

    def train_batch(
        self,
        observations: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
        batch_size: int,
        epochs: int,
    ) -> dict[str, float]:
        count = len(observations)
        indices = np.arange(count)
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []

        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        policy_t = torch.as_tensor(policy_targets, dtype=torch.float32, device=self.device)
        value_t = torch.as_tensor(value_targets, dtype=torch.float32, device=self.device)

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, count, batch_size):
                batch = torch.as_tensor(indices[start:start + batch_size], dtype=torch.int64, device=self.device)
                logits, values = self.model(obs_t[batch])
                log_probs = torch.log_softmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)
                policy_loss = -(policy_t[batch] * log_probs).sum(dim=1).mean()
                value_loss = nn.functional.mse_loss(values, value_t[batch])
                entropy = -(probs * log_probs).sum(dim=1).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "hidden_size": self.hidden_size,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
            },
            path,
        )


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
        agent: PolicyValueAgent,
        simulations: int = 100,
        c_puct: float = 1.5,
        gamma: float = 0.99,
        temperature: float = 1.0,
        network_value_weight: float = 0.5,
    ):
        self.agent = agent
        self.simulations = simulations
        self.c_puct = c_puct
        self.gamma = gamma
        self.temperature = temperature
        self.network_value_weight = network_value_weight

    def search(self, env: MiniSTSEnv) -> tuple[np.ndarray, int]:
        if env.pending_hand_choice is not None:
            return self._network_policy_action(env)

        root = PUCTNode(prior=1.0)
        self._expand(root, env)
        for _ in range(self.simulations):
            sim_env: MiniSTSEnv = copy.deepcopy(env)
            self._simulate(root, sim_env)
        policy = self._visit_policy(root, env.action_size)
        action = self._policy_action(policy)
        return policy, action

    def _network_policy_action(self, env: MiniSTSEnv) -> tuple[np.ndarray, int]:
        observation = env.observe()
        mask = env.legal_action_mask()
        policy, _ = self.agent.predict(observation, mask)
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            return policy, 0
        total = float(policy[legal].sum())
        if total <= 0:
            policy = np.zeros(env.action_size, dtype=np.float32)
            policy[legal] = 1.0 / len(legal)
        else:
            normalized = np.zeros(env.action_size, dtype=np.float32)
            normalized[legal] = policy[legal] / total
            policy = normalized
        return policy, self._policy_action(policy)

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
        priors, network_value = self.agent.predict(observation, mask)
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            return 0.0
        prior_sum = float(priors[legal].sum())
        for action in legal:
            prior = float(priors[action] / prior_sum) if prior_sum > 0 else 1.0 / len(legal)
            node.children[int(action)] = PUCTNode(prior=prior)
        node.expanded = True
        node.visits += 1
        value = (
            self.network_value_weight * network_value
            + (1.0 - self.network_value_weight) * self._heuristic_value(env)
        )
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
        visits = np.array([child.visits for child in root.children.values()], dtype=np.float32)
        if self.temperature <= 1e-6:
            best_action = max(root.children, key=lambda action: root.children[action].visits)
            policy[best_action] = 1.0
            return policy
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

    def _heuristic_value(self, env: MiniSTSEnv) -> float:
        assert env.battle_state is not None
        battle = env.battle_state
        result = battle.get_end_result()
        if result == 1:
            return 1.0
        if result == -1:
            return -1.0
        enemy_health = sum(enemy.health for enemy in battle.enemies)
        enemy_max_health = max(1, sum(enemy.max_health for enemy in battle.enemies))
        player_score = battle.player.health / battle.player.max_health
        enemy_score = enemy_health / enemy_max_health
        return float(np.clip(player_score - enemy_score, -1.0, 1.0))
