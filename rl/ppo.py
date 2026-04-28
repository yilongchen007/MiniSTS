from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class PPORollout:
    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    masks: np.ndarray
    next_observation: np.ndarray
    next_done: bool
    next_mask: np.ndarray


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(observations)
        return self.actor(features), self.critic(features).squeeze(-1)


class PPOAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        hidden_size: int = 128,
        device: str | None = None,
    ):
        self.observation_size = observation_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.hidden_size = hidden_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = ActorCritic(observation_size, action_size, hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def _distribution(self, observations: torch.Tensor, masks: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        logits, values = self.model(observations)
        logits = logits.masked_fill(~masks, -1e9)
        return Categorical(logits=logits), values

    def act(self, observation: np.ndarray, legal_mask: np.ndarray) -> tuple[int, float, float]:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist, value = self._distribution(obs, mask)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def greedy_action(self, observation: np.ndarray, legal_mask: np.ndarray) -> int:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(obs)
            logits = logits.masked_fill(~mask, -1e9)
        return int(torch.argmax(logits, dim=1).item())

    def predict(self, observation: np.ndarray, legal_mask: np.ndarray) -> tuple[np.ndarray, float]:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(obs)
            logits = logits.masked_fill(~mask, -1e9)
            policy = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return policy.astype(np.float32), float(value.item())

    def value(self, observation: np.ndarray, legal_mask: np.ndarray) -> float:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, value = self._distribution(obs, mask)
        return float(value.item())

    def train_rollout(self, rollout: PPORollout, batch_size: int, epochs: int) -> dict[str, float]:
        advantages, returns = self._advantages_and_returns(rollout)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        observations = torch.as_tensor(rollout.observations, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(rollout.actions, dtype=torch.int64, device=self.device)
        old_log_probs = torch.as_tensor(rollout.log_probs, dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(rollout.masks, dtype=torch.bool, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []

        count = len(actions)
        indices = np.arange(count)
        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, count, batch_size):
                batch_indices = indices[start:start + batch_size]
                batch = torch.as_tensor(batch_indices, dtype=torch.int64, device=self.device)
                dist, values = self._distribution(observations[batch], masks[batch])
                log_probs = dist.log_prob(actions[batch])
                ratio = torch.exp(log_probs - old_log_probs[batch])
                unclipped = ratio * advantages_t[batch]
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages_t[batch]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(values, returns_t[batch])
                entropy = dist.entropy().mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    def _advantages_and_returns(self, rollout: PPORollout) -> tuple[np.ndarray, np.ndarray]:
        rewards = rollout.rewards
        dones = rollout.dones
        values = rollout.values
        advantages = np.zeros_like(rewards, dtype=np.float32)

        next_value = 0.0 if rollout.next_done else self.value(rollout.next_observation, rollout.next_mask)
        next_advantage = 0.0
        for index in reversed(range(len(rewards))):
            if index == len(rewards) - 1:
                next_non_terminal = 0.0 if rollout.next_done else 1.0
                following_value = next_value
            else:
                next_non_terminal = 0.0 if dones[index + 1] else 1.0
                following_value = values[index + 1]
            delta = rewards[index] + self.gamma * following_value * next_non_terminal - values[index]
            next_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * next_advantage
            advantages[index] = next_advantage

        returns = advantages + values
        return advantages, returns.astype(np.float32)

    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_ratio": self.clip_ratio,
                "value_coef": self.value_coef,
                "entropy_coef": self.entropy_coef,
                "hidden_size": self.hidden_size,
            },
            path,
        )
