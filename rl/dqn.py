from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_mask: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, observation_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class DQNAgent:
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        hidden_size: int = 128,
        device: str | None = None,
    ):
        self.observation_size = observation_size
        self.action_size = action_size
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = QNetwork(observation_size, action_size, hidden_size).to(self.device)
        self.target = QNetwork(observation_size, action_size, hidden_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

    def act(self, observation: np.ndarray, legal_mask: np.ndarray, epsilon: float) -> int:
        legal_indices = np.flatnonzero(legal_mask)
        if len(legal_indices) == 0:
            raise RuntimeError("No legal actions available.")
        if random.random() < epsilon:
            return int(random.choice(legal_indices))
        with torch.no_grad():
            obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online(obs).squeeze(0).cpu().numpy()
        q_values = np.where(legal_mask, q_values, -np.inf)
        return int(np.argmax(q_values))

    def train_step(self, replay: ReplayBuffer, batch_size: int) -> float | None:
        if len(replay) < batch_size:
            return None

        batch = replay.sample(batch_size)
        states = torch.as_tensor(np.stack([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(np.stack([t.next_mask for t in batch]), dtype=torch.bool, device=self.device)

        q_values = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target(next_states)
            next_q_values = next_q_values.masked_fill(~next_masks, -1e9)
            next_best = next_q_values.max(dim=1).values
            target = rewards + (1.0 - dones) * self.gamma * next_best

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str) -> None:
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "observation_size": self.observation_size,
                "action_size": self.action_size,
                "gamma": self.gamma,
                "hidden_size": self.hidden_size,
            },
            path,
        )
