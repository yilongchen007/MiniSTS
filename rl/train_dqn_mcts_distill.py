from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import os
import random

import numpy as np
import torch
from torch import nn

from rl.dqn import DQNAgent
from rl.dqn_mcts import DQNGuidedMCTSPlanner
from rl.encoder import StateEncoder
from rl.env import MiniSTSEnv
from rl.evaluate_dqn import load_agent
from rl.experiment_config import ExperimentConfig, card_names_from_deck
from rl.mcts import MCTSPlanner


@dataclass(frozen=True)
class ImitationSample:
    observation: np.ndarray
    legal_mask: np.ndarray
    action: int


class ImitationReplay:
    def __init__(self, capacity: int):
        self.buffer: deque[ImitationSample] = deque(maxlen=capacity)

    def push(self, sample: ImitationSample) -> None:
        self.buffer.append(sample)

    def sample(self, batch_size: int) -> list[ImitationSample]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def build_env(args: argparse.Namespace, experiment_config: ExperimentConfig) -> MiniSTSEnv:
    deck = experiment_config.build_deck()
    encoder_config = experiment_config.section("encoder")
    reward_config = experiment_config.section("reward")
    encoder = StateEncoder(
        max_turns=int(encoder_config.get("max_turns", 20)),
        max_hand_size=int(encoder_config.get("max_hand_size", 10)),
        card_names=tuple(encoder_config.get("card_names", card_names_from_deck(deck))),
    )
    return MiniSTSEnv(
        encoder=encoder,
        enemy_name=args.enemy,
        deck=deck,
        relics=experiment_config.relic_names(),
        max_steps=args.max_steps,
        ascension=args.ascension,
        damage_reward_scale=float(reward_config.get("damage_reward_scale", 1.0)),
        hp_loss_penalty_scale=float(reward_config.get("hp_loss_penalty_scale", 1.0)),
        win_reward=float(reward_config.get("win_reward", 1.0)),
        loss_penalty=float(reward_config.get("loss_penalty", 1.0)),
        timeout_penalty=float(reward_config.get("timeout_penalty", 0.5)),
    )


def build_student(args: argparse.Namespace, env: MiniSTSEnv) -> DQNAgent:
    if args.init_checkpoint:
        agent = load_agent(args.init_checkpoint, env, args.device)
        agent.optimizer = torch.optim.Adam(agent.online.parameters(), lr=args.learning_rate)
        return agent
    return DQNAgent(
        observation_size=env.observation_size,
        action_size=env.action_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        hidden_size=args.hidden_size,
        double_dqn=args.double_dqn,
        device=args.device,
    )


def build_teacher(args: argparse.Namespace, env: MiniSTSEnv):
    if args.teacher == "heuristic_mcts":
        return MCTSPlanner(
            simulations=args.simulations,
            rollout_depth=args.rollout_depth,
            exploration=args.exploration,
            gamma=args.gamma,
            eval_weight=args.eval_weight,
            rollout_policy=args.rollout_policy,
        )

    checkpoint = args.teacher_checkpoint or args.init_checkpoint or args.dqn_checkpoint
    if checkpoint is None:
        raise ValueError("--teacher dqn_mcts requires --teacher-checkpoint, --init-checkpoint, or a configured DQN checkpoint.")
    teacher_agent = load_agent(checkpoint, env, args.device)
    return DQNGuidedMCTSPlanner(
        agent=teacher_agent,
        simulations=args.simulations,
        rollout_depth=args.rollout_depth,
        exploration=args.exploration,
        gamma=args.gamma,
        eval_weight=args.eval_weight,
        rollout_epsilon=args.rollout_epsilon,
        q_scale=args.q_scale,
        q_transform=args.q_transform,
    )


def train_step(agent: DQNAgent, replay: ImitationReplay, batch_size: int) -> float | None:
    if len(replay) < batch_size:
        return None
    batch = replay.sample(batch_size)
    observations = torch.as_tensor(np.stack([sample.observation for sample in batch]), dtype=torch.float32, device=agent.device)
    masks = torch.as_tensor(np.stack([sample.legal_mask for sample in batch]), dtype=torch.bool, device=agent.device)
    actions = torch.as_tensor([sample.action for sample in batch], dtype=torch.int64, device=agent.device)

    logits = agent.online(observations).masked_fill(~masks, -1e9)
    loss = nn.functional.cross_entropy(logits, actions)
    agent.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.online.parameters(), 10.0)
    agent.optimizer.step()
    return float(loss.item())


def run_teacher_episode(
    env: MiniSTSEnv,
    teacher,
    student: DQNAgent,
    replay: ImitationReplay,
    args: argparse.Namespace,
) -> tuple[int, float, int, int, list[float]]:
    observation = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    losses: list[float] = []

    while not done:
        mask = env.legal_action_mask()
        action = teacher.choose_action(env)
        replay.push(ImitationSample(observation.copy(), mask.copy(), action))

        for _ in range(args.updates_per_step):
            loss = train_step(student, replay, args.batch_size)
            if loss is not None:
                losses.append(loss)

        result = env.step_index(action)
        observation = result.observation
        total_reward += result.reward
        done = result.done
        steps += 1

    assert env.battle_state is not None
    return (
        env.battle_state.get_end_result(),
        total_reward,
        steps,
        env.battle_state.player_hp_lost_this_combat,
        losses,
    )


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    experiment_config = ExperimentConfig.load(args.config)
    env = build_env(args, experiment_config)
    student = build_student(args, env)
    teacher = build_teacher(args, env)
    replay = ImitationReplay(args.replay_size)

    wins = losses_count = timeouts = 0
    reward_sum = 0.0
    step_sum = 0
    hp_loss_sum = 0
    loss_sum = 0.0
    loss_count = 0

    for episode in range(1, args.episodes + 1):
        result, reward, steps, hp_loss, losses = run_teacher_episode(env, teacher, student, replay, args)
        wins += int(result == 1)
        losses_count += int(result == -1)
        timeouts += int(result == 0)
        reward_sum += reward
        step_sum += steps
        hp_loss_sum += hp_loss
        loss_sum += float(np.sum(losses)) if losses else 0.0
        loss_count += len(losses)

        if episode % args.target_sync_episodes == 0:
            student.sync_target()

        if episode % args.log_every == 0:
            window = args.log_every
            print(
                f"episode={episode} teacher={args.teacher} "
                f"window_win_rate={wins / window:.3f} wins={wins} losses={losses_count} timeouts={timeouts} "
                f"avg_reward={reward_sum / window:.3f} avg_steps={step_sum / window:.2f} "
                f"avg_hp_loss={hp_loss_sum / window:.2f} replay={len(replay)} "
                f"bc_loss={loss_sum / max(1, loss_count):.4f}"
            )
            wins = losses_count = timeouts = 0
            reward_sum = 0.0
            step_sum = 0
            hp_loss_sum = 0
            loss_sum = 0.0
            loss_count = 0

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        student.sync_target()
        student.save(args.save_path)


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    experiment_config = ExperimentConfig.load(pre_args.config)
    env_config = experiment_config.section("env")
    dqn_config = experiment_config.section("training")
    mcts_config = experiment_config.section("mcts")
    distill_config = experiment_config.section("distill")

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--episodes", type=int, default=distill_config.get("episodes", 500))
    parser.add_argument("--max-steps", type=int, default=env_config.get("max_steps", 200))
    parser.add_argument("--enemy", default=env_config.get("enemy", "BigJawWorm"))
    parser.add_argument("--ascension", type=int, default=env_config.get("ascension", 0))
    parser.add_argument("--teacher", choices=("heuristic_mcts", "dqn_mcts"), default=distill_config.get("teacher", "heuristic_mcts"))
    parser.add_argument("--teacher-checkpoint", type=str, default=distill_config.get("teacher_checkpoint"))
    parser.add_argument("--init-checkpoint", type=str, default=distill_config.get("init_checkpoint"))
    parser.add_argument("--dqn-checkpoint", type=str, default=dqn_config.get("save_path"))
    parser.add_argument("--simulations", type=int, default=distill_config.get("simulations", mcts_config.get("simulations", 80)))
    parser.add_argument("--rollout-depth", type=int, default=distill_config.get("rollout_depth", mcts_config.get("rollout_depth", 60)))
    parser.add_argument("--exploration", type=float, default=distill_config.get("exploration", mcts_config.get("exploration", 1.4)))
    parser.add_argument("--gamma", type=float, default=distill_config.get("gamma", dqn_config.get("gamma", 0.99)))
    parser.add_argument("--eval-weight", type=float, default=distill_config.get("eval_weight", mcts_config.get("eval_weight", 1.0)))
    parser.add_argument("--rollout-policy", choices=("heuristic", "random"), default=distill_config.get("rollout_policy", mcts_config.get("rollout_policy", "heuristic")))
    parser.add_argument("--rollout-epsilon", type=float, default=distill_config.get("rollout_epsilon", 0.0))
    parser.add_argument("--q-scale", type=float, default=distill_config.get("q_scale", 1.0))
    parser.add_argument("--q-transform", choices=("raw", "clip", "tanh"), default=distill_config.get("q_transform", "raw"))
    parser.add_argument("--batch-size", type=int, default=distill_config.get("batch_size", dqn_config.get("batch_size", 128)))
    parser.add_argument("--replay-size", type=int, default=distill_config.get("replay_size", dqn_config.get("replay_size", 100000)))
    parser.add_argument("--hidden-size", type=int, default=distill_config.get("hidden_size", dqn_config.get("hidden_size", 256)))
    parser.add_argument("--learning-rate", type=float, default=distill_config.get("learning_rate", dqn_config.get("learning_rate", 3e-4)))
    parser.add_argument("--updates-per-step", type=int, default=distill_config.get("updates_per_step", 1))
    parser.add_argument("--target-sync-episodes", type=int, default=distill_config.get("target_sync_episodes", 10))
    parser.add_argument("--log-every", type=int, default=distill_config.get("log_every", 25))
    parser.add_argument("--seed", type=int, default=distill_config.get("seed", dqn_config.get("seed", 0)))
    parser.add_argument("--double-dqn", action=argparse.BooleanOptionalAction, default=distill_config.get("double_dqn", dqn_config.get("double_dqn", False)))
    parser.add_argument("--device", type=str, default=distill_config.get("device", dqn_config.get("device")))
    parser.add_argument("--save-path", type=str, default=distill_config.get("save_path", "rl_runs/dqn_mcts_distill.pt"))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
