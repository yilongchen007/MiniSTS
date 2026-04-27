from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from rl.az import PUCTPlanner, PolicyValueAgent
from rl.encoder import StateEncoder
from rl.env import MiniSTSEnv
from rl.experiment_config import ExperimentConfig, card_names_from_deck


def build_env(args: argparse.Namespace) -> MiniSTSEnv:
    experiment_config = ExperimentConfig.load(args.config)
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
        max_steps=args.max_steps,
        ascension=args.ascension,
        damage_reward_scale=float(reward_config.get("damage_reward_scale", 1.0)),
        hp_loss_penalty_scale=float(reward_config.get("hp_loss_penalty_scale", 1.0)),
        win_reward=float(reward_config.get("win_reward", 1.0)),
        loss_penalty=float(reward_config.get("loss_penalty", 1.0)),
        timeout_penalty=float(reward_config.get("timeout_penalty", 0.5)),
    )


def discounted_value_targets(rewards: list[float], gamma: float) -> list[float]:
    targets: list[float] = []
    value = 0.0
    for reward in reversed(rewards):
        value = reward + gamma * value
        targets.append(float(np.tanh(value)))
    targets.reverse()
    return targets


def run_episode(env: MiniSTSEnv, planner: PUCTPlanner) -> tuple[int, float, int, int, list[np.ndarray], list[np.ndarray], list[float]]:
    observations: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    rewards: list[float] = []

    observation = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        policy, action = planner.search(env)
        observations.append(observation)
        policies.append(policy)
        result = env.step_index(action)
        rewards.append(result.reward)
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
        observations,
        policies,
        rewards,
    )


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = build_env(args)
    agent = PolicyValueAgent(
        observation_size=env.observation_size,
        action_size=env.action_size,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        device=args.device,
    )
    planner = PUCTPlanner(
        agent=agent,
        simulations=args.simulations,
        c_puct=args.c_puct,
        gamma=args.gamma,
        temperature=args.temperature,
        network_value_weight=args.network_value_weight,
    )

    replay_observations: list[np.ndarray] = []
    replay_policies: list[np.ndarray] = []
    replay_values: list[float] = []
    wins = losses = timeouts = 0
    reward_sum = 0.0
    step_sum = 0
    hp_loss_sum = 0

    for episode in range(1, args.episodes + 1):
        result, reward, steps, hp_loss, observations, policies, rewards = run_episode(env, planner)
        replay_observations.extend(observations)
        replay_policies.extend(policies)
        replay_values.extend(discounted_value_targets(rewards, args.gamma))
        if len(replay_values) > args.replay_size:
            overflow = len(replay_values) - args.replay_size
            del replay_observations[:overflow]
            del replay_policies[:overflow]
            del replay_values[:overflow]

        if len(replay_values) >= args.batch_size:
            indices = np.random.choice(len(replay_values), size=min(args.train_samples, len(replay_values)), replace=False)
            batch_obs = np.stack([replay_observations[index] for index in indices]).astype(np.float32)
            batch_policies = np.stack([replay_policies[index] for index in indices]).astype(np.float32)
            batch_values = np.array([replay_values[index] for index in indices], dtype=np.float32)
            losses_dict = agent.train_batch(batch_obs, batch_policies, batch_values, args.batch_size, args.epochs)
        else:
            losses_dict = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        wins += int(result == 1)
        losses += int(result == -1)
        timeouts += int(result == 0)
        reward_sum += reward
        step_sum += steps
        hp_loss_sum += hp_loss

        if episode % args.log_every == 0:
            print(
                f"episode={episode} win_rate={wins / args.log_every:.3f} "
                f"wins={wins} losses={losses} timeouts={timeouts} "
                f"avg_reward={reward_sum / args.log_every:.3f} "
                f"avg_steps={step_sum / args.log_every:.2f} "
                f"avg_hp_loss={hp_loss_sum / args.log_every:.2f} "
                f"replay={len(replay_values)} policy_loss={losses_dict['policy_loss']:.4f} "
                f"value_loss={losses_dict['value_loss']:.4f} entropy={losses_dict['entropy']:.4f}"
            )
            wins = losses = timeouts = 0
            reward_sum = 0.0
            step_sum = 0
            hp_loss_sum = 0

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        agent.save(args.save_path)


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    experiment_config = ExperimentConfig.load(pre_args.config)
    env_config = experiment_config.section("env")
    az_config = experiment_config.section("az")

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--episodes", type=int, default=az_config.get("episodes", 200))
    parser.add_argument("--max-steps", type=int, default=env_config.get("max_steps", 200))
    parser.add_argument("--enemy", default=env_config.get("enemy", "BigJawWorm"))
    parser.add_argument("--ascension", type=int, default=env_config.get("ascension", 0))
    parser.add_argument("--simulations", type=int, default=az_config.get("simulations", 100))
    parser.add_argument("--c-puct", type=float, default=az_config.get("c_puct", 1.5))
    parser.add_argument("--temperature", type=float, default=az_config.get("temperature", 1.0))
    parser.add_argument("--gamma", type=float, default=az_config.get("gamma", 0.99))
    parser.add_argument("--network-value-weight", type=float, default=az_config.get("network_value_weight", 0.5))
    parser.add_argument("--hidden-size", type=int, default=az_config.get("hidden_size", 256))
    parser.add_argument("--learning-rate", type=float, default=az_config.get("learning_rate", 1e-3))
    parser.add_argument("--batch-size", type=int, default=az_config.get("batch_size", 256))
    parser.add_argument("--epochs", type=int, default=az_config.get("epochs", 4))
    parser.add_argument("--train-samples", type=int, default=az_config.get("train_samples", 2048))
    parser.add_argument("--replay-size", type=int, default=az_config.get("replay_size", 50000))
    parser.add_argument("--value-coef", type=float, default=az_config.get("value_coef", 1.0))
    parser.add_argument("--entropy-coef", type=float, default=az_config.get("entropy_coef", 0.0))
    parser.add_argument("--log-every", type=int, default=az_config.get("log_every", 10))
    parser.add_argument("--seed", type=int, default=az_config.get("seed", 0))
    parser.add_argument("--device", type=str, default=az_config.get("device"))
    parser.add_argument("--save-path", type=str, default=az_config.get("save_path", "rl_runs/az_scenario5_guardian.pt"))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
