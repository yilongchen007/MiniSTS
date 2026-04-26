from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from rl.dqn import DQNAgent, ReplayBuffer, Transition
from rl.env import MiniSTSEnv
from rl.encoder import StateEncoder
from rl.experiment_config import ExperimentConfig, card_names_from_deck


def linear_epsilon(episode: int, episodes: int, start: float, end: float, fraction: float) -> float:
    decay_episodes = max(1, int(episodes * fraction))
    progress = min(1.0, episode / decay_episodes)
    return start + progress * (end - start)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    experiment_config = ExperimentConfig.load(args.config)
    deck = experiment_config.build_deck()
    encoder_config = experiment_config.section("encoder")
    card_names = tuple(encoder_config.get("card_names", card_names_from_deck(deck)))
    encoder = StateEncoder(
        max_turns=int(encoder_config.get("max_turns", 20)),
        max_hand_size=int(encoder_config.get("max_hand_size", 10)),
        card_names=card_names,
    )
    env = MiniSTSEnv(
        encoder=encoder,
        max_steps=args.max_steps,
        enemy_name=args.enemy,
        deck=deck,
    )
    agent = DQNAgent(
        observation_size=env.observation_size,
        action_size=env.action_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        hidden_size=args.hidden_size,
        device=args.device,
    )
    replay = ReplayBuffer(args.replay_size)

    wins = 0
    losses = 0
    timeouts = 0
    train_steps = 0

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0.0
        epsilon = linear_epsilon(episode, args.episodes, args.epsilon_start, args.epsilon_end, args.epsilon_fraction)

        while not done:
            mask = env.legal_action_mask()
            action = agent.act(state, mask, epsilon)
            result = env.step_index(action)
            next_mask = env.legal_action_mask() if not result.done else np.ones(env.action_size, dtype=np.bool_)
            replay.push(Transition(state, action, result.reward, result.observation, result.done, next_mask))

            loss = agent.train_step(replay, args.batch_size)
            if loss is not None:
                train_steps += 1
                if train_steps % args.target_sync_steps == 0:
                    agent.sync_target()

            state = result.observation
            episode_reward += result.reward
            done = result.done

        outcome = env.battle_state.get_end_result() if env.battle_state is not None else 0
        if outcome == 1:
            wins += 1
        elif outcome == -1:
            losses += 1
        else:
            timeouts += 1

        if episode % args.log_every == 0:
            total = wins + losses + timeouts
            win_rate = wins / max(1, total)
            print(
                f"episode={episode} epsilon={epsilon:.3f} "
                f"reward={episode_reward:.2f} window_win_rate={win_rate:.3f} "
                f"wins={wins} losses={losses} timeouts={timeouts} replay={len(replay)}"
            )
            wins = losses = timeouts = 0

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        agent.save(args.save_path)


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    experiment_config = ExperimentConfig.load(pre_args.config)
    training_config = experiment_config.section("training")
    env_config = experiment_config.section("env")

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--episodes", type=int, default=training_config.get("episodes", 1000))
    parser.add_argument("--max-steps", type=int, default=env_config.get("max_steps", 200))
    parser.add_argument("--enemy", choices=["jaw_worm", "big_jaw_worm"], default=env_config.get("enemy", "big_jaw_worm"))
    parser.add_argument("--batch-size", type=int, default=training_config.get("batch_size", 64))
    parser.add_argument("--replay-size", type=int, default=training_config.get("replay_size", 50000))
    parser.add_argument("--hidden-size", type=int, default=training_config.get("hidden_size", 128))
    parser.add_argument("--learning-rate", type=float, default=training_config.get("learning_rate", 1e-3))
    parser.add_argument("--gamma", type=float, default=training_config.get("gamma", 0.99))
    parser.add_argument("--epsilon-start", type=float, default=training_config.get("epsilon_start", 1.0))
    parser.add_argument("--epsilon-end", type=float, default=training_config.get("epsilon_end", 0.05))
    parser.add_argument("--epsilon-fraction", type=float, default=training_config.get("epsilon_fraction", 0.6))
    parser.add_argument("--target-sync-steps", type=int, default=training_config.get("target_sync_steps", 200))
    parser.add_argument("--log-every", type=int, default=training_config.get("log_every", 50))
    parser.add_argument("--seed", type=int, default=training_config.get("seed", 0))
    parser.add_argument("--device", type=str, default=training_config.get("device"))
    parser.add_argument("--save-path", type=str, default=training_config.get("save_path", "rl_runs/dqn_scenario5_jawworm.pt"))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
