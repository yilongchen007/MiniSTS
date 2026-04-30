from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch

from rl.encoder import StateEncoder
from rl.env import MiniSTSEnv
from rl.experiment_config import ExperimentConfig, card_names_from_deck
from rl.ppo import PPOAgent, PPORollout


def build_env(args: argparse.Namespace) -> MiniSTSEnv:
    experiment_config = ExperimentConfig.load(args.config)
    deck = experiment_config.build_deck()
    encoder_config = experiment_config.section("encoder")
    card_names = tuple(encoder_config.get("card_names", card_names_from_deck(deck)))
    encoder = StateEncoder(
        max_turns=int(encoder_config.get("max_turns", 20)),
        max_hand_size=int(encoder_config.get("max_hand_size", 10)),
        card_names=card_names,
    )
    return MiniSTSEnv(
        encoder=encoder,
        max_steps=args.max_steps,
        enemy_name=args.enemy,
        deck=deck,
        relics=experiment_config.relic_names(),
        ascension=args.ascension,
        damage_reward_scale=args.damage_reward_scale,
        hp_loss_penalty_scale=args.hp_loss_penalty_scale,
        win_reward=args.win_reward,
        loss_penalty=args.loss_penalty,
        timeout_penalty=args.timeout_penalty,
    )


def collect_rollout(env: MiniSTSEnv, agent: PPOAgent, rollout_steps: int, max_completed_episodes: int, state: np.ndarray) -> tuple[PPORollout, np.ndarray, dict[str, float]]:
    observations: list[np.ndarray] = []
    actions: list[int] = []
    log_probs: list[float] = []
    rewards: list[float] = []
    dones: list[bool] = []
    values: list[float] = []
    masks: list[np.ndarray] = []

    completed = 0
    wins = 0
    losses = 0
    timeouts = 0
    reward_sum = 0.0
    step_sum = 0
    hp_loss_sum = 0
    episode_reward = 0.0
    episode_steps = 0

    for _ in range(rollout_steps):
        mask = env.legal_action_mask()
        action, log_prob, value = agent.act(state, mask)
        result = env.step_index(action)

        observations.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(result.reward)
        dones.append(result.done)
        values.append(value)
        masks.append(mask)

        state = result.observation
        episode_reward += result.reward
        episode_steps += 1

        if result.done:
            assert env.battle_state is not None
            outcome = env.battle_state.get_end_result()
            wins += int(outcome == 1)
            losses += int(outcome == -1)
            timeouts += int(outcome == 0)
            completed += 1
            reward_sum += episode_reward
            step_sum += episode_steps
            hp_loss_sum += env.battle_state.player_hp_lost_this_combat
            state = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            if completed >= max_completed_episodes:
                break

    rollout = PPORollout(
        observations=np.stack(observations).astype(np.float32),
        actions=np.array(actions, dtype=np.int64),
        log_probs=np.array(log_probs, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        dones=np.array(dones, dtype=np.bool_),
        values=np.array(values, dtype=np.float32),
        masks=np.stack(masks).astype(np.bool_),
        next_observation=state,
        next_done=False,
        next_mask=env.legal_action_mask(),
    )
    stats = {
        "episodes": completed,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "reward_sum": reward_sum,
        "step_sum": step_sum,
        "hp_loss_sum": hp_loss_sum,
    }
    return rollout, state, stats


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = build_env(args)
    agent = PPOAgent(
        observation_size=env.observation_size,
        action_size=env.action_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        hidden_size=args.hidden_size,
        device=args.device,
    )

    state = env.reset()
    total_episodes = 0
    window_episodes = 0
    window_wins = 0
    window_losses = 0
    window_timeouts = 0
    window_reward_sum = 0.0
    window_step_sum = 0
    window_hp_loss_sum = 0
    update = 0

    while total_episodes < args.episodes:
        remaining_episodes = args.episodes - total_episodes
        remaining_log_episodes = args.log_every - window_episodes
        rollout_episode_limit = min(remaining_episodes, remaining_log_episodes)
        rollout, state, stats = collect_rollout(env, agent, args.rollout_steps, rollout_episode_limit, state)
        losses = agent.train_rollout(rollout, args.batch_size, args.epochs)
        update += 1

        total_episodes += int(stats["episodes"])
        window_episodes += int(stats["episodes"])
        window_wins += int(stats["wins"])
        window_losses += int(stats["losses"])
        window_timeouts += int(stats["timeouts"])
        window_reward_sum += stats["reward_sum"]
        window_step_sum += int(stats["step_sum"])
        window_hp_loss_sum += int(stats["hp_loss_sum"])

        if window_episodes >= args.log_every:
            print(
                f"episode={total_episodes} update={update} "
                f"window_win_rate={window_wins / max(1, window_episodes):.3f} "
                f"wins={window_wins} losses={window_losses} timeouts={window_timeouts} "
                f"avg_reward={window_reward_sum / max(1, window_episodes):.3f} "
                f"avg_steps={window_step_sum / max(1, window_episodes):.2f} "
                f"avg_hp_loss={window_hp_loss_sum / max(1, window_episodes):.2f} "
                f"policy_loss={losses['policy_loss']:.4f} value_loss={losses['value_loss']:.4f} "
                f"entropy={losses['entropy']:.4f}"
            )
            window_episodes = 0
            window_wins = window_losses = window_timeouts = 0
            window_reward_sum = 0.0
            window_step_sum = 0
            window_hp_loss_sum = 0

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        agent.save(args.save_path)


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    experiment_config = ExperimentConfig.load(pre_args.config)
    env_config = experiment_config.section("env")
    reward_config = experiment_config.section("reward")
    ppo_config = experiment_config.section("ppo")

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--episodes", type=int, default=ppo_config.get("episodes", 1500))
    parser.add_argument("--max-steps", type=int, default=env_config.get("max_steps", 200))
    parser.add_argument("--enemy", default=env_config.get("enemy", "BigJawWorm"))
    parser.add_argument("--ascension", type=int, default=env_config.get("ascension", 0))
    parser.add_argument("--damage-reward-scale", type=float, default=reward_config.get("damage_reward_scale", 1.0))
    parser.add_argument("--hp-loss-penalty-scale", type=float, default=reward_config.get("hp_loss_penalty_scale", 1.0))
    parser.add_argument("--win-reward", type=float, default=reward_config.get("win_reward", 1.0))
    parser.add_argument("--loss-penalty", type=float, default=reward_config.get("loss_penalty", 1.0))
    parser.add_argument("--timeout-penalty", type=float, default=reward_config.get("timeout_penalty", 0.5))
    parser.add_argument("--rollout-steps", type=int, default=ppo_config.get("rollout_steps", 2048))
    parser.add_argument("--batch-size", type=int, default=ppo_config.get("batch_size", 256))
    parser.add_argument("--epochs", type=int, default=ppo_config.get("epochs", 4))
    parser.add_argument("--hidden-size", type=int, default=ppo_config.get("hidden_size", 256))
    parser.add_argument("--learning-rate", type=float, default=ppo_config.get("learning_rate", 3e-4))
    parser.add_argument("--gamma", type=float, default=ppo_config.get("gamma", 0.99))
    parser.add_argument("--gae-lambda", type=float, default=ppo_config.get("gae_lambda", 0.95))
    parser.add_argument("--clip-ratio", type=float, default=ppo_config.get("clip_ratio", 0.2))
    parser.add_argument("--value-coef", type=float, default=ppo_config.get("value_coef", 0.5))
    parser.add_argument("--entropy-coef", type=float, default=ppo_config.get("entropy_coef", 0.01))
    parser.add_argument("--log-every", type=int, default=ppo_config.get("log_every", 100))
    parser.add_argument("--seed", type=int, default=ppo_config.get("seed", 0))
    parser.add_argument("--device", type=str, default=ppo_config.get("device"))
    parser.add_argument("--save-path", type=str, default=ppo_config.get("save_path", "rl_runs/ppo_scenario5_guardian.pt"))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
