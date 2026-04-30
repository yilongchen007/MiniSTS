from __future__ import annotations

import argparse

import numpy as np
import torch

from rl.encoder import StateEncoder
from rl.env import MiniSTSEnv
from rl.experiment_config import ExperimentConfig, card_names_from_deck
from rl.ppo import PPOAgent
from rl.evaluate_dqn import describe_action, print_state


def load_agent(path: str, env: MiniSTSEnv, device: str | None = None) -> PPOAgent:
    checkpoint = torch.load(path, map_location=device or "cpu")
    agent = PPOAgent(
        observation_size=env.observation_size,
        action_size=env.action_size,
        hidden_size=int(checkpoint.get("hidden_size", 256)),
        gamma=float(checkpoint.get("gamma", 0.99)),
        gae_lambda=float(checkpoint.get("gae_lambda", 0.95)),
        clip_ratio=float(checkpoint.get("clip_ratio", 0.2)),
        value_coef=float(checkpoint.get("value_coef", 0.5)),
        entropy_coef=float(checkpoint.get("entropy_coef", 0.01)),
        device=device,
    )
    agent.model.load_state_dict(checkpoint["model"])
    return agent


def run_episode(agent: PPOAgent, env: MiniSTSEnv, trace: bool) -> tuple[int, float, int, int]:
    observation = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        mask = env.legal_action_mask()
        action = agent.greedy_action(observation, mask)

        if trace:
            print("\n" + "=" * 72)
            print_state(env)
            print("Legal actions:")
            for index in np.flatnonzero(mask):
                print(f"  {index:2d}: {describe_action(env, int(index))}")
            print(f"Chosen: {action} | {describe_action(env, action)}")

        result = env.step_index(action)
        total_reward += result.reward
        observation = result.observation
        done = result.done
        steps += 1

        if trace:
            print(f"Step reward: {result.reward:.4f}")

    assert env.battle_state is not None
    if trace:
        print("\n" + "=" * 72)
        print_state(env)
        print(
            f"Episode done: result={env.battle_state.get_end_result()} "
            f"reward={total_reward:.4f} steps={steps} hp_loss={env.battle_state.player_hp_lost_this_combat}"
        )
    return env.battle_state.get_end_result(), total_reward, steps, env.battle_state.player_hp_lost_this_combat


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    experiment_config = ExperimentConfig.load(pre_args.config)
    env_config = experiment_config.section("env")
    reward_config = experiment_config.section("reward")
    ppo_config = experiment_config.section("ppo")
    evaluation_config = experiment_config.section("evaluation")

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--checkpoint", default=ppo_config.get("save_path", "rl_runs/ppo_scenario5_guardian.pt"))
    parser.add_argument("--episodes", type=int, default=evaluation_config.get("episodes", 100))
    parser.add_argument("--trace", action="store_true", default=evaluation_config.get("trace", False))
    parser.add_argument("--enemy", default=env_config.get("enemy", "BigJawWorm"))
    parser.add_argument("--ascension", type=int, default=env_config.get("ascension", 0))
    parser.add_argument("--damage-reward-scale", type=float, default=reward_config.get("damage_reward_scale", 1.0))
    parser.add_argument("--hp-loss-penalty-scale", type=float, default=reward_config.get("hp_loss_penalty_scale", 1.0))
    parser.add_argument("--win-reward", type=float, default=reward_config.get("win_reward", 1.0))
    parser.add_argument("--loss-penalty", type=float, default=reward_config.get("loss_penalty", 1.0))
    parser.add_argument("--timeout-penalty", type=float, default=reward_config.get("timeout_penalty", 0.5))
    parser.add_argument("--device", default=ppo_config.get("device"))
    args = parser.parse_args()

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
        enemy_name=args.enemy,
        deck=deck,
        relics=experiment_config.relic_names(),
        max_steps=int(env_config.get("max_steps", 200)),
        ascension=args.ascension,
        damage_reward_scale=args.damage_reward_scale,
        hp_loss_penalty_scale=args.hp_loss_penalty_scale,
        win_reward=args.win_reward,
        loss_penalty=args.loss_penalty,
        timeout_penalty=args.timeout_penalty,
    )
    agent = load_agent(args.checkpoint, env, args.device)

    wins = 0
    losses = 0
    timeouts = 0
    rewards: list[float] = []
    steps: list[int] = []
    hp_losses: list[int] = []

    for episode in range(args.episodes):
        result, reward, step_count, hp_loss = run_episode(agent, env, trace=args.trace and episode == 0)
        wins += int(result == 1)
        losses += int(result == -1)
        timeouts += int(result == 0)
        rewards.append(reward)
        steps.append(step_count)
        hp_losses.append(hp_loss)

    print(
        f"episodes={args.episodes} win_rate={wins / args.episodes:.3f} "
        f"wins={wins} losses={losses} timeouts={timeouts} "
        f"avg_reward={np.mean(rewards):.3f} avg_steps={np.mean(steps):.2f} "
        f"avg_hp_loss={np.mean(hp_losses):.2f}"
    )


if __name__ == "__main__":
    main()
