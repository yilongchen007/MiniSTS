from __future__ import annotations

import argparse

import numpy as np
import torch

from rl.az import PUCTPlanner, PolicyValueAgent
from rl.encoder import StateEncoder
from rl.env import MiniSTSEnv
from rl.evaluate_dqn import describe_action, print_state
from rl.experiment_config import ExperimentConfig, card_names_from_deck


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


def load_agent(path: str, env: MiniSTSEnv, device: str | None = None) -> PolicyValueAgent:
    checkpoint = torch.load(path, map_location=device or "cpu")
    agent = PolicyValueAgent(
        observation_size=env.observation_size,
        action_size=env.action_size,
        hidden_size=int(checkpoint.get("hidden_size", 256)),
        value_coef=float(checkpoint.get("value_coef", 1.0)),
        entropy_coef=float(checkpoint.get("entropy_coef", 0.0)),
        device=device,
    )
    agent.model.load_state_dict(checkpoint["model"])
    return agent


def choose_action(
    agent: PolicyValueAgent,
    planner: PUCTPlanner | None,
    env: MiniSTSEnv,
    observation: np.ndarray,
    mode: str,
) -> tuple[int, np.ndarray | None]:
    if mode == "puct":
        assert planner is not None
        policy, action = planner.search(env)
        return action, policy
    mask = env.legal_action_mask()
    policy, _ = agent.predict(observation, mask)
    return int(np.argmax(policy)), policy


def run_episode(
    agent: PolicyValueAgent,
    env: MiniSTSEnv,
    planner: PUCTPlanner | None,
    mode: str,
    trace: bool,
) -> tuple[int, float, int, int]:
    observation = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action, policy = choose_action(agent, planner, env, observation, mode)
        if trace:
            print("\n" + "=" * 72)
            print_state(env)
            print("Legal action policy:")
            for index in np.flatnonzero(env.legal_action_mask()):
                probability = 0.0 if policy is None else float(policy[index])
                print(f"  {index:2d}: {probability: .4f} | {describe_action(env, int(index))}")
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
    az_config = experiment_config.section("az")
    evaluation_config = experiment_config.section("evaluation")

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--checkpoint", default=az_config.get("save_path", "rl_runs/az_scenario5_guardian.pt"))
    parser.add_argument("--episodes", type=int, default=evaluation_config.get("episodes", 100))
    parser.add_argument("--max-steps", type=int, default=env_config.get("max_steps", 200))
    parser.add_argument("--enemy", default=env_config.get("enemy", "BigJawWorm"))
    parser.add_argument("--ascension", type=int, default=env_config.get("ascension", 0))
    parser.add_argument("--mode", choices=("puct", "greedy"), default=az_config.get("eval_mode", "puct"))
    parser.add_argument("--simulations", type=int, default=az_config.get("eval_simulations", az_config.get("simulations", 100)))
    parser.add_argument("--c-puct", type=float, default=az_config.get("c_puct", 1.5))
    parser.add_argument("--gamma", type=float, default=az_config.get("gamma", 0.99))
    parser.add_argument("--network-value-weight", type=float, default=az_config.get("network_value_weight", 0.5))
    parser.add_argument("--seed", type=int, default=az_config.get("seed", 0))
    parser.add_argument("--device", default=az_config.get("device"))
    parser.add_argument("--trace", action="store_true", default=evaluation_config.get("trace", False))
    parser.add_argument("--quiet", action="store_true", default=az_config.get("quiet", False))
    args = parser.parse_args()

    np.random.seed(args.seed)
    env = build_env(args, experiment_config)
    agent = load_agent(args.checkpoint, env, args.device)
    planner = None
    if args.mode == "puct":
        planner = PUCTPlanner(
            agent=agent,
            simulations=args.simulations,
            c_puct=args.c_puct,
            gamma=args.gamma,
            temperature=1e-6,
            network_value_weight=args.network_value_weight,
        )

    wins = 0
    losses = 0
    timeouts = 0
    rewards: list[float] = []
    steps: list[int] = []
    hp_losses: list[int] = []

    for episode in range(args.episodes):
        result, reward, step_count, hp_loss = run_episode(
            agent,
            env,
            planner,
            args.mode,
            trace=args.trace and episode == 0,
        )
        wins += int(result == 1)
        losses += int(result == -1)
        timeouts += int(result == 0)
        rewards.append(reward)
        steps.append(step_count)
        hp_losses.append(hp_loss)
        if not args.quiet:
            print(
                f"episode={episode + 1}/{args.episodes} "
                f"result={result} win_rate={wins / (episode + 1):.3f} "
                f"reward={reward:.3f} steps={step_count} hp_loss={hp_loss}"
            )

    print(
        f"episodes={args.episodes} win_rate={wins / args.episodes:.3f} "
        f"wins={wins} losses={losses} timeouts={timeouts} "
        f"avg_reward={np.mean(rewards):.3f} avg_steps={np.mean(steps):.2f} "
        f"avg_hp_loss={np.mean(hp_losses):.2f} mode={args.mode} simulations={args.simulations}"
    )


if __name__ == "__main__":
    main()
