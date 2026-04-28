from __future__ import annotations

import argparse
import random

import numpy as np

from rl.dqn_mcts import DQNGuidedMCTSPlanner
from rl.encoder import StateEncoder
from rl.env import MiniSTSEnv
from rl.evaluate_dqn import describe_action, load_agent, print_state
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
        max_steps=args.max_steps,
        ascension=args.ascension,
        damage_reward_scale=float(reward_config.get("damage_reward_scale", 1.0)),
        hp_loss_penalty_scale=float(reward_config.get("hp_loss_penalty_scale", 1.0)),
        win_reward=float(reward_config.get("win_reward", 1.0)),
        loss_penalty=float(reward_config.get("loss_penalty", 1.0)),
        timeout_penalty=float(reward_config.get("timeout_penalty", 0.5)),
    )


def run_episode(env: MiniSTSEnv, planner: DQNGuidedMCTSPlanner, trace: bool) -> tuple[int, float, int, int]:
    observation = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        action = planner.choose_action(env)
        if trace:
            print("\n" + "=" * 72)
            print_state(env)
            print("Legal actions:")
            for legal_action in env.legal_actions():
                index = env.to_action_index(legal_action)
                print(f"  {index:2d}: {describe_action(env, index)}")
            print(f"Chosen: {action} | {describe_action(env, action)}")

        result = env.step_index(action)
        observation = result.observation
        total_reward += result.reward
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
    dqn_config = experiment_config.section("training")
    mcts_config = experiment_config.section("dqn_mcts")
    evaluation_config = experiment_config.section("evaluation")

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--checkpoint", default=dqn_config.get("save_path", evaluation_config.get("checkpoint", "rl_runs/dqn_scenario5_guardian.pt")))
    parser.add_argument("--episodes", type=int, default=evaluation_config.get("episodes", 100))
    parser.add_argument("--max-steps", type=int, default=env_config.get("max_steps", 200))
    parser.add_argument("--enemy", default=env_config.get("enemy", "BigJawWorm"))
    parser.add_argument("--ascension", type=int, default=env_config.get("ascension", 0))
    parser.add_argument("--simulations", type=int, default=mcts_config.get("simulations", 30))
    parser.add_argument("--rollout-depth", type=int, default=mcts_config.get("rollout_depth", 60))
    parser.add_argument("--exploration", type=float, default=mcts_config.get("exploration", 1.4))
    parser.add_argument("--gamma", type=float, default=mcts_config.get("gamma", dqn_config.get("gamma", 0.99)))
    parser.add_argument("--eval-weight", type=float, default=mcts_config.get("eval_weight", 1.0))
    parser.add_argument("--rollout-epsilon", type=float, default=mcts_config.get("rollout_epsilon", 0.0))
    parser.add_argument("--q-scale", type=float, default=mcts_config.get("q_scale", 1.0))
    parser.add_argument("--q-transform", choices=("raw", "clip", "tanh"), default=mcts_config.get("q_transform", "raw"))
    parser.add_argument("--seed", type=int, default=mcts_config.get("seed", 0))
    parser.add_argument("--device", default=evaluation_config.get("device"))
    parser.add_argument("--trace", action="store_true", default=evaluation_config.get("trace", False))
    parser.add_argument("--quiet", action="store_true", default=mcts_config.get("quiet", False))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = build_env(args, experiment_config)
    agent = load_agent(args.checkpoint, env, args.device)
    planner = DQNGuidedMCTSPlanner(
        agent=agent,
        simulations=args.simulations,
        rollout_depth=args.rollout_depth,
        exploration=args.exploration,
        gamma=args.gamma,
        eval_weight=args.eval_weight,
        rollout_epsilon=args.rollout_epsilon,
        q_scale=args.q_scale,
        q_transform=args.q_transform,
    )

    wins = 0
    losses = 0
    timeouts = 0
    rewards: list[float] = []
    steps: list[int] = []
    hp_losses: list[int] = []

    for episode in range(args.episodes):
        result, reward, step_count, hp_loss = run_episode(env, planner, trace=args.trace and episode == 0)
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
        f"avg_hp_loss={np.mean(hp_losses):.2f} simulations={args.simulations} "
        f"rollout_depth={args.rollout_depth} q_transform={args.q_transform}"
    )


if __name__ == "__main__":
    main()
