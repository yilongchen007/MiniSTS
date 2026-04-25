from __future__ import annotations

import argparse

import numpy as np
import torch

from rl.dqn import DQNAgent
from rl.env import MiniSTSEnv


def load_agent(path: str, env: MiniSTSEnv, device: str | None = None) -> DQNAgent:
    checkpoint = torch.load(path, map_location=device or "cpu")
    agent = DQNAgent(
        observation_size=env.observation_size,
        action_size=env.action_size,
        device=device,
    )
    agent.online.load_state_dict(checkpoint["online"])
    agent.target.load_state_dict(checkpoint["target"])
    return agent


def describe_action(env: MiniSTSEnv, action_index: int) -> str:
    assert env.battle_state is not None
    if action_index == 0:
        return "End turn"
    hand_index = action_index - 1
    if not 0 <= hand_index < len(env.battle_state.hand):
        return f"Play missing hand slot {hand_index}"
    return f"Play hand {hand_index}: {env.battle_state.hand[hand_index].get_name()}"


def print_state(env: MiniSTSEnv) -> None:
    assert env.battle_state is not None
    battle = env.battle_state
    enemy = battle.enemies[0] if battle.enemies else None
    print(
        f"Turn {battle.turn} | mana {battle.mana}/{battle.game_state.max_mana} | "
        f"player hp {battle.player.health}/{battle.player.max_health} block {battle.player.block}"
    )
    if enemy is not None:
        print(
            f"Enemy {enemy.name}: hp {enemy.health}/{enemy.max_health} block {enemy.block} "
            f"intent [{enemy.get_intention(battle.game_state, battle)}]"
        )
    print("Hand:", ", ".join(f"{i}:{card.get_name()}" for i, card in enumerate(battle.hand)) or "-empty-")
    print("Draw:", ", ".join(card.get_name() for card in battle.draw_pile) or "-empty-")
    print("Discard:", ", ".join(card.get_name() for card in battle.discard_pile) or "-empty-")


def q_values(agent: DQNAgent, observation: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        obs = torch.as_tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)
        return agent.online(obs).squeeze(0).cpu().numpy()


def run_episode(agent: DQNAgent, env: MiniSTSEnv, trace: bool) -> tuple[int, float, int]:
    observation = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        mask = env.legal_action_mask()
        qs = q_values(agent, observation)
        masked_qs = np.where(mask, qs, -np.inf)
        action = int(np.argmax(masked_qs))

        if trace:
            print("\n" + "=" * 72)
            print_state(env)
            print("Legal action Q-values:")
            for index in np.flatnonzero(mask):
                print(f"  {index:2d}: {qs[index]: .4f} | {describe_action(env, int(index))}")
            print(f"Chosen: {action} | {describe_action(env, action)}")

        result = env.step_index(action)
        total_reward += result.reward
        observation = result.observation
        done = result.done
        steps += 1

        if trace:
            print(f"Step reward: {result.reward:.4f}")

    if trace:
        print("\n" + "=" * 72)
        print_state(env)
        print(f"Episode done: result={env.battle_state.get_end_result()} reward={total_reward:.4f} steps={steps}")

    assert env.battle_state is not None
    return env.battle_state.get_end_result(), total_reward, steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="rl_runs/dqn_scenario5_jawworm.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--enemy", choices=["jaw_worm", "big_jaw_worm"], default="big_jaw_worm")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    env = MiniSTSEnv(enemy_name=args.enemy)
    agent = load_agent(args.checkpoint, env, args.device)

    wins = 0
    losses = 0
    timeouts = 0
    rewards: list[float] = []
    steps: list[int] = []

    for episode in range(args.episodes):
        result, reward, step_count = run_episode(agent, env, trace=args.trace and episode == 0)
        wins += int(result == 1)
        losses += int(result == -1)
        timeouts += int(result == 0)
        rewards.append(reward)
        steps.append(step_count)

    print(
        f"episodes={args.episodes} win_rate={wins / args.episodes:.3f} "
        f"wins={wins} losses={losses} timeouts={timeouts} "
        f"avg_reward={np.mean(rewards):.3f} avg_steps={np.mean(steps):.2f}"
    )


if __name__ == "__main__":
    main()
