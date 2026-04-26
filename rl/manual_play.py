from __future__ import annotations

import argparse
import random

from rl.env import MiniSTSEnv
from rl.encoder import StateEncoder
from rl.experiment_config import ExperimentConfig, card_names_from_deck


def describe_action(env: MiniSTSEnv, action_index: int) -> str:
    assert env.battle_state is not None
    battle = env.battle_state
    if env.pending_hand_choice is not None:
        if action_index == 0:
            return "invalid during hand choice"
        hand_index = action_index - 1
        if not 0 <= hand_index < len(battle.hand):
            return f"choose missing hand slot {hand_index}"
        return (
            f"choose hand {hand_index} for {env.pending_hand_choice.purpose}: "
            f"{battle.hand[hand_index].get_name()}"
        )
    if action_index == 0:
        return "end turn"
    hand_index = action_index - 1
    if not 0 <= hand_index < len(battle.hand):
        return f"play missing hand slot {hand_index}"
    return f"play hand {hand_index}: {battle.hand[hand_index].get_name()}"


def print_state(env: MiniSTSEnv) -> None:
    assert env.battle_state is not None
    battle = env.battle_state
    print("\n" + "=" * 72)
    print(
        f"step {env.steps}/{env.max_steps} | turn {battle.turn} | "
        f"mana {battle.mana}/{battle.game_state.max_mana} | "
        f"hp {battle.player.health}/{battle.player.max_health} | "
        f"block {battle.player.block} | status {battle.player.status_effect_state}"
    )
    for index, enemy in enumerate(battle.enemies):
        print(
            f"enemy {index} {enemy.name}: hp {enemy.health}/{enemy.max_health} "
            f"block {enemy.block} status {enemy.status_effect_state} "
            f"intent [{enemy.get_intention(battle.game_state, battle)}]"
        )
    if env.pending_hand_choice is not None:
        print(
            f"pending: {env.pending_hand_choice.purpose} | "
            f"legal hand slots {list(env.pending_hand_choice.hand_indices)}"
        )
    print("hand:")
    for index, card in enumerate(battle.hand):
        playable = card.is_playable(battle.game_state, battle)
        marker = "" if playable else " (not playable)"
        print(f"  {index}: {card}{marker}")
    print("draw:", ", ".join(card.get_name() for card in battle.draw_pile) or "-empty-")
    print("discard:", ", ".join(card.get_name() for card in battle.discard_pile) or "-empty-")
    print("exhaust:", ", ".join(card.get_name() for card in battle.exhaust_pile) or "-empty-")


def print_legal_actions(env: MiniSTSEnv) -> None:
    legal_indices = sorted(env.to_action_index(action) for action in env.legal_actions())
    print("legal actions:")
    for action_index in legal_indices:
        print(f"  {action_index}: {describe_action(env, action_index)}")


def read_action(env: MiniSTSEnv) -> int | None:
    legal_indices = {env.to_action_index(action) for action in env.legal_actions()}
    while True:
        raw = input("action index, h for help, q to quit: ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            return None
        if raw in {"h", "help", "?"}:
            print_legal_actions(env)
            continue
        try:
            action_index = int(raw)
        except ValueError:
            print("Please enter an integer action index.")
            continue
        if action_index not in legal_indices:
            print(f"Illegal action {action_index}.")
            print_legal_actions(env)
            continue
        return action_index


def build_env(args: argparse.Namespace) -> MiniSTSEnv:
    experiment_config = ExperimentConfig.load(args.config)
    env_config = experiment_config.section("env")
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
        ascension=args.ascension,
    )


def parse_actions(value: str | None) -> list[int]:
    if value is None or value.strip() == "":
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="rl_configs/scenario5_big_jaw_worm.json")
    pre_args, _ = pre_parser.parse_known_args()
    experiment_config = ExperimentConfig.load(pre_args.config)
    env_config = experiment_config.section("env")

    parser = argparse.ArgumentParser(parents=[pre_parser])
    parser.add_argument("--enemy", default=env_config.get("enemy", "BigJawWorm"))
    parser.add_argument("--ascension", type=int, default=env_config.get("ascension", 0))
    parser.add_argument("--max-steps", type=int, default=env_config.get("max_steps", 200))
    parser.add_argument("--seed", type=int, default=env_config.get("seed", 0))
    parser.add_argument(
        "--actions",
        default=None,
        help="Optional comma-separated action indices for non-interactive smoke runs.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    env = build_env(args)
    env.reset()
    scripted_actions = parse_actions(args.actions)
    scripted_mode = args.actions is not None
    action_cursor = 0

    while env.battle_state is not None and not env.battle_state.ended() and env.steps < env.max_steps:
        print_state(env)
        print_legal_actions(env)
        if action_cursor < len(scripted_actions):
            action_index = scripted_actions[action_cursor]
            action_cursor += 1
            print(f"scripted action: {action_index} | {describe_action(env, action_index)}")
        elif scripted_mode:
            print("scripted actions exhausted")
            return
        else:
            action_index = read_action(env)
            if action_index is None:
                print("manual play stopped")
                return
        result = env.step_index(action_index)
        print(f"reward {result.reward:.4f}")
        if result.done:
            break

    assert env.battle_state is not None
    result = env.battle_state.get_end_result()
    outcome = "WIN" if result == 1 else "LOSE" if result == -1 else "TIMEOUT"
    print_state(env)
    print(
        f"{outcome} | steps={env.steps} | "
        f"hp_loss={env.battle_state.player_hp_lost_this_combat}"
    )


if __name__ == "__main__":
    main()
