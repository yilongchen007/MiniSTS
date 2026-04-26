import argparse
import time

from game import GameState
from battle import BattleState
from config import Character, Verbose
from agent import MONSTER_FACTORIES, MONSTER_NAME_ZH, create_monster_by_name
from card import CardGen, CardRepo
from encounters import EXORDIUM_SUPPORTED_ENCOUNTERS, create_exordium_encounter
from ggpa.human_input import HumanInput
from ggpa.backtrack import BacktrackBot
from deck_builder import build_deck

def create_battle_enemies(enemy_name: str, game_state: GameState):
    if enemy_name.startswith("exordium:"):
        return create_exordium_encounter(enemy_name.split(":", 1)[1], game_state)
    if enemy_name in MONSTER_FACTORIES:
        return [create_monster_by_name(enemy_name, game_state)]
    supported_monsters = ", ".join(sorted(MONSTER_FACTORIES))
    supported_encounters = ", ".join(f"exordium:{name}" for name in sorted(EXORDIUM_SUPPORTED_ENCOUNTERS))
    raise ValueError(
        f"Unknown enemy {enemy_name!r}.\n"
        f"Monsters: {supported_monsters}\n"
        f"Encounters: {supported_encounters}"
    )

def print_available_battles() -> None:
    print("Monsters:")
    for name in sorted(MONSTER_FACTORIES):
        print(f"  {name}: {MONSTER_NAME_ZH[name]}")
    print("Exordium encounters:")
    for name in sorted(EXORDIUM_SUPPORTED_ENCOUNTERS):
        print(f"  exordium:{name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enemy", default="JawWorm", help="Monster class name, or exordium:<encounter name>.")
    parser.add_argument("--list-enemies", action="store_true", help="Print supported monster and encounter names.")
    args = parser.parse_args()
    if args.list_enemies:
        print_available_battles()
        return

    agent = HumanInput(True)
    # agent = BacktrackBot(4, False)
    # from ggpa.chatgpt_bot import ChatGPTBot
    # from ggpa.prompt2 import PromptOption
    # agent = ChatGPTBot(ChatGPTBot.ModelName.GPT_Turbo_35, PromptOption.CoT, 0, False, 1)
    game_state = GameState(Character.IRON_CLAD, agent, 0)
    game_state.set_deck(*CardRepo.get_scenario_0()[1])
    # game_state.set_deck(CardGen.Strike(), CardGen.Defend(), CardGen.Defend())
    # game_state.add_to_deck(CardGen.Strike(), CardGen.Defend(), CardGen.Defend())
    build_deck(game_state)
    battle_state = BattleState(game_state, *create_battle_enemies(args.enemy, game_state), verbose=Verbose.LOG)
    start = time.time()
    battle_state.run()
    end = time.time()
    print(f"run ended in {end-start} seconds")
    # to save all the requests and responses for the ChatGPTBot agent, use:
    # agent.dump_history("bot_history")

if __name__ == '__main__':
    main()
