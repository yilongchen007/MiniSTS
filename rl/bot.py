from __future__ import annotations

from typing import TYPE_CHECKING

from ggpa.ggpa import GGPA

if TYPE_CHECKING:
    from action.action import EndAgentTurn, PlayCard
    from agent import Agent
    from battle import BattleState
    from card import Card
    from game import GameState


class RLBattleBot(GGPA):
    """Bot shim used by cards that ask the player to choose a target."""

    def __init__(self):
        super().__init__("RLBattleBot")
        self.next_agent_target_index = 0
        self.next_card_target_index = 0

    def set_agent_target(self, target_index: int) -> None:
        self.next_agent_target_index = target_index

    def choose_card(self, game_state: GameState, battle_state: BattleState) -> EndAgentTurn | PlayCard:
        raise RuntimeError("RLBattleBot is driven by MiniSTSEnv.step(), not choose_card().")

    def choose_agent_target(self, battle_state: BattleState, list_name: str, agent_list: list[Agent]) -> Agent:
        index = min(max(self.next_agent_target_index, 0), len(agent_list) - 1)
        return agent_list[index]

    def choose_card_target(self, battle_state: BattleState, list_name: str, card_list: list[Card]) -> Card:
        index = min(max(self.next_card_target_index, 0), len(card_list) - 1)
        return card_list[index]
