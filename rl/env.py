from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from action.action import EndAgentTurn, PlayCard
from agent import BigJawWorm, JawWorm
from battle import BattleState
from card import CardRepo
from config import Character, Verbose
from game import GameState
from rl.actions import RLAction, RLActionType
from rl.bot import RLBattleBot
from rl.encoder import StateEncoder


@dataclass(frozen=True)
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, object]


class MiniSTSEnv:
    """DQN-friendly environment. Initial supported task: scenario 5 vs BigJawWorm."""

    def __init__(self, encoder: StateEncoder | None = None, max_steps: int = 200, enemy_name: str = "big_jaw_worm"):
        self.encoder = encoder or StateEncoder()
        self.max_steps = max_steps
        self.enemy_name = enemy_name
        self.bot = RLBattleBot()
        self.game_state: GameState | None = None
        self.battle_state: BattleState | None = None
        self.steps = 0
        self.previous_player_health = 0

    @property
    def observation_size(self) -> int:
        return self.encoder.size

    @property
    def action_size(self) -> int:
        # 0=end turn, 1..10=play hand index 0..9. Scenario 5 starts with 6 cards.
        return 11

    def reset(self) -> np.ndarray:
        self.bot = RLBattleBot()
        self.game_state = GameState(Character.IRON_CLAD, self.bot, 0)
        self.game_state.set_deck(*CardRepo.get_scenario_5()[1])
        self.battle_state = BattleState(self.game_state, self._create_enemy(), verbose=Verbose.NO_LOG)
        self.steps = 0
        self.battle_state.mana = self.game_state.max_mana
        self.battle_state.turn = 1
        self.battle_state.turn_phase = 0
        self.battle_state.draw_hand()
        self.previous_player_health = self.battle_state.player.health
        return self.observe()

    def _create_enemy(self):
        assert self.game_state is not None
        if self.enemy_name == "jaw_worm":
            return JawWorm(self.game_state)
        if self.enemy_name == "big_jaw_worm":
            return BigJawWorm(self.game_state)
        raise ValueError(f"Unknown enemy_name: {self.enemy_name}")

    def observe(self) -> np.ndarray:
        assert self.battle_state is not None
        return self.encoder.encode(self.battle_state)

    def legal_actions(self) -> list[RLAction]:
        assert self.game_state is not None and self.battle_state is not None
        actions = [RLAction.end_turn()]
        for index, card in enumerate(self.battle_state.hand):
            if card.is_playable(self.game_state, self.battle_state):
                actions.append(RLAction.play_card(index, 0))
        return actions

    def legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=np.bool_)
        for action in self.legal_actions():
            mask[self.to_action_index(action)] = True
        return mask

    def to_action_index(self, action: RLAction) -> int:
        if action.action_type == RLActionType.END_TURN:
            return 0
        assert action.hand_index is not None
        return action.hand_index + 1

    def from_action_index(self, action_index: int) -> RLAction:
        if action_index == 0:
            return RLAction.end_turn()
        return RLAction.play_card(action_index - 1, 0)

    def step_index(self, action_index: int) -> StepResult:
        return self.step(self.from_action_index(action_index))

    def step(self, action: RLAction) -> StepResult:
        assert self.game_state is not None and self.battle_state is not None
        if self.battle_state.ended():
            return StepResult(self.observe(), 0.0, True, {"already_done": True})

        legal_indices = {self.to_action_index(legal_action) for legal_action in self.legal_actions()}
        action_index = self.to_action_index(action)
        if action_index not in legal_indices:
            return StepResult(self.observe(), -1.0, True, {"invalid_action": action_index})

        previous_health = self.battle_state.player.health
        if action.action_type == RLActionType.END_TURN:
            self.battle_state.tick_player(EndAgentTurn())
        else:
            assert action.hand_index is not None
            self.bot.set_agent_target(action.target_index or 0)
            self.battle_state.tick_player(PlayCard(action.hand_index))

        self.steps += 1
        done = self.battle_state.ended() or self.steps >= self.max_steps
        reward = self._reward(previous_health, done)
        self.previous_player_health = self.battle_state.player.health
        return StepResult(self.observe(), reward, done, {"result": self.battle_state.get_end_result()})

    def _reward(self, previous_health: int, done: bool) -> float:
        assert self.battle_state is not None
        player = self.battle_state.player
        health_loss = max(0, previous_health - player.health)
        reward = -health_loss / player.max_health

        if not done:
            return reward

        result = self.battle_state.get_end_result()
        if result == 1:
            return reward + 1.0
        if result == -1:
            return reward - 1.0
        return reward - 0.5
