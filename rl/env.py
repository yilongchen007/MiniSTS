from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from action.action import Action, EndAgentTurn
from agent import MONSTER_FACTORIES, create_monster_by_name
from battle import BattleState
from card import Card
from config import CardType, Character, Verbose
from encounters import create_exordium_encounter
from game import GameState
from rl.actions import RLAction, RLActionType
from rl.bot import RLBattleBot
from rl.encoder import StateEncoder
from rl.experiment_config import DEFAULT_DECK, build_deck
from status_effecs import StatusEffectRepo


@dataclass(frozen=True)
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, object]


@dataclass
class PendingHandChoice:
    purpose: str
    hand_indices: tuple[int, ...]
    resolve: Callable[[int], None]


class EnvPlayCard(Action):
    def __init__(self, env: MiniSTSEnv, hand_index: int):
        super().__init__()
        self.env = env
        self.hand_index = hand_index

    def play(self, by, game_state: GameState, battle_state: BattleState) -> None:
        self.env._play_card_or_start_pending(self.hand_index)

    def __repr__(self) -> str:
        return f"PlayCard({self.hand_index})"


class MiniSTSEnv:
    """DQN-friendly battle environment for fixed-deck experiments."""

    def __init__(
        self,
        encoder: StateEncoder | None = None,
        max_steps: int = 200,
        enemy_name: str = "BigJawWorm",
        deck: list[Card] | None = None,
        ascension: int = 0,
        damage_reward_scale: float = 1.0,
    ):
        self.encoder = encoder or StateEncoder()
        self.max_steps = max_steps
        self.enemy_name = enemy_name
        self.deck = deck
        self.ascension = ascension
        self.damage_reward_scale = damage_reward_scale
        self.bot = RLBattleBot()
        self.game_state: GameState | None = None
        self.battle_state: BattleState | None = None
        self.pending_hand_choice: PendingHandChoice | None = None
        self.steps = 0
        self.previous_player_health = 0
        self.previous_enemy_health = 0

    @property
    def observation_size(self) -> int:
        return self.encoder.size

    @property
    def action_size(self) -> int:
        # 0=end turn, 1..10=play hand index 0..9.
        return 11

    def reset(self) -> np.ndarray:
        self.bot = RLBattleBot()
        self.pending_hand_choice = None
        self.game_state = GameState(Character.IRON_CLAD, self.bot, self.ascension)
        self.game_state.set_deck(*(self.deck if self.deck is not None else build_deck(DEFAULT_DECK)))
        self.battle_state = BattleState(self.game_state, *self._create_enemies(), verbose=Verbose.NO_LOG)
        self.steps = 0
        self.battle_state.mana = self.game_state.max_mana
        self.battle_state.turn = 1
        self.battle_state.turn_phase = 0
        self.battle_state.draw_hand()
        self.previous_player_health = self.battle_state.player.health
        self.previous_enemy_health = self._enemy_health_total()
        return self.observe()

    def _create_enemies(self):
        assert self.game_state is not None
        if self.enemy_name.startswith("exordium:"):
            return create_exordium_encounter(self.enemy_name.split(":", 1)[1], self.game_state)
        if self.enemy_name in MONSTER_FACTORIES:
            return [create_monster_by_name(self.enemy_name, self.game_state)]
        aliases = {
            "cultist": "Cultist",
            "jaw_worm": "Jaw Worm",
            "small_slimes": "Small Slimes",
            "blue_slaver": "Blue Slaver",
            "red_slaver": "Red Slaver",
            "looter": "Looter",
            "louses": "2 Louse",
            "three_louses": "3 Louse",
            "large_slime": "Large Slime",
            "lots_of_slimes": "Lots of Slimes",
            "exordium_thugs": "Exordium Thugs",
            "exordium_wildlife": "Exordium Wildlife",
            "fungi_beasts": "2 Fungi Beasts",
            "gremlin_gang": "Gremlin Gang",
            "gremlin_nob": "Gremlin Nob",
            "lagavulin": "Lagavulin",
            "sentries": "3 Sentries",
            "slime_boss": "Slime Boss",
            "hexaghost": "Hexaghost",
            "guardian": "The Guardian",
        }
        if self.enemy_name in aliases:
            return create_exordium_encounter(aliases[self.enemy_name], self.game_state)
        raise ValueError(f"Unknown enemy_name: {self.enemy_name}")

    def observe(self) -> np.ndarray:
        assert self.battle_state is not None
        return self.encoder.encode(self.battle_state, self.pending_hand_choice)

    def legal_actions(self) -> list[RLAction]:
        assert self.game_state is not None and self.battle_state is not None
        if self.pending_hand_choice is not None:
            return [
                RLAction.choose_hand_card(index)
                for index in self.pending_hand_choice.hand_indices
                if 0 <= index < len(self.battle_state.hand)
            ]

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
        if self.pending_hand_choice is not None:
            return RLAction.choose_hand_card(action_index - 1)
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

        if self.pending_hand_choice is None and action.action_type == RLActionType.CHOOSE_HAND_CARD:
            return StepResult(self.observe(), -1.0, True, {"invalid_action": action_index})

        previous_enemy_health = self._enemy_health_total()
        if self.pending_hand_choice is not None:
            assert action.hand_index is not None
            pending = self.pending_hand_choice
            self.pending_hand_choice = None
            pending.resolve(action.hand_index)
        elif action.action_type == RLActionType.END_TURN:
            self.battle_state.tick_player(EndAgentTurn())
        else:
            assert action.hand_index is not None
            self.bot.set_agent_target(action.target_index or 0)
            self.battle_state.tick_player(EnvPlayCard(self, action.hand_index))

        self.steps += 1
        done = self.battle_state.ended() or self.steps >= self.max_steps
        reward = self._reward(previous_enemy_health)
        self.previous_player_health = self.battle_state.player.health
        self.previous_enemy_health = self._enemy_health_total()
        return StepResult(self.observe(), reward, done, {"result": self.battle_state.get_end_result()})

    def _play_card_or_start_pending(self, hand_index: int) -> None:
        assert self.game_state is not None and self.battle_state is not None
        card = self.battle_state.hand[hand_index]
        if card.name == "True Grit" and card.upgrade_count > 0:
            self._play_true_grit_plus(hand_index)
        elif card.name == "Burning Pact":
            self._play_burning_pact(hand_index)
        elif card.name == "Warcry":
            self._play_warcry(hand_index)
        elif card.name == "Armaments" and card.upgrade_count == 0:
            self._play_armaments(hand_index)
        elif card.name == "Dual Wield":
            self._play_dual_wield(hand_index)
        else:
            self.battle_state.play_card(hand_index)

    def _start_card(self, hand_index: int) -> Card:
        assert self.game_state is not None and self.battle_state is not None
        card = self.battle_state.hand.pop(hand_index)
        self.battle_state.add_to_mana(-card.effective_cost(self.game_state, self.battle_state))
        return card

    def _set_pending_hand_choice(self, purpose: str, hand_indices: list[int], resolve: Callable[[int], None]) -> None:
        assert self.battle_state is not None
        valid_indices = tuple(index for index in hand_indices if 0 <= index < len(self.battle_state.hand))
        if len(valid_indices) == 0:
            return
        self.pending_hand_choice = PendingHandChoice(purpose, valid_indices, resolve)

    def _finish_played_card(self, card: Card) -> None:
        assert self.game_state is not None and self.battle_state is not None
        BattleState.card_play_event.broadcast_after((self.battle_state.player, self.game_state, self.battle_state, card))
        if not self.battle_state.is_present(card) and card.card_type != CardType.POWER:
            if card.card_type == CardType.SKILL and self.battle_state.player.status_effect_state.has(StatusEffectRepo.CORRUPTION):
                self.battle_state.exhaust(card)
            elif card.exhaust_on_play:
                self.battle_state.exhaust(card)
            else:
                card.cost_override = None
                self.battle_state.discard_pile.append(card)

    def _play_true_grit_plus(self, hand_index: int) -> None:
        assert self.game_state is not None and self.battle_state is not None
        card = self._start_card(hand_index)
        card.actions[0].play(self.battle_state.player, self.game_state, self.battle_state)
        if len(self.battle_state.hand) == 0:
            self._finish_played_card(card)
            return

        def resolve(selected_index: int) -> None:
            assert self.battle_state is not None
            self.battle_state.exhaust(self.battle_state.hand[selected_index])
            self._finish_played_card(card)

        self._set_pending_hand_choice("exhaust_hand_card", list(range(len(self.battle_state.hand))), resolve)

    def _play_burning_pact(self, hand_index: int) -> None:
        assert self.battle_state is not None
        card = self._start_card(hand_index)

        def finish_without_choice() -> None:
            assert self.battle_state is not None
            for action in card.actions:
                if action.__class__.__name__ == "BurningPactAction":
                    self.battle_state.draw(action.draw_count.get())
                    break
            self._finish_played_card(card)

        if len(self.battle_state.hand) == 0:
            finish_without_choice()
            return

        def resolve(selected_index: int) -> None:
            assert self.battle_state is not None
            self.battle_state.exhaust(self.battle_state.hand[selected_index])
            finish_without_choice()

        self._set_pending_hand_choice("exhaust_hand_card", list(range(len(self.battle_state.hand))), resolve)

    def _play_warcry(self, hand_index: int) -> None:
        assert self.battle_state is not None
        card = self._start_card(hand_index)
        for action in card.actions:
            if action.__class__.__name__ == "WarcryAction":
                self.battle_state.draw(action.draw_count.get())
                break
        if len(self.battle_state.hand) == 0:
            self._finish_played_card(card)
            return

        def resolve(selected_index: int) -> None:
            assert self.battle_state is not None
            selected = self.battle_state.hand[selected_index]
            self.battle_state.remove_card(selected)
            self.battle_state.draw_pile.append(selected)
            self._finish_played_card(card)

        self._set_pending_hand_choice("topdeck_hand_card", list(range(len(self.battle_state.hand))), resolve)

    def _play_armaments(self, hand_index: int) -> None:
        assert self.game_state is not None and self.battle_state is not None
        card = self._start_card(hand_index)
        card.actions[0].play(self.battle_state.player, self.game_state, self.battle_state)
        if len(self.battle_state.hand) == 0:
            self._finish_played_card(card)
            return

        def resolve(selected_index: int) -> None:
            assert self.battle_state is not None
            self.battle_state.hand[selected_index].upgrade()
            self._finish_played_card(card)

        self._set_pending_hand_choice("upgrade_hand_card", list(range(len(self.battle_state.hand))), resolve)

    def _play_dual_wield(self, hand_index: int) -> None:
        assert self.battle_state is not None
        card = self._start_card(hand_index)
        hand_indices = [
            index
            for index, target in enumerate(self.battle_state.hand)
            if target.card_type in (CardType.ATTACK, CardType.POWER)
        ]
        if len(hand_indices) == 0:
            self._finish_played_card(card)
            return

        def resolve(selected_index: int) -> None:
            assert self.battle_state is not None

            selected = self.battle_state.hand[selected_index]
            duplicate_count = 1
            for action in card.actions:
                if action.__class__.__name__ == "DualWieldAction":
                    duplicate_count = action.count.get()
                    break
            for _ in range(duplicate_count):
                self.battle_state.hand.append(copy.deepcopy(selected))
            self._finish_played_card(card)

        self._set_pending_hand_choice("duplicate_hand_card", hand_indices, resolve)

    def _enemy_health_total(self) -> int:
        assert self.battle_state is not None
        return sum(enemy.health for enemy in self.battle_state.enemies)

    def _enemy_max_health_total(self) -> int:
        assert self.battle_state is not None
        return max(1, sum(enemy.max_health for enemy in self.battle_state.enemies))

    def _reward(self, previous_enemy_health: int) -> float:
        damage_dealt = max(0, previous_enemy_health - self._enemy_health_total())
        return self.damage_reward_scale * damage_dealt / self._enemy_max_health_total()
