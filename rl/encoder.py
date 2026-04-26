from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from card import Card
from status_effecs import StatusEffectRepo


CARD_NAMES = (
    "Shrug It Off",
    "Defend",
    "Strike",
    "Bash",
    "Bloodletting",
    "Pommel Strike",
)

INTENT_NAMES = (
    "Deal 11 attack damage to player",
    "Deal 7 attack damage to player and Add 5 block to self",
    "Apply 3 Strength and Add 5 block to self",
)

STATUS_EFFECTS = (
    StatusEffectRepo.VULNERABLE,
    StatusEffectRepo.WEAK,
    StatusEffectRepo.STRENGTH,
    StatusEffectRepo.VIGOR,
    StatusEffectRepo.TOLERANCE,
    StatusEffectRepo.BOMB,
)


@dataclass(frozen=True)
class StateEncoder:
    """Fixed-size encoder for configurable fixed-deck battles."""

    max_turns: int = 20
    max_hand_size: int = 10
    card_names: tuple[str, ...] = CARD_NAMES
    intent_names: tuple[str, ...] = INTENT_NAMES

    @property
    def size(self) -> int:
        scalar_count = 8
        status_count = len(STATUS_EFFECTS) * 2
        hand_slot_count = self.max_hand_size * (len(self.card_names) + 1 + 3)
        pile_count = len(self.card_names) * 2 * 3
        return scalar_count + status_count + hand_slot_count + pile_count + len(self.intent_names)

    def encode(self, battle_state) -> np.ndarray:
        player = battle_state.player
        enemy = battle_state.enemies[0] if battle_state.enemies else None

        features: list[float] = [
            player.health / player.max_health,
            player.block / 100.0,
            battle_state.mana / max(1, battle_state.game_state.max_mana),
            min(battle_state.turn, self.max_turns) / self.max_turns,
            0.0 if enemy is None else enemy.health / enemy.max_health,
            0.0 if enemy is None else enemy.block / 100.0,
            len(battle_state.hand) / 10.0,
            len(battle_state.draw_pile) / 20.0,
        ]

        features.extend(self._status_features(player))
        features.extend(self._status_features(enemy))

        features.extend(self._hand_slot_features(battle_state))

        for pile in (battle_state.draw_pile, battle_state.discard_pile, battle_state.exhaust_pile):
            features.extend(self._pile_features(pile))

        intent = "" if enemy is None else repr(enemy.get_intention(battle_state.game_state, battle_state))
        features.extend(1.0 if intent == name else 0.0 for name in self.intent_names)

        return np.array(features, dtype=np.float32)

    def _status_features(self, agent) -> list[float]:
        if agent is None:
            return [0.0 for _ in STATUS_EFFECTS]
        return [agent.status_effect_state.get(status) / 20.0 for status in STATUS_EFFECTS]

    def _pile_features(self, cards: list[Card]) -> list[float]:
        counts = {(name, upgraded): 0 for name in self.card_names for upgraded in (False, True)}
        for card in cards:
            key = (card.name, card.upgrade_count > 0)
            if key in counts:
                counts[key] += 1
        return [counts[(name, upgraded)] / 10.0 for name in self.card_names for upgraded in (False, True)]

    def _hand_slot_features(self, battle_state) -> list[float]:
        features: list[float] = []
        for index in range(self.max_hand_size):
            card = battle_state.hand[index] if index < len(battle_state.hand) else None
            features.extend(self._card_slot_features(card, battle_state))
        return features

    def _card_slot_features(self, card: Card | None, battle_state) -> list[float]:
        if card is None:
            return [1.0] + [0.0 for _ in self.card_names] + [0.0, 0.0, 0.0]

        card_one_hot = [1.0 if card.name == name else 0.0 for name in self.card_names]
        return (
            [0.0]
            + card_one_hot
            + [
                1.0 if card.upgrade_count > 0 else 0.0,
                card.mana_cost.peek() / 5.0,
                1.0 if card.is_playable(battle_state.game_state, battle_state) else 0.0,
            ]
        )
