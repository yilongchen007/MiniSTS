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
    "NoAction()",
    "Sleep",
    "Stunned",
    "Escape",
    "Split",
    "Add 3 status card(s) to discard",
    "Add 5 status card(s) to discard",
    "Add 1 status card(s) to discard",
    "Add 2 status card(s) to discard",
    "Apply 1 Weak to player",
    "Apply 3 Ritual to self",
    "Apply 4 Ritual to self",
    "Apply 5 Ritual to self",
    "Apply 3 Strength to self",
    "Apply 4 Strength to self",
    "Apply 2 Anger to self",
    "Apply 3 Anger to self",
    "Apply 2 Weak to player and Apply 2 Vulnerable to player",
    "Deal 11 attack damage to player",
    "Deal 12 attack damage to player",
    "Deal 13 attack damage to player",
    "Deal 14 attack damage to player",
    "Deal 16 attack damage to player",
    "Deal 18 attack damage to player",
    "Deal 20 attack damage to player",
    "Deal 22 attack damage to player",
    "Deal 24 attack damage to player",
    "Deal 25 attack damage to player",
    "Deal 30 attack damage to player",
    "Deal 32 attack damage to player",
    "Deal 35 attack damage to player",
    "Deal 36 attack damage to player",
    "Deal 38 attack damage to player",
    "Deal 3 attack damage to player",
    "Deal 4 attack damage to player",
    "Deal 5 attack damage to player",
    "Deal 6 attack damage to player",
    "Deal 7 attack damage to player",
    "Deal 8 attack damage to player",
    "Deal 9 attack damage to player",
    "Deal 10 attack damage to player",
    "Deal 2 attack damage 6 times to player",
    "Deal 3 attack damage 6 times to player",
    "Deal 5 attack damage 2 times to player",
    "Deal 6 attack damage 2 times to player",
    "Deal 5 attack damage 4 times to player",
    "Deal 8 attack damage 2 times to player",
    "Deal 9 attack damage 4 times to player",
    "Deal 10 attack damage 2 times to player",
    "Deal 7 attack damage to player and Add 5 block to self",
    "Deal 14 attack damage to player and Add 10 block to self",
    "Deal 4 attack damage to player and Apply 1 Weak to player and Apply 1 Frail to player",
    "Deal 7 attack damage to player and Apply 1 Weak to player",
    "Deal 8 attack damage to player and Apply 1 Weak to player",
    "Deal 8 attack damage to player and Apply 1 Vulnerable to player",
    "Deal 9 attack damage to player and Apply 1 Vulnerable to player",
    "Deal 6 attack damage to player and Apply 2 Vulnerable to player",
    "Deal 8 attack damage to player and Apply 2 Vulnerable to player",
    "Deal 10 attack damage to player and Add 1 status card(s) to discard",
    "Deal 11 attack damage to player and Add 1 status card(s) to discard",
    "Deal 11 attack damage to player and Add 2 status card(s) to discard",
    "Deal 12 attack damage to player and Add 2 status card(s) to discard",
    "Deal 16 attack damage to player and Add 2 status card(s) to discard",
    "Deal 18 attack damage to player and Add 2 status card(s) to discard",
    "Apply 3 Strength and Add 6 block to self",
    "Apply 4 Strength and Add 6 block to self",
    "Apply 5 Strength and Add 9 block to self",
    "Apply 6 Strength and Add 10 block to self",
    "Apply 8 Strength and Add 10 block to self",
    "Apply 10 Strength and Add 10 block to self",
    "Apply -1 Strength to player and Apply -1 Dexterity to player",
    "Apply -2 Strength to player and Apply -2 Dexterity to player",
    "Add 7 block to all enemies",
    "Add 9 block to self",
    "Add 11 block to all enemies",
    "Apply 3 Sharp Hide to self and Add 20 block to self",
)

STATUS_EFFECTS = (
    StatusEffectRepo.VULNERABLE,
    StatusEffectRepo.WEAK,
    StatusEffectRepo.STRENGTH,
    StatusEffectRepo.DEXTERITY,
    StatusEffectRepo.VIGOR,
    StatusEffectRepo.TOLERANCE,
    StatusEffectRepo.BOMB,
    StatusEffectRepo.RITUAL,
    StatusEffectRepo.FRAIL,
    StatusEffectRepo.ENTANGLED,
    StatusEffectRepo.CURL_UP,
    StatusEffectRepo.ANGER,
    StatusEffectRepo.SHARP_HIDE,
    StatusEffectRepo.ARTIFACT,
)

HAND_CHOICE_PURPOSES = (
    "exhaust_hand_card",
    "upgrade_hand_card",
    "topdeck_hand_card",
    "duplicate_hand_card",
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
        pending_hand_choice_count = 2 + len(HAND_CHOICE_PURPOSES) + self.max_hand_size
        return scalar_count + status_count + hand_slot_count + pile_count + len(self.intent_names) + pending_hand_choice_count

    def encode(self, battle_state, pending_hand_choice=None) -> np.ndarray:
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

        features.extend(self._pending_hand_choice_features(pending_hand_choice))

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

    def _pending_hand_choice_features(self, pending_hand_choice) -> list[float]:
        if pending_hand_choice is None:
            return [1.0, 0.0] + [0.0 for _ in HAND_CHOICE_PURPOSES] + [0.0 for _ in range(self.max_hand_size)]

        purpose = pending_hand_choice.purpose
        hand_indices = set(pending_hand_choice.hand_indices)
        return (
            [0.0, 1.0]
            + [1.0 if purpose == name else 0.0 for name in HAND_CHOICE_PURPOSES]
            + [1.0 if index in hand_indices else 0.0 for index in range(self.max_hand_size)]
        )
