from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RLActionType(Enum):
    END_TURN = "end_turn"
    PLAY_CARD = "play_card"


@dataclass(frozen=True)
class RLAction:
    action_type: RLActionType
    hand_index: int | None = None
    target_index: int | None = None

    @staticmethod
    def end_turn() -> RLAction:
        return RLAction(RLActionType.END_TURN)

    @staticmethod
    def play_card(hand_index: int, target_index: int = 0) -> RLAction:
        return RLAction(RLActionType.PLAY_CARD, hand_index, target_index)
