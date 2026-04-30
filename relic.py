from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent
    from battle import BattleState
    from card import Card
    from game import GameState
from status_effecs import StatusEffectRepo
from config import CardType


class Relic:
    name = "Relic"

    def at_battle_start(self, game_state: GameState, battle_state: BattleState) -> None:
        return

    def on_card_play(self, card: Card, game_state: GameState, battle_state: BattleState) -> None:
        return

    def after_card_play(self, card: Card, game_state: GameState, battle_state: BattleState) -> None:
        return

    def at_turn_end(self, agent: Agent, game_state: GameState, battle_state: BattleState) -> None:
        return

    def at_turn_start(self, agent: Agent, game_state: GameState, battle_state: BattleState) -> None:
        return

    def on_attacked(
        self,
        target: Agent,
        attacker: Agent,
        amount: int,
        game_state: GameState,
        battle_state: BattleState,
    ) -> None:
        return

    def on_hp_loss(self, target: Agent, amount: int, from_card: bool, game_state: GameState, battle_state: BattleState) -> None:
        return

    def on_heal(self, target: Agent, amount: int, game_state: GameState, battle_state: BattleState) -> None:
        return

    def modify_attack_damage(
        self,
        amount: int,
        attacker: Agent,
        target: Agent,
        game_state: GameState,
        battle_state: BattleState,
    ) -> int:
        return amount


class Anchor(Relic):
    name = "Anchor"

    def at_battle_start(self, game_state: GameState, battle_state: BattleState) -> None:
        battle_state.gain_block(battle_state.player, 10)


class Vajra(Relic):
    name = "Vajra"

    def at_battle_start(self, game_state: GameState, battle_state: BattleState) -> None:
        battle_state.player.status_effect_state.apply_status(StatusEffectRepo.STRENGTH, 1)


class OddlySmoothStone(Relic):
    name = "Oddly Smooth Stone"

    def at_battle_start(self, game_state: GameState, battle_state: BattleState) -> None:
        battle_state.player.status_effect_state.apply_status(StatusEffectRepo.DEXTERITY, 1)


class PaperPhrog(Relic):
    name = "Paper Phrog"


class Lantern(Relic):
    name = "Lantern"

    def at_turn_start(self, agent: Agent, game_state: GameState, battle_state: BattleState) -> None:
        if agent is game_state.player and battle_state.turn == 1:
            battle_state.add_to_mana(1)


class HappyFlower(Relic):
    name = "Happy Flower"

    def __init__(self):
        self.counter = 0

    def at_turn_start(self, agent: Agent, game_state: GameState, battle_state: BattleState) -> None:
        if agent is not game_state.player:
            return
        self.counter += 1
        if self.counter >= 3:
            self.counter = 0
            battle_state.add_to_mana(1)


class Orichalcum(Relic):
    name = "Orichalcum"

    def at_turn_end(self, agent: Agent, game_state: GameState, battle_state: BattleState) -> None:
        if agent is game_state.player and agent.block == 0:
            battle_state.gain_block(agent, 6)


class BronzeScales(Relic):
    name = "Bronze Scales"

    def on_attacked(
        self,
        target: Agent,
        attacker: Agent,
        amount: int,
        game_state: GameState,
        battle_state: BattleState,
    ) -> None:
        if target is game_state.player and amount > 0 and not attacker.is_dead():
            attacker.get_damaged(3)


class BagOfPreparation(Relic):
    name = "Bag of Preparation"

    def at_turn_start(self, agent: Agent, game_state: GameState, battle_state: BattleState) -> None:
        if agent is game_state.player and battle_state.turn == 1:
            battle_state.draw(2)


class PenNib(Relic):
    name = "Pen Nib"

    def __init__(self):
        self.counter = 0
        self.active = False

    def on_card_play(self, card: Card, game_state: GameState, battle_state: BattleState) -> None:
        if card.card_type != CardType.ATTACK:
            return
        self.counter += 1
        if self.counter >= 10:
            self.counter = 0
            self.active = True

    def after_card_play(self, card: Card, game_state: GameState, battle_state: BattleState) -> None:
        self.active = False

    def modify_attack_damage(
        self,
        amount: int,
        attacker: Agent,
        target: Agent,
        game_state: GameState,
        battle_state: BattleState,
    ) -> int:
        if self.active and attacker is game_state.player:
            return amount * 2
        return amount


class Nunchaku(Relic):
    name = "Nunchaku"

    def __init__(self):
        self.counter = 0

    def on_card_play(self, card: Card, game_state: GameState, battle_state: BattleState) -> None:
        if card.card_type != CardType.ATTACK:
            return
        self.counter += 1
        if self.counter >= 10:
            self.counter = 0
            battle_state.add_to_mana(1)


class RedSkull(Relic):
    name = "Red Skull"

    def __init__(self):
        self.active = False

    def at_battle_start(self, game_state: GameState, battle_state: BattleState) -> None:
        self._update(game_state)

    def on_hp_loss(self, target: Agent, amount: int, from_card: bool, game_state: GameState, battle_state: BattleState) -> None:
        if target is game_state.player:
            self._update(game_state)

    def on_heal(self, target: Agent, amount: int, game_state: GameState, battle_state: BattleState) -> None:
        if target is game_state.player:
            self._update(game_state)

    def _update(self, game_state: GameState) -> None:
        player = game_state.player
        if not self.active and player.health <= player.max_health // 2:
            player.status_effect_state.apply_status(StatusEffectRepo.STRENGTH, 3)
            self.active = True
        elif self.active and player.health > player.max_health // 2:
            player.status_effect_state.apply_status(StatusEffectRepo.STRENGTH, -3)
            self.active = False


class PreservedInsect(Relic):
    name = "Preserved Insect"
    ELITE_NAMES = {"GremlinNob", "Lagavulin", "Sentry"}

    def at_battle_start(self, game_state: GameState, battle_state: BattleState) -> None:
        for enemy in battle_state.enemies:
            if enemy.name not in self.ELITE_NAMES:
                continue
            enemy.max_health = max(1, int(enemy.max_health * 0.75))
            enemy.health = min(enemy.health, enemy.max_health)


RELIC_FACTORIES = {
    "Anchor": Anchor,
    "Bag of Preparation": BagOfPreparation,
    "Bronze Scales": BronzeScales,
    "Happy Flower": HappyFlower,
    "Lantern": Lantern,
    "Nunchaku": Nunchaku,
    "Oddly Smooth Stone": OddlySmoothStone,
    "Orichalcum": Orichalcum,
    "Paper Phrog": PaperPhrog,
    "Pen Nib": PenNib,
    "Preserved Insect": PreservedInsect,
    "Red Skull": RedSkull,
    "Vajra": Vajra,
}


def create_relic_by_name(name: str) -> Relic:
    try:
        return RELIC_FACTORIES[name]()
    except KeyError as exc:
        supported = ", ".join(sorted(RELIC_FACTORIES))
        raise ValueError(f"Unsupported relic {name!r}. Supported: {supported}") from exc
