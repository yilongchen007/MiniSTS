from __future__ import annotations
import copy
import random
from collections.abc import Callable
from value import Value, ConstValue
from status_effecs import StatusEffectRepo
from target.card_target import CardPile
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from battle import BattleState
    from card import Card
    from game import GameState
    from agent import Agent

class Action:
    def __init__(self, *values: Value) -> None:
        self.values = values

    def And(self, other: Action) -> Action:
        return AndAction(self, other)
    
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        raise NotImplementedError("The \"play\" method is not implemented for action {}.".format(self.__class__.__name__))

    def __repr__(self) -> str:
        return self.__class__.__name__ + "({})".format('-'.join([value.__repr__() for value in self.values]))
    
class AndAction(Action):
    def __init__(self, *actions: Action):
        super().__init__(*[value for action in actions for value in action.values])
        self.actions = actions
    
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState):
        for action in self.actions:
            action.play(by, game_state, battle_state)
    
    def __repr__(self) -> str:
        parts = [repr(action) for action in self.actions]
        return ' and '.join(part for part in parts if part)

class AddMana(Action):
    def __init__(self, val: Value):
        super().__init__(val)
        self.val = val
    
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        battle_state.add_to_mana(self.val.get())


class DrawCard(Action):
    def __init__(self, val: Value):
        super().__init__(val)
        self.val = val

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        if by.status_effect_state.has(StatusEffectRepo.NO_DRAW):
            return
        battle_state.draw(self.val.get())

    def __repr__(self) -> str:
        amount = self.val.peek()
        return f"Draw {amount} {'card' if amount == 1 else 'cards'}"

class MakeTempCard(Action):
    def __init__(self, card_factory: Callable[[], Card], card_pile: CardPile, count: Value = ConstValue(1)):
        super().__init__(count)
        self.card_factory = card_factory
        self.card_pile = card_pile
        self.count = count

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        for _ in range(self.count.get()):
            battle_state.add_card_to_pile(self.card_factory(), self.card_pile)

    def __repr__(self) -> str:
        return f"Add {self.count.peek()} temporary card(s) to {self.card_pile.name.lower()}"

class UpgradeAllCardsInHand(Action):
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        for card in list(battle_state.hand):
            card.upgrade()

    def __repr__(self) -> str:
        return "Upgrade all cards in hand"

class DoubleBlock(Action):
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        battle_state.gain_block(by, by.block)

    def __repr__(self) -> str:
        return "Double your Block"

class LimitBreakAction(Action):
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        strength = by.status_effect_state.get(StatusEffectRepo.STRENGTH)
        if strength > 0:
            by.status_effect_state.apply_status(StatusEffectRepo.STRENGTH, strength)

    def __repr__(self) -> str:
        return "Double your Strength"

class ExhaustNonAttackCards(Action):
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        from config import CardType
        for card in list(battle_state.hand):
            if card.card_type != CardType.ATTACK:
                battle_state.exhaust(card)

    def __repr__(self) -> str:
        return "Exhaust all non-Attack cards in hand"

class SecondWindAction(Action):
    def __init__(self, block_per_card: Value):
        super().__init__(block_per_card)
        self.block_per_card = block_per_card

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        from config import CardType
        cards = [card for card in list(battle_state.hand) if card.card_type != CardType.ATTACK]
        for card in cards:
            battle_state.exhaust(card)
        battle_state.gain_block(by, self.block_per_card.get() * len(cards))

    def __repr__(self) -> str:
        return f"Exhaust non-Attacks; gain {self.block_per_card.peek()} Block each"

class SeverSoulAction(Action):
    def __init__(self, damage: Value):
        super().__init__(damage)
        self.damage = damage

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        from config import CardType
        target = battle_state.get_player_agent_target("enemy", battle_state.enemies)
        for card in list(battle_state.hand):
            if card.card_type not in (CardType.ATTACK, CardType.POWER):
                battle_state.exhaust(card)
        battle_state.deal_attack_damage(by, target, self.damage.get())

    def __repr__(self) -> str:
        return f"Exhaust non-Attack non-Power cards; deal {self.damage.peek()} damage"

class FiendFireAction(Action):
    def __init__(self, damage: Value):
        super().__init__(damage)
        self.damage = damage

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        target = battle_state.get_player_agent_target("enemy", battle_state.enemies)
        cards = list(battle_state.hand)
        for card in cards:
            battle_state.exhaust(card)
            battle_state.deal_attack_damage(by, target, self.damage.get())

    def __repr__(self) -> str:
        return f"Exhaust hand; deal {self.damage.peek()} damage per card"

class WarcryAction(Action):
    def __init__(self, draw_count: Value):
        super().__init__(draw_count)
        self.draw_count = draw_count

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        battle_state.draw(self.draw_count.get())
        if len(battle_state.hand) == 0:
            return
        card = battle_state.get_player_card_target("hand", battle_state.hand)
        battle_state.remove_card(card)
        battle_state.draw_pile.append(card)

    def __repr__(self) -> str:
        return f"Draw {self.draw_count.peek()}; put a hand card on top of draw pile"

class HeadbuttAction(Action):
    def __init__(self, damage: Value):
        super().__init__(damage)
        self.damage = damage

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        target = battle_state.get_player_agent_target("enemy", battle_state.enemies)
        battle_state.deal_attack_damage(by, target, self.damage.get())
        if len(battle_state.discard_pile) == 0:
            return
        card = battle_state.get_player_card_target("discard", battle_state.discard_pile)
        battle_state.remove_card(card)
        battle_state.draw_pile.append(card)

    def __repr__(self) -> str:
        return f"Deal {self.damage.peek()} damage; put a discard card on top of draw pile"

class ExhumeAction(Action):
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        if len(battle_state.exhaust_pile) == 0:
            return
        card = battle_state.get_player_card_target("exhaust", battle_state.exhaust_pile)
        battle_state.remove_card(card)
        battle_state.hand.append(card)

    def __repr__(self) -> str:
        return "Put a card from exhaust into hand"

class DualWieldAction(Action):
    def __init__(self, count: Value):
        super().__init__(count)
        self.count = count

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        from config import CardType
        cards = [card for card in battle_state.hand if card.card_type in (CardType.ATTACK, CardType.POWER)]
        if len(cards) == 0:
            return
        card = battle_state.get_player_card_target("hand", cards)
        for _ in range(self.count.get()):
            battle_state.hand.append(copy.deepcopy(card))

    def __repr__(self) -> str:
        return f"Duplicate an Attack or Power {self.count.peek()} time(s)"

class BurningPactAction(Action):
    def __init__(self, draw_count: Value):
        super().__init__(draw_count)
        self.draw_count = draw_count

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        if len(battle_state.hand) > 0:
            card = battle_state.get_player_card_target("hand", battle_state.hand)
            battle_state.exhaust(card)
        battle_state.draw(self.draw_count.get())

    def __repr__(self) -> str:
        return f"Exhaust a card; draw {self.draw_count.peek()}"

class HavocAction(Action):
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        if len(battle_state.draw_pile) == 0:
            battle_state.reshuffle()
        if len(battle_state.draw_pile) == 0:
            return
        card = battle_state.draw_pile.pop()
        if card.is_playable(game_state, battle_state, ignore_mana=True):
            card.play_actions(game_state, battle_state)
        battle_state.exhaust(card)

    def __repr__(self) -> str:
        return "Play the top card of draw pile and exhaust it"

class InfernalBladeAction(Action):
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        from card import CardGen
        attacks = [
            CardGen.Strike, CardGen.Bash, CardGen.Anger, CardGen.Pommel_Strike,
            CardGen.Cleave, CardGen.Wild_Strike, CardGen.Bludgeon, CardGen.Clothesline,
            CardGen.Thunderclap, CardGen.Twin_Strike, CardGen.Uppercut, CardGen.Iron_Wave,
            CardGen.Pummel, CardGen.Hemokinesis, CardGen.Body_Slam, CardGen.Carnage,
            CardGen.Clash, CardGen.Dropkick, CardGen.Heavy_Blade, CardGen.Headbutt,
            CardGen.Immolate, CardGen.Perfected_Strike, CardGen.Rampage, CardGen.Reckless_Charge,
            CardGen.Sever_Soul, CardGen.Sword_Boomerang,
        ]
        card = random.choice(attacks)()
        card.cost_override = 0
        battle_state.hand.append(card)

    def __repr__(self) -> str:
        return "Add a random Attack to hand; it costs 0 this turn"

class WhirlwindAction(Action):
    def __init__(self, damage: Value):
        super().__init__(damage)
        self.damage = damage

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        times = battle_state.mana
        battle_state.mana = 0
        for _ in range(times):
            for enemy in list(battle_state.enemies):
                battle_state.deal_attack_damage(by, enemy, self.damage.get())
            battle_state.enemies = [enemy for enemy in battle_state.enemies if not enemy.is_dead()]

    def __repr__(self) -> str:
        return f"Spend all energy; deal {self.damage.peek()} to all enemies per energy"

class PlayCard(Action):
    def __init__(self, card_index: int):
        super().__init__()
        self.card_index = card_index
    
    def get_card_index(self):
        return self.card_index

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        battle_state.play_card(self.card_index)
    
    def __repr__(self) -> str:
        return f"Play card {self.card_index} from your hand"

class NoAction(Action):
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        pass

class EndAgentTurn(Action):
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        battle_state.end_agent_turn()
    
    def __repr__(self) -> str:
        return "End turn"
