from __future__ import annotations
import random
from config import CardType, MAX_STATUS
from typing import Callable
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from battle import BattleState
    from card import Card
    from game import GameState
    from agent import Agent

class StatusEffectDefinition:
    def __init__(self, name,
                 stack: Callable[[list[StatusEffectObject]], list[bool]],
                 end_turn: Callable[[StatusEffectObject], None],
                 done: Callable[[StatusEffectObject], bool],
                 repr: Callable[[StatusEffectObject], str]|None):
        self.name = name
        self.stack = stack
        self.end_turn = end_turn
        self.done = done
        if repr is not None:
            self.is_hidden = False
            self.repr = repr
        else:
            self.is_hidden = True
            self.repr = SEDef._hidden_repr
    
    @staticmethod
    def zero_done(se: StatusEffectObject) -> bool:
        return se.val == 0

    @staticmethod
    def never_done(se: StatusEffectObject) -> bool:
        return False

    @staticmethod
    def always_done(se: StatusEffectObject) -> bool:
        return True

    @staticmethod
    def add_stack(se_list: list[StatusEffectObject]) -> list[bool]:
        keep = [False for _ in se_list]
        total = min(MAX_STATUS, sum([se.val for se in se_list]))
        se_list[0].val = total
        keep[0] = True
        return keep
        
    @staticmethod
    def no_stack(se_list: list[StatusEffectObject]) -> list[bool]:
        keep = [False for _ in se_list]
        keep[0] = True
        return keep
    
    @staticmethod
    def unique_stack(se_list: list[StatusEffectObject]) -> list[bool]:
        return [True for _ in se_list]

    @staticmethod
    def decrease(se: StatusEffectObject, amount: int = 1):
        se.val -= amount
    
    @staticmethod
    def increase(se: StatusEffectObject, amount: int = 1):
        se.val += amount
    
    @staticmethod
    def get_decrease(amount: int = 1) -> Callable[[StatusEffectObject], None]:
        return lambda x: StatusEffectDefinition.decrease(x, amount)
    
    @staticmethod
    def get_increase(amount: int = 1):
        return lambda x: StatusEffectDefinition.increase(x, amount)
    
    @staticmethod
    def remove(se: StatusEffectObject):
        se.done = lambda: True
    
    @staticmethod
    def no_change(se: StatusEffectObject):
        return
    
    @staticmethod
    def key_value_repr(se: StatusEffectObject):
        return f"<{se.definition.name}>: {se.val}"
    
    @staticmethod
    def _hidden_repr(se: StatusEffectObject):
        raise Exception(f"Hidden status effect {se.definition.name} does not have a representation.")
    
    def __repr__(self):
        return self.name
SEDef = StatusEffectDefinition

class StatusEffectRepo:
    VULNERABLE = SEDef("Vulnerable", SEDef.add_stack, SEDef.get_decrease(1), SEDef.zero_done, SEDef.key_value_repr)
    WEAK = SEDef("Weak", SEDef.add_stack, SEDef.get_decrease(1), SEDef.zero_done, SEDef.key_value_repr)
    STRENGTH = SEDef("Strength", SEDef.add_stack, SEDef.no_change, SEDef.zero_done, SEDef.key_value_repr)
    VIGOR = SEDef("Vigor", SEDef.add_stack, SEDef.no_change, SEDef.zero_done, SEDef.key_value_repr)
    TOLERANCE = SEDef("Tolerance", SEDef.no_stack, SEDef.get_increase(2), SEDef.zero_done, SEDef.key_value_repr)
    BOMB = SEDef("Bomb", SEDef.unique_stack, SEDef.get_decrease(1), SEDef.zero_done, SEDef.key_value_repr)
    NO_DRAW = SEDef("No Draw", SEDef.no_stack, SEDef.remove, SEDef.never_done, SEDef.key_value_repr)
    BARRICADE = SEDef("Barricade", SEDef.no_stack, SEDef.no_change, SEDef.never_done, SEDef.key_value_repr)
    METALLICIZE = SEDef("Metallicize", SEDef.add_stack, SEDef.no_change, SEDef.zero_done, SEDef.key_value_repr)
    LOSE_STRENGTH = SEDef("Lose Strength", SEDef.add_stack, SEDef.no_change, SEDef.zero_done, SEDef.key_value_repr)
    BERSERK = SEDef("Berserk", SEDef.add_stack, SEDef.no_change, SEDef.zero_done, SEDef.key_value_repr)
    DEMON_FORM = SEDef("Demon Form", SEDef.add_stack, SEDef.no_change, SEDef.zero_done, SEDef.key_value_repr)
    BRUTALITY = SEDef("Brutality", SEDef.add_stack, SEDef.no_change, SEDef.zero_done, SEDef.key_value_repr)
    RAGE = SEDef("Rage", SEDef.add_stack, SEDef.remove, SEDef.never_done, SEDef.key_value_repr)
    FEEL_NO_PAIN = SEDef("Feel No Pain", SEDef.add_stack, SEDef.no_change, SEDef.zero_done, SEDef.key_value_repr)
    DARK_EMBRACE = SEDef("Dark Embrace", SEDef.add_stack, SEDef.no_change, SEDef.zero_done, SEDef.key_value_repr)
    EVOLVE = SEDef("Evolve", SEDef.add_stack, SEDef.no_change, SEDef.zero_done, SEDef.key_value_repr)
    FIRE_BREATHING = SEDef("Fire Breathing", SEDef.add_stack, SEDef.no_change, SEDef.zero_done, SEDef.key_value_repr)
    JUGGERNAUT = SEDef("Juggernaut", SEDef.add_stack, SEDef.no_change, SEDef.zero_done, SEDef.key_value_repr)
    FLAME_BARRIER = SEDef("Flame Barrier", SEDef.add_stack, SEDef.remove, SEDef.never_done, SEDef.key_value_repr)
    COMBUST = SEDef("Combust", SEDef.add_stack, SEDef.no_change, SEDef.zero_done, SEDef.key_value_repr)
    CORRUPTION = SEDef("Corruption", SEDef.no_stack, SEDef.no_change, SEDef.never_done, SEDef.key_value_repr)
    DOUBLE_TAP = SEDef("Double Tap", SEDef.add_stack, SEDef.remove, SEDef.zero_done, SEDef.key_value_repr)
    RUPTURE = SEDef("Rupture", SEDef.add_stack, SEDef.no_change, SEDef.zero_done, SEDef.key_value_repr)

class StatusEffectObject:
    def __init__(self, definition: StatusEffectDefinition, val: int):
        self.val = val
        self.definition = definition
    
    def done(self):
        return self.definition.done(self)
    
    def __repr__(self) -> str:
        return self.definition.repr(self)

class StatusEffectState:
    def __init__(self):
        self.status_effects: list[StatusEffectObject] = []
    
    def get(self, status: StatusEffectDefinition) -> int:
        values = self._get_obj(status)
        if len(values) > 1:
            raise Exception(f"Cannot return a single value for {status}")
        elif len(values) == 0:
            return 0
        return values[0].val

    def has(self, status: StatusEffectDefinition) -> bool:
        return len(self._get_obj(status)) > 0

    def _get_obj(self, status: StatusEffectDefinition) -> list[StatusEffectObject]:
        ret: list[StatusEffectObject] = []
        for se_obj in self.status_effects:
            if se_obj.definition.name == status.name:
                ret.append(se_obj)
        return ret
    
    def end_turn(self):
        for se in self.status_effects:
            se.definition.end_turn(se)
        self.clean()
        
    def remove_status(self, sed: StatusEffectDefinition):
        find = self._get_obj(sed)
        for se in find:
            se.done = lambda: True
        self.clean()
    
    def apply_status(self, definition: StatusEffectDefinition, amount: int):
        self.status_effects.append(StatusEffectObject(definition, amount))
        find = self._get_obj(definition)
        keep = definition.stack(find)
        for se, should_keep in zip(find, keep):
            if not should_keep:
                se.done = lambda: True
        self.clean()

    def clean_up(self):
        self.status_effects = []

    def clean(self):
        self.status_effects = [se for se in self.status_effects if not se.done()]

    def __repr__(self) -> str:
        return f'[{",".join([repr(se) for se in self.status_effects if not se.definition.is_hidden])}]'

def tolerance_after(__: None, additional_info: tuple[Agent, GameState, BattleState, list[Agent]]):
    by, _, battle_state, _ = additional_info
    battle_state.gain_block(by, by.status_effect_state.get(StatusEffectRepo.TOLERANCE))

def bomb_after(__: None, additional_info: tuple[Agent, GameState, BattleState, list[Agent]]):
    by, _, _, other_side = additional_info
    bomb = [se for se in by.status_effect_state._get_obj(StatusEffectRepo.BOMB) if se.val == 1]
    for _ in bomb:
        for agent in other_side:
            agent.get_damaged(40)

def metallicize_end(__: None, additional_info: tuple[Agent, GameState, BattleState, list[Agent]]):
    by, _, battle_state, _ = additional_info
    amount = by.status_effect_state.get(StatusEffectRepo.METALLICIZE)
    if amount > 0:
        battle_state.gain_block(by, amount)

def lose_strength_end(__: None, additional_info: tuple[Agent, GameState, BattleState, list[Agent]]):
    by, _, _, _ = additional_info
    amount = by.status_effect_state.get(StatusEffectRepo.LOSE_STRENGTH)
    if amount > 0:
        by.status_effect_state.apply_status(StatusEffectRepo.STRENGTH, -amount)
        by.status_effect_state.remove_status(StatusEffectRepo.LOSE_STRENGTH)

def berserk_start(__: None, additional_info: tuple[Agent, GameState, BattleState, list[Agent]]):
    by, _, battle_state, _ = additional_info
    amount = by.status_effect_state.get(StatusEffectRepo.BERSERK)
    if amount > 0:
        battle_state.add_to_mana(amount)

def demon_form_start(__: None, additional_info: tuple[Agent, GameState, BattleState, list[Agent]]):
    by, _, _, _ = additional_info
    amount = by.status_effect_state.get(StatusEffectRepo.DEMON_FORM)
    if amount > 0:
        by.status_effect_state.apply_status(StatusEffectRepo.STRENGTH, amount)

def brutality_start(__: None, additional_info: tuple[Agent, GameState, BattleState, list[Agent]]):
    by, _, battle_state, _ = additional_info
    amount = by.status_effect_state.get(StatusEffectRepo.BRUTALITY)
    if amount > 0:
        battle_state.lose_hp(by, amount, from_card=False)
        battle_state.draw(amount)

def combust_end(__: None, additional_info: tuple[Agent, GameState, BattleState, list[Agent]]):
    by, _, battle_state, other_side = additional_info
    amount = by.status_effect_state.get(StatusEffectRepo.COMBUST)
    if amount > 0:
        battle_state.lose_hp(by, 1, from_card=False)
        for agent in list(other_side):
            agent.get_damaged(amount)
        battle_state.enemies = [enemy for enemy in battle_state.enemies if not enemy.is_dead()]

def rage_play(__: None, additional_info: tuple[Agent, GameState, BattleState, Card]):
    by, _, battle_state, card = additional_info
    amount = by.status_effect_state.get(StatusEffectRepo.RAGE)
    if amount > 0 and card.card_type == CardType.ATTACK:
        battle_state.gain_block(by, amount)

def feel_no_pain_exhaust(__: None, additional_info: tuple[Agent, GameState, BattleState, Card]):
    by, _, battle_state, _ = additional_info
    amount = by.status_effect_state.get(StatusEffectRepo.FEEL_NO_PAIN)
    if amount > 0:
        battle_state.gain_block(by, amount)

def dark_embrace_exhaust(__: None, additional_info: tuple[Agent, GameState, BattleState, Card]):
    by, _, battle_state, _ = additional_info
    amount = by.status_effect_state.get(StatusEffectRepo.DARK_EMBRACE)
    if amount > 0 and not battle_state.ended():
        battle_state.draw(amount)

def evolve_draw(__: None, additional_info: tuple[Agent, GameState, BattleState, Card]):
    by, _, battle_state, card = additional_info
    amount = by.status_effect_state.get(StatusEffectRepo.EVOLVE)
    if amount > 0 and card.card_type == CardType.STATUS and not by.status_effect_state.has(StatusEffectRepo.NO_DRAW):
        battle_state.draw(amount)

def fire_breathing_draw(__: None, additional_info: tuple[Agent, GameState, BattleState, Card]):
    by, _, battle_state, card = additional_info
    amount = by.status_effect_state.get(StatusEffectRepo.FIRE_BREATHING)
    if amount > 0 and card.card_type in (CardType.STATUS, CardType.CURSE):
        for enemy in list(battle_state.enemies):
            enemy.get_damaged(amount)
        battle_state.enemies = [enemy for enemy in battle_state.enemies if not enemy.is_dead()]

def juggernaut_block(__: None, additional_info: tuple[Agent, GameState, BattleState, int]):
    by, _, battle_state, gained = additional_info
    amount = by.status_effect_state.get(StatusEffectRepo.JUGGERNAUT)
    if amount > 0 and gained > 0 and len(battle_state.enemies) > 0:
        random.choice(battle_state.enemies).get_damaged(amount)
        battle_state.enemies = [enemy for enemy in battle_state.enemies if not enemy.is_dead()]

def flame_barrier_attacked(__: None, additional_info: tuple[Agent, GameState, BattleState, Agent, int]):
    by, _, _, attacker, _ = additional_info
    amount = by.status_effect_state.get(StatusEffectRepo.FLAME_BARRIER)
    if amount > 0 and attacker is not by:
        attacker.get_damaged(amount)

def rupture_hp_loss(__: None, additional_info: tuple[Agent, GameState, BattleState, int, bool]):
    by, _, _, _, from_card = additional_info
    amount = by.status_effect_state.get(StatusEffectRepo.RUPTURE)
    if amount > 0 and from_card:
        by.status_effect_state.apply_status(StatusEffectRepo.STRENGTH, amount)

def strength_apply(amount: int, additional_info: tuple[Agent, GameState, BattleState, Agent]):
    by, _, _, _ = additional_info
    amount += by.status_effect_state.get(StatusEffectRepo.STRENGTH)
    return amount

def vigor_apply(amount: int, additional_info: tuple[Agent, GameState, BattleState, Agent]):
    by, _, _, _ = additional_info
    amount += by.status_effect_state.get(StatusEffectRepo.VIGOR)
    return amount

def vigor_after(_, additional_info: tuple[Agent, GameState, BattleState, Agent]):
    by, _, _, _ = additional_info
    # TODO this should be applied for multiple damages on the same card
    by.status_effect_state.remove_status(StatusEffectRepo.VIGOR)

def vulnerable_apply(amount: int, additional_info: tuple[Agent, GameState, BattleState, Agent]):
    _, _, _, target = additional_info
    if target.status_effect_state.get(StatusEffectRepo.VULNERABLE) > 0:
        amount = int(amount * 1.5)
    return amount

def weak_apply(amount: int, additional_info: tuple[Agent, GameState, BattleState, Agent]):
    by, _, _, _ = additional_info
    if by.status_effect_state.get(StatusEffectRepo.WEAK) > 0:
        amount = int(amount * 0.75)
    return amount
