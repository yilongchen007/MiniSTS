from __future__ import annotations
from value import Value, ConstValue
from status_effecs import StatusEffectDefinition, StatusEffectRepo, strength_apply, vigor_apply, vulnerable_apply, weak_apply, vigor_after
from utility import Event
from typing import TYPE_CHECKING
from action.action import Action
if TYPE_CHECKING:
    from battle import BattleState
    from game import GameState
    from agent import Agent
    from target.agent_target import AgentTarget

class AgentTargetedAction(Action):
    def __init__(self, targeted: AgentTargeted, target: AgentTarget):
        super().__init__(*targeted.values)
        self.targeted = targeted
        self.target = target
    
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        self.targeted.play_many(by, game_state, battle_state, self.target.get(by, battle_state))
    
    def __repr__(self) -> str:
        return self.targeted.__repr__() + " to " + self.target.__repr__()

class AgentTargeted:
    def __init__(self, *values: Value) -> None:
        self.values = values

    def To(self, target: AgentTarget):
        return AgentTargetedAction(self, target)

    def And(self, other: AgentTargeted) -> AgentTargeted:
        return AndAgentTargeted(self, other)
    
    def play_many(self, by: Agent, game_state: GameState, battle_state: BattleState, targets: list[Agent]) -> None:
        for target in targets:
            self.play(by, game_state, battle_state, target)

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState, target: Agent) -> None:
        raise NotImplementedError("The \"play\" method is not implemented for {}.".format(self.__class__.__name__))
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + "({})".format('-'.join([value.__repr__() for value in self.values]))

class AndAgentTargeted(AgentTargeted):
    def __init__(self, *targeted_set: AgentTargeted):
        super().__init__(*[value for targeted in targeted_set for value in targeted.values])
        self.targeted_set = targeted_set
    
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState, target: Agent):
        for targeted in self.targeted_set:
            targeted.play(by, game_state, battle_state, target)
    
    def __repr__(self) -> str:
        return ' and '.join([targeted.__repr__() for targeted in self.targeted_set])

class DealAttackDamage(AgentTargeted):
    event: Event[int, tuple[Agent, GameState, BattleState, Agent]] = Event()
    def __init__(self, val: Value, times: Value = ConstValue(1)):
        super().__init__(val, times)
        self.val = val
        self.times = times
    
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState, target: Agent) -> None:
        self.event.broadcast_before((by, game_state, battle_state, target))
        amount = self.val.get()
        amount = self.event.broadcast_apply(amount, (by, game_state, battle_state, target))
        times = self.times.get()
        for _ in range(times):
            battle_state.deal_attack_damage(by, target, round(amount))
        self.event.broadcast_after((by, game_state, battle_state, target))
    
    def __repr__(self) -> str:
        if self.times.peek() != 1:
            return f"Deal {self.val.peek()} attack damage {self.times.peek()} times"
        else:
            return f"Deal {self.val.peek()} attack damage"

class DealBodySlamDamage(AgentTargeted):
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState, target: Agent) -> None:
        DealAttackDamage(ConstValue(by.block)).play(by, game_state, battle_state, target)

    def __repr__(self) -> str:
        return "Deal attack damage equal to your Block"

class DealHeavyBladeDamage(AgentTargeted):
    def __init__(self, base: Value, strength_multiplier: Value):
        super().__init__(base, strength_multiplier)
        self.base = base
        self.strength_multiplier = strength_multiplier

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState, target: Agent) -> None:
        strength = by.status_effect_state.get(StatusEffectRepo.STRENGTH)
        damage = self.base.get() + strength * max(0, self.strength_multiplier.get() - 1)
        DealAttackDamage(ConstValue(damage)).play(by, game_state, battle_state, target)

    def __repr__(self) -> str:
        return f"Deal {self.base.peek()} damage; Strength affects it {self.strength_multiplier.peek()}x"

class Dropkick(AgentTargeted):
    def __init__(self, damage: Value):
        super().__init__(damage)
        self.damage = damage

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState, target: Agent) -> None:
        DealAttackDamage(self.damage).play(by, game_state, battle_state, target)
        if target.status_effect_state.has(StatusEffectRepo.VULNERABLE):
            battle_state.add_to_mana(1)
            battle_state.draw(1)

    def __repr__(self) -> str:
        return f"Deal {self.damage.peek()} damage; if target is Vulnerable, gain energy and draw"

class Feed(AgentTargeted):
    def __init__(self, damage: Value, max_hp_gain: Value):
        super().__init__(damage, max_hp_gain)
        self.damage = damage
        self.max_hp_gain = max_hp_gain

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState, target: Agent) -> None:
        DealAttackDamage(self.damage).play(by, game_state, battle_state, target)
        if target.is_dead():
            by.max_health += self.max_hp_gain.get()
            by.health += self.max_hp_gain.peek()

    def __repr__(self) -> str:
        return f"Deal {self.damage.peek()} damage; fatal raises max HP"

class Reaper(AgentTargeted):
    def __init__(self, damage: Value):
        super().__init__(damage)
        self.damage = damage

    def play_many(self, by: Agent, game_state: GameState, battle_state: BattleState, targets: list[Agent]) -> None:
        healed = 0
        for target in list(targets):
            healed += battle_state.deal_attack_damage(by, target, self.damage.get())
        by.get_healed(healed)

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState, target: Agent) -> None:
        healed = battle_state.deal_attack_damage(by, target, self.damage.get())
        by.get_healed(healed)

    def __repr__(self) -> str:
        return f"Deal {self.damage.peek()} to enemies; heal unblocked damage"

class PerfectedStrike(AgentTargeted):
    def __init__(self, base: Value, per_strike: Value):
        super().__init__(base, per_strike)
        self.base = base
        self.per_strike = per_strike

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState, target: Agent) -> None:
        piles = [game_state.deck, battle_state.hand, battle_state.draw_pile, battle_state.discard_pile, battle_state.exhaust_pile]
        strike_count = sum(1 for pile in piles for card in pile if "Strike" in card.name)
        DealAttackDamage(ConstValue(self.base.get() + self.per_strike.get() * strike_count)).play(by, game_state, battle_state, target)

    def __repr__(self) -> str:
        return f"Deal {self.base.peek()} plus {self.per_strike.peek()} per Strike card"

class Rampage(AgentTargeted):
    def __init__(self, damage: Value, increase: Value):
        super().__init__(damage, increase)
        self.damage = damage
        self.increase = increase

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState, target: Agent) -> None:
        DealAttackDamage(self.damage).play(by, game_state, battle_state, target)
        if hasattr(self.damage, "val"):
            self.damage.val += self.increase.get()

    def __repr__(self) -> str:
        return f"Deal {self.damage.peek()} damage; increase this combat"

class SwordBoomerang(AgentTargeted):
    def __init__(self, damage: Value, times: Value):
        super().__init__(damage, times)
        self.damage = damage
        self.times = times

    def play_many(self, by: Agent, game_state: GameState, battle_state: BattleState, targets: list[Agent]) -> None:
        for _ in range(self.times.get()):
            living = [enemy for enemy in battle_state.enemies if not enemy.is_dead()]
            if len(living) == 0:
                return
            import random
            DealAttackDamage(self.damage).play(by, game_state, battle_state, random.choice(living))

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState, target: Agent) -> None:
        self.play_many(by, game_state, battle_state, [target])

    def __repr__(self) -> str:
        return f"Deal {self.damage.peek()} to random enemy {self.times.peek()} times"

class SpotWeakness(AgentTargeted):
    def __init__(self, strength: Value):
        super().__init__(strength)
        self.strength = strength

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState, target: Agent) -> None:
        if "attack damage" in repr(target.get_intention(game_state, battle_state)):
            by.status_effect_state.apply_status(StatusEffectRepo.STRENGTH, self.strength.get())

    def __repr__(self) -> str:
        return f"If target intends to attack, gain {self.strength.peek()} Strength"
        

class DealDamage(AgentTargeted):
    def __init__(self, val: Value, times: Value = ConstValue(1)):
        super().__init__(val, times)
        self.val = val
        self.times = times
    
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState, target: Agent) -> None:
        amount = self.val.get()
        times = self.times.get()
        for _ in range(times):
            target.get_damaged(round(amount))
    
    def __repr__(self) -> str:
        if self.times.peek() != 1:
            return f"Deal {self.val.peek()} damage {self.times.peek()} times"
        else:
            return f"Deal {self.val.peek()} damage"

class LoseHP(AgentTargeted):
    def __init__(self, val: Value):
        super().__init__(val)
        self.val = val

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState, target: Agent) -> None:
        battle_state.lose_hp(target, self.val.get(), from_card=True)

    def __repr__(self) -> str:
        return f"Lose {self.val.peek()} HP"
        

class Heal(AgentTargeted):
    def __init__(self, val: Value):
        super().__init__(val)
        self.val = val
    
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState, target: Agent) -> None:
        target.get_healed(self.val.get())
    
    def __repr__(self) -> str:
        return f"Apply {self.val.peek()} heal"

class AddBlock(AgentTargeted):
    def __init__(self, val: Value):
        super().__init__(val)
        self.val = val
    
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState, target: Agent) -> None:
        battle_state.gain_block(target, self.val.get())
    
    def __repr__(self) -> str:
        return f"Add {self.val.peek()} block"

class ApplyStatus(AgentTargeted):
    def __init__(self, val: Value, status_effect: StatusEffectDefinition):
        super().__init__(val)
        self.val = val
        self.status_effect = status_effect
    
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState, target: Agent) -> None:
        target.status_effect_state.apply_status(self.status_effect, self.val.get())

    def __repr__(self) -> str:
        return f"Apply {self.val.peek()} {str(self.status_effect)}"
        # return self.__class__.__name__ + "({}-{})".format('-'.join([value.__repr__() for value in self.values]), self.status_effect)

# TODO order
DealAttackDamage.event.subscribe_values(strength_apply)
DealAttackDamage.event.subscribe_values(vigor_apply)
DealAttackDamage.event.subscribe_after(vigor_after)
DealAttackDamage.event.subscribe_values(vulnerable_apply)
DealAttackDamage.event.subscribe_values(weak_apply)
