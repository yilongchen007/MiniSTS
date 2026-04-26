from __future__ import annotations
import random
from action.action import Action, MakeTempCard, NoAction
from config import Character, MAX_HEALTH
from value import RandomUniformRange, ConstValue
from utility import RoundRobin, ItemSet, ItemSequence, RandomizedItemSet, PreventRepeats
from action.action import EndAgentTurn
from action.agent_targeted_action import DealAttackDamage, AddBlock, ApplyStatus
from target.agent_target import PlayerAgentTarget, SelfAgentTarget
from target.card_target import CardPile
from config import MAX_BLOCK, CHARACTER_NAME
from status_effecs import StatusEffectState, StatusEffectRepo
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from battle import BattleState
    from game import GameState
    from action.action import Action
    from ggpa.ggpa import GGPA

class Agent:
    def __init__(self, name: str, max_health: int):
        self.max_health = max_health
        self.health = max_health
        self.block = 0
        self.status_effect_state = StatusEffectState()
        self.name = name
        self.prev_action: Action|None = None
        self.death_handled = False
        self.last_attack_hp_loss = 0
    
    def set_name(self) -> None:
        raise NotImplementedError("Set name is not implemented for {}.".format(self.__class__.__name__))
    
    def is_dead(self):
        return self.health <= 0

    def get_damaged(self, amount: int) -> int:
        assert amount >= 0, "Damage amount cannot be less than 0"
        blocked = min(self.block, amount)
        amount -= blocked
        self.block -= blocked
        self.health -= amount
        self.last_attack_hp_loss = amount
        if self.health <= 0:
            self.health = 0
        return amount

    def lose_health(self, amount: int) -> int:
        assert amount >= 0, "HP loss amount cannot be less than 0"
        lost = min(self.health, amount)
        self.health -= amount
        if self.health <= 0:
            self.health = 0
        return lost
    
    def clear_block(self):
        if self.status_effect_state.has(StatusEffectRepo.BARRICADE):
            return
        self.block = 0

    def clean_up(self):
        self.status_effect_state.clean_up()
        self.block = 0

    def gain_block(self, amount: int):
        assert amount >= 0, "Block amount cannot be less than 0"
        self.block += amount
        if self.block >= MAX_BLOCK:
            self.block = MAX_BLOCK
        
    def get_healed(self, amount: int):
        assert amount >= 0, "Heal amount cannot be less than 0"
        self.health += amount
        if self.health >= self.max_health:
            self.health = self.max_health
    
    def _get_action(self, game_state: GameState, battle_state: BattleState) -> Action:
        raise NotImplementedError("The \"_get_action\" method is not implemented for {}.".format(self.__class__.__name__))
    
    def play(self, game_state: GameState, battle_state: BattleState) -> None:
        self.prev_action = self._get_action(game_state, battle_state)
        self.prev_action.play(self, game_state, battle_state)

    def on_death(self, game_state: GameState, battle_state: BattleState) -> None:
        if self.death_handled:
            return
        self.death_handled = True
    
    def __repr__(self) -> str:
        return "{}-hp:[{}/{}]-block:{}-status:{}".format(
            self.name, self.health, self.max_health, self.block, self.status_effect_state
        )

class Player(Agent):
    def __init__(self, character: Character, bot: GGPA):
        self.character = character
        self.bot = bot
        super().__init__(CHARACTER_NAME[self.character], MAX_HEALTH[self.character])
    
    def _get_action(self, game_state: GameState, battle_state: BattleState):
        return self.bot.choose_card(game_state, battle_state)

class Enemy(Agent):
    def __init__(self, name: str, max_health: int, action_set: ItemSet[Action]):
        super().__init__(name, max_health)
        self.action_set = action_set

    def _get_action(self, game_state: GameState, battle_state: BattleState) -> Action:
        return self.action_set.get().And(EndAgentTurn())

    def get_intention(self, game_state: GameState, battle_state: BattleState) -> Action:
        return self.action_set.peek()

class RepeatItem(ItemSet[Action]):
    def __init__(self, item: Action):
        super().__init__()
        self.item = item

    def _sample(self) -> Action:
        return self.item

class JawWormActionSet(ItemSet[Action]):
    def __init__(self, chomp: Action, bellow: Action, thrash: Action, first_move: bool = True):
        super().__init__()
        self.chomp = chomp
        self.bellow = bellow
        self.thrash = thrash
        self.first_move = first_move
        self.history: list[Action] = []

    def get(self) -> Action:
        ret = super().get()
        self.history.append(ret)
        return ret

    def _last_move(self, action: Action) -> bool:
        return len(self.history) >= 1 and self.history[-1] is action

    def _last_two_moves(self, action: Action) -> bool:
        return len(self.history) >= 2 and self.history[-1] is action and self.history[-2] is action

    def _sample(self) -> Action:
        if self.first_move:
            self.first_move = False
            return self.chomp

        roll = random.randrange(100)
        if roll < 25:
            if self._last_move(self.chomp):
                return self.bellow if random.random() < 0.5625 else self.thrash
            return self.chomp
        if roll < 55:
            if self._last_two_moves(self.thrash):
                return self.chomp if random.random() < 0.357 else self.bellow
            return self.thrash
        if self._last_move(self.bellow):
            return self.chomp if random.random() < 0.416 else self.thrash
        return self.bellow

class ScriptedEnemy(Enemy):
    def __init__(self, name: str, max_health: int):
        super().__init__(name, max_health, RepeatItem(NoAction()))
        self.move_history: list[str] = []
        self._next_move_cache: tuple[str, Action] | None = None

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        raise NotImplementedError(f"{self.__class__.__name__}.choose_move is not implemented")

    def get_intention(self, game_state: GameState, battle_state: BattleState) -> Action:
        if self._next_move_cache is None:
            self._next_move_cache = self.choose_move(game_state, battle_state)
        return self._next_move_cache[1]

    def _get_action(self, game_state: GameState, battle_state: BattleState) -> Action:
        if self._next_move_cache is None:
            self._next_move_cache = self.choose_move(game_state, battle_state)
        move_id, action = self._next_move_cache
        self._next_move_cache = None
        self.move_history.append(move_id)
        return action.And(EndAgentTurn())

    def last_move(self, move_id: str) -> bool:
        return len(self.move_history) > 0 and self.move_history[-1] == move_id

    def last_two_moves(self, move_id: str) -> bool:
        return len(self.move_history) >= 2 and self.move_history[-1] == move_id and self.move_history[-2] == move_id

    def last_three_moves(self, move_id: str) -> bool:
        return len(self.move_history) >= 3 and all(move == move_id for move in self.move_history[-3:])

class EscapeAction(Action):
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        by.health = 0
        by.death_handled = True
        battle_state.enemies = [enemy for enemy in battle_state.enemies if enemy is not by]

    def __repr__(self) -> str:
        return "Escape"

class SpawnEnemiesAction(Action):
    def __init__(self, *factories):
        super().__init__()
        self.factories = factories

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        battle_state.enemies.extend(factory(game_state) for factory in self.factories)

    def __repr__(self) -> str:
        return "Spawn enemies"

class SplitIntoAction(Action):
    def __init__(self, *factories):
        super().__init__()
        self.factories = factories

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        split_health = by.health
        by.health = 0
        by.death_handled = True
        battle_state.enemies = [enemy for enemy in battle_state.enemies if enemy is not by]
        children = [factory(game_state) for factory in self.factories]
        for child in children:
            child.max_health = max(1, split_health)
            child.health = child.max_health
        battle_state.enemies.extend(children)

    def __repr__(self) -> str:
        return "Split"

class SetStatusAction(Action):
    def __init__(self, status, amount: int):
        super().__init__()
        self.status = status
        self.amount = amount

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        by.status_effect_state.apply_status(self.status, self.amount)

    def __repr__(self) -> str:
        return f"Apply {self.amount} {self.status} to self"

class RemoveStatusAction(Action):
    def __init__(self, status):
        super().__init__()
        self.status = status

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        by.status_effect_state.remove_status(self.status)

    def __repr__(self) -> str:
        return f"Remove {self.status} from self"

class StunAction(Action):
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        if hasattr(by, "stun_next"):
            by.stun_next = False
        return

    def __repr__(self) -> str:
        return "Stunned"

class LagavulinSleepAction(Action):
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        by.idle_count += 1
        if by.idle_count >= 3:
            by._open()
            by.force_attack_once = True

    def __repr__(self) -> str:
        return "Sleep"

class LagavulinAttackAction(Action):
    def __init__(self, damage: int, forced: bool = False):
        super().__init__()
        self.damage = damage
        self.forced = forced

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        if self.forced:
            by.force_attack_once = False
        by.debuff_turn_count += 1
        DealAttackDamage(ConstValue(self.damage)).To(PlayerAgentTarget()).play(by, game_state, battle_state)

    def __repr__(self) -> str:
        return f"Deal {self.damage} attack damage"

class LagavulinDebuffAction(Action):
    def __init__(self, amount: int):
        super().__init__()
        self.amount = amount

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        by.debuff_turn_count = 0
        ApplyStatus(ConstValue(-self.amount), StatusEffectRepo.STRENGTH).To(PlayerAgentTarget()).And(
            ApplyStatus(ConstValue(-self.amount), StatusEffectRepo.DEXTERITY).To(PlayerAgentTarget())
        ).play(by, game_state, battle_state)

    def __repr__(self) -> str:
        return f"Apply -{self.amount} Strength to player and Apply -{self.amount} Dexterity to player"

class AddStatusCardAction(Action):
    def __init__(self, card_factory, card_pile: CardPile, count: int):
        super().__init__()
        self.card_factory = card_factory
        self.card_pile = card_pile
        self.count = count

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        for _ in range(self.count):
            battle_state.add_card_to_pile(self.card_factory(), self.card_pile)

    def __repr__(self) -> str:
        return f"Add {self.count} status card(s) to {self.card_pile.name.lower()}"

class SetForcedMoveAction(Action):
    def __init__(self, move_id: str | None):
        super().__init__()
        self.move_id = move_id

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        by.forced_move = self.move_id

    def __repr__(self) -> str:
        return f"Set next move to {self.move_id}"

class AcidSlimeSmall(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        max_health = RandomUniformRange(8, 12) if game_state.ascention < 7 else RandomUniformRange(9, 13)
        super().__init__("AcidSlime(S)", max_health.get())
        self.attack_damage = 3 if game_state.ascention < 2 else 4

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        tackle = DealAttackDamage(ConstValue(self.attack_damage)).To(PlayerAgentTarget())
        lick = ApplyStatus(ConstValue(1), StatusEffectRepo.WEAK).To(PlayerAgentTarget())
        if random.randrange(100) < 50:
            if self.last_two_moves("tackle"):
                return "lick", lick
            return "tackle", tackle
        if game_state.ascention >= 17:
            if self.last_move("lick"):
                return "tackle", tackle
        elif self.last_two_moves("lick"):
            return "tackle", tackle
        return "lick", lick

class SpikeSlimeSmall(Enemy):
    def __init__(self, game_state: GameState):
        max_health = RandomUniformRange(10, 14) if game_state.ascention < 7 else RandomUniformRange(11, 15)
        action_set: ItemSet[Action] = RoundRobin(0, DealAttackDamage(ConstValue(5 if game_state.ascention < 2 else 6)).To(PlayerAgentTarget()))
        super().__init__("SpikeSlime(S)", max_health.get(), action_set)

class JawWorm(Enemy):
    def __init__(self, game_state: GameState):
        max_health = RandomUniformRange(40, 44) if game_state.ascention < 7 else RandomUniformRange(42, 46)
        chomp: Action = DealAttackDamage(ConstValue(11 if game_state.ascention < 2 else 12)).To(PlayerAgentTarget())
        thrash: Action = DealAttackDamage(ConstValue(7)).To(PlayerAgentTarget()).And(AddBlock(ConstValue(5)).To(SelfAgentTarget()))
        bellow_block = 6 if game_state.ascention < 17 else 9
        bellow: Action = ApplyStatus(ConstValue(3 if game_state.ascention < 2 else 4 if game_state.ascention < 17 else 5), StatusEffectRepo.STRENGTH).And(AddBlock(ConstValue(bellow_block))).To(SelfAgentTarget())
        action_set = JawWormActionSet(chomp, bellow, thrash)
        super().__init__("JawWorm", max_health.get(), action_set)

class Cultist(Enemy):
    def __init__(self, game_state: GameState):
        max_health = RandomUniformRange(48, 54) if game_state.ascention < 7 else RandomUniformRange(50, 56)
        ritual_amount = 3 if game_state.ascention < 2 else 4
        if game_state.ascention >= 17:
            ritual_amount += 1
        incantation: Action = ApplyStatus(ConstValue(ritual_amount), StatusEffectRepo.RITUAL).To(SelfAgentTarget())
        dark_strike: Action = DealAttackDamage(ConstValue(6)).To(PlayerAgentTarget())
        action_set: ItemSet[Action] = ItemSequence(incantation, RepeatItem(dark_strike))
        super().__init__("Cultist", max_health.get(), action_set)

class AcidSlimeMedium(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = RandomUniformRange(28, 32) if game_state.ascention < 7 else RandomUniformRange(29, 34)
        super().__init__("AcidSlime(M)", hp.get())
        self.spit_damage = 7 if game_state.ascention < 2 else 8
        self.attack_damage = 10 if game_state.ascention < 2 else 12
        self.weak_amount = 1

    def get_damaged(self, amount: int) -> int:
        dealt = super().get_damaged(amount)
        if self.health > 0 and self.health <= self.max_health // 2:
            self._next_move_cache = ("split", SplitIntoAction(AcidSlimeSmall, AcidSlimeSmall))
        return dealt

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        if self.health <= self.max_health // 2:
            return "split", SplitIntoAction(AcidSlimeSmall, AcidSlimeSmall)
        from card import CardGen
        roll = random.randrange(100)
        spit = DealAttackDamage(ConstValue(self.spit_damage)).To(PlayerAgentTarget()).And(
            AddStatusCardAction(CardGen.Slimed, CardPile.DISCARD, 1)
        )
        tackle = DealAttackDamage(ConstValue(self.attack_damage)).To(PlayerAgentTarget())
        lick = ApplyStatus(ConstValue(self.weak_amount), StatusEffectRepo.WEAK).To(PlayerAgentTarget())
        if roll < 40:
            if self.last_two_moves("corrosive_spit"):
                return ("tackle", tackle) if random.random() < 0.5 else ("lick", lick)
            return "corrosive_spit", spit
        if roll < 80:
            if self.last_two_moves("tackle"):
                return ("corrosive_spit", spit) if random.random() < 0.5 else ("lick", lick)
            return "tackle", tackle
        if self.last_move("lick"):
            return ("corrosive_spit", spit) if random.random() < 0.5 else ("tackle", tackle)
        return "lick", lick

class SpikeSlimeMedium(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = RandomUniformRange(28, 32) if game_state.ascention < 7 else RandomUniformRange(29, 34)
        super().__init__("SpikeSlime(M)", hp.get())
        self.attack_damage = 8 if game_state.ascention < 2 else 10
        self.frail_amount = 1

    def get_damaged(self, amount: int) -> int:
        dealt = super().get_damaged(amount)
        if self.health > 0 and self.health <= self.max_health // 2:
            self._next_move_cache = ("split", SplitIntoAction(SpikeSlimeSmall, SpikeSlimeSmall))
        return dealt

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        if self.health <= self.max_health // 2:
            return "split", SplitIntoAction(SpikeSlimeSmall, SpikeSlimeSmall)
        from card import CardGen
        flame_tackle = DealAttackDamage(ConstValue(self.attack_damage)).To(PlayerAgentTarget()).And(
            AddStatusCardAction(CardGen.Slimed, CardPile.DISCARD, 1)
        )
        frail = ApplyStatus(ConstValue(self.frail_amount), StatusEffectRepo.FRAIL).To(PlayerAgentTarget())
        roll = random.randrange(100)
        if roll < 30:
            if self.last_two_moves("flame_tackle"):
                return "frail_lick", frail
            return "flame_tackle", flame_tackle
        if game_state.ascention >= 17:
            if self.last_move("frail_lick"):
                return "flame_tackle", flame_tackle
        elif self.last_two_moves("frail_lick"):
            return "flame_tackle", flame_tackle
        return "frail_lick", frail

class AcidSlimeLarge(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = RandomUniformRange(65, 69) if game_state.ascention < 7 else RandomUniformRange(68, 72)
        super().__init__("AcidSlime(L)", hp.get())
        self.tackle_damage = 16 if game_state.ascention < 2 else 18
        self.spit_damage = 11 if game_state.ascention < 2 else 12
        self.weak_amount = 2

    def get_damaged(self, amount: int) -> int:
        dealt = super().get_damaged(amount)
        if self.health > 0 and self.health <= self.max_health // 2:
            self._next_move_cache = ("split", SplitIntoAction(AcidSlimeMedium, AcidSlimeMedium))
        return dealt

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        if self.health <= self.max_health // 2:
            return "split", SplitIntoAction(AcidSlimeMedium, AcidSlimeMedium)
        roll = random.randrange(100)
        from card import CardGen
        spit = DealAttackDamage(ConstValue(self.spit_damage)).To(PlayerAgentTarget()).And(
            AddStatusCardAction(CardGen.Slimed, CardPile.DISCARD, 2)
        )
        tackle = DealAttackDamage(ConstValue(self.tackle_damage)).To(PlayerAgentTarget())
        lick = ApplyStatus(ConstValue(self.weak_amount), StatusEffectRepo.WEAK).To(PlayerAgentTarget())
        if roll < 40:
            if self.last_two_moves("corrosive_spit"):
                return ("tackle", tackle) if random.random() < 0.5 else ("lick", lick)
            return "corrosive_spit", spit
        if roll < 80:
            if self.last_two_moves("tackle"):
                return ("corrosive_spit", spit) if random.random() < 0.5 else ("lick", lick)
            return "tackle", tackle
        if self.last_move("lick"):
            return ("corrosive_spit", spit) if random.random() < 0.5 else ("tackle", tackle)
        return "lick", lick

class SpikeSlimeLarge(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = RandomUniformRange(64, 70) if game_state.ascention < 7 else RandomUniformRange(67, 73)
        super().__init__("SpikeSlime(L)", hp.get())
        self.attack_damage = 16 if game_state.ascention < 2 else 18
        self.frail_amount = 1

    def get_damaged(self, amount: int) -> int:
        dealt = super().get_damaged(amount)
        if self.health > 0 and self.health <= self.max_health // 2:
            self._next_move_cache = ("split", SplitIntoAction(SpikeSlimeMedium, SpikeSlimeMedium))
        return dealt

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        if self.health <= self.max_health // 2:
            return "split", SplitIntoAction(SpikeSlimeMedium, SpikeSlimeMedium)
        from card import CardGen
        flame_tackle = DealAttackDamage(ConstValue(self.attack_damage)).To(PlayerAgentTarget()).And(
            AddStatusCardAction(CardGen.Slimed, CardPile.DISCARD, 2)
        )
        frail = ApplyStatus(ConstValue(self.frail_amount), StatusEffectRepo.FRAIL).To(PlayerAgentTarget())
        roll = random.randrange(100)
        if roll < 30:
            if self.last_two_moves("flame_tackle"):
                return "frail_lick", frail
            return "flame_tackle", flame_tackle
        if game_state.ascention >= 17:
            if self.last_move("frail_lick"):
                return "flame_tackle", flame_tackle
        elif self.last_two_moves("frail_lick"):
            return "flame_tackle", flame_tackle
        return "frail_lick", frail

class LouseNormal(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = RandomUniformRange(10, 15) if game_state.ascention < 7 else RandomUniformRange(11, 16)
        super().__init__("Louse(N)", hp.get())
        self.attack_damage = RandomUniformRange(5, 7).get() if game_state.ascention < 2 else RandomUniformRange(6, 8).get()
        self.strength_amount = 3 if game_state.ascention < 17 else 4
        if game_state.ascention >= 17:
            self.curl_amount = RandomUniformRange(9, 12).get()
        elif game_state.ascention >= 7:
            self.curl_amount = RandomUniformRange(4, 8).get()
        else:
            self.curl_amount = RandomUniformRange(3, 7).get()
        self.status_effect_state.apply_status(StatusEffectRepo.CURL_UP, self.curl_amount)

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        roll = random.randrange(100)
        bite = DealAttackDamage(ConstValue(self.attack_damage)).To(PlayerAgentTarget())
        strengthen = ApplyStatus(ConstValue(self.strength_amount), StatusEffectRepo.STRENGTH).To(SelfAgentTarget())
        if roll < 25:
            if game_state.ascention >= 17:
                if self.last_move("strengthen"):
                    return "bite", bite
            elif self.last_two_moves("strengthen"):
                return "bite", bite
            return "strengthen", strengthen
        if self.last_two_moves("bite"):
            return "strengthen", strengthen
        return "bite", bite

class LouseDefensive(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = RandomUniformRange(11, 16) if game_state.ascention < 7 else RandomUniformRange(12, 18)
        super().__init__("Louse(D)", hp.get())
        self.attack_damage = RandomUniformRange(5, 7).get() if game_state.ascention < 2 else RandomUniformRange(6, 8).get()
        self.weak_amount = 2
        if game_state.ascention >= 17:
            self.curl_amount = RandomUniformRange(9, 12).get()
        elif game_state.ascention >= 7:
            self.curl_amount = RandomUniformRange(4, 8).get()
        else:
            self.curl_amount = RandomUniformRange(3, 7).get()
        self.status_effect_state.apply_status(StatusEffectRepo.CURL_UP, self.curl_amount)

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        roll = random.randrange(100)
        bite = DealAttackDamage(ConstValue(self.attack_damage)).To(PlayerAgentTarget())
        weaken = ApplyStatus(ConstValue(self.weak_amount), StatusEffectRepo.WEAK).To(PlayerAgentTarget())
        if roll < 25:
            if game_state.ascention >= 17:
                if self.last_move("weaken"):
                    return "bite", bite
            elif self.last_two_moves("weaken"):
                return "bite", bite
            return "weaken", weaken
        if self.last_two_moves("bite"):
            return "weaken", weaken
        return "bite", bite

class FungiBeast(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = RandomUniformRange(22, 28) if game_state.ascention < 7 else RandomUniformRange(24, 28)
        super().__init__("FungiBeast", hp.get())
        self.attack_damage = 6
        self.grow_strength = 3 if game_state.ascention < 2 else 4
        if game_state.ascention >= 17:
            self.grow_strength += 1
        self.vulnerable_on_death = 2

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        roll = random.randrange(100)
        bite = DealAttackDamage(ConstValue(self.attack_damage)).To(PlayerAgentTarget())
        grow = ApplyStatus(ConstValue(self.grow_strength), StatusEffectRepo.STRENGTH).To(SelfAgentTarget())
        if roll < 60:
            if self.last_two_moves("bite"):
                return "grow", grow
            return "bite", bite
        if self.last_move("grow"):
            return "bite", bite
        return "grow", grow

    def on_death(self, game_state: GameState, battle_state: BattleState) -> None:
        if self.death_handled:
            return
        super().on_death(game_state, battle_state)
        battle_state.player.status_effect_state.apply_status(StatusEffectRepo.VULNERABLE, self.vulnerable_on_death)

class SlaverBlue(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = RandomUniformRange(46, 50) if game_state.ascention < 7 else RandomUniformRange(48, 52)
        super().__init__("BlueSlaver", hp.get())
        self.stab_damage = 12 if game_state.ascention < 2 else 13
        self.rake_damage = 7 if game_state.ascention < 2 else 8
        self.weak_amount = 1 if game_state.ascention < 17 else 2

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        roll = random.randrange(100)
        stab = DealAttackDamage(ConstValue(self.stab_damage)).To(PlayerAgentTarget())
        rake = DealAttackDamage(ConstValue(self.rake_damage)).To(PlayerAgentTarget()).And(
            ApplyStatus(ConstValue(self.weak_amount), StatusEffectRepo.WEAK).To(PlayerAgentTarget())
        )
        if roll >= 40 and not self.last_two_moves("stab"):
            return "stab", stab
        if game_state.ascention >= 17:
            if not self.last_move("rake"):
                return "rake", rake
        elif not self.last_two_moves("rake"):
            return "rake", rake
        return "stab", stab

class SlaverRed(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = RandomUniformRange(46, 50) if game_state.ascention < 7 else RandomUniformRange(48, 52)
        super().__init__("RedSlaver", hp.get())
        self.stab_damage = 13 if game_state.ascention < 2 else 14
        self.scrape_damage = 8 if game_state.ascention < 2 else 9
        self.vulnerable_amount = 1 if game_state.ascention < 17 else 2
        self.used_entangle = False

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        if len(self.move_history) == 0:
            return "stab", DealAttackDamage(ConstValue(self.stab_damage)).To(PlayerAgentTarget())
        roll = random.randrange(100)
        if roll >= 75 and not self.used_entangle:
            self.used_entangle = True
            return "entangle", ApplyStatus(ConstValue(1), StatusEffectRepo.ENTANGLED).To(PlayerAgentTarget())
        if roll >= 55 and self.used_entangle and not self.last_two_moves("stab"):
            return "stab", DealAttackDamage(ConstValue(self.stab_damage)).To(PlayerAgentTarget())
        if game_state.ascention >= 17:
            if self.last_move("scrape"):
                return "stab", DealAttackDamage(ConstValue(self.stab_damage)).To(PlayerAgentTarget())
        elif self.last_two_moves("scrape"):
            return "stab", DealAttackDamage(ConstValue(self.stab_damage)).To(PlayerAgentTarget())
        return "scrape", DealAttackDamage(ConstValue(self.scrape_damage)).To(PlayerAgentTarget()).And(
            ApplyStatus(ConstValue(self.vulnerable_amount), StatusEffectRepo.VULNERABLE).To(PlayerAgentTarget())
        )

class Looter(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = RandomUniformRange(44, 48) if game_state.ascention < 7 else RandomUniformRange(46, 50)
        super().__init__("Looter", hp.get())
        self.mug_damage = 10 if game_state.ascention < 2 else 11
        self.lunge_damage = 12 if game_state.ascention < 2 else 14
        self.escape_block = 6
        self.after_second_mug_move: str | None = None

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        turn = len(self.move_history)
        if turn <= 1:
            return "mug", DealAttackDamage(ConstValue(self.mug_damage)).To(PlayerAgentTarget())
        if turn == 2:
            if self.after_second_mug_move is None:
                self.after_second_mug_move = "smoke_bomb" if random.random() < 0.5 else "lunge"
            if self.after_second_mug_move == "smoke_bomb":
                return "smoke_bomb", AddBlock(ConstValue(self.escape_block)).To(SelfAgentTarget())
            return "lunge", DealAttackDamage(ConstValue(self.lunge_damage)).To(PlayerAgentTarget())
        if turn == 3:
            return "escape" if self.after_second_mug_move == "smoke_bomb" else "smoke_bomb", (
                EscapeAction() if self.after_second_mug_move == "smoke_bomb" else AddBlock(ConstValue(self.escape_block)).To(SelfAgentTarget())
            )
        return "escape", EscapeAction()

class GremlinWarrior(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = RandomUniformRange(20, 24) if game_state.ascention < 7 else RandomUniformRange(21, 25)
        super().__init__("MadGremlin", hp.get())
        self.attack_damage = 4 if game_state.ascention < 2 else 5
        self.status_effect_state.apply_status(StatusEffectRepo.ANGRY, 1 if game_state.ascention < 17 else 2)

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        return "scratch", DealAttackDamage(ConstValue(self.attack_damage)).To(PlayerAgentTarget())

class GremlinFat(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = RandomUniformRange(13, 17) if game_state.ascention < 7 else RandomUniformRange(14, 18)
        super().__init__("FatGremlin", hp.get())
        self.attack_damage = 4 if game_state.ascention < 2 else 5
        self.frail_amount = 1 if game_state.ascention >= 17 else 0

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        smash = DealAttackDamage(ConstValue(self.attack_damage)).To(PlayerAgentTarget()).And(
            ApplyStatus(ConstValue(1), StatusEffectRepo.WEAK).To(PlayerAgentTarget())
        )
        if self.frail_amount:
            smash = smash.And(ApplyStatus(ConstValue(self.frail_amount), StatusEffectRepo.FRAIL).To(PlayerAgentTarget()))
        return "smash", smash

class GremlinThief(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = RandomUniformRange(10, 14) if game_state.ascention < 7 else RandomUniformRange(11, 15)
        super().__init__("SneakyGremlin", hp.get())
        self.attack_damage = 9 if game_state.ascention < 2 else 10

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        return "puncture", DealAttackDamage(ConstValue(self.attack_damage)).To(PlayerAgentTarget())

class GremlinTsundere(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = RandomUniformRange(13, 17) if game_state.ascention < 7 else RandomUniformRange(14, 18)
        super().__init__("ShieldGremlin", hp.get())
        self.block_amount = 7 if game_state.ascention < 17 else 11

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        return "protect", ShieldAllEnemiesAction(self.block_amount)

class GremlinWizard(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = RandomUniformRange(23, 25) if game_state.ascention < 7 else RandomUniformRange(24, 26)
        super().__init__("GremlinWizard", hp.get())
        self.attack_damage = 25 if game_state.ascention < 2 else 30
        self.fast_recharge = game_state.ascention >= 17

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        if self.fast_recharge and len(self.move_history) >= 3:
            return "ultimate_blast", DealAttackDamage(ConstValue(self.attack_damage)).To(PlayerAgentTarget())
        if len(self.move_history) % 4 < 3:
            return "charging", NoAction()
        return "ultimate_blast", DealAttackDamage(ConstValue(self.attack_damage)).To(PlayerAgentTarget())

class ShieldAllEnemiesAction(Action):
    def __init__(self, amount: int):
        super().__init__()
        self.amount = amount

    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        for enemy in battle_state.enemies:
            battle_state.gain_block(enemy, self.amount)

    def __repr__(self) -> str:
        return f"Add {self.amount} block to all enemies"

class GremlinNob(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = RandomUniformRange(82, 86) if game_state.ascention < 8 else RandomUniformRange(85, 90)
        super().__init__("GremlinNob", hp.get())
        self.attack_damage = 14 if game_state.ascention < 2 else 16
        self.skull_bash_damage = 6 if game_state.ascention < 2 else 8
        self.enrage_amount = 2 if game_state.ascention < 17 else 3

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        if len(self.move_history) == 0:
            return "bellow", ApplyStatus(ConstValue(self.enrage_amount), StatusEffectRepo.ANGER).To(SelfAgentTarget())
        skull_bash = DealAttackDamage(ConstValue(self.skull_bash_damage)).To(PlayerAgentTarget()).And(
            ApplyStatus(ConstValue(2), StatusEffectRepo.VULNERABLE).To(PlayerAgentTarget())
        )
        rush = DealAttackDamage(ConstValue(self.attack_damage)).To(PlayerAgentTarget())
        if game_state.ascention >= 18:
            if not self.last_move("skull_bash") and not (len(self.move_history) >= 2 and self.move_history[-2] == "skull_bash"):
                return "skull_bash", skull_bash
            if self.last_two_moves("rush"):
                return "skull_bash", skull_bash
            return "rush", rush
        if random.randrange(100) < 33:
            return "skull_bash", skull_bash
        if self.last_two_moves("rush"):
            return "skull_bash", skull_bash
        return "rush", rush

class Lagavulin(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = RandomUniformRange(109, 111) if game_state.ascention < 8 else RandomUniformRange(112, 115)
        super().__init__("Lagavulin", hp.get())
        self.attack_damage = 18 if game_state.ascention < 3 else 20
        self.debuff_amount = 1 if game_state.ascention < 18 else 2
        self.idle_count = 0
        self.debuff_turn_count = 0
        self.is_out = False
        self.is_out_triggered = False
        self.force_attack_once = False
        self.stun_next = False
        self.block = 8
        self.status_effect_state.apply_status(StatusEffectRepo.METALLICIZE, 8)

    def _open(self) -> None:
        self.is_out = True
        self.is_out_triggered = True
        self.status_effect_state.remove_status(StatusEffectRepo.METALLICIZE)

    def get_damaged(self, amount: int) -> int:
        before_health = self.health
        dealt = super().get_damaged(amount)
        if self.health != before_health and not self.is_out_triggered:
            self._open()
            self.stun_next = True
            self._next_move_cache = ("stun", StunAction())
        return dealt

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        if self.stun_next:
            return "stun", StunAction()
        if not self.is_out:
            return "sleep", LagavulinSleepAction()
        if self.force_attack_once:
            return "attack", LagavulinAttackAction(self.attack_damage, forced=True)
        if self.debuff_turn_count >= 2 or self.last_two_moves("attack"):
            return "siphon_soul", LagavulinDebuffAction(self.debuff_amount)
        return "attack", LagavulinAttackAction(self.attack_damage)

class Sentry(ScriptedEnemy):
    def __init__(self, game_state: GameState, starts_with_bolt: bool = False):
        hp = RandomUniformRange(38, 42) if game_state.ascention < 8 else RandomUniformRange(39, 45)
        super().__init__("Sentry", hp.get())
        self.attack_damage = 9 if game_state.ascention < 3 else 10
        self.dazed_count = 2 if game_state.ascention < 18 else 3
        self.starts_with_bolt = starts_with_bolt
        self.status_effect_state.apply_status(StatusEffectRepo.ARTIFACT, 1)

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        if (len(self.move_history) == 0 and self.starts_with_bolt) or self.last_move("beam"):
            from card import CardGen
            return "bolt", AddStatusCardAction(CardGen.Dazed, CardPile.DISCARD, self.dazed_count)
        return "beam", DealAttackDamage(ConstValue(self.attack_damage)).To(PlayerAgentTarget())

class SlimeBoss(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = ConstValue(140 if game_state.ascention < 9 else 150)
        super().__init__("SlimeBoss", hp.get())
        self.slam_damage = 35 if game_state.ascention < 4 else 38
        self.slimed_count = 3 if game_state.ascention < 19 else 5

    def get_damaged(self, amount: int) -> int:
        dealt = super().get_damaged(amount)
        if self.health > 0 and self.health <= self.max_health // 2 and not self.last_move("split"):
            self._next_move_cache = ("split", SplitIntoAction(SpikeSlimeLarge, AcidSlimeLarge))
        return dealt

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        if self.health <= self.max_health // 2:
            return "split", SplitIntoAction(SpikeSlimeLarge, AcidSlimeLarge)
        turn = len(self.move_history) % 3
        if turn == 0:
            from card import CardGen
            return "goop_spray", AddStatusCardAction(CardGen.Slimed, CardPile.DISCARD, self.slimed_count)
        if turn == 1:
            return "preparing", NoAction()
        return "slam", DealAttackDamage(ConstValue(self.slam_damage)).To(PlayerAgentTarget())

class Hexaghost(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = ConstValue(250 if game_state.ascention < 9 else 264)
        super().__init__("Hexaghost", hp.get())
        self.divider_damage = 0
        self.sear_damage = 6
        self.tackle_damage = 5 if game_state.ascention < 4 else 6
        self.inferno_damage = 2 if game_state.ascention < 4 else 3
        self.inflame_strength = 2 if game_state.ascention < 19 else 3
        self.sear_burn_count = 1 if game_state.ascention < 19 else 2

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        from card import CardGen
        if len(self.move_history) == 0:
            return "activate", NoAction()
        if len(self.move_history) == 1:
            self.divider_damage = max(1, battle_state.player.health // 12 + 1)
            return "divider", DealAttackDamage(ConstValue(self.divider_damage), ConstValue(6)).To(PlayerAgentTarget())
        cycle = (len(self.move_history) - 2) % 7
        if cycle in (0, 2, 5):
            return "sear", DealAttackDamage(ConstValue(self.sear_damage)).To(PlayerAgentTarget()).And(
                AddStatusCardAction(CardGen.Burn, CardPile.DISCARD, self.sear_burn_count)
            )
        if cycle in (1, 4):
            return "tackle", DealAttackDamage(ConstValue(self.tackle_damage), ConstValue(2)).To(PlayerAgentTarget())
        if cycle == 3:
            return "inflame", ApplyStatus(ConstValue(self.inflame_strength), StatusEffectRepo.STRENGTH).To(SelfAgentTarget()).And(
                AddBlock(ConstValue(12)).To(SelfAgentTarget())
            )
        return "inferno", DealAttackDamage(ConstValue(self.inferno_damage), ConstValue(6)).To(PlayerAgentTarget()).And(
            AddStatusCardAction(CardGen.Burn, CardPile.DISCARD, 3)
        )

class GuardianDefensiveModeAction(Action):
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        battle_state.gain_block(by, by.defensive_block)
        by.mode_shift_threshold += by.mode_shift_threshold_increase
        by.mode_shift_counter = by.mode_shift_threshold
        by.damage_taken_for_mode = 0
        by.mode_shift_pending = False
        by.close_up_triggered = False
        by.is_open = False
        by.forced_move = "close_up"

    def __repr__(self) -> str:
        return "Enter Defensive Mode"

class GuardianOffensiveModeAction(Action):
    def play(self, by: Agent, game_state: GameState, battle_state: BattleState) -> None:
        by.is_open = True
        by.close_up_triggered = False
        by.damage_taken_for_mode = 0
        by.mode_shift_counter = by.mode_shift_threshold
        by.block = 0
        by.forced_move = "whirlwind"

    def __repr__(self) -> str:
        return "Enter Offensive Mode"

class TheGuardian(ScriptedEnemy):
    def __init__(self, game_state: GameState):
        hp = ConstValue(240 if game_state.ascention < 9 else 250)
        super().__init__("TheGuardian", hp.get())
        if game_state.ascention >= 19:
            self.mode_shift_threshold = 40
        elif game_state.ascention >= 9:
            self.mode_shift_threshold = 35
        else:
            self.mode_shift_threshold = 30
        self.mode_shift_threshold_increase = 10
        self.mode_shift_counter = self.mode_shift_threshold
        self.damage_taken_for_mode = 0
        self.mode_shift_pending = False
        self.close_up_triggered = False
        self.is_open = True
        self.forced_move: str | None = None
        self.whirlwind_damage = 5
        self.whirlwind_count = 4
        self.fierce_bash_damage = 32 if game_state.ascention < 4 else 36
        self.twin_slam_damage = 8 if game_state.ascention < 4 else 10
        self.roll_damage = 9 if game_state.ascention < 4 else 10
        self.block_amount = 9
        self.defensive_block = 20
        self.sharp_hide_amount = 3 if game_state.ascention < 19 else 4
        self.vent_debuff = 2

    def get_damaged(self, amount: int) -> int:
        before_health = self.health
        dealt = super().get_damaged(amount)
        hp_lost = before_health - self.health
        if self.is_open and not self.close_up_triggered and hp_lost > 0 and self.health > 0:
            self.damage_taken_for_mode += hp_lost
            self.mode_shift_counter = max(0, self.mode_shift_counter - hp_lost)
            if self.damage_taken_for_mode >= self.mode_shift_threshold:
                self.damage_taken_for_mode = 0
                self.mode_shift_pending = True
                self.close_up_triggered = True
                self._next_move_cache = ("defensive_mode", GuardianDefensiveModeAction())
        return dealt

    def choose_move(self, game_state: GameState, battle_state: BattleState) -> tuple[str, Action]:
        if self.mode_shift_pending:
            return "defensive_mode", GuardianDefensiveModeAction()
        if self.forced_move is not None:
            move = self.forced_move
            self.forced_move = None
        elif self.is_open:
            move = "charging_up" if len(self.move_history) == 0 else "charging_up"
        else:
            move = "roll_attack"

        if move == "close_up":
            return "close_up", ApplyStatus(ConstValue(self.sharp_hide_amount), StatusEffectRepo.SHARP_HIDE).To(SelfAgentTarget()).And(
                SetForcedMoveAction("roll_attack")
            )
        if move == "roll_attack":
            return "roll_attack", DealAttackDamage(ConstValue(self.roll_damage)).To(PlayerAgentTarget()).And(
                SetForcedMoveAction("twin_slam")
            )
        if move == "twin_slam":
            return "twin_slam", DealAttackDamage(ConstValue(self.twin_slam_damage), ConstValue(2)).To(PlayerAgentTarget()).And(
                RemoveStatusAction(StatusEffectRepo.SHARP_HIDE)
            ).And(GuardianOffensiveModeAction())
        if move == "whirlwind":
            return "whirlwind", DealAttackDamage(ConstValue(self.whirlwind_damage), ConstValue(self.whirlwind_count)).To(PlayerAgentTarget()).And(
                SetForcedMoveAction("charging_up")
            )
        if move == "fierce_bash":
            return "fierce_bash", DealAttackDamage(ConstValue(self.fierce_bash_damage)).To(PlayerAgentTarget()).And(
                SetForcedMoveAction("vent_steam")
            )
        if move == "vent_steam":
            return "vent_steam", ApplyStatus(ConstValue(self.vent_debuff), StatusEffectRepo.WEAK).To(PlayerAgentTarget()).And(
                ApplyStatus(ConstValue(self.vent_debuff), StatusEffectRepo.VULNERABLE).To(PlayerAgentTarget())
            ).And(SetForcedMoveAction("whirlwind"))
        return "charging_up", AddBlock(ConstValue(self.block_amount)).To(SelfAgentTarget()).And(
            SetForcedMoveAction("fierce_bash")
        )

class BigJawWorm(Enemy):
    def __init__(self, game_state: GameState):
        max_health = RandomUniformRange(80, 88) if game_state.ascention < 7 else RandomUniformRange(84, 92)
        chomp: Action = DealAttackDamage(ConstValue(22 if game_state.ascention < 2 else 24)).To(PlayerAgentTarget())
        thrash: Action = DealAttackDamage(ConstValue(14)).To(PlayerAgentTarget()).And(AddBlock(ConstValue(10)).To(SelfAgentTarget()))
        bellow: Action = ApplyStatus(ConstValue(6 if game_state.ascention < 2 else 8 if game_state.ascention < 17 else 10), StatusEffectRepo.STRENGTH).And(AddBlock(ConstValue(10))).To(SelfAgentTarget())
        regular_turn: ItemSet[Action] = RandomizedItemSet((bellow, 0.45), (thrash, 0.30), (chomp, 0.25))
        all_turns: ItemSet[Action] = ItemSequence(chomp, regular_turn)
        action_set = PreventRepeats(all_turns, (bellow, 2), (thrash, 3), (chomp, 2), consecutive=True)
        super().__init__("BigJawWorm", max_health.get(), action_set)
	        
class Goblin(Enemy):
    def __init__(self, game_state: GameState):
        max_health = ConstValue(44)
        slash: Action = DealAttackDamage(ConstValue(11)).To(PlayerAgentTarget())
        stand: Action = DealAttackDamage(ConstValue(7)).To(PlayerAgentTarget()).And(AddBlock(ConstValue(5)).To(SelfAgentTarget()))
        action_set: ItemSet[Action] = RoundRobin(0, slash, stand)
        super().__init__("Goblin", max_health.get(), action_set)

class HobGoblin(Enemy):
    def __init__(self, game_state: GameState):
        max_health = ConstValue(44)
        slash: Action = DealAttackDamage(ConstValue(22)).To(PlayerAgentTarget())
        stand: Action = DealAttackDamage(ConstValue(10)).To(PlayerAgentTarget()).And(AddBlock(ConstValue(10)).To(SelfAgentTarget()))
        action_set: ItemSet[Action] = RoundRobin(0, slash, stand)
        super().__init__("Goblin", max_health.get(), action_set)

class Leech(Enemy):
    def __init__(self, game_state: GameState):
        max_health = ConstValue(70)
        drink: Action = DealAttackDamage(ConstValue(1)).To(PlayerAgentTarget()).And(ApplyStatus(ConstValue(1), StatusEffectRepo.WEAK).To(PlayerAgentTarget()))
        bite: Action = DealAttackDamage(ConstValue(4)).To(PlayerAgentTarget())
        action_set: ItemSet[Action] = RoundRobin(0, drink, bite)
        super().__init__("Leach", max_health.get(), action_set)


# MiniSTS 单怪注册表。RL 配置中的 env.enemy 优先使用这里的 key，也就是上面定义的怪物类名。
# 中文名来自 STS 解包资源 localization/zhs/monsters.json。
MONSTER_NAME_ZH: dict[str, str] = {
    "AcidSlimeSmall": "酸液史莱姆（小）",
    "SpikeSlimeSmall": "尖刺史莱姆（小）",
    "AcidSlimeMedium": "酸液史莱姆（中）",
    "SpikeSlimeMedium": "尖刺史莱姆（中）",
    "AcidSlimeLarge": "酸液史莱姆（大）",
    "SpikeSlimeLarge": "尖刺史莱姆（大）",
    "JawWorm": "大颚虫",
    "Cultist": "邪教徒",
    "LouseNormal": "虱虫",
    "LouseDefensive": "虱虫",
    "FungiBeast": "真菌兽",
    "SlaverBlue": "奴隶贩子",
    "SlaverRed": "奴隶贩子",
    "Looter": "抢劫的",
    "GremlinWarrior": "火大地精",
    "GremlinFat": "胖地精",
    "GremlinThief": "卑鄙地精",
    "GremlinTsundere": "持盾地精",
    "GremlinWizard": "地精法师",
    "GremlinNob": "地精大块头",
    "Lagavulin": "乐加维林",
    "Sentry": "哨卫",
    "SlimeBoss": "史莱姆老大",
    "Hexaghost": "六火亡魂",
    "TheGuardian": "守护者",
    "BigJawWorm": "实验大颚虫",
    "Goblin": "实验哥布林",
    "HobGoblin": "实验大哥布林",
    "Leech": "实验水蛭",
}

MONSTER_FACTORIES = {
    "AcidSlimeSmall": AcidSlimeSmall,
    "SpikeSlimeSmall": SpikeSlimeSmall,
    "AcidSlimeMedium": AcidSlimeMedium,
    "SpikeSlimeMedium": SpikeSlimeMedium,
    "AcidSlimeLarge": AcidSlimeLarge,
    "SpikeSlimeLarge": SpikeSlimeLarge,
    "JawWorm": JawWorm,
    "Cultist": Cultist,
    "LouseNormal": LouseNormal,
    "LouseDefensive": LouseDefensive,
    "FungiBeast": FungiBeast,
    "SlaverBlue": SlaverBlue,
    "SlaverRed": SlaverRed,
    "Looter": Looter,
    "GremlinWarrior": GremlinWarrior,
    "GremlinFat": GremlinFat,
    "GremlinThief": GremlinThief,
    "GremlinTsundere": GremlinTsundere,
    "GremlinWizard": GremlinWizard,
    "GremlinNob": GremlinNob,
    "Lagavulin": Lagavulin,
    "Sentry": Sentry,
    "SlimeBoss": SlimeBoss,
    "Hexaghost": Hexaghost,
    "TheGuardian": TheGuardian,
    "BigJawWorm": BigJawWorm,
    "Goblin": Goblin,
    "HobGoblin": HobGoblin,
    "Leech": Leech,
}


def create_monster_by_name(name: str, game_state: GameState) -> Enemy:
    try:
        return MONSTER_FACTORIES[name](game_state)
    except KeyError as exc:
        supported = ", ".join(sorted(MONSTER_FACTORIES))
        raise ValueError(f"Unsupported monster {name!r}. Supported: {supported}") from exc
