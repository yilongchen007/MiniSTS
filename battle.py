from __future__ import annotations
import copy
import os.path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Enemy, Agent
    from game import GameState
    from card import Card
    from action.action import Action
from config import MAX_MANA, Verbose
from card import CardType
from utility import get_unique_filename, Event
from status_effecs import (
    berserk_start,
    bomb_after,
    brutality_start,
    combust_end,
    curl_up_attacked,
    demon_form_start,
    dark_embrace_exhaust,
    evolve_draw,
    feel_no_pain_exhaust,
    flame_barrier_attacked,
    fire_breathing_draw,
    angry_attacked,
    juggernaut_block,
    lose_strength_end,
    metallicize_end,
    angry_skill_play,
    rage_play,
    ritual_end,
    rupture_hp_loss,
    sharp_hide_attacked,
    StatusEffectRepo,
    tolerance_after,
)

import random

class BattleState:
    side_turn_event: Event[None, tuple[Agent, GameState, BattleState, list[Agent]]] = Event()
    turn_start_event: Event[None, tuple[Agent, GameState, BattleState, list[Agent]]] = Event()
    turn_end_event: Event[None, tuple[Agent, GameState, BattleState, list[Agent]]] = Event()
    card_play_event: Event[None, tuple[Agent, GameState, BattleState, Card]] = Event()
    card_exhaust_event: Event[None, tuple[Agent, GameState, BattleState, Card]] = Event()
    card_draw_event: Event[None, tuple[Agent, GameState, BattleState, Card]] = Event()
    block_gain_event: Event[None, tuple[Agent, GameState, BattleState, int]] = Event()
    attacked_event: Event[None, tuple[Agent, GameState, BattleState, Agent, int]] = Event()
    hp_loss_event: Event[None, tuple[Agent, GameState, BattleState, int, bool]] = Event()
    def __init__(self, game_state: GameState, *enemies: Enemy, verbose: Verbose, log_filename: str|None = None):
        self.player = game_state.player
        self.enemies = [enemy for enemy in enemies]
        self.game_state = game_state
        self.turn = 0
        self.mana = 0
        self.agent_turn_ended = False
        self.player_turn_started = False
        self.turn_phase = 0
        self.draw_pile: list[Card] = []
        self.discard_pile: list[Card] = [copy.deepcopy(card) for card in self.game_state.deck]
        self.hand: list[Card] = []
        self.exhaust_pile: list[Card] = []
        self.player_hp_lost_this_combat = 0
        self.verbose = verbose
        self.log_filename = log_filename

    def copy_undeterministic(self) -> BattleState:
        battle_state_copy = copy.deepcopy(self)
        random.shuffle(battle_state_copy.draw_pile)
        return battle_state_copy
    
    def get_undeterministic_repr_hash(self) -> str:
        import hashlib
        combined_hash = hashlib.sha256()
        for agent in [self.player] + self.enemies:
            combined_hash.update(hashlib.sha256(agent.__repr__().encode()).digest())
        combined_hash.update(hashlib.sha256(self.turn.__repr__().encode()).digest())
        combined_hash.update(hashlib.sha256(self.mana.__repr__().encode()).digest())
        combined_hash.update(hashlib.sha256(self.agent_turn_ended.__repr__().encode()).digest())
        combined_hash.update(hashlib.sha256(self.turn_phase.__repr__().encode()).digest())
        sorted_draw_pile = sorted(self.draw_pile, key=lambda card: repr(card))
        sorted_discard_pile = sorted(self.discard_pile, key=lambda card: repr(card))
        sorted_hand = sorted(self.hand, key=lambda card: repr(card))
        sorted_exhaust_pile = sorted(self.exhaust_pile, key=lambda card: repr(card))
        for card in sorted_draw_pile:
            combined_hash.update(hashlib.sha256(card.__repr__().encode()).digest())
        combined_hash.update(hashlib.sha256('-'.encode()).digest())
        for card in sorted_discard_pile:
            combined_hash.update(hashlib.sha256(card.__repr__().encode()).digest())
        combined_hash.update(hashlib.sha256('-'.encode()).digest())
        for card in sorted_hand:
            combined_hash.update(hashlib.sha256(card.__repr__().encode()).digest())
        combined_hash.update(hashlib.sha256('-'.encode()).digest())
        for card in sorted_exhaust_pile:
            combined_hash.update(hashlib.sha256(card.__repr__().encode()).digest())
        return combined_hash.hexdigest()

    def discard_hand(self):
        for card in list(self.hand):
            if card.name == "Burn":
                self.lose_hp(self.player, 2, from_card=False)
                self.discard(card)
            elif card.ethereal:
                self.exhaust(card)
            else:
                self.discard(card)

    def reshuffle(self):
        self.draw_pile, self.discard_pile = self.draw_pile + self.discard_pile, []
        random.shuffle(self.draw_pile)

    def draw_one(self):
        if len(self.draw_pile) == 0:
            self.reshuffle()
        if len(self.draw_pile) > 0:
            card = self.draw_pile.pop()
            self.hand.append(card)
            BattleState.card_draw_event.broadcast_after((self.player, self.game_state, self, card))
        else:
            #discard+draw+hand is empty
            pass

    def draw(self, count: int):
        if self.player.status_effect_state.has(StatusEffectRepo.NO_DRAW):
            return
        for _ in range(count):
            self.draw_one()

    def draw_hand(self):
        draw_count = self.game_state.draw_count
        self.draw(draw_count)
    
    def get_hand(self):
        return self.hand
    
    def play_card(self, card_index: int):
        assert card_index < len(self.hand) and card_index >= 0, "Card index {} out of range for hand {}".format(card_index, self.hand)
        card = self.hand.pop(card_index)
        card.play(self.game_state, self)
        BattleState.card_play_event.broadcast_after((self.player, self.game_state, self, card))
        double_tap = self.player.status_effect_state.get(StatusEffectRepo.DOUBLE_TAP)
        if card.card_type == CardType.ATTACK and double_tap > 0 and not self.ended():
            self.player.status_effect_state.apply_status(StatusEffectRepo.DOUBLE_TAP, -1)
            card.play_actions(self.game_state, self)
        if not self.is_present(card) and not card.card_type == CardType.POWER:
            if card.card_type == CardType.SKILL and self.player.status_effect_state.has(StatusEffectRepo.CORRUPTION):
                self.exhaust(card)
            elif card.exhaust_on_play:
                self.exhaust(card)
            else:
                card.cost_override = None
                self.discard_pile.append(card)

    def is_present(self, card: Card):
        if card in self.hand:
            return True
        if card in self.draw_pile:
            return True
        if card in self.discard_pile:
            return True
        if card in self.exhaust_pile:
            return True
        return False
    
    def remove_card(self, card: Card):
        if card in self.hand:
            self.hand.remove(card)
        if card in self.draw_pile:
            self.draw_pile.remove(card)
        if card in self.discard_pile:
            self.discard_pile.remove(card)
        if card in self.exhaust_pile:
            self.exhaust_pile.remove(card)
    
    def exhaust(self, card: Card):
        self.remove_card(card)
        self.exhaust_pile.append(card)
        if card.name == "Sentinel":
            self.add_to_mana(3 if card.upgrade_count > 0 else 2)
        BattleState.card_exhaust_event.broadcast_after((self.player, self.game_state, self, card))

    def gain_block(self, agent: Agent, amount: int):
        amount += agent.status_effect_state.get(StatusEffectRepo.DEXTERITY)
        if agent.status_effect_state.has(StatusEffectRepo.FRAIL):
            amount = int(amount * 0.75)
        amount = max(0, amount)
        before = agent.block
        agent.gain_block(amount)
        gained = agent.block - before
        if gained > 0:
            BattleState.block_gain_event.broadcast_after((agent, self.game_state, self, gained))

    def lose_hp(self, target: Agent, amount: int, from_card: bool = False):
        lost = target.lose_health(amount)
        if target is self.player:
            self.player_hp_lost_this_combat += lost
        if lost > 0:
            BattleState.hp_loss_event.broadcast_after((target, self.game_state, self, lost, from_card))
        return lost

    def deal_attack_damage(self, attacker: Agent, target: Agent, amount: int):
        dealt = target.get_damaged(amount)
        if target is self.player:
            self.player_hp_lost_this_combat += dealt
        if amount > 0:
            BattleState.attacked_event.broadcast_after((target, self.game_state, self, attacker, amount))
        if target.is_dead() and hasattr(target, "on_death"):
            target.on_death(self.game_state, self)
            self.enemies = [enemy for enemy in self.enemies if not enemy.is_dead()]
        return dealt

    def discard(self, card: Card):
        if card in self.discard_pile:
            return
        self.remove_card(card)
        card.cost_override = None
        self.discard_pile.append(card)

    def add_card_to_pile(self, card: Card, card_pile):
        from target.card_target import CardPile
        if card_pile == CardPile.HAND:
            self.hand.append(card)
        elif card_pile == CardPile.DISCARD:
            self.discard_pile.append(card)
        elif card_pile == CardPile.DRAW:
            self.draw_pile.append(card)
        elif card_pile == CardPile.EXHAUST:
            self.exhaust_pile.append(card)
        else:
            raise Exception(f"Unrecognized CardPile {card_pile}")

    def get_player_card_target(self, name: str, card_list: list[Card]) -> Card:
        card = self.player.bot.choose_card_target(self, name, card_list)
        self.log(f"Card choice {repr(card)}\n")
        return card
    
    def get_player_agent_target(self, name: str, agent_list: list[Agent]) -> Agent:
        agent = self.player.bot.choose_agent_target(self, name, agent_list)
        self.log(f"Agent choice {repr(agent)}\n")
        return agent

    def log(self, log: str):
        if self.verbose == Verbose.NO_LOG:
            return
        if self.log_filename is None:
            print(log)
        else:
            with open(self.log_filename, 'a') as f:
                f.write(log)
    
    def get_visualization(self):
        log = ''
        log += "*Turn {} - {}*\n".format(self.turn, "Player" if self.turn_phase == 0 else "Enemy {}".format(self.turn_phase-1))
        log += "mana: {}/{}\n".format(self.mana, self.game_state.max_mana)
        log += str(self.player) + '\n'
        log += '\n'.join('{}:{}---{}'.format(i, enemy, enemy.get_intention(self.game_state, self)) for i, enemy in enumerate(self.enemies)) + '\n'
        log += "exhaust pile: "
        log += ' '.join(['{}:{}'.format(i, card.get_name()) for i, card in enumerate(self.exhaust_pile)]) + '\n'
        log += "discard pile: "
        log += ' '.join(['{}:{}'.format(i, card.get_name()) for i, card in enumerate(self.discard_pile)]) + '\n'
        log += "draw pile: "
        sorted_draw: list[Card] = sorted(self.draw_pile, key=lambda card: repr(card))
        log += ' '.join('{}:{}'.format(i, card.get_name()) for i, card in enumerate(sorted_draw)) + '\n'
        log += "hand: "
        log += ' '.join(['{}:{}'.format(i, card.get_name()) for i, card in enumerate(self.hand)]) + '\n'
        return log
    
    def visualize(self):
        if self.verbose == Verbose.NO_LOG:
            return
        self.log(self.get_visualization())

    def end_agent_turn(self):
        self.agent_turn_ended = True
    
    def add_to_mana(self, amount: int):
        self.mana += amount
        if self.mana > MAX_MANA:
            self.mana = MAX_MANA
        assert self.mana >= 0, "Mana value cannot be negative"

    def _step_agent(self, agent: Agent) -> bool:
        if agent.is_dead() or self.ended() or self.agent_turn_ended:
            return False
        self.visualize()
        agent.play(self.game_state, self)
        assert agent.prev_action is not None, "Action taken is not recorded for agent {}".format(agent.name)
        self.log(str(agent.prev_action) + '\n')
        return True

    def _take_agent_turn(self, agent: Agent):
        self.agent_turn_ended = False
        while self._step_agent(agent):
            self.enemies: list[Enemy] = [enemy for enemy in self.enemies if not enemy.is_dead()]
        self.turn_phase += 1

    def _start_agent_turn(self, agent: Agent, other_side: list[Agent]):
        BattleState.side_turn_event.broadcast_before((agent, self.game_state, self, other_side))
        BattleState.turn_start_event.broadcast_after((agent, self.game_state, self, other_side))

    def _end_agent_turn(self, agent: Agent, other_side: list[Agent]):
        BattleState.side_turn_event.broadcast_after((agent, self.game_state, self, other_side))
        BattleState.turn_end_event.broadcast_after((agent, self.game_state, self, other_side))
    
    def _play_side(self, side: list[Agent], other_side: list[Agent]):
        for agent in side:
            self._start_agent_turn(agent, other_side)
        for agent in side:
            self._take_agent_turn(agent)
        for agent in side:
            self._end_agent_turn(agent, other_side)
        for agent in side:
            agent.status_effect_state.end_turn()
        for agent in other_side:
            agent.clear_block()

    def take_turn(self):
        self.mana = self.game_state.max_mana
        self.turn += 1
        self.turn_phase = 0
        self.draw_hand()
        self._play_side([self.player], [enemy for enemy in self.enemies])
        self._play_side([enemy for enemy in self.enemies], [self.player])
        self.discard_hand()

    def tick_player(self, action: Action) -> bool:
        if self.ended():
            return False
        other_side: list[Agent] = [enemy for enemy in self.enemies]
        if not self.player_turn_started:
            self._start_agent_turn(self.player, other_side)
            self.player_turn_started = True
        action.play(self.player, self.game_state, self)
        self.enemies: list[Enemy] = [enemy for enemy in self.enemies if not enemy.is_dead()]
        if not self.agent_turn_ended:
            return True
        self.turn_phase += 1
        other_side: list[Agent] = [enemy for enemy in self.enemies]
        self._end_agent_turn(self.player, other_side)
        self.player_turn_started = False
        self.player.status_effect_state.end_turn()
        for enemy in self.enemies:
            enemy.clear_block()
        self._play_side([enemy for enemy in self.enemies], [self.player])
        self.discard_hand()
        self.mana = self.game_state.max_mana
        self.turn += 1
        self.turn_phase = 0
        self.draw_hand()
        self.agent_turn_ended = False
        self.player_turn_started = False
        return True
        
    def ended(self):
        return self.get_end_result() != 0
    
    def get_end_result(self):
        if self.player.is_dead():
            return -1
        for enemy in self.enemies:
            if not enemy.is_dead():
                return 0
        return 1
    
    def initiate_log(self):
        if self.verbose == Verbose.LOG and self.log_filename is not None:
            self.log_filename = get_unique_filename(self.log_filename, 'log')
        self.log(f'version {1.0}\n')
        for card in self.game_state.deck:
            self.log(f'{card}\n')
    
    def run(self):
        self.initiate_log()
        while not self.ended():
            self.take_turn()
        self.player.clean_up()
        self.visualize()
        self.log("WIN\n" if self.get_end_result() == 1 else "LOSE\n")

BattleState.side_turn_event.subscribe_after(tolerance_after)
BattleState.side_turn_event.subscribe_after(bomb_after)
BattleState.turn_end_event.subscribe_after(metallicize_end)
BattleState.turn_end_event.subscribe_after(lose_strength_end)
BattleState.turn_end_event.subscribe_after(combust_end)
BattleState.turn_end_event.subscribe_after(ritual_end)
BattleState.turn_start_event.subscribe_after(berserk_start)
BattleState.turn_start_event.subscribe_after(demon_form_start)
BattleState.turn_start_event.subscribe_after(brutality_start)
BattleState.hp_loss_event.subscribe_after(rupture_hp_loss)
BattleState.card_play_event.subscribe_after(rage_play)
BattleState.card_play_event.subscribe_after(angry_skill_play)
BattleState.card_exhaust_event.subscribe_after(feel_no_pain_exhaust)
BattleState.card_exhaust_event.subscribe_after(dark_embrace_exhaust)
BattleState.card_draw_event.subscribe_after(evolve_draw)
BattleState.card_draw_event.subscribe_after(fire_breathing_draw)
BattleState.block_gain_event.subscribe_after(juggernaut_block)
BattleState.attacked_event.subscribe_after(flame_barrier_attacked)
BattleState.attacked_event.subscribe_after(curl_up_attacked)
BattleState.attacked_event.subscribe_after(angry_attacked)
BattleState.attacked_event.subscribe_after(sharp_hide_attacked)
