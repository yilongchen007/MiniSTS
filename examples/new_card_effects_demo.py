from __future__ import annotations

import random
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agent import BigJawWorm
from battle import BattleState
from card import Card, CardGen
from config import Character, Verbose
from game import GameState
from ggpa.ggpa import GGPA
from status_effecs import StatusEffectRepo


class FirstChoiceBot(GGPA):
    def __init__(self):
        super().__init__("FirstChoiceBot")

    def choose_card(self, game_state, battle_state):
        return self.get_choose_card_options(game_state, battle_state)[0]

    def choose_agent_target(self, battle_state, list_name, agent_list):
        return agent_list[0]

    def choose_card_target(self, battle_state, list_name, card_list):
        return card_list[0]


def names(cards: list[Card]) -> str:
    return "[" + ", ".join(card.get_name() for card in cards) + "]"


def snapshot(battle: BattleState) -> str:
    player = battle.player
    enemy = battle.enemies[0]
    return (
        f"mana={battle.mana} | "
        f"P hp={player.health} block={player.block} status={player.status_effect_state} | "
        f"E hp={enemy.health} block={enemy.block} status={enemy.status_effect_state} | "
        f"hand={names(battle.hand)} draw={names(battle.draw_pile)} "
        f"discard={names(battle.discard_pile)} exhaust={names(battle.exhaust_pile)}"
    )


def make_battle(card: Card, *, hand_extra: list[Card] | None = None, draw: list[Card] | None = None, player_block: int = 0) -> tuple[GameState, BattleState]:
    game = GameState(Character.IRON_CLAD, FirstChoiceBot(), 0)
    game.set_deck(card)
    battle = BattleState(game, BigJawWorm(game), verbose=Verbose.NO_LOG)
    battle.turn = 1
    battle.mana = game.max_mana
    battle.hand = [card] + (hand_extra or [])
    battle.draw_pile = draw or []
    battle.discard_pile = []
    battle.exhaust_pile = []
    battle.player.block = player_block
    return game, battle


def show(
    card_factory: Callable[[], Card],
    note: str,
    *,
    hand_extra: list[Card] | None = None,
    draw: list[Card] | None = None,
    player_block: int = 0,
    setup: Callable[[GameState, BattleState], None] | None = None,
    trigger: Callable[[GameState, BattleState], str] | None = None,
) -> None:
    random.seed(0)
    card = card_factory()
    game, battle = make_battle(card, hand_extra=hand_extra, draw=draw, player_block=player_block)
    if setup is not None:
        setup(game, battle)

    print(f"\n{card.get_name()} - {note}")
    print(f"  before: {snapshot(battle)}")
    if not card.is_playable(game, battle):
        print("  after : unplayable, cannot be played from hand")
        return

    battle.play_card(0)
    print(f"  after : {snapshot(battle)}")
    if trigger is not None:
        label = trigger(game, battle)
        print(f"  {label}: {snapshot(battle)}")


def end_turn_trigger(game: GameState, battle: BattleState) -> str:
    BattleState.turn_end_event.broadcast_after((battle.player, game, battle, battle.enemies))
    return "turn end"


def start_turn_trigger(game: GameState, battle: BattleState) -> str:
    BattleState.turn_start_event.broadcast_after((battle.player, game, battle, battle.enemies))
    return "turn start"


def clear_block_trigger(game: GameState, battle: BattleState) -> str:
    battle.player.clear_block()
    return "clear block"


def play_next_card_trigger(game: GameState, battle: BattleState) -> str:
    battle.play_card(0)
    return "play next"


def exhaust_next_card_trigger(game: GameState, battle: BattleState) -> str:
    battle.exhaust(battle.hand[0])
    return "exhaust next"


def exhaust_discard_trigger(game: GameState, battle: BattleState) -> str:
    battle.exhaust(battle.discard_pile[0])
    return "exhaust discard"


def draw_one_trigger(game: GameState, battle: BattleState) -> str:
    battle.draw(1)
    return "draw one"


def enemy_attack_trigger(game: GameState, battle: BattleState) -> str:
    battle.deal_attack_damage(battle.enemies[0], battle.player, 1)
    return "enemy attack"


def vulnerable_enemy_setup(game: GameState, battle: BattleState) -> None:
    battle.enemies[0].status_effect_state.apply_status(StatusEffectRepo.VULNERABLE, 1)


def strength_setup(game: GameState, battle: BattleState) -> None:
    battle.player.status_effect_state.apply_status(StatusEffectRepo.STRENGTH, 2)


def lost_hp_setup(game: GameState, battle: BattleState) -> None:
    battle.lose_hp(battle.player, 2, from_card=False)


def near_dead_enemy_setup(game: GameState, battle: BattleState) -> None:
    battle.enemies[0].health = 8


def true_grit_plus() -> Card:
    card = CardGen.True_Grit()
    card.upgrade()
    return card


def combust_turn_end_trigger(game: GameState, battle: BattleState) -> str:
    BattleState.turn_end_event.broadcast_after((battle.player, game, battle, battle.enemies))
    return "turn end"


def main() -> None:
    print("MiniSTS new card effect demo")

    demos: list[tuple[Callable[[], Card], str, dict]] = [
        (CardGen.Stimulate, "Gain 4 Vigor.", {}),
        (CardGen.Batter, "Deal 0 damage 10 times.", {}),
        (CardGen.Tolerate, "Gain Tolerance.", {}),
        (CardGen.Bomb, "Gain Bomb countdown.", {}),
        (CardGen.Suffer, "Deal 15 damage.", {}),
        (CardGen.Bloodletting, "Lose 3 HP and gain energy.", {}),
        (CardGen.Bludgeon, "Deal heavy damage.", {}),
        (CardGen.Clothesline, "Deal damage and apply Weak.", {}),
        (CardGen.Thunderclap, "Deal damage to all enemies and apply Vulnerable.", {}),
        (CardGen.Twin_Strike, "Deal damage twice.", {}),
        (CardGen.Uppercut, "Deal damage and apply Weak/Vulnerable.", {}),
        (CardGen.Iron_Wave, "Gain Block and deal damage.", {}),
        (CardGen.Pummel, "Deal damage multiple times and exhaust.", {}),
        (CardGen.Hemokinesis, "Lose HP and deal damage.", {}),
        (CardGen.Offering, "Lose HP, gain energy, draw, and exhaust.", {"draw": [CardGen.Strike(), CardGen.Defend(), CardGen.Bash()]}),
        (CardGen.Disarm, "Reduce enemy Strength and exhaust.", {}),
        (CardGen.Shockwave, "Apply Weak and Vulnerable to all enemies, then exhaust.", {}),
        (CardGen.Berserk, "Gain Vulnerable and start-turn energy.", {"trigger": start_turn_trigger}),
        (CardGen.Demon_Form, "Gain Strength at start of turn.", {"trigger": start_turn_trigger}),
        (CardGen.Brutality, "At start of turn, lose HP and draw.", {"draw": [CardGen.Strike()], "trigger": start_turn_trigger}),
        (CardGen.Armaments, "Gain Block and upgrade a card in hand.", {"hand_extra": [CardGen.Strike()]}),
        (CardGen.Blood_for_Blood, "Cost is reduced by HP lost this combat.", {"setup": lost_hp_setup}),
        (CardGen.Body_Slam, "Deal damage equal to current Block.", {"player_block": 12}),
        (CardGen.Burning_Pact, "Exhaust a hand card and draw.", {"hand_extra": [CardGen.Strike()], "draw": [CardGen.Defend(), CardGen.Bash()]}),
        (CardGen.Carnage, "Deal heavy damage; Ethereal if left in hand.", {}),
        (CardGen.Clash, "Playable only when hand has only Attack cards.", {"hand_extra": [CardGen.Strike()]}),
        (CardGen.Combust, "At turn end, lose HP and damage enemies.", {"trigger": combust_turn_end_trigger}),
        (CardGen.Corruption, "Skills become free and exhaust when played.", {"hand_extra": [CardGen.Defend()], "trigger": play_next_card_trigger}),
        (CardGen.Double_Tap, "Next Attack is played twice.", {"hand_extra": [CardGen.Strike()], "trigger": play_next_card_trigger}),
        (CardGen.Dropkick, "Vulnerable target grants energy and draw.", {"setup": vulnerable_enemy_setup, "draw": [CardGen.Strike()]}),
        (CardGen.Dual_Wield, "Duplicate an Attack or Power in hand.", {"hand_extra": [CardGen.Strike()]}),
        (CardGen.Entrench, "Double current Block.", {"player_block": 9}),
        (CardGen.Exhume, "Move a card from exhaust to hand.", {"setup": lambda game, battle: battle.exhaust_pile.append(CardGen.Strike())}),
        (CardGen.Feed, "Fatal damage increases max HP.", {"setup": near_dead_enemy_setup}),
        (CardGen.Fiend_Fire, "Exhaust hand and deal damage for each exhausted card.", {"hand_extra": [CardGen.Strike(), CardGen.Defend()]}),
        (CardGen.Ghostly_Armor, "Gain Block; Ethereal if left in hand.", {}),
        (CardGen.Havoc, "Play top draw-pile card and exhaust it.", {"draw": [CardGen.Strike()]}),
        (CardGen.Headbutt, "Deal damage and put discard card on top of draw pile.", {"setup": lambda game, battle: battle.discard_pile.append(CardGen.Bash())}),
        (CardGen.Heavy_Blade, "Strength affects damage multiple times.", {"setup": strength_setup}),
        (CardGen.Immolate, "Damage all enemies and add Burn to discard.", {}),
        (CardGen.Infernal_Blade, "Add a random Attack that costs 0 this turn.", {}),
        (CardGen.Intimidate, "Apply Weak to all enemies and exhaust.", {}),
        (CardGen.Limit_Break, "Double current Strength.", {"setup": strength_setup}),
        (CardGen.Perfected_Strike, "Damage scales with Strike cards in deck/piles.", {"hand_extra": [CardGen.Strike()]}),
        (CardGen.Rampage, "Damage increases after use this combat.", {}),
        (CardGen.Reaper, "Damage enemies and heal unblocked damage.", {"setup": lost_hp_setup}),
        (CardGen.Reckless_Charge, "Deal damage and add Dazed to draw pile.", {}),
        (CardGen.Rupture, "HP loss from cards grants Strength.", {"hand_extra": [CardGen.Bloodletting()], "trigger": play_next_card_trigger}),
        (CardGen.Second_Wind, "Exhaust non-Attacks and gain Block per card.", {"hand_extra": [CardGen.Defend(), CardGen.Strike()]}),
        (CardGen.Sentinel, "Gain Block; when exhausted, gain energy.", {"trigger": exhaust_discard_trigger}),
        (CardGen.Sever_Soul, "Exhaust non-Attack non-Power cards and deal damage.", {"hand_extra": [CardGen.Defend(), CardGen.Strike()]}),
        (CardGen.Spot_Weakness, "Gain Strength if target intends to attack.", {}),
        (CardGen.Sword_Boomerang, "Hit random enemies multiple times.", {}),
        (CardGen.Warcry, "Draw then put a hand card on top of draw pile.", {"draw": [CardGen.Strike()]}),
        (CardGen.Whirlwind, "Spend all energy to hit all enemies repeatedly.", {}),
        (CardGen.Battle_Trance, "Draw 3 cards and apply No Draw.", {"draw": [CardGen.Strike(), CardGen.Defend(), CardGen.Strike()]}),
        (CardGen.Flex, "Gain temporary Strength.", {"trigger": end_turn_trigger}),
        (CardGen.Inflame, "Gain Strength.", {}),
        (CardGen.Metallicize, "Gain block at turn end.", {"trigger": end_turn_trigger}),
        (CardGen.Barricade, "Keep Block when it would be cleared.", {"player_block": 8, "trigger": clear_block_trigger}),
        (CardGen.Seeing_Red, "Gain 2 energy and exhaust.", {}),
        (CardGen.True_Grit, "Gain block and exhaust a random card in hand.", {"hand_extra": [CardGen.Strike(), CardGen.Defend()]}),
        (true_grit_plus, "Upgraded: gain more block and choose the exhausted card.", {"hand_extra": [CardGen.Strike(), CardGen.Defend()]}),
        (CardGen.Wild_Strike, "Deal damage and add Wound to draw pile.", {}),
        (CardGen.Power_Through, "Gain block and add 2 Wounds to hand.", {}),
        (CardGen.Rage, "This turn, Attack cards gain Block.", {"hand_extra": [CardGen.Strike()], "trigger": play_next_card_trigger}),
        (CardGen.Feel_No_Pain, "When a card is exhausted, gain Block.", {"hand_extra": [CardGen.Strike()], "trigger": exhaust_next_card_trigger}),
        (CardGen.Dark_Embrace, "When a card is exhausted, draw.", {"hand_extra": [CardGen.Strike()], "draw": [CardGen.Defend()], "trigger": exhaust_next_card_trigger}),
        (CardGen.Evolve, "When a Status is drawn, draw extra.", {"draw": [CardGen.Strike(), CardGen.Wound()], "trigger": draw_one_trigger}),
        (CardGen.Fire_Breathing, "When a Status/Curse is drawn, damage all enemies.", {"draw": [CardGen.Wound()], "trigger": draw_one_trigger}),
        (CardGen.Juggernaut, "When Block is gained, damage a random enemy.", {"hand_extra": [CardGen.Defend()], "trigger": play_next_card_trigger}),
        (CardGen.Flame_Barrier, "Gain Block; attackers take return damage this turn.", {"trigger": enemy_attack_trigger}),
        (CardGen.Wound, "Status card.", {}),
        (CardGen.Dazed, "Status card.", {}),
        (CardGen.Burn, "Status card.", {}),
    ]

    for factory, note, kwargs in demos:
        show(factory, note, **kwargs)


if __name__ == "__main__":
    main()
