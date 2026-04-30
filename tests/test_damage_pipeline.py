import unittest

from agent import Cultist, TheGuardian
from battle import BattleState
from card import CardGen
from config import Character, Verbose
from game import GameState
from ggpa.random_bot import RandomBot
from status_effecs import StatusEffectRepo


class DamagePipelineTests(unittest.TestCase):
    def test_reaper_uses_attack_damage_modifiers(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        enemy = Cultist(game_state)
        battle_state = BattleState(game_state, enemy, verbose=Verbose.NO_LOG)
        game_state.player.status_effect_state.apply_status(StatusEffectRepo.STRENGTH, 1)

        CardGen.Reaper().actions[0].targeted.play_many(
            game_state.player,
            game_state,
            battle_state,
            [enemy],
        )

        self.assertEqual(enemy.max_health - 5, enemy.health)

    def test_paper_phrog_increases_vulnerable_attack_bonus(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        game_state.set_relics("Paper Phrog")
        enemy = Cultist(game_state)
        battle_state = BattleState(game_state, enemy, verbose=Verbose.NO_LOG)
        enemy.status_effect_state.apply_status(StatusEffectRepo.VULNERABLE, 1)

        CardGen.Strike().actions[0].targeted.play(
            game_state.player,
            game_state,
            battle_state,
            enemy,
        )

        self.assertEqual(enemy.max_health - 10, enemy.health)

    def test_pen_nib_doubles_every_tenth_attack_card(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        game_state.set_relics("Pen Nib")
        enemy = TheGuardian(game_state)
        battle_state = BattleState(game_state, enemy, verbose=Verbose.NO_LOG)
        battle_state.hand = [CardGen.Strike() for _ in range(10)]

        for _ in range(10):
            battle_state.mana = 1
            battle_state.play_card(0)

        self.assertEqual(enemy.max_health - 66, enemy.health)


if __name__ == "__main__":
    unittest.main()
