import unittest

from battle import BattleState
from card import CardGen
from config import Character, Verbose
from game import GameState
from agent import Cultist, GremlinNob, JawWorm, TheGuardian
from action.action import EndAgentTurn
from ggpa.random_bot import RandomBot
from rl.env import MiniSTSEnv
from status_effecs import StatusEffectRepo


class RelicTests(unittest.TestCase):
    def test_anchor_grants_block_at_combat_start(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        game_state.set_relics("Anchor")

        battle_state = BattleState(game_state, TheGuardian(game_state), verbose=Verbose.NO_LOG)

        self.assertEqual(10, battle_state.player.block)

    def test_env_reset_applies_configured_relics(self):
        env = MiniSTSEnv(enemy_name="Cultist", relics=["Anchor"])

        env.reset()

        self.assertEqual(10, env.battle_state.player.block)

    def test_vajra_grants_strength_at_combat_start(self):
        env = MiniSTSEnv(enemy_name="Cultist", relics=["Vajra"])

        env.reset()

        strength = env.battle_state.player.status_effect_state.get(StatusEffectRepo.STRENGTH)
        self.assertEqual(1, strength)

    def test_oddly_smooth_stone_grants_dexterity_at_combat_start(self):
        env = MiniSTSEnv(enemy_name="Cultist", relics=["Oddly Smooth Stone"])

        env.reset()

        dexterity = env.battle_state.player.status_effect_state.get(StatusEffectRepo.DEXTERITY)
        self.assertEqual(1, dexterity)

    def test_nunchaku_grants_energy_on_every_tenth_attack_card(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        game_state.set_relics("Nunchaku")
        battle_state = BattleState(game_state, TheGuardian(game_state), verbose=Verbose.NO_LOG)
        battle_state.hand = [CardGen.Strike() for _ in range(10)]

        for _ in range(9):
            battle_state.mana = 1
            battle_state.play_card(0)
        battle_state.mana = 1
        battle_state.play_card(0)

        self.assertEqual(1, battle_state.mana)

    def test_orichalcum_grants_block_before_enemy_turn_when_player_has_none(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        game_state.set_relics("Orichalcum")
        battle_state = BattleState(game_state, JawWorm(game_state), verbose=Verbose.NO_LOG)

        battle_state.tick_player(EndAgentTurn())

        self.assertEqual(game_state.player.max_health - 6, game_state.player.health)

    def test_bronze_scales_damages_attackers(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        game_state.set_relics("Bronze Scales")
        enemy = JawWorm(game_state)
        battle_state = BattleState(game_state, enemy, verbose=Verbose.NO_LOG)

        battle_state.tick_player(EndAgentTurn())

        self.assertEqual(enemy.max_health - 3, enemy.health)

    def test_red_skull_grants_strength_at_half_health(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        game_state.set_relics("Red Skull")
        battle_state = BattleState(game_state, Cultist(game_state), verbose=Verbose.NO_LOG)

        battle_state.lose_hp(game_state.player, game_state.player.max_health // 2)

        strength = game_state.player.status_effect_state.get(StatusEffectRepo.STRENGTH)
        self.assertEqual(3, strength)

    def test_red_skull_removes_strength_when_healed_above_half_health(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        game_state.set_relics("Red Skull")
        battle_state = BattleState(game_state, Cultist(game_state), verbose=Verbose.NO_LOG)

        battle_state.lose_hp(game_state.player, game_state.player.max_health // 2)
        battle_state.heal(game_state.player, 1)

        strength = game_state.player.status_effect_state.get(StatusEffectRepo.STRENGTH)
        self.assertEqual(0, strength)

    def test_preserved_insect_reduces_elite_health(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        game_state.set_relics("Preserved Insect")
        enemy = GremlinNob(game_state)
        original_max_health = enemy.max_health

        BattleState(game_state, enemy, verbose=Verbose.NO_LOG)

        self.assertEqual(int(original_max_health * 0.75), enemy.max_health)
        self.assertEqual(enemy.max_health, enemy.health)

    def test_lantern_grants_extra_energy_on_first_turn(self):
        env = MiniSTSEnv(enemy_name="Cultist", relics=["Lantern"])

        env.reset()

        self.assertEqual(4, env.battle_state.mana)

    def test_bag_of_preparation_draws_two_extra_cards_on_first_turn(self):
        deck = [CardGen.Strike() for _ in range(10)]
        env = MiniSTSEnv(enemy_name="Cultist", deck=deck, relics=["Bag of Preparation"])

        env.reset()

        self.assertEqual(7, len(env.battle_state.hand))

    def test_happy_flower_grants_energy_every_three_turns(self):
        env = MiniSTSEnv(enemy_name="Cultist", relics=["Happy Flower"])

        env.reset()
        env.step_index(0)
        env.step_index(0)

        self.assertEqual(3, env.battle_state.turn)
        self.assertEqual(4, env.battle_state.mana)


if __name__ == "__main__":
    unittest.main()
