import unittest
from unittest.mock import patch

from action.agent_targeted_action import DealAttackDamage
from agent import (
    AcidSlimeLarge,
    AcidSlimeMedium,
    AcidSlimeSmall,
    Cultist,
    FungiBeast,
    GremlinFat,
    GremlinThief,
    GremlinTsundere,
    GremlinWarrior,
    GremlinWizard,
    GremlinNob,
    Hexaghost,
    JawWorm,
    Lagavulin,
    Looter,
    LouseDefensive,
    LouseNormal,
    Sentry,
    SlaverBlue,
    SlaverRed,
    SlimeBoss,
    SpikeSlimeLarge,
    SpikeSlimeMedium,
    SpikeSlimeSmall,
    TheGuardian,
)
from battle import BattleState
from config import Character, Verbose
from game import GameState
from ggpa.random_bot import RandomBot
from status_effecs import StatusEffectRepo
from value import ConstValue


class EnemyA20Tests(unittest.TestCase):
    def test_boss_a20_base_stats_match_sts(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)

        guardian = TheGuardian(game_state)
        hexaghost = Hexaghost(game_state)
        slime_boss = SlimeBoss(game_state)

        self.assertEqual(250, guardian.max_health)
        self.assertEqual(40, guardian.mode_shift_threshold)
        self.assertEqual(36, guardian.fierce_bash_damage)
        self.assertEqual(10, guardian.roll_damage)
        self.assertEqual(4, guardian.sharp_hide_amount)
        self.assertEqual(264, hexaghost.max_health)
        self.assertEqual(3, hexaghost.inferno_damage)
        self.assertEqual(2, hexaghost.sear_burn_count)
        self.assertEqual(3, hexaghost.inflame_strength)
        self.assertEqual(150, slime_boss.max_health)
        self.assertEqual(38, slime_boss.slam_damage)
        self.assertEqual(5, slime_boss.slimed_count)

    def test_elite_a20_base_stats_match_sts(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)

        nob = GremlinNob(game_state)
        lagavulin = Lagavulin(game_state)
        sentry = Sentry(game_state)

        self.assertGreaterEqual(nob.max_health, 85)
        self.assertLessEqual(nob.max_health, 90)
        self.assertEqual(16, nob.attack_damage)
        self.assertEqual(8, nob.skull_bash_damage)
        self.assertEqual(3, nob.enrage_amount)
        self.assertGreaterEqual(lagavulin.max_health, 112)
        self.assertLessEqual(lagavulin.max_health, 115)
        self.assertEqual(20, lagavulin.attack_damage)
        self.assertEqual(2, lagavulin.debuff_amount)
        self.assertEqual(8, lagavulin.block)
        self.assertEqual(8, lagavulin.status_effect_state.get(StatusEffectRepo.METALLICIZE))
        self.assertGreaterEqual(sentry.max_health, 39)
        self.assertLessEqual(sentry.max_health, 45)
        self.assertEqual(10, sentry.attack_damage)
        self.assertEqual(3, sentry.dazed_count)
        self.assertEqual(1, sentry.status_effect_state.get(StatusEffectRepo.ARTIFACT))

    def test_normal_enemy_a20_base_stats_match_sts(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)

        cultist = Cultist(game_state)
        jaw_worm = JawWorm(game_state)
        acid_small = AcidSlimeSmall(game_state)
        spike_small = SpikeSlimeSmall(game_state)
        acid_medium = AcidSlimeMedium(game_state)
        spike_medium = SpikeSlimeMedium(game_state)
        acid_large = AcidSlimeLarge(game_state)
        spike_large = SpikeSlimeLarge(game_state)
        louse_normal = LouseNormal(game_state)
        louse_defensive = LouseDefensive(game_state)
        fungi = FungiBeast(game_state)
        blue_slaver = SlaverBlue(game_state)
        red_slaver = SlaverRed(game_state)
        looter = Looter(game_state)
        mad_gremlin = GremlinWarrior(game_state)
        fat_gremlin = GremlinFat(game_state)
        sneaky_gremlin = GremlinThief(game_state)
        shield_gremlin = GremlinTsundere(game_state)
        gremlin_wizard = GremlinWizard(game_state)

        self.assertGreaterEqual(cultist.max_health, 50)
        self.assertLessEqual(cultist.max_health, 56)
        self.assertGreaterEqual(jaw_worm.max_health, 42)
        self.assertLessEqual(jaw_worm.max_health, 46)
        self.assertEqual(4, acid_small.attack_damage)
        self.assertEqual(6, spike_small.action_set.peek().targeted.val.peek())
        self.assertEqual(8, acid_medium.spit_damage)
        self.assertEqual(12, acid_medium.attack_damage)
        self.assertEqual(10, spike_medium.attack_damage)
        self.assertEqual(18, acid_large.tackle_damage)
        self.assertEqual(12, acid_large.spit_damage)
        self.assertEqual(18, spike_large.attack_damage)
        self.assertGreaterEqual(louse_normal.attack_damage, 6)
        self.assertLessEqual(louse_normal.attack_damage, 8)
        self.assertEqual(4, louse_normal.strength_amount)
        self.assertGreaterEqual(louse_defensive.attack_damage, 6)
        self.assertLessEqual(louse_defensive.attack_damage, 8)
        self.assertEqual(5, fungi.grow_strength)
        self.assertEqual(6, fungi.attack_damage)
        self.assertEqual(13, blue_slaver.stab_damage)
        self.assertEqual(8, blue_slaver.rake_damage)
        self.assertEqual(2, blue_slaver.weak_amount)
        self.assertEqual(14, red_slaver.stab_damage)
        self.assertEqual(9, red_slaver.scrape_damage)
        self.assertEqual(2, red_slaver.vulnerable_amount)
        self.assertEqual(11, looter.mug_damage)
        self.assertEqual(14, looter.lunge_damage)
        self.assertEqual(5, mad_gremlin.attack_damage)
        self.assertEqual(2, mad_gremlin.status_effect_state.get(StatusEffectRepo.ANGRY))
        self.assertEqual(5, fat_gremlin.attack_damage)
        self.assertEqual(1, fat_gremlin.frail_amount)
        self.assertEqual(10, sneaky_gremlin.attack_damage)
        self.assertGreaterEqual(shield_gremlin.max_health, 13)
        self.assertLessEqual(shield_gremlin.max_health, 17)
        self.assertEqual(11, shield_gremlin.block_amount)
        self.assertEqual(8, shield_gremlin.bash_damage)
        self.assertGreaterEqual(gremlin_wizard.max_health, 22)
        self.assertLessEqual(gremlin_wizard.max_health, 26)
        self.assertEqual(30, gremlin_wizard.attack_damage)
        self.assertTrue(gremlin_wizard.fast_recharge)

    def test_gremlin_nob_a20_action_sequence(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        nob = GremlinNob(game_state)
        battle_state = BattleState(game_state, nob, verbose=Verbose.NO_LOG)

        expected_fragments = [
            "Apply 3 Anger",
            "Deal 8 attack damage",
            "Deal 16 attack damage",
            "Deal 16 attack damage",
            "Deal 8 attack damage",
        ]
        for fragment in expected_fragments:
            self.assertIn(fragment, repr(nob.get_intention(game_state, battle_state)))
            nob.play(game_state, battle_state)

    def test_lagavulin_a20_sleep_wake_and_debuff_cycle(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        lagavulin = Lagavulin(game_state)
        battle_state = BattleState(game_state, lagavulin, verbose=Verbose.NO_LOG)

        for _ in range(3):
            self.assertEqual("Sleep", repr(lagavulin.get_intention(game_state, battle_state)))
            lagavulin.play(game_state, battle_state)

        self.assertIn("Deal 20 attack damage", repr(lagavulin.get_intention(game_state, battle_state)))
        lagavulin.play(game_state, battle_state)
        self.assertIn("Deal 20 attack damage", repr(lagavulin.get_intention(game_state, battle_state)))
        lagavulin.play(game_state, battle_state)
        self.assertIn("Apply -2 Strength", repr(lagavulin.get_intention(game_state, battle_state)))
        lagavulin.play(game_state, battle_state)
        self.assertEqual(-2, game_state.player.status_effect_state.get(StatusEffectRepo.STRENGTH))
        self.assertEqual(-2, game_state.player.status_effect_state.get(StatusEffectRepo.DEXTERITY))

    def test_sentries_a20_opening_intents(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        left = Sentry(game_state)
        center = Sentry(game_state, starts_with_bolt=True)
        right = Sentry(game_state)
        battle_state = BattleState(game_state, left, center, right, verbose=Verbose.NO_LOG)

        self.assertIn("Deal 10 attack damage", repr(left.get_intention(game_state, battle_state)))
        self.assertIn("Add 3 status card", repr(center.get_intention(game_state, battle_state)))
        self.assertIn("Deal 10 attack damage", repr(right.get_intention(game_state, battle_state)))
        center.play(game_state, battle_state)
        self.assertEqual(3, sum(1 for card in battle_state.discard_pile if card.name == "Dazed"))

    def test_gremlin_wizard_a20_charges_twice_then_attacks_every_turn(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        wizard = GremlinWizard(game_state)
        battle_state = BattleState(game_state, wizard, verbose=Verbose.NO_LOG)

        expected_fragments = [
            "NoAction()",
            "NoAction()",
            "Deal 30 attack damage",
            "Deal 30 attack damage",
            "Deal 30 attack damage",
        ]
        for fragment in expected_fragments:
            self.assertIn(fragment, repr(wizard.get_intention(game_state, battle_state)))
            wizard.play(game_state, battle_state)

    def test_acid_slime_small_a20_opens_with_weak_then_alternates(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        slime = AcidSlimeSmall(game_state)
        battle_state = BattleState(game_state, slime, verbose=Verbose.NO_LOG)

        expected_fragments = [
            "Apply 1 Weak",
            "Deal 4 attack damage",
            "Apply 1 Weak",
            "Deal 4 attack damage",
        ]
        for fragment in expected_fragments:
            self.assertIn(fragment, repr(slime.get_intention(game_state, battle_state)))
            slime.play(game_state, battle_state)

    def test_acid_slime_large_a20_roll_75_uses_weak_lick(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        slime = AcidSlimeLarge(game_state)
        battle_state = BattleState(game_state, slime, verbose=Verbose.NO_LOG)

        with patch("agent.random.randrange", return_value=75):
            self.assertIn("Apply 2 Weak", repr(slime.get_intention(game_state, battle_state)))

    def test_spike_slime_large_a20_frail_lick_applies_three_frail(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        slime = SpikeSlimeLarge(game_state)
        battle_state = BattleState(game_state, slime, verbose=Verbose.NO_LOG)

        with patch("agent.random.randrange", return_value=50):
            self.assertIn("Apply 3 Frail", repr(slime.get_intention(game_state, battle_state)))
            slime.play(game_state, battle_state)

        self.assertEqual(3, game_state.player.status_effect_state.get(StatusEffectRepo.FRAIL))

    def test_shield_gremlin_a20_attacks_after_protect_when_alone(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        gremlin = GremlinTsundere(game_state)
        battle_state = BattleState(game_state, gremlin, verbose=Verbose.NO_LOG)

        self.assertIn("Add 11 block", repr(gremlin.get_intention(game_state, battle_state)))
        gremlin.play(game_state, battle_state)
        self.assertEqual(11, gremlin.block)
        self.assertIn("Deal 8 attack damage", repr(gremlin.get_intention(game_state, battle_state)))

    def test_guardian_enters_defensive_mode_after_a20_threshold_damage(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        guardian = TheGuardian(game_state)
        battle_state = BattleState(game_state, guardian, verbose=Verbose.NO_LOG)

        DealAttackDamage(ConstValue(40)).play(game_state.player, game_state, battle_state, guardian)
        intent = repr(guardian.get_intention(game_state, battle_state))

        self.assertIn("Defensive Mode", intent)
        guardian.play(game_state, battle_state)
        self.assertEqual(20, guardian.block)
        self.assertEqual(50, guardian.mode_shift_threshold)
        self.assertEqual(50, guardian.mode_shift_counter)
        self.assertIn("Sharp Hide", repr(guardian.get_intention(game_state, battle_state)))

        guardian.play(game_state, battle_state)
        self.assertEqual(4, guardian.status_effect_state.get(StatusEffectRepo.SHARP_HIDE))

    def test_slime_boss_a20_opening_cycle_and_split(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        slime_boss = SlimeBoss(game_state)
        battle_state = BattleState(game_state, slime_boss, verbose=Verbose.NO_LOG)

        self.assertIn("Add 5 status card", repr(slime_boss.get_intention(game_state, battle_state)))
        slime_boss.play(game_state, battle_state)
        self.assertEqual(5, sum(1 for card in battle_state.discard_pile if card.name == "Slimed"))
        self.assertEqual("NoAction()", repr(slime_boss.get_intention(game_state, battle_state)))
        slime_boss.play(game_state, battle_state)
        self.assertIn("Deal 38 attack damage", repr(slime_boss.get_intention(game_state, battle_state)))

        DealAttackDamage(ConstValue(75)).play(game_state.player, game_state, battle_state, slime_boss)

        self.assertEqual("Split", repr(slime_boss.get_intention(game_state, battle_state)))

    def test_hexaghost_a20_opening_and_cycle_match_sts(self):
        game_state = GameState(Character.IRON_CLAD, RandomBot(), 20)
        hexaghost = Hexaghost(game_state)
        battle_state = BattleState(game_state, hexaghost, verbose=Verbose.NO_LOG)
        game_state.player.health = 72

        self.assertEqual("NoAction()", repr(hexaghost.get_intention(game_state, battle_state)))
        hexaghost.play(game_state, battle_state)
        self.assertIn("Deal 7 attack damage 6 times", repr(hexaghost.get_intention(game_state, battle_state)))
        hexaghost.play(game_state, battle_state)

        expected_fragments = [
            "Deal 6 attack damage",
            "Deal 6 attack damage 2 times",
            "Deal 6 attack damage",
            "Apply 3 Strength",
            "Deal 6 attack damage 2 times",
            "Deal 6 attack damage",
            "Deal 3 attack damage 6 times",
        ]
        for fragment in expected_fragments:
            self.assertIn(fragment, repr(hexaghost.get_intention(game_state, battle_state)))
            hexaghost.play(game_state, battle_state)


if __name__ == "__main__":
    unittest.main()
