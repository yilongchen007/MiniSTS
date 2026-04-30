import unittest

from rl.experiment_config import ExperimentConfig
from rl.env import MiniSTSEnv


class ExperimentConfigTests(unittest.TestCase):
    def test_relic_names_come_from_top_level_config(self):
        config = ExperimentConfig({"relics": ["Anchor", "Pen Nib"]})

        self.assertEqual(["Anchor", "Pen Nib"], config.relic_names())

    def test_repo_rl_config_declares_relics_for_experiments(self):
        config = ExperimentConfig.load("rl_configs/scenario5_big_jaw_worm.json")

        self.assertIn("relics", config.raw)
        self.assertEqual([], config.relic_names())

    def test_configured_relics_apply_through_rl_env_reset(self):
        config = ExperimentConfig({"relics": ["Anchor"], "env": {"enemy": "Cultist", "ascension": 20}})
        env_config = config.section("env")
        env = MiniSTSEnv(
            enemy_name=env_config["enemy"],
            ascension=env_config["ascension"],
            relics=config.relic_names(),
        )

        env.reset()

        self.assertEqual(10, env.battle_state.player.block)


if __name__ == "__main__":
    unittest.main()
