import random
import unittest

import numpy as np

from rl.encoder import StateEncoder
from rl.env import MiniSTSEnv


class RelicEncoderTests(unittest.TestCase):
    def test_observation_encodes_relic_ownership(self):
        encoder = StateEncoder(relic_names=("Paper Phrog",))

        random.seed(0)
        without_relic = MiniSTSEnv(encoder=encoder, enemy_name="Cultist")
        without_observation = without_relic.reset()

        random.seed(0)
        with_relic = MiniSTSEnv(encoder=encoder, enemy_name="Cultist", relics=["Paper Phrog"])
        with_observation = with_relic.reset()

        self.assertFalse(np.array_equal(without_observation, with_observation))


if __name__ == "__main__":
    unittest.main()
