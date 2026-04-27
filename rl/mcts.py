from __future__ import annotations

import copy
import math
import random
import re
from dataclasses import dataclass

from config import CardType
from rl.env import MiniSTSEnv


@dataclass
class ActionStats:
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class MCTSPlanner:
    def __init__(
        self,
        simulations: int = 200,
        rollout_depth: int = 80,
        exploration: float = 1.4,
        gamma: float = 0.99,
        eval_weight: float = 1.0,
        rollout_policy: str = "heuristic",
    ):
        self.simulations = simulations
        self.rollout_depth = rollout_depth
        self.exploration = exploration
        self.gamma = gamma
        self.eval_weight = eval_weight
        self.rollout_policy = rollout_policy

    def choose_action(self, env: MiniSTSEnv) -> int:
        legal = self._legal_indices(env)
        if len(legal) == 1:
            return legal[0]
        if env.pending_hand_choice is not None:
            return self._heuristic_action(env, legal)

        stats = {action: ActionStats() for action in legal}
        for _ in range(self.simulations):
            action = self._select_ucb(stats)
            value = self._simulate(env, action)
            stats[action].visits += 1
            stats[action].value_sum += value

        return max(legal, key=lambda action: (stats[action].mean, stats[action].visits))

    def _select_ucb(self, stats: dict[int, ActionStats]) -> int:
        unvisited = [action for action, action_stats in stats.items() if action_stats.visits == 0]
        if unvisited:
            return random.choice(unvisited)
        total_visits = sum(action_stats.visits for action_stats in stats.values())
        return max(
            stats,
            key=lambda action: stats[action].mean
            + self.exploration * math.sqrt(math.log(total_visits + 1) / stats[action].visits),
        )

    def _simulate(self, root_env: MiniSTSEnv, first_action: int) -> float:
        env: MiniSTSEnv = copy.deepcopy(root_env)
        result = env.step_index(first_action)
        total = result.reward
        discount = self.gamma
        depth = 0

        while not result.done and depth < self.rollout_depth:
            legal = self._legal_indices(env)
            action = self._rollout_action(env, legal)
            result = env.step_index(action)
            total += discount * result.reward
            discount *= self.gamma
            depth += 1

        if not result.done:
            total += discount * self.eval_weight * self._evaluate(env)
        return total

    def _rollout_action(self, env: MiniSTSEnv, legal: list[int]) -> int:
        if self.rollout_policy == "random":
            return random.choice(legal)
        return self._heuristic_action(env, legal)

    def _heuristic_action(self, env: MiniSTSEnv, legal: list[int]) -> int:
        assert env.battle_state is not None and env.game_state is not None
        battle = env.battle_state

        if env.pending_hand_choice is not None:
            return self._choose_pending(env, legal)

        incoming = self._incoming_attack(env)
        if incoming > battle.player.block:
            for name in ("Shrug It Off", "True Grit", "Defend"):
                action = self._first_card_action(env, legal, name)
                if action is not None:
                    return action

        if battle.player.health > 40 and battle.mana <= 1:
            action = self._first_card_action(env, legal, "Bloodletting")
            if action is not None:
                return action

        for name in ("Bludgeon", "Pommel Strike", "Strike"):
            action = self._first_card_action(env, legal, name)
            if action is not None:
                return action

        for name in ("Shrug It Off", "True Grit", "Defend"):
            action = self._first_card_action(env, legal, name)
            if action is not None:
                return action
        return 0 if 0 in legal else legal[0]

    def _choose_pending(self, env: MiniSTSEnv, legal: list[int]) -> int:
        assert env.battle_state is not None
        purpose = env.pending_hand_choice.purpose if env.pending_hand_choice is not None else ""
        preferred = {
            "exhaust_hand_card": ("Bloodletting", "Strike", "Defend", "Shrug It Off", "Pommel Strike", "Bludgeon"),
            "topdeck_hand_card": ("Bludgeon", "Pommel Strike", "Shrug It Off", "True Grit", "Defend", "Strike"),
            "upgrade_hand_card": ("Bludgeon", "Pommel Strike", "True Grit", "Shrug It Off", "Defend", "Strike"),
            "duplicate_hand_card": ("Bludgeon", "Pommel Strike", "Strike"),
        }.get(purpose, ())
        for name in preferred:
            action = self._first_card_action(env, legal, name)
            if action is not None:
                return action
        return legal[0]

    def _first_card_action(self, env: MiniSTSEnv, legal: list[int], card_name: str) -> int | None:
        assert env.battle_state is not None
        for index, card in enumerate(env.battle_state.hand):
            action = index + 1
            if action in legal and card.name == card_name:
                return action
        return None

    def _incoming_attack(self, env: MiniSTSEnv) -> int:
        assert env.game_state is not None and env.battle_state is not None
        total = 0
        for enemy in env.battle_state.enemies:
            intent = repr(enemy.get_intention(env.game_state, env.battle_state))
            for damage, times in re.findall(r"Deal (\d+) attack damage(?: (\d+) times)?", intent):
                total += int(damage) * (int(times) if times else 1)
        return total

    def _evaluate(self, env: MiniSTSEnv) -> float:
        assert env.battle_state is not None
        battle = env.battle_state
        result = battle.get_end_result()
        if result == 1:
            return 1.0
        if result == -1:
            return -1.0
        enemy_health = sum(enemy.health for enemy in battle.enemies)
        enemy_max_health = max(1, sum(enemy.max_health for enemy in battle.enemies))
        player_score = battle.player.health / battle.player.max_health
        enemy_score = enemy_health / enemy_max_health
        return player_score - enemy_score

    def _legal_indices(self, env: MiniSTSEnv) -> list[int]:
        return [env.to_action_index(action) for action in env.legal_actions()]
