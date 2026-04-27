from __future__ import annotations

import argparse
import json
import mimetypes
import random
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rl.env import MiniSTSEnv
from rl.encoder import StateEncoder
from rl.experiment_config import ExperimentConfig, card_names_from_deck


ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = ROOT / "rl_web" / "manual_play"


def describe_action(env: MiniSTSEnv, action_index: int) -> str:
    assert env.battle_state is not None
    battle = env.battle_state
    if env.pending_hand_choice is not None:
        if action_index == 0:
            return "Invalid during hand choice"
        hand_index = action_index - 1
        if not 0 <= hand_index < len(battle.hand):
            return f"Choose missing hand slot {hand_index}"
        return f"Choose {battle.hand[hand_index].get_name()} for {env.pending_hand_choice.purpose}"
    if action_index == 0:
        return "End turn"
    hand_index = action_index - 1
    if not 0 <= hand_index < len(battle.hand):
        return f"Play missing hand slot {hand_index}"
    return f"Play {battle.hand[hand_index].get_name()}"


def build_env(config_path: str, enemy: str | None, ascension: int | None, max_steps: int | None) -> MiniSTSEnv:
    experiment_config = ExperimentConfig.load(config_path)
    env_config = experiment_config.section("env")
    deck = experiment_config.build_deck()
    encoder_config = experiment_config.section("encoder")
    card_names = tuple(encoder_config.get("card_names", card_names_from_deck(deck)))
    encoder = StateEncoder(
        max_turns=int(encoder_config.get("max_turns", 20)),
        max_hand_size=int(encoder_config.get("max_hand_size", 10)),
        card_names=card_names,
    )
    return MiniSTSEnv(
        encoder=encoder,
        max_steps=int(max_steps if max_steps is not None else env_config.get("max_steps", 200)),
        enemy_name=str(enemy if enemy is not None else env_config.get("enemy", "BigJawWorm")),
        deck=deck,
        ascension=int(ascension if ascension is not None else env_config.get("ascension", 0)),
    )


class ManualPlaySession:
    def __init__(
        self,
        config_path: str,
        enemy: str | None,
        ascension: int | None,
        max_steps: int | None,
        seed: int,
        record_path: str | None = None,
    ):
        self.config_path = config_path
        self.enemy = enemy
        self.ascension = ascension
        self.max_steps = max_steps
        self.seed = seed
        self.record_path = record_path
        self.actions: list[int] = []
        self.env = build_env(config_path, enemy, ascension, max_steps)
        self.reset()

    def reset(self) -> dict[str, Any]:
        random.seed(self.seed)
        self.actions = []
        self.env = build_env(self.config_path, self.enemy, self.ascension, self.max_steps)
        self.env.reset()
        return self.state()

    def act(self, action_index: int) -> dict[str, Any]:
        result = self.env.step_index(action_index)
        self.actions.append(action_index)
        state = self.state()
        state["last_reward"] = result.reward
        state["last_info"] = result.info
        self.save_record()
        return state

    def state(self) -> dict[str, Any]:
        assert self.env.game_state is not None and self.env.battle_state is not None
        battle = self.env.battle_state
        legal_indices = sorted(self.env.to_action_index(action) for action in self.env.legal_actions())
        result = battle.get_end_result()
        return {
            "config_path": self.config_path,
            "enemy_name": self.env.enemy_name,
            "ascension": self.env.ascension,
            "max_steps": self.env.max_steps,
            "steps": self.env.steps,
            "turn": battle.turn,
            "mana": battle.mana,
            "max_mana": battle.game_state.max_mana,
            "done": battle.ended() or self.env.steps >= self.env.max_steps,
            "result": result,
            "outcome": "WIN" if result == 1 else "LOSE" if result == -1 else "RUNNING",
            "hp_loss": battle.player_hp_lost_this_combat,
            "player": {
                "name": battle.player.name,
                "health": battle.player.health,
                "max_health": battle.player.max_health,
                "block": battle.player.block,
                "status": repr(battle.player.status_effect_state),
            },
            "enemies": [self._agent(enemy, index) for index, enemy in enumerate(battle.enemies)],
            "hand": [self._card(card, index) for index, card in enumerate(battle.hand)],
            "draw_pile": [card.get_name() for card in battle.draw_pile],
            "discard_pile": [card.get_name() for card in battle.discard_pile],
            "exhaust_pile": [card.get_name() for card in battle.exhaust_pile],
            "pending": None if self.env.pending_hand_choice is None else {
                "purpose": self.env.pending_hand_choice.purpose,
                "hand_indices": list(self.env.pending_hand_choice.hand_indices),
            },
            "legal_actions": [
                {
                    "index": index,
                    "label": describe_action(self.env, index),
                }
                for index in legal_indices
            ],
            "actions": list(self.actions),
        }

    def save_record(self) -> None:
        if self.record_path is None:
            return
        assert self.env.battle_state is not None
        result = self.env.battle_state.get_end_result()
        payload = {
            "config": self.config_path,
            "enemy": self.env.enemy_name,
            "ascension": self.env.ascension,
            "seed": self.seed,
            "result": result,
            "outcome": "WIN" if result == 1 else "LOSE" if result == -1 else "RUNNING",
            "steps": self.env.steps,
            "hp_loss": self.env.battle_state.player_hp_lost_this_combat,
            "actions": list(self.actions),
        }
        path = Path(self.record_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))

    def _agent(self, agent: Any, index: int) -> dict[str, Any]:
        assert self.env.game_state is not None and self.env.battle_state is not None
        return {
            "index": index,
            "name": agent.name,
            "health": agent.health,
            "max_health": agent.max_health,
            "block": agent.block,
            "status": repr(agent.status_effect_state),
            "intent": repr(agent.get_intention(self.env.game_state, self.env.battle_state)),
        }

    def _card(self, card: Any, index: int) -> dict[str, Any]:
        assert self.env.game_state is not None and self.env.battle_state is not None
        return {
            "index": index,
            "name": card.name,
            "display_name": card.get_name(),
            "type": card.card_type.name,
            "rarity": card.rarity.name,
            "cost": card.effective_cost(self.env.game_state, self.env.battle_state),
            "base_cost": card.mana_cost.peek(),
            "playable": card.is_playable(self.env.game_state, self.env.battle_state),
            "exhaust": card.exhaust_on_play,
            "ethereal": card.ethereal,
            "desc": str(card.desc),
        }


class RequestHandler(BaseHTTPRequestHandler):
    session: ManualPlaySession

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/state":
            self.write_json(self.session.state())
            return
        if path == "/":
            path = "/index.html"
        self.serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/reset":
            self.write_json(self.session.reset())
            return
        if path == "/api/action":
            payload = self.read_json()
            try:
                action_index = int(payload["action"])
            except (KeyError, TypeError, ValueError):
                self.write_json({"error": "Expected JSON body with integer field 'action'."}, HTTPStatus.BAD_REQUEST)
                return
            self.write_json(self.session.act(action_index))
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length else "{}"
        return json.loads(body)

    def write_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def serve_static(self, path: str) -> None:
        relative = path.lstrip("/")
        file_path = (STATIC_ROOT / relative).resolve()
        if not file_path.is_file() or STATIC_ROOT.resolve() not in file_path.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(str(file_path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rl_configs/scenario5_big_jaw_worm.json")
    parser.add_argument("--enemy", default=None)
    parser.add_argument("--ascension", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--record-path", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    handler = RequestHandler
    handler.session = ManualPlaySession(args.config, args.enemy, args.ascension, args.max_steps, args.seed, args.record_path)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"MiniSTS manual play UI running at http://{args.host}:{server.server_port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
