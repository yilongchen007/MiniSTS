from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from card import Card, CardGen


DEFAULT_DECK: list[dict[str, Any]] = [
    {"name": "Shrug It Off"},
    {"name": "Defend"},
    {"name": "Strike"},
    {"name": "Bash"},
    {"name": "Bloodletting"},
    {"name": "Pommel Strike", "upgrades": 1},
]


def _normalize_name(name: str) -> str:
    return name.lower().replace("_", " ").replace("-", " ").strip()


CARD_FACTORIES: dict[str, Callable[[], Card]] = {
    _normalize_name("Strike"): CardGen.Strike,
    _normalize_name("Defend"): CardGen.Defend,
    _normalize_name("SearingBlow"): CardGen.Searing_Blow,
    _normalize_name("Searing Blow"): CardGen.Searing_Blow,
    _normalize_name("Bash"): CardGen.Bash,
    _normalize_name("Anger"): CardGen.Anger,
    _normalize_name("Pommel Strike"): CardGen.Pommel_Strike,
    _normalize_name("Shrug It Off"): CardGen.Shrug_It_Off,
    _normalize_name("Bloodletting"): CardGen.Bloodletting,
    _normalize_name("Cleave"): CardGen.Cleave,
    _normalize_name("Impervious"): CardGen.Impervious,
    _normalize_name("Battle Trance"): CardGen.Battle_Trance,
    _normalize_name("Flex"): CardGen.Flex,
    _normalize_name("Inflame"): CardGen.Inflame,
    _normalize_name("Metallicize"): CardGen.Metallicize,
    _normalize_name("Barricade"): CardGen.Barricade,
    _normalize_name("Seeing Red"): CardGen.Seeing_Red,
    _normalize_name("True Grit"): CardGen.True_Grit,
    _normalize_name("Wild Strike"): CardGen.Wild_Strike,
    _normalize_name("Power Through"): CardGen.Power_Through,
    _normalize_name("Rage"): CardGen.Rage,
    _normalize_name("Feel No Pain"): CardGen.Feel_No_Pain,
    _normalize_name("Dark Embrace"): CardGen.Dark_Embrace,
    _normalize_name("Evolve"): CardGen.Evolve,
    _normalize_name("Fire Breathing"): CardGen.Fire_Breathing,
    _normalize_name("Juggernaut"): CardGen.Juggernaut,
    _normalize_name("Flame Barrier"): CardGen.Flame_Barrier,
    _normalize_name("Bludgeon"): CardGen.Bludgeon,
    _normalize_name("Clothesline"): CardGen.Clothesline,
    _normalize_name("Thunderclap"): CardGen.Thunderclap,
    _normalize_name("Twin Strike"): CardGen.Twin_Strike,
    _normalize_name("Uppercut"): CardGen.Uppercut,
    _normalize_name("Iron Wave"): CardGen.Iron_Wave,
    _normalize_name("Pummel"): CardGen.Pummel,
    _normalize_name("Hemokinesis"): CardGen.Hemokinesis,
    _normalize_name("Offering"): CardGen.Offering,
    _normalize_name("Disarm"): CardGen.Disarm,
    _normalize_name("Shockwave"): CardGen.Shockwave,
    _normalize_name("Berserk"): CardGen.Berserk,
    _normalize_name("Demon Form"): CardGen.Demon_Form,
    _normalize_name("Brutality"): CardGen.Brutality,
    _normalize_name("Armament"): CardGen.Armaments,
    _normalize_name("Armaments"): CardGen.Armaments,
    _normalize_name("Blood for Blood"): CardGen.Blood_for_Blood,
    _normalize_name("Body Slam"): CardGen.Body_Slam,
    _normalize_name("Burning Pact"): CardGen.Burning_Pact,
    _normalize_name("Carnage"): CardGen.Carnage,
    _normalize_name("Clash"): CardGen.Clash,
    _normalize_name("Combust"): CardGen.Combust,
    _normalize_name("Corruption"): CardGen.Corruption,
    _normalize_name("Double Tap"): CardGen.Double_Tap,
    _normalize_name("Dropkick"): CardGen.Dropkick,
    _normalize_name("Dual Wield"): CardGen.Dual_Wield,
    _normalize_name("Entrench"): CardGen.Entrench,
    _normalize_name("Exhume"): CardGen.Exhume,
    _normalize_name("Feed"): CardGen.Feed,
    _normalize_name("Fiend Fire"): CardGen.Fiend_Fire,
    _normalize_name("Ghostly Armor"): CardGen.Ghostly_Armor,
    _normalize_name("Havoc"): CardGen.Havoc,
    _normalize_name("Headbutt"): CardGen.Headbutt,
    _normalize_name("Heavy Blade"): CardGen.Heavy_Blade,
    _normalize_name("Immolate"): CardGen.Immolate,
    _normalize_name("Infernal Blade"): CardGen.Infernal_Blade,
    _normalize_name("Intimidate"): CardGen.Intimidate,
    _normalize_name("Limit Break"): CardGen.Limit_Break,
    _normalize_name("Perfected Strike"): CardGen.Perfected_Strike,
    _normalize_name("Rampage"): CardGen.Rampage,
    _normalize_name("Reaper"): CardGen.Reaper,
    _normalize_name("Reckless Charge"): CardGen.Reckless_Charge,
    _normalize_name("Rupture"): CardGen.Rupture,
    _normalize_name("Second Wind"): CardGen.Second_Wind,
    _normalize_name("Sentinel"): CardGen.Sentinel,
    _normalize_name("Sever Soul"): CardGen.Sever_Soul,
    _normalize_name("Spot Weakness"): CardGen.Spot_Weakness,
    _normalize_name("Sword Boomerang"): CardGen.Sword_Boomerang,
    _normalize_name("Warcry"): CardGen.Warcry,
    _normalize_name("Whirlwind"): CardGen.Whirlwind,
    _normalize_name("Strike_R"): CardGen.Strike_R,
    _normalize_name("Defend_R"): CardGen.Defend_R,
    _normalize_name("Stimulate"): CardGen.Stimulate,
    _normalize_name("Batter"): CardGen.Batter,
    _normalize_name("Tolerate"): CardGen.Tolerate,
    _normalize_name("Bomb"): CardGen.Bomb,
    _normalize_name("Suffer"): CardGen.Suffer,
    _normalize_name("Wound"): CardGen.Wound,
    _normalize_name("Slimed"): CardGen.Slimed,
    _normalize_name("Dazed"): CardGen.Dazed,
    _normalize_name("Burn"): CardGen.Burn,
}


@dataclass(frozen=True)
class ExperimentConfig:
    raw: dict[str, Any]

    @staticmethod
    def load(path: str | None) -> ExperimentConfig:
        if path is None:
            return ExperimentConfig({})
        with Path(path).open() as f:
            return ExperimentConfig(json.load(f))

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"Config section {name!r} must be an object.")
        return value

    def get(self, section: str, key: str, default: Any) -> Any:
        return self.section(section).get(key, default)

    def build_deck(self) -> list[Card]:
        if "deck" in self.raw:
            return build_deck(self.raw["deck"])

        return build_deck(DEFAULT_DECK)

    def relic_names(self) -> list[str]:
        specs = self.raw.get("relics", [])
        if not isinstance(specs, list):
            raise ValueError("Config field 'relics' must be a list.")
        for spec in specs:
            if not isinstance(spec, str):
                raise ValueError(f"Relic spec must be a string, got: {spec!r}")
        return list(specs)


def build_deck(specs: Any) -> list[Card]:
    if not isinstance(specs, list):
        raise ValueError("Config field 'deck' must be a list.")

    cards: list[Card] = []
    for spec in specs:
        cards.extend(build_cards(spec))
    return cards


def build_cards(spec: Any) -> list[Card]:
    if isinstance(spec, str):
        spec = {"name": spec}
    if not isinstance(spec, dict):
        raise ValueError(f"Deck card spec must be a string or object, got: {spec!r}")

    name = spec.get("name")
    if not isinstance(name, str):
        raise ValueError(f"Deck card spec needs a string name: {spec!r}")

    count = int(spec.get("count", 1))
    upgrades = int(spec.get("upgrades", 0))
    if count < 0 or upgrades < 0:
        raise ValueError(f"Deck card count/upgrades must be non-negative: {spec!r}")

    factory = CARD_FACTORIES.get(_normalize_name(name))
    if factory is None:
        available = ", ".join(sorted({card.name for card in [factory() for factory in CARD_FACTORIES.values()]}))
        raise ValueError(f"Unknown card name {name!r}. Available cards: {available}")

    cards = [factory() for _ in range(count)]
    for card in cards:
        if upgrades:
            card.upgrade(upgrades)
    return cards


def card_names_from_deck(deck: list[Card]) -> tuple[str, ...]:
    names: list[str] = []
    for card in deck:
        if card.name not in names:
            names.append(card.name)
    return tuple(names)
