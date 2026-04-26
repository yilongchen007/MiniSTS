from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from agent import (
    AcidSlimeLarge,
    AcidSlimeMedium,
    AcidSlimeSmall,
    Cultist,
    Enemy,
    FungiBeast,
    GremlinFat,
    GremlinNob,
    GremlinThief,
    GremlinTsundere,
    GremlinWarrior,
    GremlinWizard,
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
from game import GameState


@dataclass(frozen=True)
class EncounterInfo:
    name: str
    weight: float


EXORDIUM_WEAK_ENCOUNTERS: tuple[EncounterInfo, ...] = (
    EncounterInfo("Cultist", 2.0),
    EncounterInfo("Jaw Worm", 2.0),
    EncounterInfo("2 Louse", 2.0),
    EncounterInfo("Small Slimes", 2.0),
)

EXORDIUM_STRONG_ENCOUNTERS: tuple[EncounterInfo, ...] = (
    EncounterInfo("Blue Slaver", 2.0),
    EncounterInfo("Gremlin Gang", 1.0),
    EncounterInfo("Looter", 2.0),
    EncounterInfo("Large Slime", 2.0),
    EncounterInfo("Lots of Slimes", 1.0),
    EncounterInfo("Exordium Thugs", 1.5),
    EncounterInfo("Exordium Wildlife", 1.5),
    EncounterInfo("Red Slaver", 1.0),
    EncounterInfo("3 Louse", 2.0),
    EncounterInfo("2 Fungi Beasts", 2.0),
)

EXORDIUM_ELITE_ENCOUNTERS: tuple[EncounterInfo, ...] = (
    EncounterInfo("Gremlin Nob", 1.0),
    EncounterInfo("Lagavulin", 1.0),
    EncounterInfo("3 Sentries", 1.0),
)


EXORDIUM_SUPPORTED_ENCOUNTERS: dict[str, Callable[[GameState], list[Enemy]]] = {
    "Cultist": lambda game_state: [Cultist(game_state)],
    "Jaw Worm": lambda game_state: [JawWorm(game_state)],
    "2 Louse": lambda game_state: [LouseNormal(game_state), LouseDefensive(game_state)],
    "Small Slimes": lambda game_state: random_small_slimes(game_state),
    "Blue Slaver": lambda game_state: [SlaverBlue(game_state)],
    "Gremlin Gang": lambda game_state: random_gremlin_gang(game_state),
    "Looter": lambda game_state: [Looter(game_state)],
    "Large Slime": lambda game_state: [random_large_slime(game_state)],
    "Lots of Slimes": lambda game_state: random_many_small_slimes(game_state),
    "Exordium Thugs": lambda game_state: [Looter(game_state), SlaverBlue(game_state)],
    "Exordium Wildlife": lambda game_state: [FungiBeast(game_state), JawWorm(game_state)],
    "Red Slaver": lambda game_state: [SlaverRed(game_state)],
    "3 Louse": lambda game_state: [LouseNormal(game_state), LouseDefensive(game_state), LouseNormal(game_state)],
    "2 Fungi Beasts": lambda game_state: [FungiBeast(game_state), FungiBeast(game_state)],
    "Gremlin Nob": lambda game_state: [GremlinNob(game_state)],
    "Lagavulin": lambda game_state: [Lagavulin(game_state)],
    "3 Sentries": lambda game_state: [Sentry(game_state), Sentry(game_state, starts_with_bolt=True), Sentry(game_state)],
    "The Guardian": lambda game_state: [TheGuardian(game_state)],
    "Hexaghost": lambda game_state: [Hexaghost(game_state)],
    "Slime Boss": lambda game_state: [SlimeBoss(game_state)],
}


def random_large_slime(game_state: GameState) -> Enemy:
    if random.random() < 0.5:
        return AcidSlimeLarge(game_state)
    return SpikeSlimeLarge(game_state)


def random_small_slimes(game_state: GameState) -> list[Enemy]:
    if random.random() < 0.5:
        return [SpikeSlimeSmall(game_state), AcidSlimeMedium(game_state)]
    return [AcidSlimeSmall(game_state), SpikeSlimeMedium(game_state)]


def random_many_small_slimes(game_state: GameState) -> list[Enemy]:
    pool = [SpikeSlimeSmall, SpikeSlimeSmall, SpikeSlimeSmall, AcidSlimeSmall, AcidSlimeSmall]
    random.shuffle(pool)
    return [factory(game_state) for factory in pool]


def random_gremlin_gang(game_state: GameState) -> list[Enemy]:
    pool = [
        GremlinWarrior,
        GremlinWarrior,
        GremlinThief,
        GremlinThief,
        GremlinFat,
        GremlinFat,
        GremlinTsundere,
        GremlinWizard,
    ]
    return [factory(game_state) for factory in random.sample(pool, 4)]


def create_exordium_encounter(name: str, game_state: GameState) -> list[Enemy]:
    try:
        return EXORDIUM_SUPPORTED_ENCOUNTERS[name](game_state)
    except KeyError as exc:
        supported = ", ".join(sorted(EXORDIUM_SUPPORTED_ENCOUNTERS))
        raise ValueError(f"Unsupported Exordium encounter {name!r}. Supported: {supported}") from exc
