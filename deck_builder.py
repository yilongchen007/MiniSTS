from __future__ import annotations

from typing import Callable

from card import Card, CardGen, CardRepo
from config import Character
from game import GameState


CardFactory = Callable[[], Card]


def _get_card_catalog() -> dict[str, CardFactory]:
    catalog: dict[str, CardFactory] = {}
    for name in dir(CardGen):
        if name.startswith("_"):
            continue
        factory = getattr(CardGen, name)
        if not callable(factory):
            continue
        try:
            factory()
        except Exception as error:
            print(f"Skipping unavailable card {name}: {error}")
            continue
        catalog[name.lower()] = factory
    return dict(sorted(catalog.items()))


def _print_deck(deck: list[Card]) -> None:
    print("\nCurrent deck:")
    if not deck:
        print("  -empty-")
        return
    for index, card in enumerate(deck):
        print(f"  {index}: {card.get_name()} ({card.card_type.name}, cost {card.mana_cost.peek()})")


def _print_catalog(catalog: dict[str, CardFactory]) -> None:
    print("\nAvailable cards:")
    for index, name in enumerate(catalog):
        card = catalog[name]()
        print(f"  {index}: {name} -> {card.get_name()} ({card.card_type.name}, cost {card.mana_cost.peek()})")


def _choose_card_factory(catalog: dict[str, CardFactory]) -> CardFactory | None:
    _print_catalog(catalog)
    value = input("Card name or index to add, blank to cancel: ").strip().lower()
    if value == "":
        return None
    if value.isdigit():
        index = int(value)
        names = list(catalog)
        if 0 <= index < len(names):
            return catalog[names[index]]
        print("Invalid card index.")
        return None
    if value in catalog:
        return catalog[value]
    print("Invalid card name.")
    return None


def _choose_deck_index(deck: list[Card], action_name: str) -> int | None:
    _print_deck(deck)
    value = input(f"Card index to {action_name}, blank to cancel: ").strip()
    if value == "":
        return None
    if not value.isdigit():
        print("Please enter an integer index.")
        return None
    index = int(value)
    if not 0 <= index < len(deck):
        print("Invalid deck index.")
        return None
    return index


def _load_scenario(deck: list[Card]) -> list[Card]:
    scenarios = [
        CardRepo.get_scenario_0,
        CardRepo.get_scenario_1,
        CardRepo.get_scenario_2,
        CardRepo.get_scenario_3,
        CardRepo.get_scenario_4,
        CardRepo.get_scenario_5,
    ]
    print("\nScenarios:")
    for index, scenario in enumerate(scenarios):
        name, cards = scenario()
        print(f"  {index}: {name} ({len(cards)} cards)")
    value = input("Scenario index, blank to cancel: ").strip()
    if value == "":
        return deck
    if not value.isdigit():
        print("Please enter an integer index.")
        return deck
    index = int(value)
    if not 0 <= index < len(scenarios):
        print("Invalid scenario index.")
        return deck
    _, cards = scenarios[index]()
    return cards


def build_deck(game_state: GameState) -> None:
    catalog = _get_card_catalog()
    deck = list(game_state.deck)

    while True:
        _print_deck(deck)
        print(
            "\nDeck builder commands:\n"
            "  a: add card\n"
            "  r: remove card\n"
            "  u: upgrade card\n"
            "  s: load scenario\n"
            "  i: reset to IronClad starter\n"
            "  d: done"
        )
        command = input("Choose command: ").strip().lower()

        if command in ("d", "done", ""):
            game_state.set_deck(*deck)
            return
        if command in ("a", "add"):
            factory = _choose_card_factory(catalog)
            if factory is not None:
                deck.append(factory())
            continue
        if command in ("r", "remove"):
            index = _choose_deck_index(deck, "remove")
            if index is not None:
                del deck[index]
            continue
        if command in ("u", "upgrade"):
            index = _choose_deck_index(deck, "upgrade")
            if index is not None:
                deck[index].upgrade()
            continue
        if command in ("s", "scenario"):
            deck = _load_scenario(deck)
            continue
        if command in ("i", "starter"):
            deck = CardRepo.get_starter(Character.IRON_CLAD)
            continue

        print("Unknown command.")
