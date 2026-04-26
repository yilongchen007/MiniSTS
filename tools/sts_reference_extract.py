from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_JAR = (
    Path.home()
    / "Library/Application Support/Steam/steamapps/common/SlayTheSpire"
    / "SlayTheSpire.app/Contents/Resources/desktop-1.0.jar"
)


INT_OPS = {
    "iconst_m1": -1,
    "iconst_0": 0,
    "iconst_1": 1,
    "iconst_2": 2,
    "iconst_3": 3,
    "iconst_4": 4,
    "iconst_5": 5,
}


def run(command: list[str]) -> str:
    return subprocess.check_output(command, text=True)


def list_card_classes(jar_path: Path, colors: set[str]) -> list[str]:
    entries = run(["jar", "tf", str(jar_path)]).splitlines()
    classes: list[str] = []
    for entry in entries:
        match = re.fullmatch(r"com/megacrit/cardcrawl/cards/([^/]+)/([^/$]+)\.class", entry)
        if not match:
            continue
        color, _ = match.groups()
        if color in colors:
            classes.append(entry[:-6].replace("/", "."))
    return sorted(classes)


def parse_int_instruction(line: str) -> int | None:
    text = line.strip()
    for op, value in INT_OPS.items():
        if re.search(rf"\b{op}\b", text):
            return value
    match = re.search(r"\b(?:bipush|sipush)\s+(-?\d+)\b", text)
    if match:
        return int(match.group(1))
    match = re.search(r"// int (-?\d+)", text)
    if match:
        return int(match.group(1))
    return None


def previous_int(lines: list[str], index: int) -> int | None:
    for line in reversed(lines[:index]):
        value = parse_int_instruction(line)
        if value is not None:
            return value
    return None


def parse_method(lines: list[str], method_marker: str) -> list[str]:
    for index, line in enumerate(lines):
        if method_marker in line:
            for start in range(index + 1, len(lines)):
                if lines[start].strip() == "Code:":
                    break
            else:
                return []
            end = len(lines)
            for cursor in range(start + 1, len(lines)):
                stripped = lines[cursor].strip()
                if stripped.startswith("public ") or stripped.startswith("private ") or stripped.startswith("protected ") or stripped.startswith("static {}"):
                    end = cursor
                    break
            return lines[start + 1 : end]
    return []


def parse_constructor(lines: list[str], fqcn: str) -> dict[str, Any]:
    short_name = fqcn.rsplit(".", 1)[-1]
    body = parse_method(lines, f"public {fqcn}();")
    data: dict[str, Any] = {
        "class": fqcn,
        "class_name": short_name,
        "id": None,
        "asset": None,
        "cost": None,
        "type": None,
        "color": None,
        "rarity": None,
        "target": None,
        "base_fields": {},
    }

    for index, line in enumerate(body):
        string_match = re.search(r"// String (.+)$", line)
        if string_match:
            value = string_match.group(1)
            if data["id"] is None:
                data["id"] = value
            elif "/" in value and data["asset"] is None:
                data["asset"] = value
                for next_line in body[index + 1 :]:
                    cost = parse_int_instruction(next_line)
                    if cost is not None:
                        data["cost"] = cost
                        break

        enum_match = re.search(r"AbstractCard\$Card(Type|Color|Rarity|Target)\.([A-Z_]+)", line)
        if enum_match:
            field = enum_match.group(1).lower()
            data[field] = enum_match.group(2)

        field_match = re.search(r"// Field (base[A-Za-z0-9_]+):I", line)
        if field_match and "putfield" in line:
            value = previous_int(body, index)
            if value is not None:
                data["base_fields"][field_match.group(1)] = value

    return data


def parse_upgrade(lines: list[str]) -> list[dict[str, Any]]:
    body = parse_method(lines, "public void upgrade();")
    upgrades: list[dict[str, Any]] = []
    for index, line in enumerate(body):
        match = re.search(r"// Method (upgrade[A-Za-z0-9_]+):", line)
        if not match:
            continue
        method = match.group(1)
        if method == "upgradeName":
            continue
        upgrades.append({"method": method, "amount": previous_int(body, index)})
    return upgrades


def parse_references(lines: list[str]) -> dict[str, list[str]]:
    body = parse_method(lines, "public void use(")
    actions: set[str] = set()
    powers: set[str] = set()
    for line in body:
        match = re.search(r"// class (com/megacrit/cardcrawl/[^ ]+)", line)
        if not match:
            continue
        fqcn = match.group(1).replace("/", ".")
        if ".actions." in fqcn:
            actions.add(fqcn)
        elif ".powers." in fqcn:
            powers.add(fqcn)
    return {"actions": sorted(actions), "powers": sorted(powers)}


def parse_card(jar_path: Path, fqcn: str) -> dict[str, Any]:
    output = run(["javap", "-classpath", str(jar_path), "-c", "-p", fqcn])
    lines = output.splitlines()
    data = parse_constructor(lines, fqcn)
    data["upgrade"] = parse_upgrade(lines)
    data["use_refs"] = parse_references(lines)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jar", type=Path, default=DEFAULT_JAR)
    parser.add_argument("--colors", nargs="+", default=["red"], help="Card packages, e.g. red green blue purple colorless curses status.")
    parser.add_argument("--out", type=Path, default=Path("local_refs/sts_cards_red_summary.json"))
    args = parser.parse_args()

    classes = list_card_classes(args.jar, set(args.colors))
    cards = [parse_card(args.jar, fqcn) for fqcn in classes]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cards, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(cards)} card summaries to {args.out}")


if __name__ == "__main__":
    main()
