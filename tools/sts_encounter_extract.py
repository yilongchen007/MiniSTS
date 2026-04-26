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

DUNGEON_CLASSES = {
    "exordium": "com.megacrit.cardcrawl.dungeons.Exordium",
    "city": "com.megacrit.cardcrawl.dungeons.TheCity",
    "beyond": "com.megacrit.cardcrawl.dungeons.TheBeyond",
}


def run(command: list[str]) -> str:
    return subprocess.check_output(command, text=True)


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
                if (
                    stripped.startswith("public ")
                    or stripped.startswith("private ")
                    or stripped.startswith("protected ")
                    or stripped.startswith("static {}")
                ):
                    end = cursor
                    break
            return lines[start + 1 : end]
    return []


def parse_float(line: str) -> float | None:
    text = line.strip()
    constants = {
        "fconst_0": 0.0,
        "fconst_1": 1.0,
        "fconst_2": 2.0,
    }
    for op, value in constants.items():
        if re.search(rf"\b{op}\b", text):
            return value
    match = re.search(r"// float (-?\d+(?:\.\d+)?)f?", text)
    if match:
        return float(match.group(1))
    return None


def next_float(lines: list[str], index: int) -> float | None:
    for line in lines[index + 1 : index + 8]:
        value = parse_float(line)
        if value is not None:
            return value
    return None


def parse_encounters(lines: list[str], method_marker: str) -> list[dict[str, Any]]:
    body = parse_method(lines, method_marker)
    encounters: list[dict[str, Any]] = []
    for index, line in enumerate(body):
        match = re.search(r"// String (.+)$", line)
        if not match:
            continue
        weight = next_float(body, index)
        if weight is not None:
            encounters.append({"name": match.group(1), "weight": weight})
    return encounters


def parse_bosses(lines: list[str]) -> list[str]:
    body = parse_method(lines, "protected void initializeBoss();")
    bosses: list[str] = []
    for index, line in enumerate(body):
        match = re.search(r"// String (.+)$", line)
        if not match:
            continue
        value = match.group(1)
        if value.isupper() or "empty" in value.lower():
            continue
        nearby = "\n".join(body[index + 1 : index + 8])
        if "java/util/ArrayList.add" in nearby and value not in bosses:
            bosses.append(value)
    return bosses


def parse_dungeon(jar_path: Path, dungeon: str) -> dict[str, Any]:
    fqcn = DUNGEON_CLASSES[dungeon]
    output = run(["javap", "-classpath", str(jar_path), "-c", "-p", fqcn])
    lines = output.splitlines()
    return {
        "dungeon": dungeon,
        "class": fqcn,
        "weak": parse_encounters(lines, "protected void generateWeakEnemies(int);"),
        "strong": parse_encounters(lines, "protected void generateStrongEnemies(int);"),
        "elite": parse_encounters(lines, "protected void generateElites(int);"),
        "bosses": parse_bosses(lines),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jar", type=Path, default=DEFAULT_JAR)
    parser.add_argument("--dungeons", nargs="+", choices=sorted(DUNGEON_CLASSES), default=["exordium"])
    parser.add_argument("--out", type=Path, default=Path("local_refs/sts_encounters_summary.json"))
    args = parser.parse_args()

    data = [parse_dungeon(args.jar, dungeon) for dungeon in args.dungeons]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(f"wrote encounter summaries for {len(data)} dungeon(s) to {args.out}")


if __name__ == "__main__":
    main()
