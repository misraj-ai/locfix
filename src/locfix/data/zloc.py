"""The localization z: a ranked list of code locations, serialized as the localizer's answer.

    {"locations": [{"path": "arrow/factory.py", "symbol": "ArrowFactory.get",
                    "lines": [90, 186], "role": "edit"}, ...]}

Rules (spec 04 §3): at most `max_locations` entries; edit locations always included in full; the
code of all spans together within `ctx_budget_tokens`.
"""
from __future__ import annotations

import json
from collections.abc import Callable

from .locations import Location, normalize_path


def z_to_json(locs: list[Location]) -> str:
    return json.dumps({"locations": [{"path": loc.path, "symbol": loc.symbol,
                                      "lines": [loc.start, loc.end], "role": loc.role}
                                     for loc in locs]}, ensure_ascii=False)


def parse_z(text: str) -> list[Location]:
    """Parse a localizer answer. Raises ValueError on anything but the documented JSON."""
    obj = json.loads(text)
    if not isinstance(obj, dict) or not isinstance(obj.get("locations"), list):
        raise ValueError("z must be an object with a 'locations' list")
    out = []
    for e in obj["locations"]:
        lines = e["lines"]
        if not (isinstance(lines, list) and len(lines) == 2):
            raise ValueError(f"bad lines in {e}")
        out.append(Location(normalize_path(str(e["path"])), str(e["symbol"]), int(lines[0]), int(lines[1]),
                            str(e.get("role", "context"))))
    return out


def key(loc: Location) -> tuple[str, str, int]:
    return (normalize_path(loc.path), loc.symbol, loc.start)


def union(*parts: list[Location]) -> list[Location]:
    """Ordered union by (path, symbol, start); the first occurrence wins, except that a location
    marked `edit` anywhere keeps role `edit`."""
    seen: dict[tuple, Location] = {}
    for part in parts:
        for loc in part:
            k = key(loc)
            if k not in seen:
                seen[k] = loc
            elif loc.role == "edit" and seen[k].role != "edit":
                seen[k] = Location(seen[k].path, seen[k].symbol, seen[k].start, seen[k].end, "edit")
    return list(seen.values())


def minus(a: list[Location], b: list[Location]) -> list[Location]:
    kb = {key(x) for x in b}
    return [x for x in a if key(x) not in kb]


def span_text(loc: Location, files: dict[str, str]) -> str:
    text = files.get(normalize_path(loc.path))
    if text is None:
        return ""
    lines = text.splitlines()
    if loc.symbol == "<file>":
        return "\n".join(lines)
    return "\n".join(lines[max(0, loc.start - 1): loc.end])


def chars_tokens(text: str) -> int:
    return len(text) // 4


def prune(locs: list[Location], files: dict[str, str], max_locations: int = 8,
          budget_tokens: int = 6000, count_tokens: Callable[[str], int] = chars_tokens) -> list[Location]:
    """Edit locations first (all kept, even beyond the limits), then context locations in order
    while both limits hold. Overlapping spans in the same file are counted once."""
    edit = [x for x in locs if x.role == "edit"]
    ctx = [x for x in locs if x.role != "edit"]
    kept = list(edit)
    covered: dict[str, set[int]] = {}

    def lines_of(loc: Location) -> set[int]:
        text = files.get(normalize_path(loc.path))
        if text is None:
            return set()
        n = len(text.splitlines())
        a, b = (1, n) if loc.symbol == "<file>" else (max(1, loc.start), min(loc.end, n))
        return set(range(a, b + 1))

    def cost(loc: Location) -> int:
        new = lines_of(loc) - covered.get(normalize_path(loc.path), set())
        if not new:
            return 0
        lines = files[normalize_path(loc.path)].splitlines()
        return count_tokens("\n".join(lines[i - 1] for i in sorted(new)))

    used = 0
    for loc in edit:
        used += cost(loc)
        covered.setdefault(normalize_path(loc.path), set()).update(lines_of(loc))
    for loc in ctx:
        if len(kept) >= max_locations:
            break
        c = cost(loc)
        if used + c > budget_tokens:
            continue
        kept.append(loc)
        used += c
        covered.setdefault(normalize_path(loc.path), set()).update(lines_of(loc))
    return kept
