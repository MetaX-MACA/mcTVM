#!/usr/bin/env python3
"""Audit CMakeCache.txt for required MACA build configuration keys."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REQUIRED = ['MACA_HOME', 'USE_MACA', 'CMAKE_CXX_COMPILER']


def parse_cache(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("#") or "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        key = key_type.split(":", 1)[0].strip()
        values[key] = value.strip()
    return values


def audit(path: Path) -> dict[str, object]:
    values = parse_cache(path)
    missing = [key for key in REQUIRED if not values.get(key)]
    return {"ok": not missing, "missing": missing, "values": {key: values.get(key, "") for key in REQUIRED}}


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        sample = Path(tmp) / "CMakeCache.txt"
        sample.write_text(" MACA_HOME:PATH = /opt/maca \n", encoding="utf-8")
        data = audit(sample)
        if "missing" not in data or data["values"]["MACA_HOME"] != "/opt/maca":
            raise RuntimeError("self-test failed: cache parser did not normalize values")
        print(json.dumps({"ok": True, "missing": len(data["missing"])}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", nargs="?", default="CMakeCache.txt")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    cache = Path(args.cache)
    if not cache.is_file():
        print(f"CMake cache file not found: {cache}", file=sys.stderr)
        return 2
    print(json.dumps(audit(cache), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
