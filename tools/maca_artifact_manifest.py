#!/usr/bin/env python3
"""Build a reproducible artifact manifest with file sizes and hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PATTERNS = ['build/**/*.so', 'build/**/*.o', 'build/**/*.log', '*.txt']


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect(root: Path) -> dict[str, object]:
    seen: set[str] = set()
    artifacts: list[dict[str, object]] = []
    for pattern in PATTERNS:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            artifacts.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return {"root": str(root), "count": len(artifacts), "artifacts": artifacts}


def self_test() -> None:
    sample = Path("_artifact_manifest_sample.txt")
    sample.write_text("maca artifact\n", encoding="utf-8")
    try:
        data = collect(Path.cwd())
        assert any(item["path"] == sample.name for item in data["artifacts"])
        print(json.dumps({"ok": True, "count": data["count"]}, ensure_ascii=False))
    finally:
        sample.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    print(json.dumps(collect(Path(args.root)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
