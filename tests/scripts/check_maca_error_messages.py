from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MACA_COMMON = ROOT / "src" / "runtime" / "maca" / "maca_common.h"


def main() -> None:
    text = MACA_COMMON.read_text(encoding="utf-8")
    if "MACA MACA" in text:
        raise SystemExit("stale duplicated MACA error wording found")
    for expected in ("MACA driver call failed", "MACA runtime call failed"):
        if expected not in text:
            raise SystemExit(f"missing expected error wording: {expected}")


if __name__ == "__main__":
    main()
