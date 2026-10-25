"""Build the five-slide Task 1.3 pitch deck.

    .venv313/bin/python tools/build_pitch13_figures.py
    .venv313/bin/python tools/build_pitch13_deck.py
    .venv313/bin/python tools/verify_task1_deck.py pitch13_deck.html

Writes pitch13_deck.html and pitch13-assets/. The full 23-slide deck
(task1_deck.html) stays the backup for questions.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_task1_deck import write_deck  # noqa: E402
from tools.task1_deck import slides_pitch13  # noqa: E402


def main() -> None:
    write_deck(ROOT / "pitch13_deck.html", "Cat Qubit 1.3 Pitch", slides_pitch13.SLIDES,
               slides_pitch13.ASSET_DIR, slides_pitch13.EXTRA_CSS)


if __name__ == "__main__":
    main()
