"""Build the Track 1 slide deck from results/task1/claims.json.

    .venv313/bin/python tools/build_task1_deck.py

Writes task1_deck.html and copies the notebook figures it uses into
task1_deck-assets/. Every number on a slide is rendered by a helper in
tools/task1_deck/claims.py; run tools/verify_task1_deck.py afterwards.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.task1_deck import slides_cat, slides_intro  # noqa: E402
from tools.task1_deck.claims import Claims  # noqa: E402
from tools.task1_deck.layout import ASSET_DIR  # noqa: E402

DECK_DIR = ROOT / "tools" / "task1_deck"
OUT = ROOT / "task1_deck.html"
FIGURES = ROOT / "figures"
VIEWPORT_CSS = DECK_DIR / "viewport-base.css"
FONTS = ("https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800"
         "&family=Nunito:wght@400;600;700&display=swap")

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{fonts}" rel="stylesheet">
<style>
/* === VIEWPORT BASE (frontend-slides, included in full) === */
{viewport}
/* === DECK STYLE (Swiss Modern) === */
{style}
</style>
</head>
<body>
<div class="edit-hotzone" aria-hidden="true"></div>
<button id="editToggle" type="button" aria-label="Toggle text editing">Edit</button>
<div class="progress" aria-hidden="true"></div>
<main class="deck-viewport">
<div class="deck-stage">
{slides}
</div>
</main>
<script>
{script}
</script>
</body>
</html>
"""


def render_slides(claims: Claims, builders: list) -> list[str]:
    total = len(builders)
    pages = []
    for number, build in enumerate(builders, start=1):
        html = build(claims)
        if "{page}" not in html:
            raise ValueError(f"slide {build.__name__} has no footer page slot")
        pages.append(html.replace("{page}", f"{number:02d} / {total:02d}"))
    return pages


def copy_figures(pages: list[str], asset_dir: str = ASSET_DIR) -> list[str]:
    """Copy each referenced figure once; a figure used twice is an error."""
    names = []
    for page in pages:
        for chunk in page.split(f'src="{asset_dir}/')[1:]:
            names.append(chunk.split('"', 1)[0])
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"figures used on more than one slide: {repeated}")
    assets = ROOT / asset_dir
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir()
    for name in names:
        source = FIGURES / name
        if not source.exists():
            raise FileNotFoundError(f"figure not produced by the notebook: {source}")
        shutil.copy2(source, assets / name)
    return names


def write_deck(out: Path, title: str, builders: list, asset_dir: str, extra_css: str = "") -> None:
    """Render the slides, copy their figures into asset_dir and write one HTML file."""
    pages = render_slides(Claims.load(), builders)
    names = copy_figures(pages, asset_dir)
    html = PAGE.format(
        title=title,
        fonts=FONTS,
        viewport=VIEWPORT_CSS.read_text(),
        style=(DECK_DIR / "style.css").read_text() + extra_css,
        slides="\n".join(pages),
        script=(DECK_DIR / "deck.js").read_text(),
    )
    out.write_text(html)
    print(f"{len(pages)} slides, {len(names)} figures -> {out.relative_to(ROOT)}, {asset_dir}/")


def main() -> None:
    write_deck(OUT, "Cat Qubit Track 1", [*slides_intro.SLIDES, *slides_cat.SLIDES], ASSET_DIR)


if __name__ == "__main__":
    main()
