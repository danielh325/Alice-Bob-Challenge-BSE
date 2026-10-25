"""HTML building blocks for the Track 1 deck (Swiss Modern layout)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

ASSET_DIR = "task1_deck-assets"


_TASK_CODE = re.compile(r"^(\d\.\d(?:\([a-c]\))?)")


def header(tag: str, title: str) -> str:
    tag = _TASK_CODE.sub(r'<span class="code">\1</span>', tag)
    return (f'<header class="hd reveal d1"><div class="tag">{tag}</div>'
            f'<h2>{title}</h2></header>')


def footer(section: str, cells: Sequence[int], extra: str = "") -> str:
    """Footer names the notebook cells whose printed output the slide uses."""
    if cells:
        listed = ", ".join(str(c) for c in cells)
        src = f'Numbers: intro_track.ipynb cell{"s" if len(cells) > 1 else ""} {listed}{extra}'
    else:
        src = extra
    return (f'<footer class="ft"><span class="sec">{section}</span>'
            f'<span class="src ref">{src}</span><span class="pg">{{page}}</span></footer>')


def figure(name: str, alt: str, caption: str, width: int, size: tuple[int, int], delay: int = 3,
           asset_dir: str = ASSET_DIR, ext: str = "png") -> str:
    """A notebook figure; size is the image's pixel size, used for the aspect ratio."""
    height = round(width * size[1] / size[0])
    return (f'<figure class="plot reveal d{delay}" style="width:{width}px">'
            f'<img src="{asset_dir}/{name}.{ext}" alt="{alt}" width="{width}" height="{height}">'
            f'<figcaption>{caption}</figcaption></figure>')


def stat(number: str, label: str, unit: str = "", hot: bool = False) -> str:
    unit_html = f"<small>{unit}</small>" if unit else ""
    return (f'<div class="stat{" hot" if hot else ""}"><div class="n">{number}{unit_html}</div>'
            f'<div class="l">{label}</div></div>')


def stats(items: Iterable[str], columns: int = 1, delay: int = 2, width: int | None = None) -> str:
    style = f"grid-template-columns:repeat({columns},1fr)"
    if width:
        style += f";width:{width}px"
    return f'<div class="stats reveal d{delay}" style="{style}">{"".join(items)}</div>'


def table(head: Sequence[str], rows: Sequence[Sequence[str]], right_from: int = 1,
          highlight: Sequence[int] = (), compact: bool = False, delay: int = 3) -> str:
    """Columns from `right_from` on are right-aligned numbers."""
    def cell(tag: str, k: int, text: str) -> str:
        return f'<{tag} class="r">{text}</{tag}>' if k >= right_from else f"<{tag}>{text}</{tag}>"

    head_html = "".join(cell("th", k, h) for k, h in enumerate(head))
    body = []
    for r, row in enumerate(rows):
        cls = ' class="hi"' if r in highlight else ""
        body.append(f"<tr{cls}>" + "".join(cell("td", k, t) for k, t in enumerate(row)) + "</tr>")
    return (f'<table class="tbl{" compact" if compact else ""} reveal d{delay}">'
            f'<thead><tr>{head_html}</tr></thead><tbody>{"".join(body)}</tbody></table>')


def cols(template: str, *blocks: str) -> str:
    return f'<div class="cols" style="grid-template-columns:{template}">{"".join(blocks)}</div>'


def stack(*blocks: str, delay: int = 2) -> str:
    return f'<div class="stack reveal d{delay}">{"".join(blocks)}</div>'


def lede(text: str) -> str:
    return f'<p class="lede">{text}</p>'


def note(text: str) -> str:
    return f'<p class="note">{text}</p>'


def kicker(text: str) -> str:
    return f'<p class="kicker">{text}</p>'


def slide(label: str, content: str, notes: str = "", extra_class: str = "") -> str:
    notes_html = f'<aside class="notes" hidden>{notes}</aside>' if notes else ""
    return (f'<section class="slide {extra_class}" aria-label="{label}">'
            f"{content}{notes_html}</section>")


def body(content: str) -> str:
    return f'<div class="body">{content}</div>'
