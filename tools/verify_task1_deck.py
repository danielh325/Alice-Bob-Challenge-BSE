"""Check every number on task1_deck.html against results/task1/claims.json.

    .venv313/bin/python tools/verify_task1_deck.py [DECK.html]

1. Each tagged number (data-claim / data-derived / data-input / data-meta) must
   parse back to the value in claims.json, to the precision shown.
   data-study spans name a study output, results/task1/<study>.json, and the
   exact row (data-match) whose error they show.
2. data-claim spans must also carry the notebook cell and printed line.
3. Any digit in visible slide text outside a tagged span is reported as
   untraced. Exempt spans: .set (sweep labels), .math (formula constants),
   .ref (file names, versions), .pg (page numbers), .tag (section numbers),
   .sec (footer section name), .idx (list ordinals 01, 02, ...).
Exit status 1 if anything fails.
"""

from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.task1_deck.claims import study_row  # noqa: E402

DECK = ROOT / "task1_deck.html"
CLAIMS = ROOT / "results" / "task1" / "claims.json"
TESTS = ROOT / "results" / "task1" / "pytest_summary.txt"

EXEMPT_CLASSES = {"set", "math", "ref", "pg", "tag", "sec", "idx"}
SKIP_TAGS = {"script", "style", "aside", "title", "head"}
VOID_TAGS = {"img", "br", "meta", "link", "hr", "input"}
DIGIT = re.compile(r"\d")
TAG_ATTRS = ("data-claim", "data-derived", "data-input", "data-meta", "data-study")


class DeckParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, dict, bool]] = []  # (tag, attrs, exempt)
        self.numbers: list[dict] = []  # tagged spans with collected text
        self.loose: list[tuple[int, str]] = []  # (slide index, text) with untraced digits
        self.slide = 0
        self._open_numbers: list[dict] = []

    def _exempt(self) -> bool:
        return any(exempt for _, _, exempt in self.stack)

    def handle_starttag(self, tag: str, attrs_list: list) -> None:
        if tag in VOID_TAGS:
            return
        attrs = dict(attrs_list)
        classes = set((attrs.get("class") or "").split())
        if tag == "section" and "slide" in classes:
            self.slide += 1
        tagged = any(k in attrs for k in TAG_ATTRS)
        exempt = tag in SKIP_TAGS or bool(classes & EXEMPT_CLASSES) or tagged
        if tagged:
            record = {"attrs": attrs, "text": "", "slide": self.slide}
            self.numbers.append(record)
            self._open_numbers.append(record)
        self.stack.append((tag, attrs, exempt))

    def handle_endtag(self, tag: str) -> None:
        while self.stack:
            open_tag, attrs, _ = self.stack.pop()
            if any(k in attrs for k in TAG_ATTRS):
                self._open_numbers.pop()
            if open_tag == tag:
                break

    def handle_data(self, data: str) -> None:
        for record in self._open_numbers:
            record["text"] += data
        if self.slide and not self._exempt() and DIGIT.search(data):
            self.loose.append((self.slide, data.strip()))


def parse_shown(text: str) -> tuple[float, float]:
    """Displayed text -> (value, half of the last shown digit)."""
    t = text.replace("−", "-").replace(" ", " ").replace("×", "x").strip()
    m = re.fullmatch(r"([+-]?\d+(?:\.(\d+))?)(?:\s*x\s*10([+-]?\d+))?", t)
    if not m:
        raise ValueError(f"cannot parse displayed number {text!r}")
    decimals = len(m.group(2) or "")
    exponent = int(m.group(3)) if m.group(3) else 0
    value = float(m.group(1)) * 10.0 ** exponent
    return value, 0.5 * 10.0 ** (exponent - decimals)


def check_number(record: dict, payload: dict, tests_line: str) -> str | None:
    a, shown = record["attrs"], record["text"]
    if "data-meta" in a:
        expected = {"claim_count": str(len(payload["claims"])),
                    "tests_passed": re.search(r"(\d+) passed", tests_line).group(1)}[a["data-meta"]]
        if shown.strip() == expected:
            return None
        return f"{a['data-meta']}: shows {shown!r}, expected {expected}"
    if "data-claim" in a:
        key = a["data-claim"]
        claim = payload["claims"].get(key)
        if claim is None:
            return f"claim {key!r} not in claims.json"
        if str(claim["cell"]) != a.get("data-cell") or claim["line"] != a.get("data-line"):
            return f"claim {key!r}: cell/line attributes do not match claims.json"
        expected = claim["value"]
    elif "data-derived" in a:
        key = a["data-derived"]
        if key not in payload["derived"]:
            return f"derived {key!r} not in claims.json"
        expected = payload["derived"][key]["value"]
    elif "data-study" in a:
        key = f"{a['data-study']}{a['data-match']}"
        try:
            expected = study_row(a["data-study"], json.loads(a["data-match"]))["error"]
        except ValueError as err:
            return str(err)
    else:
        key = a["data-input"]
        if key not in payload["inputs"]:
            return f"input {key!r} not in claims.json"
        expected = payload["inputs"][key]["value"]
    value, half_ulp = parse_shown(shown)
    if abs(value - expected) > half_ulp * (1 + 1e-9):
        return f"{key!r}: shows {shown!r} = {value:g}, data says {expected:g}"
    return None


def main() -> int:
    payload = json.loads(CLAIMS.read_text())
    tests_line = TESTS.read_text().strip().splitlines()[-1]
    parser = DeckParser()
    deck = Path(sys.argv[1]) if len(sys.argv) > 1 else DECK
    parser.feed(deck.read_text())
    failures = [f"slide {r['slide']}: {msg}" for r in parser.numbers
                if (msg := check_number(r, payload, tests_line))]
    failures += [f"slide {s}: untraced digits in {text!r}" for s, text in parser.loose]
    kinds = {k: sum(k in r["attrs"] for r in parser.numbers)
             for k in TAG_ATTRS}
    print(f"{parser.slide} slides, {len(parser.numbers)} tagged numbers "
          f"({', '.join(f'{k[5:]} {n}' for k, n in kinds.items())})")
    for line in failures:
        print("FAIL", line)
    ok = "OK: every number traces to a saved simulation output"
    print(ok if not failures else f"{len(failures)} problems")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
