"""Render numbers for the Track 1 deck straight from results/task1/claims.json.

Every number on a slide goes through one of these helpers, so it carries the
key, the notebook cell and the exact printed line it came from. The verifier
(tools/verify_task1_deck.py) parses the displayed text back and checks it.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLAIMS_PATH = ROOT / "results" / "task1" / "claims.json"
TESTS_PATH = ROOT / "results" / "task1" / "pytest_summary.txt"

_SCI = re.compile(r"^([+-]?\d+(?:\.\d+)?)e([+-]?\d+)$")
_MINUS = "−"


def pretty_number(text: str, sign: bool = False) -> str:
    """Printed text -> slide text: 4.6e-02 -> 4.6 x 10^-2, '-' -> real minus."""
    text = text.strip()
    if not sign:
        text = text.lstrip("+")
    match = _SCI.match(text)
    if match:
        mantissa, exponent = match.group(1), int(match.group(2))
        mantissa = mantissa.replace("-", _MINUS)
        power = str(exponent).replace("-", _MINUS)
        return f"{mantissa}&#8239;&times;&#8239;10<sup>{power}</sup>"
    return text.replace("-", _MINUS)


def round_text(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def sci_text(value: float, digits: int) -> str:
    return f"{value:.{digits}e}"


@dataclass(frozen=True)
class Claims:
    claims: dict
    derived: dict
    inputs: dict
    source: dict
    tests_line: str

    @classmethod
    def load(cls) -> Claims:
        payload = json.loads(CLAIMS_PATH.read_text())
        tests_line = TESTS_PATH.read_text().strip().splitlines()[-1]
        return cls(payload["claims"], payload["derived"], payload["inputs"], payload["source"], tests_line)

    def v(self, key: str, sign: bool = False) -> str:
        """A simulation result exactly as the notebook printed it."""
        claim = self.claims[key]
        line = html.escape(claim["line"], quote=True)
        return (f'<span class="num" data-claim="{html.escape(key, quote=True)}" '
                f'data-cell="{claim["cell"]}" data-line="{line}">'
                f'{pretty_number(claim["text"], sign)}</span>')

    def d(self, key: str, digits: int, sci: bool = False) -> str:
        """Arithmetic on results (ratios, percentages), rounded for display."""
        item = self.derived[key]
        text = sci_text(item["value"], digits) if sci else round_text(item["value"], digits)
        formula = html.escape(item["formula"], quote=True)
        return (f'<span class="num" data-derived="{html.escape(key, quote=True)}" '
                f'title="derived: {formula}">{pretty_number(text)}</span>')

    def i(self, key: str, sci: bool = False, digits: int = 1) -> str:
        """A chosen simulation setting, with the source line that sets it."""
        item = self.inputs[key]
        value = item["value"]
        text = sci_text(value, digits) if sci else f"{value:g}"
        source = html.escape(item["source"], quote=True)
        return (f'<span class="num" data-input="{html.escape(key, quote=True)}" '
                f'title="setting: {source}">{pretty_number(text)}</span>')

    def r(self, study: str, digits: int = 1, **match) -> str:
        """One row of a study output, results/task1/<study>.json, matched on
        exactly these keys (rows are sparse: a key left out means its default)."""
        error = study_row(study, match)["error"]
        text = pretty_number(sci_text(error, digits))
        where = html.escape(json.dumps(match, sort_keys=True), quote=True)
        return (f'<span class="num" data-study="{study}" data-match="{where}" '
                f'title="results/task1/{study}.json">{text}</span>')

    def count(self) -> str:
        """Number of printed results in claims.json."""
        return f'<span class="num" data-meta="claim_count">{len(self.claims)}</span>'

    def tests(self) -> str:
        """Passed-test count from the saved pytest summary line."""
        passed = re.search(r"(\d+) passed", self.tests_line)
        if passed is None:
            raise ValueError(f"no pass count in {TESTS_PATH}: {self.tests_line!r}")
        line = html.escape(self.tests_line, quote=True)
        return f'<span class="num" data-meta="tests_passed" title="{line}">{passed.group(1)}</span>'

    def lib(self, name: str) -> str:
        return f'<span class="ref">{name} {self.source["libraries"][name]}</span>'


def study_row(study: str, match: dict) -> dict:
    """The single row of results/task1/<study>.json whose keys, apart from
    'error', are exactly `match`."""
    rows = json.loads((ROOT / "results" / "task1" / f"{study}.json").read_text())["rows"]
    found = [row for row in rows
             if set(row) - {"error"} == set(match) and all(row[k] == v for k, v in match.items())]
    if len(found) != 1:
        raise ValueError(f"{study}.json: {match} matched {len(found)} rows")
    return found[0]


def setting(text: str) -> str:
    """A label that names a sweep point or grid size (part of the experiment
    design, visible in tools/build_notebook.py), not a measured result."""
    return f'<span class="set">{text}</span>'


def math(text: str) -> str:
    """Formula text whose digits are mathematical constants, not data."""
    return f'<span class="math">{text}</span>'
