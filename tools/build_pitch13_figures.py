"""Draw the two Task 1.3 pitch figures from saved simulation output.

    .venv313/bin/python tools/build_pitch13_figures.py

figures/pitch13_ladder.png  preparation error for four starts (two-mode model)
figures/pitch13_push.png    error vs push duration, pump off vs pump on

Every plotted value and every label number is read from results/task1/*.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.task1_deck.claims import study_row  # noqa: E402

FIGURES = ROOT / "figures"
INK, MUTED, RULE, ACCENT = "#0b0b0b", "#8a8580", "#e4e0dc", "#ccd9f5"
DURATIONS_US = (0.05, 0.2, 1.0)

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 22,
    "axes.edgecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "savefig.dpi": 120,
})


def sci(x: float) -> str:
    mantissa, exponent = f"{x:.1e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def claim(key: str) -> float:
    claims = json.loads((ROOT / "results" / "task1" / "claims.json").read_text())["claims"]
    return claims[key]["value"]


def ladder() -> None:
    def base(start: str, beta: float) -> float:
        row = {"start": start, "beta": beta, "model": "model"}
        return study_row("baseline_compare", row)["error"]

    rows = [
        ("Start in vacuum", base("vacuum", 0.0), INK),
        ("Push to β = 1.5", base("A_1.5", 1.5), INK),
        ("Best ramped push", claim("robust_c_before"), INK),
        ("Push to β = α", base("D_alpha", 2.1), ACCENT),
    ]
    fig, ax = plt.subplots(figsize=(14, 6.2))
    for y, (_, err, color) in enumerate(reversed(rows)):
        ax.barh(y, err, color=color, height=0.56, left=1e-7)
        ax.text(err * 1.6, y, f"{err:.0%}" if err > 0.1 else sci(err), va="center",
                fontsize=26, fontweight="bold", color=color)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=26)
    ax.get_yticklabels()[0].set_color(ACCENT)
    ax.set_xscale("log")
    ax.set_xlim(1e-7, 30)
    ax.set_xlabel("preparation error, 1 − F(|+α⟩)", color=MUTED, labelpad=12)
    ax.tick_params(axis="y", length=0, pad=14)
    ax.grid(axis="x", color=RULE, lw=1)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "pitch13_ladder.png", transparent=False, facecolor="white")
    plt.close(fig)


def push() -> None:
    def err(kind: str, duration: float) -> float:
        return study_row("finite_push", {"kind": kind, "duration": duration})["error"]

    ideal = study_row("finite_push", {"kind": "ideal"})["error"]
    ns = [1000 * d for d in DURATIONS_US]
    on = [err("pump_on", d) for d in DURATIONS_US]
    off = [err("pump_off", d) for d in DURATIONS_US]
    fig, ax = plt.subplots(figsize=(14, 6.2))
    ax.axhline(ideal, color=MUTED, lw=1.5, ls=(0, (4, 4)))
    ax.plot(ns, on, "-o", color=INK, lw=4, ms=14)
    ax.plot(ns, off, "-o", color=ACCENT, lw=4, ms=14)
    ax.text(ns[-1] * 1.12, on[-1], "pump on\nduring push", va="center", fontsize=24, color=INK,
            fontweight="bold")
    ax.text(ns[-1] * 1.12, off[-1], "pump off\nduring push", va="center", fontsize=24,
            color=ACCENT, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns, [f"{n:.0f} ns" for n in ns])
    ax.minorticks_off()
    ax.set_xlim(ns[0] / 1.5, ns[-1] * 3.2)
    ax.set_ylim(ideal / 4, 2)
    ax.set_xlabel("push duration", color=MUTED, labelpad=12)
    ax.set_ylabel("preparation error", color=MUTED, labelpad=12)
    ax.grid(axis="y", color=RULE, lw=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURES / "pitch13_push.png", transparent=False, facecolor="white")
    plt.close(fig)


def main() -> None:
    ladder()
    push()
    print("wrote figures/pitch13_ladder.png, figures/pitch13_push.png")


if __name__ == "__main__":
    main()
