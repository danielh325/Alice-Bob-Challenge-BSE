"""Eight-slide Task 1.3 pitch: one idea per slide, one figure or one row of numbers.

Numbers come from results/task1/claims.json (c.v) or from a study output row
(c.r); the two figures are drawn by tools/build_pitch13_figures.py from the
same files. The two GIFs are drawn from saved simulation states by
studies/task13_idea_animation.py and studies/task13_protocol_animation.py.
"""

from __future__ import annotations

import json

from tools.task1_deck.claims import ROOT, Claims, math, setting
from tools.task1_deck.layout import (
    body,
    cols,
    figure,
    footer,
    header,
    lede,
    slide,
    stack,
    stat,
    stats,
    table,
)

ASSET_DIR = "pitch13-assets"
CODEWORD = "|+α⟩"
PUSH_SHORT = {"kind": "pump_off", "duration": 0.05}
PUSH_LONG = {"kind": "pump_off", "duration": 1.0}
PUMP_ON_SHORT = {"kind": "pump_on", "duration": 0.05}
PUMP_ON_LONG = {"kind": "pump_on", "duration": 1.0}
# memory detuning during a pump-off push, as saved by studies/task13_push_realism.py
STARK_OFF = -json.loads((ROOT / "results/task1/push_realism.json").read_text())[
    "stark_a_rad_per_us"]["floquet"]


def even_cat(c: Claims) -> str:
    left = figure("1_3a_observables", "Parity, coherence, fidelities and purity versus time",
                  f"Two-mode model: {math('⟨P⟩')}, fidelity with {math('|C₊⟩')} and purity over time.",
                  700, (775, 542), delay=2, asset_dir=ASSET_DIR)
    right = stack(
        lede(f"Vacuum has {math('⟨P⟩ = +1')}. The drive, buffer loss and {math('a²b†')} change the memory "
             f"photon number by two {setting('(1.0(b))')}, so parity is conserved and "
             f"{math('|0⟩')} can only grow {math('|C₊⟩ ∝ |+α⟩ + |−α⟩')}."),
        stats([stat(c.v("parity_control"), f"drift of {math('⟨P⟩')} with {setting('κ<sub>a</sub> = 0')}", hot=True),
               stat(c.v("cat_t.1.fid_cat"), f"{math('F(C₊)')} at {setting('1 µs')}: memory loss sets in"),
               stat(c.v("cat_t.6.purity"), f"purity at {setting('6 µs')}: an equal mixture of the two codewords")],
              delay=3),
        delay=2)
    content = header("1.3(a)", "Vacuum can only become the <em>even cat</em>") + body(cols("700px 1fr", left, right))
    return slide("Even cat", content + footer("Task 1.3", [22]))


def superposition(c: Claims) -> str:
    content = header("1.3(a) <span class='dim'>· not a mixture</span>",
                     "A superposition, <em>not a mixture</em>") + body(stack(
        figure("1_3a_superposition_vs_mixture", "Wigner functions: superposition with fringes, mixture without",
               f"Same photon number, same overlap with {CODEWORD}. Only the superposition has fringes.",
               1500, (1766, 524), delay=2, asset_dir=ASSET_DIR),
        stats([stat(c.v("lifetime_us"), "superposition lifetime, from the fringe decay", unit="µs", hot=True),
               stat(c.v("coh_decay"), "fringe decay rate, measured", unit="/µs"),
               stat(c.v("gamma_pred"), f"predicted {math('2κ<sub>a</sub>α²')}", unit="/µs"),
               stat(c.v("liou_lifetime"), "independent check (Liouvillian gap)", unit="µs")],
             columns=4, delay=3),
        delay=2))
    return slide("Superposition", content + footer("Task 1.3", [22]))


def idea(c: Claims) -> str:
    animation = figure("pitch13_idea", "Animated Wigner functions: vacuum, displaced start and method D",
                       f"Simulated, stabilization on for {setting('2 µs')}: vacuum, a short push "
                       f"{math('β = 1.5')}, and a push to exactly {math('β = α')}.",
                       960, (1280, 720), delay=2, asset_dir=ASSET_DIR, ext="gif")
    right = stack(
        lede(f"Push the memory to exactly {math('β = α')} with the pump off. On {math('|α, 0⟩')} "
             f"the pair exchange cancels the buffer drive, so the codeword is a <b>dark state</b>:"),
        f'<p class="pitch-eq reveal d2">{math("g₂α² − ε<sub>d</sub> = 0")}</p>',
        stats([stat(c.v("d_err"), "error, two-mode model", hot=True),
               stat(c.v("d_err_lossless"), f"with {setting('κ<sub>a</sub> = 0')}: the error is all memory loss")],
              delay=3),
        delay=2)
    content = header("1.3(b) <span class='dim'>· the idea</span>",
                     "Don't roll into the valley. <em>Start at the bottom.</em>") + body(
        cols("960px 1fr", animation, right))
    return slide("Dark state", content + footer("Task 1.3", [43], " · studies/task13_idea_animation.py"))


def protocol(c: Claims) -> str:
    content = header("1.3(b) <span class='dim'>· the protocol</span>",
                     "Push with the pump off, <em>then</em> stabilize") + body(stack(
        figure("pitch13_protocol", "Animated protocol: stabilization off, microwave push, pump on, hold",
               f"Every frame is a saved two-mode state. Time axis stretched during the {setting('50 ns')} push.",
               1140, (1280, 720), delay=2, asset_dir=ASSET_DIR, ext="gif"),
        delay=2))
    return slide("Protocol", content + footer("Task 1.3", [], "studies/task13_protocol_animation.py"))


def ladder(c: Claims) -> str:
    content = header("1.3(b) <span class='dim'>· result</span>",
                     "Lowest error of every preparation we tested") + body(stack(
        figure("pitch13_ladder", "Preparation error on a log scale for four starts",
               f"{math(f'1 − F({CODEWORD})')} after {setting('2 µs')} of stabilization, two-mode model.",
               1060, (1680, 744), delay=2, asset_dir=ASSET_DIR),
        stats([stat(c.v("d_vs_c"), "lower than the best ramped push", unit="×", hot=True),
               stat(c.v("d_err_circuit"), "with real-circuit Kerr terms"),
               stat(c.v("d_vs_c_circuit"), "lower, with circuit terms", unit="×")],
             columns=3, delay=3),
        delay=2))
    return slide("Result", content + footer("Task 1.3", [40, 43],
                                            " · studies/task13_baseline_compare.py"))


def finite_push(c: Claims) -> str:
    content = header("1.3(b) <span class='dim'>· a real push</span>",
                     "Turn the pump <em>off</em> while you push") + body(cols(
        "1.45fr 1fr",
        figure("pitch13_push", "Preparation error against push duration, pump off and pump on",
               "Dashed line: the ideal, instantaneous push.", 1050, (1680, 744), delay=2,
               asset_dir=ASSET_DIR),
        stats([stat(c.r("finite_push", **PUSH_SHORT), f"pump off, {setting('50 ns')} push", hot=True),
               stat(c.r("finite_push", **PUSH_LONG), f"pump off, {setting('1 µs')} push"),
               stat(c.r("finite_push", **PUMP_ON_SHORT), f"pump on, {setting('50 ns')} push"),
               stat(c.r("finite_push", **PUMP_ON_LONG), f"pump on, {setting('1 µs')} push")],
              delay=3)))
    return slide("Finite push", content + footer("Task 1.3", [], "Numbers: studies/task13_finite_push.py"))


def hardware(c: Claims) -> str:
    def realism(**match) -> str:
        return c.r("push_realism", **match)

    shift = {"effect": "detuning", "duration": 0.2, "detuning": STARK_OFF}
    smooth = {"effect": "lowpass", "tau": 0.01}
    rows = [
        [f"Memory frequency shift with the pump off, {setting('200 ns')} push",
         realism(**shift, calibration="none"), realism(**shift, calibration="second_order"),
         "calibrate the push frequency"],
        [f"Pulse smoothing in the wiring, {setting('τ = 10 ns')}",
         realism(**smooth, guard=0.0), realism(**smooth, guard=0.05),
         f"switch the pump on {setting('5τ')} later"],
        [f"Reflected echo, {setting('10%')} arriving {setting('20 ns')} later",
         realism(effect="echo", echo=[0.1, 0.02], guard=0.0),
         realism(effect="echo", echo=[0.1, 0.02], guard=0.02),
         "the same delayed pump switch"],
    ]
    combined = {"effect": "combined", "extra": 0.1, "n_th": 0.001, "circuit": True}
    content = header("1.3(c) <span class='dim'>· real hardware</span>",
                     "It survives the effects the ideal model <em>leaves out</em>") + body(stack(
        lede(f"Simulated in the same two-mode model: {math(f'1 − F({CODEWORD})')} after "
             f"{setting('2 µs')}. Each effect costs one calibration step."),
        cols("1.35fr 1fr",
             table(["effect", "ignored", "handled", "fix"], rows, right_from=1, delay=3),
             stats([stat(realism(**combined, detuning=STARK_OFF, calibration="second_order",
                                 tau=0.005, guard=0.025),
                         f"method D, all effects at once plus circuit terms "
                         f"({setting('+10%')} {math('κ<sub>a</sub>')}, {setting('10⁻³')} thermal photons)",
                         hot=True),
                    stat(realism(**combined, start="instant_A"), "method A under the same conditions"),
                    stat(realism(effect="heat_loss", start="push", extra=1.0, n_th=0.01),
                         f"method D with a very hot drive line ({setting('2')}{math('κ<sub>a</sub>')}, "
                         f"{setting('10⁻²')} thermal photons)")],
                   delay=4)),
        delay=2))
    return slide("Real hardware", content + footer("Task 1.3", [],
                                                   "Numbers: studies/task13_push_realism.py"))


def costs(c: Claims) -> str:
    rows = [
        ("One memory drive line", "the same line any displacement needs"),
        ("A fast pump switch", "pump off during the push"),
        ("Phase calibration", f"{setting('22.5°')} off → {c.v('d_phase.22.5.err')}"),
        ("Amplitude is forgiving", f"±{setting('10%')} → {c.v('d_amp.0.9.err')}, {c.v('d_amp.1.1.err')}"),
    ]
    items = "".join(f'<div class="pitch-cost"><b>{k}</b><span>{v}</span></div>' for k, v in rows)
    content = header("1.3(c)", "What it costs") + body(stack(
        f'<div class="pitch-costs reveal d2">{items}</div>',
        lede(f"Real circuit: the memory Kerr sets the floor at {c.v('d_err_circuit')}. "
             f"Removing it needs a drive we have not shown the device can make, so we report it "
             f"as an extension, not a result."),
        delay=2))
    return slide("Costs", content + footer("Task 1.3", [43]))


SLIDES = [even_cat, superposition, idea, protocol, ladder, finite_push, hardware, costs]

EXTRA_CSS = """
/* === PITCH 1.3 ACCENT ===
   The pitch deck's accent is rgb(204, 217, 245) in place of the red; white page kept. */
:root { --red: rgb(204, 217, 245); }

/* === PITCH 1.3 === */
.pitch-eq { font-family: var(--display); font-size: 84px; font-weight: 800; color: var(--red);
  margin: 28px 0; }
.pitch-costs { display: grid; grid-template-columns: 1fr 1fr; gap: 28px 64px; }
.pitch-cost { border-top: 4px solid var(--ink); padding-top: 18px; }
.pitch-cost b { display: block; font-family: var(--display); font-size: 40px; font-weight: 800; }
.pitch-cost > span { display: block; font-size: 28px; color: var(--grey); margin-top: 8px; }
"""
EXTRA_CSS += """.slide .lede { font-size: 30px; line-height: 1.4; }
.slide figcaption { font-size: 20px; }
"""
