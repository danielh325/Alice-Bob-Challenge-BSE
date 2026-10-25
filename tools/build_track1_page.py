"""Build the Track 1 results page from saved simulation output only.

Every number on the page is read from a results JSON file at build time and
rendered with its source (file and key) in a tooltip. Nothing is typed in.

    .venv313/bin/python tools/build_track1_page.py OUT_DIR
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "task1"
FIGURES = ("1_1_inflation.png", "1_3a_fringes.png")
CSS = (Path(__file__).with_name("track1_page.css")).read_text()


def load(name: str) -> dict:
    return json.loads((RES / f"{name}.json").read_text())


CLAIMS = load("claims")
BASE = load("baseline_compare")
KERR = load("kerr_floor")
PUSH = load("finite_push")


def fmt(x: float, digits: int, sci: bool) -> str:
    if not sci:
        return f"{x:.{digits}f}".replace("-", "−")
    m, e = f"{x:.{digits}e}".split("e")
    return f"{m}×10<sup>{int(e)}</sup>".replace("-", "−")


def span(x: float, src: str, digits: int = 2, sci: bool = False, unit: str = "") -> str:
    u = f"&#8239;{unit}" if unit else ""
    return f'<span class="n" title="{escape(src)}">{fmt(x, digits, sci)}{u}</span>'


def c(key: str, digits: int = 2, sci: bool = False, unit: str = "") -> str:
    """A printed output of intro_track.ipynb, via results/task1/claims.json."""
    for group in ("claims", "derived", "inputs"):
        if key in CLAIMS[group]:
            item = CLAIMS[group][key]
            where = f"cell {item['cell']}, line {item['line']}" if "cell" in item else item.get(
                "source", item.get("formula", ""))
            return span(item["value"], f"intro_track.ipynb {where} · claims.json:{group}.{key}",
                        digits, sci, unit)
    raise KeyError(key)


def row_value(data: dict, name: str, **match) -> float:
    rows = [r for r in data["rows"] if all(r.get(k, False if v is True else None) == v
                                           for k, v in match.items())]
    rows = [r for r in rows if set(r) - {"error"} == set(match)]
    if len(rows) != 1:
        raise ValueError(f"{name}: {match} matched {len(rows)} rows")
    return rows[0]["error"]


def r(data: dict, name: str, digits: int = 1, **match) -> str:
    value = row_value(data, name, **match)
    keys = ", ".join(f"{k}={v}" for k, v in match.items())
    return span(value, f"results/task1/{name}.json rows[{keys}].error", digits, sci=True)


def base(start: str, model: str) -> float:
    return row_value(BASE, "baseline_compare", start=start,
                     beta=next(x["beta"] for x in BASE["rows"] if x["start"] == start), model=model)


def b(start: str, model: str) -> str:
    beta = next(x["beta"] for x in BASE["rows"] if x["start"] == start)
    return r(BASE, "baseline_compare", start=start, beta=beta, model=model)


def k(case: str) -> str:
    return r(KERR, "kerr_floor", case=case)


def p(**match) -> str:
    return r(PUSH, "finite_push", **match)


# ---------------------------------------------------------------- chart

METHODS = (
    ("Vacuum start", lambda m: base("vacuum", m)),
    ("A · push β = 1.5", lambda m: base("A_1.5", m)),
    ("B · bias pulse", lambda m: CLAIMS["claims"][f"robust_b_{'before' if m == 'model' else 'after'}"]["value"]),
    ("C · push during ramp", lambda m: CLAIMS["claims"][f"robust_c_{'before' if m == 'model' else 'after'}"]["value"]),
    ("A · push β = 2.0", lambda m: base("A_2.0", m)),
    ("D · push β = α", lambda m: base("D_alpha", m)),
)


def chart() -> str:
    """Horizontal log-scale dot plot: preparation error, two-mode model vs circuit terms."""
    w, left, right, top, row_h = 880, 190, 30, 40, 44
    lo, hi = -6, 0
    h = top + row_h * len(METHODS) + 34

    def x(v: float) -> float:
        return left + (math.log10(v) - lo) / (hi - lo) * (w - left - right)

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Preparation error by method, log scale">']
    for e in range(lo, hi + 1):
        xe = x(10.0 ** e)
        parts.append(f'<line x1="{xe:.1f}" y1="{top - 14}" x2="{xe:.1f}" y2="{h - 30}" class="grid"/>')
        parts.append(f'<text x="{xe:.1f}" y="{h - 10}" class="tick" text-anchor="middle">10{_sup(e)}</text>')
    for i, (label, f) in enumerate(METHODS):
        y = top + i * row_h + row_h / 2 - 8
        vm, vc = f("model"), f("circuit")
        hl = " hl" if label.startswith("D") else ""
        parts.append(f'<text x="{left - 14}" y="{y + 5:.1f}" class="lab{hl}" text-anchor="end">{escape(label)}</text>')
        parts.append(f'<line x1="{x(vm):.1f}" y1="{y:.1f}" x2="{x(vc):.1f}" y2="{y:.1f}" class="link"/>')
        parts.append(f'<circle cx="{x(vc):.1f}" cy="{y:.1f}" r="6.5" class="dot-c"><title>{label}, circuit terms: {vc:.2e}</title></circle>')
        parts.append(f'<circle cx="{x(vm):.1f}" cy="{y:.1f}" r="6.5" class="dot-m{hl}"><title>{label}, two-mode model: {vm:.2e}</title></circle>')
    parts.append("</svg>")
    return "\n".join(parts)


def _sup(e: int) -> str:
    table = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return str(e).translate(table)


# ---------------------------------------------------------------- page

def page() -> str:
    src = CLAIMS["source"]
    libs = " · ".join(f"{name} {ver}" for name, ver in src["libraries"].items())
    return f"""<title>Cat Qubit Track 1</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=Nunito:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap">
<style>{CSS}</style>
<main class="wrap">

<header class="hero">
  <p class="eyebrow">DuQuantum 2026 · Track 1 · Intro track</p>
  <h1>A dissipative cat qubit, simulated end to end</h1>
  <p class="lede">Memory and buffer modes in the full two-mode Lindblad model, with the device parameters of
  Réglade et al. (α = {c('alpha', 1)}). Every number below is read from a saved simulation output at build
  time; hover any figure to see the file and line it came from.</p>
  <dl class="stats">
    <div><dt>Steady photon number</dt><dd>{c('n_3us', 4)}</dd><p>at 3 µs, N<sub>a</sub>×N<sub>b</sub> = {c('na_conv', 0)}×{c('nb_conv', 0)}</p></div>
    <div><dt>Superposition lifetime</dt><dd>{c('lifetime_us', 2, unit='µs')}</dd><p>Liouvillian gap: {c('liou_lifetime', 2, unit='µs')}</p></div>
    <div><dt>Best codeword preparation</dt><dd>{c('d_err', 1, sci=True)}</dd><p>1 − F, method D, two-mode model</p></div>
    <div><dt>Same, with circuit terms</dt><dd>{c('d_err_circuit', 1, sci=True)}</dd><p>{c('d_vs_c_circuit', 0)}× below method C</p></div>
  </dl>
</header>

<section>
  <h2><span class="code">1.0</span> Warm-ups against known answers</h2>
  <div class="tbl"><table>
    <thead><tr><th>Check</th><th>Result</th><th>Reference</th></tr></thead>
    <tbody>
    <tr><td>(a) Damped, detuned coherent state stays coherent</td><td>min overlap {c('a0_min_fid', 6)}</td><td>max error vs e<sup>−κt</sup>: {c('a0_max_err', 1, sci=True)}</td></tr>
    <tr><td>(b) Two-photon loss from |α₀ = 2⟩ stalls</td><td>⟨n⟩ → {c('b0_n_final', 5)}</td><td>odd-parity share {c('b0_odd_pop', 5)}</td></tr>
    <tr><td>(c) Pair drive + loss grows a cat</td><td>κ<sub>2</sub> = {c('kappa2_mhz', 3, unit='MHz')}</td><td>state depends on κ<sub>2</sub>t only when κ<sub>a</sub> = 0</td></tr>
    </tbody>
  </table></div>
</section>

<section class="split">
  <div>
    <h2><span class="code">1.1–1.2</span> Inflation and truncation</h2>
    <p>Starting from vacuum, the memory overshoots to {c('peak_n', 3)} photons at {c('peak_t_ns', 0, unit='ns')},
    settles within 1% from {c('settle_us', 2, unit='µs')}, and ends at {c('n_3us', 4)}. The buffer peaks at
    {c('b_peak', 2)} photons at {c('b_peak_t_ns', 0, unit='ns')} and empties to {c('b_end', 1, sci=True)}.</p>
    <p>The steady state converges from N<sub>a</sub> = {c('na_steady_from', 0)}, N<sub>b</sub> = {c('nb_steady_from', 0)};
    the overshoot needs N<sub>a</sub> = {c('na_peak_from', 0)}, N<sub>b</sub> = {c('nb_peak_from', 0)}. The transient,
    not the end state, sets the truncation. Memory loss moves ⟨a†a⟩ by {c('n_ss_pct', 2)}%.</p>
  </div>
  <figure><img src="1_1_inflation.png" alt="Memory and buffer photon number during inflation, 0 to 3 µs">
    <figcaption>Notebook figure, cell 20.</figcaption></figure>
</section>

<section class="split">
  <div>
    <h2><span class="code">1.3(a)</span> Vacuum inflates to the even cat</h2>
    <p>The drive, buffer loss and pair exchange a²b† only change memory photon number in pairs, so memory parity is
    conserved and vacuum starts at ⟨P⟩ = +1. With memory loss switched off, parity drifts by only
    {c('parity_control', 1, sci=True)}.</p>
    <p>The fringes show a superposition, not a mixture: an equal mixture has the same photon number and
    F(|+α⟩) = {c('mix_fid', 1)}, but zero parity and purity {c('mix_purity', 1)}. The coherence decays at
    {c('coh_decay', 4, unit='/µs')}, against 2κ<sub>a</sub>|α|² = {c('gamma_pred', 4, unit='/µs')}.</p>
  </div>
  <figure><img src="1_3a_fringes.png" alt="Wigner function fringes of the even cat fading over time">
    <figcaption>Notebook figure, cell 22.</figcaption></figure>
</section>

<section>
  <h2><span class="code">1.3(b)</span> Preparing |+α⟩: push straight onto the codeword</h2>
  <p class="wide">Displace the memory to exactly β = α with the pump off, then switch the stabilization on.
  The state |α, 0⟩ is a dark state of the full two-mode model: g₂a²b† gives g₂α²b†, which the drive
  −ε<sub>d</sub>b† cancels because ε<sub>d</sub> = g₂α², and b|α, 0⟩ = 0. Nothing pulls the state toward −α,
  so the only error left is memory loss. With κ<sub>a</sub> = 0 the error drops to {c('d_err_lossless', 1, sci=True)}.</p>
  <figure class="chart">
    {chart()}
    <figcaption><span class="key m"></span> two-mode model <span class="key c"></span> with circuit terms (Task 2.4(a) Kerr and
    cross-Kerr) · 1 − F(|+α⟩) after a 2 µs hold, log scale. Method D is method A with the push set to β = α.</figcaption>
  </figure>
  <div class="tbl"><table>
    <thead><tr><th>Push β</th><th>1.8</th><th>α = 2.1</th><th>2.3</th><th>2.5</th></tr></thead>
    <tbody><tr><td>1 − F, two-mode model</td><td>{c('d_beta.1.80.err', 2, sci=True)}</td><td class="best">{c('d_beta.2.10.err', 2, sci=True)}</td>
    <td>{c('d_beta.2.30.err', 2, sci=True)}</td><td>{c('d_beta.2.50.err', 2, sci=True)}</td></tr></tbody>
  </table></div>
  <p class="note">The minimum sits at β = α, as the dark-state argument predicts. The fair comparison is against
  method A at β = 2.0: {b('A_2.0', 'model')} → {b('D_alpha', 'model')} in the model, and {b('A_2.0', 'circuit')} →
  {b('D_alpha', 'circuit')} with circuit terms.</p>
</section>

<section>
  <h2><span class="code">1.3(b)</span> A real, finite push</h2>
  <p class="wide">The ideal start is replaced by a memory drive of duration T, followed by the same 2 µs hold.
  The pump state during the push decides everything.</p>
  <div class="tbl"><table>
    <thead><tr><th>Push</th><th>T = 50 ns</th><th>200 ns</th><th>1 µs</th></tr></thead>
    <tbody>
    <tr><td>Ideal start (reference)</td><td colspan="3">{p(kind='ideal')}</td></tr>
    <tr><td>Pump off during push</td><td class="best">{p(kind='pump_off', duration=0.05)}</td><td>{p(kind='pump_off', duration=0.2)}</td><td>{p(kind='pump_off', duration=1.0)}</td></tr>
    <tr><td>Pump on during push</td><td class="bad">{p(kind='pump_on', duration=0.05)}</td><td class="bad">{p(kind='pump_on', duration=0.2)}</td><td class="bad">{p(kind='pump_on', duration=1.0)}</td></tr>
    <tr><td>Moving target: ε<sub>d</sub>(t) = g₂α(t)²</td><td>{p(kind='moving', duration=0.05)}</td><td>{p(kind='moving', duration=0.2)}</td><td>{p(kind='moving', duration=1.0)}</td></tr>
    <tr><td>Pump off, circuit terms</td><td>{p(kind='pump_off', duration=0.05, circuit=True)}</td><td>{p(kind='pump_off', duration=0.2, circuit=True)}</td><td>{p(kind='pump_off', duration=1.0, circuit=True)}</td></tr>
    </tbody>
  </table></div>
  <p class="note">With the pump off, a finite push costs nothing measurable. Leaving the pump on while pushing is the
  failure mode: the stabilization drags the half-displaced state back toward the centre. Raising the pump in step
  with the displacement also works, but needs a shaped pump instead of an on/off switch.</p>
</section>

<section>
  <h2><span class="code">1.3(c)</span> What it costs</h2>
  <ul class="costs">
    <li><strong>One memory drive line.</strong> The same line method A needs; no extra hardware beyond it.</li>
    <li><strong>A fast pump switch.</strong> The pump must be off during the push (see the table above).</li>
    <li><strong>Phase calibration.</strong> The push must point at the codeword: 22.5° off costs
      {c('d_phase.22.5.err', 1, sci=True)}, 45° costs {c('d_phase.45.0.err', 1, sci=True)}.</li>
    <li><strong>Amplitude is forgiving.</strong> ±10% gives {c('d_amp.0.9.err', 1, sci=True)} and {c('d_amp.1.1.err', 1, sci=True)}.</li>
    <li><strong>Stabilization is unchanged.</strong> The push ends on a state the stabilization already leaves alone.</li>
  </ul>
</section>

<section class="ext">
  <p class="eyebrow">Extension · beyond the scope of 1.3</p>
  <h2>What limits it on the real circuit</h2>
  <p class="wide">With the Task 2.4(a) circuit terms, the memory Kerr sets the floor; buffer Kerr and cross-Kerr barely
  matter. A Kerr-cat correction −(K/2)α²(a†² + a²) removes the floor in simulation, but it needs a two-photon
  drive on the memory that we have not shown the device can make, and K must be known: the second-order and Floquet
  estimates differ, and correcting for the wrong one is worse than not correcting.</p>
  <div class="tbl"><table>
    <thead><tr><th>Terms added to the two-mode model</th><th>1 − F</th></tr></thead>
    <tbody>
    <tr><td>None</td><td>{k('none')}</td></tr>
    <tr><td>Memory Kerr only</td><td>{k('kerr_a only')}</td></tr>
    <tr><td>Buffer Kerr only</td><td>{k('kerr_b only')}</td></tr>
    <tr><td>Cross-Kerr only</td><td>{k('chi_ab only')}</td></tr>
    <tr><td>All (Task 2.4(a))</td><td>{k('all (2.4a)')}</td></tr>
    <tr><td>All, Floquet memory Kerr</td><td class="bad">{k('all, kerr_a Floquet')}</td></tr>
    <tr><td>All + Kerr-cat correction</td><td class="best">{k('all + correction')}</td></tr>
    <tr><td>Floquet Kerr, corrected for second-order value</td><td>{k('all, kerr_a Floquet + correction for 2nd order')}</td></tr>
    </tbody>
  </table></div>
</section>

<footer class="prov">
  <h2>Where the numbers come from</h2>
  <ul>
    <li><code>intro_track.ipynb</code> (sha256 {src['sha256'][:12]}…), executed; parsed by <code>tools/extract_task1_claims.py</code> into <code>results/task1/claims.json</code>.</li>
    <li><code>studies/task13_baseline_compare.py</code>, <code>task13_kerr_floor.py</code>, <code>task13_finite_push.py</code>, double precision, N<sub>a</sub>×N<sub>b</sub> = {PUSH['truncation'][0]}×{PUSH['truncation'][1]}.</li>
    <li>Lindblad solver dynamiqs, tolerance 10<sup>−9</sup>. {escape(libs)}.</li>
    <li>Not modelled: thermal photons, pure dephasing, readout, pulse distortion; one parameter set.</li>
  </ul>
</footer>
</main>
"""


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "track1_page"
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page())
    for name in FIGURES:
        shutil.copy(ROOT / "figures" / name, out / name)
    print("wrote", out / "index.html")


if __name__ == "__main__":
    main()
