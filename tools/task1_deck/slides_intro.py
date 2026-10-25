"""Slides 1-11: device, provenance, warm-ups 1.0, inflation 1.1, convergence 1.2."""

from __future__ import annotations

from tools.task1_deck.claims import Claims, math, setting
from tools.task1_deck.layout import (
    body,
    cols,
    figure,
    footer,
    header,
    kicker,
    lede,
    note,
    slide,
    stack,
    stat,
    stats,
    table,
)

KA, KB, K2 = "κ<sub>a</sub>", "κ<sub>b</sub>", "κ₂"
G2, ED, NA, NB = "g₂", "ε<sub>d</sub>", "N<sub>a</sub>", "N<sub>b</sub>"
KA_OFF = setting(f"{KA} = 0")
KET0, KET1, PER_2PI = math("|0⟩"), math("|1⟩"), math("/2π")


def cover(c: Claims) -> str:
    content = (
        '<div class="tag reveal d1" style="position:absolute;left:160px;top:96px">'
        'Alice &amp; Bob challenge <span class="dim">· Track 1 · DuQuantum</span></div>'
        '<h1 class="reveal d2">Building a cat qubit, <em>two photons at a time</em></h1>'
        '<div class="sub reveal d3">'
        + lede("A two-mode simulation of the dissipative cat: inflation, convergence, parity and the "
               "lifetime of the superposition, and four ways to prepare a codeword.")
        + note(f"Every number on these slides is a printed output of an executed cell of "
               f"<b>intro_track.ipynb</b>, run with {c.lib('dynamiqs')} on {c.lib('jax')}. "
               f"Hover a number to see the line it came from.")
        + "</div>"
        + stats([
            stat(c.v("n_3us"), "photons in the memory after the inflation", hot=True),
            stat(c.v("lifetime_us"), "lifetime of the cat superposition", "µs"),
            stat(c.v("d_err"), "best codeword preparation error (method D)"),
        ], delay=4)
        + footer("Track 1 · intro track", [20, 22, 43])
    )
    return slide("Cover", content, extra_class="cover")


def device(c: Claims) -> str:
    rows = [
        [f"Two-photon coupling {G2}", c.i("g2_mhz"), c.v("g2")],
        [f"Buffer loss {KB}", c.i("kappa_b_mhz"), c.v("kappa_b")],
        [f"Memory loss {KA}", c.i("kappa_a_mhz", sci=True), c.v("kappa_a")],
        [f"Buffer drive {ED} = α²{G2}", "", c.v("eps_d")],
        [f"Two-photon loss {K2} = {math('4')}{G2}²/{KB}", c.v("kappa2_mhz"), c.v("kappa2")],
    ]
    left = stack(
        kicker("The model we simulate"),
        lede("Memory <b>a</b> and buffer <b>b</b>: the buffer is driven and lossy, and the coupling "
             "turns one buffer photon into two memory photons."),
        f'<p class="eq">{math("H = g₂* a² b† + g₂ a†² b − ε<sub>d</sub> b† − ε<sub>d</sub>* b")}</p>',
        f'<p class="eq">{math(f"L<sub>b</sub> = √{KB} b,  L<sub>a</sub> = √{KA} a")}</p>',
        note(f"Target cat size α = {c.i('alpha')}. Rates from Réglade et al. "
             f'<span class="ref">(arXiv:2307.06617, Table S2)</span>, set in '
             f'<span class="ref">catqubit/params.py</span>.'),
    )
    right = stack(
        table(["Parameter", f"{PER_2PI} (MHz)", "Simulated (rad/µs)"], rows, right_from=1, delay=2),
        stats([
            stat(c.v("g2_over_kb"), f"{G2}/{KB}: not small, so we keep the buffer as a full mode"),
            stat(c.v("epsd_over_kb"), f"{ED}/{KB}: the drive is strong on the buffer's scale", hot=True),
        ], columns=2, delay=3),
        delay=2,
    )
    content = header("Setup <span class='dim'>· device parameters</span>",
                     "Real device rates in, <em>two modes</em> simulated") + body(cols("1fr 1.15fr", left, right))
    return slide("Device parameters", content + footer("Setup", [1, 20]))


def provenance(c: Claims) -> str:
    steps = [
        ("01", "Settings", "catqubit/params.py<br>tools/build_notebook.py"),
        ("02", "Simulate", f"intro_track.ipynb executed<br>{c.lib('dynamiqs')} · {c.lib('jax')}"),
        ("03", "Extract", "tools/extract_task1_claims.py<br>→ results/task1/claims.json"),
        ("04", "Build", "tools/build_task1_deck.py<br>→ task1_deck.html"),
        ("05", "Verify", "tools/verify_task1_deck.py<br>every number vs the JSON"),
    ]
    chain = "".join(f'<div><div class="k idx">{k}</div><div class="t">{t}</div><code class="ref">{s}</code></div>'
                    for k, t, s in steps)
    sha = c.source["sha256"][:12]
    content = (
        header("Method <span class='dim'>· where the numbers come from</span>",
               "No number is typed by hand: <em>each one is a printed result</em>")
        + body(
            f'<div class="chain reveal d2">{chain}</div><div style="margin-top:56px">'
            + cols("1fr 1fr 1fr",
                   stack(kicker("Results"), lede(f"{c.count()} printed lines parsed into "
                                                 f"<b>claims.json</b> with cell and line."), delay=3),
                   stack(kicker("Derived"), lede("Ratios and percentages are computed from those "
                                                 "results, and the formula is stored next to them."), delay=3),
                   stack(kicker("Refusal"), lede("The extractor stops if any cell has an error or did "
                                                 "not run, so stale output cannot reach a slide."), delay=4)) + "</div>"
        )
        + footer("Method", [], f'Notebook sha256 <span class="ref">{sha}…</span> · '
                 f'{c.lib("numpy")} · {c.lib("scipy")}')
    )
    return slide("Provenance", content)


def warmup_a(c: Claims) -> str:
    left = figure("1_0a_photon_number", "Photon number decays exponentially and matches the analytic curve",
                  "Simulated ⟨a†a⟩ on top of |α₀|²e<sup>−κt</sup>.", 560, (672, 495))
    right = stack(
        figure("1_0a_wigner", "Wigner snapshots: a single blob spiralling into the origin",
               "The blob turns and shrinks toward the origin but stays round.", 1000, (2689, 545), delay=2),
        cols("1fr 1fr",
             stats([stat(c.v("a0_max_err", sign=False), "largest gap between simulation and |α₀|²e<sup>−κt</sup>",
                         hot=True)], delay=3),
             stats([stat(c.v("a0_min_fid"), "lowest overlap with a coherent state over the run")], delay=4)),
        note(f"Detuned damped cavity, Δ{PER_2PI} = κ{PER_2PI} = {c.i('a0_delta_mhz')} MHz, start |α₀ = {c.i('a0_alpha0')}⟩: "
             f"loss shrinks the amplitude, detuning turns it, and a coherent state stays coherent."),
        delay=2,
    )
    content = header("1.0(a) <span class='dim'>· warm-up</span>",
                     "A coherent state <em>stays coherent</em> under one-photon loss") + body(cols("560px 1fr", left, right))
    return slide("Warm-up 1.0a", content + footer("Warm-up", [3]))


def warmup_b(c: Claims) -> str:
    left = figure("1_0b_photon_number", "Photon number under two-photon loss stalls at one half",
                  f"⟨a†a⟩ under loss of pairs only, with the stall line {math('(1 − e<sup>−2|α₀|²</sup>)/2')}.",
                  700, (692, 495))
    right = stack(
        lede(f"Loss of photon <b>pairs</b> (operator a²) cannot change parity. Odd states fall to {KET1}, "
             f"even states to {KET0}, and a² does nothing to either."),
        stats([
            stat(c.v("b0_n_final"), f"⟨a†a⟩ after {c.i('b0_t_us')} µs", hot=True),
            stat(c.v("b0_p01_init"), f"starting weight on {KET0} and {KET1}"),
            stat(c.v("b0_p0_final"), f"final weight on {KET0}"),
            stat(c.v("b0_p1_final"), f"final weight on {KET1} = odd part of the start"),
        ], columns=2, delay=3),
        note(f"The stall equals the odd part of |α₀ = {c.i('a0_alpha0')}⟩, {math('(1 − e<sup>−8</sup>)/2')}. "
             f"This parity rule is what protects the cat in {setting('1.3')}."),
    )
    content = header("1.0(b) <span class='dim'>· warm-up</span>",
                     "Losing photons <em>in pairs</em> freezes the parity") + body(cols("700px 1fr", left, right))
    return slide("Warm-up 1.0b", content + footer("Warm-up", [5]))


def warmup_c(c: Claims) -> str:
    rows = [[setting(f"{K2} × {m}"), c.v(f"c0.{k}.n2us"), c.v(f"c0.{k}.t90_ns"),
             c.v(f"c0.{k}.t90k2"), c.v(f"c0.{k}.t90k2_lossless")]
            for m, k in (("0.25", "0.25"), ("1", "1.0"), ("4", "4.0"))]
    left = stack(
        figure("1_0c_wigner", "Wigner snapshots: two blobs with fringes form from vacuum",
               "Vacuum splits into two blobs at ±α, with interference fringes between them.", 1600, (2689, 545), delay=2),
        delay=2,
    )
    lower = cols("1fr 0.62fr",
                 table(["Two-photon loss", f"⟨a†a⟩ at {c.i('c0_t_us')} µs", "t₉₀ (ns)", f"t₉₀·{K2}",
                        f"t₉₀·{K2}, {KA_OFF}"], rows, highlight=[1], compact=True),
                 stack(note(f"With {KA_OFF} the dissipator is {math(f'{K2}·D[a² − α²]')}, so the state depends "
                            f"only on {K2}t and t₉₀·{K2} is the same for every rate."),
                       delay=4))
    content = (header("1.0(c) <span class='dim'>· warm-up</span>",
                      f"Pair drive and loss <em>grow a cat</em> at a speed set by {K2}")
               + body(left + f'<div style="margin-top:28px">{lower}</div>'))
    return slide("Warm-up 1.0c", content + footer("Warm-up", [7]))


def inflation(c: Claims) -> str:
    left = figure("1_1_inflation", "Memory photon number overshoots then settles; buffer fills and empties",
                  f"Left: memory, buffer and the one-mode {setting('1.0(c)')} model (dashed). Right: populations of the top levels.",
                  1000, (1249, 518))
    right = stats([
        stat(c.v("peak_n"), f"peak ⟨a†a⟩ at {c.v('peak_t_ns')} ns, {c.d('peak_over_alpha2_pct', 0)}% above α²", hot=True),
        stat(c.v("settle_us"), f"within {setting('1%')} of the final value from here on", "µs"),
        stat(c.v("n_3us"), f"⟨a†a⟩ at {c.i('sim_t_us')} µs, the end of the run"),
        stat(c.v("b_peak"), f"buffer peak at {c.v('b_peak_t_ns')} ns, then {c.v('b_end')} at the end"),
    ], delay=2, width=480)
    content = (header("1.1 <span class='dim'>· inflation</span>",
                      "Switching the drive on <em>overshoots</em> before it settles")
               + body(cols("1000px 1fr", left, right)
                      + note(f"Top two memory levels stay below {c.v('top2_mem')} and the top buffer level below "
                             f"{c.v('top_buffer')}: the box is big enough.").replace(
                          'class="note"', 'class="note reveal d4" style="position:absolute;left:0;top:560px;width:1000px"', 1)))
    return slide("Inflation", content + footer("Inflation", [20]))


def inflation_wigner(c: Claims) -> str:
    left = figure("1_1_wigner", "Wigner snapshots of the memory during inflation",
                  "Memory Wigner function during the inflation (two-mode simulation, buffer traced out).",
                  1160, (2239, 950))
    right = stack(
        lede("Two blobs at ±α with fringes between them: an <b>even cat</b>, because the vacuum we start "
             "from is even and pair processes keep it so."),
        stats([stat(c.v("n_ss"), "final ⟨a†a⟩ with memory loss", hot=True),
               stat(c.d("alpha2", 2), "α² for comparison")], delay=3),
        delay=2,
    )
    content = header("1.1 <span class='dim'>· phase space</span>",
                     "The memory ends as <em>an even cat</em> of the target size") + body(cols("1160px 1fr", left, right))
    return slide("Inflation Wigner", content + footer("Inflation", [18]))


def convergence(c: Claims) -> str:
    na_rows = [[setting(f"{NA} = {n}"), c.v(f"sweep.na.{n}.steady"), c.v(f"sweep.na.{n}.peak")]
               for n in (16, 20, 24, 28, 32)]
    nb_rows = [[setting(f"{NB} = {n}"), c.v(f"sweep.nb.{n}.steady"), c.v(f"sweep.nb.{n}.peak")]
               for n in (4, 6, 8, 10, 12, 14)]
    left = stack(
        figure("1_2_convergence", "Relative change of the observables versus truncation",
               f"Relative change against the largest box; tolerance line at {c.i('rtol', sci=True, digits=0)}.",
               760, (1169, 495), delay=2),
        note(f"The final value converges at {NA} = {c.v('na_steady_from')}, {NB} = {c.v('nb_steady_from')}. "
             f"The overshoot needs more levels, {NA} = {c.v('na_peak_from')}, {NB} = {c.v('nb_peak_from')}, "
             f"because the memory briefly holds far more photons than α²."),
        delay=2,
    )
    right = cols("1fr 1fr",
                 table([setting(f"{NB} = 14"), "final", "peak"], na_rows, highlight=[3], compact=True),
                 table([setting(f"{NA} = 32"), "final", "peak"], nb_rows, highlight=[4], compact=True, delay=4))
    content = (header("1.2 <span class='dim'>· convergence</span>",
                      f"Chosen box: <em>{NA} = {c.v('na_conv')}, {NB} = {c.v('nb_conv')}</em>, within {setting('0.1%')} on both")
               + body(cols("760px 1fr", left, right)))
    return slide("Convergence", content + footer("Convergence", [11, 12]))


def tolerance(c: Claims) -> str:
    rows = [
        [setting("solver tolerance 10<sup>−6</sup>"), c.v("tol6_steady"), c.v("tol6_min", sign=True)],
        [setting("solver tolerance 10<sup>−9</sup>"), c.v("tol9_steady"), c.v("tol9_min", sign=True)],
    ]
    left = stack(
        kicker("Solver tolerance"),
        table(["Run", "final ⟨a†a⟩", "most negative population"], rows, highlight=[1], delay=2),
        note("A loose tolerance gives a wrong final value and negative probabilities. The tight one "
             "agrees with the converged sweep."),
        delay=2,
    )
    right = stack(
        kicker(f"Starter box {setting(NB + ' = 6')}"),
        stats([
            stat(c.v("starter_peak"), f"peak ⟨a†a⟩, against {c.v('peak_n')} in the converged box", hot=True),
            stat(c.v("starter_top_buffer"), "population of the top buffer level: not negligible"),
            stat(c.v("starter_b_peak"), f"buffer peak, against {c.v('b_peak')} converged"),
        ], delay=3),
        delay=3,
    )
    content = (header("1.2 <span class='dim'>· two traps</span>",
                      "A small box and a loose solver <em>both</em> give wrong numbers")
               + body(cols("1fr 1fr", left, right)))
    return slide("Tolerance and truncation", content + footer("Convergence", [9, 15, 20]))


def steady(c: Claims) -> str:
    content = (
        header("1.1 <span class='dim'>· final value</span>",
               f"Memory loss moves ⟨a†a⟩ by <em>{c.v('n_ss_pct')}%</em>; mean-field predicts the shift to {c.d('n_shift_agree_pct', 0)}%")
        + body(cols("1fr 1fr 1fr",
                    stats([stat(c.v("n_ss"), f"final ⟨a†a⟩ with {KA}", hot=True)], delay=2),
                    stats([stat(c.v("n_lossless"), f"final ⟨a†a⟩ with {KA_OFF}"),
                           stat(c.v("n_tanh"), f"analytic {math('α² tanh(α²)')}")], delay=3),
                    stats([stat(c.v("n_shift", sign=True), "shift from memory loss, simulated"),
                           stat(c.v("n_shift_mf", sign=True), "shift from mean-field theory")], delay=4))
               + note(f"The lossless run matches the exact {math('α² tanh(α²)')} to every printed digit. The loss "
                      f"shift agrees with the mean-field estimate to {c.d('n_shift_agree_pct', 0)}%, which is "
                      f"expected for a first-order estimate.").replace(
                   'class="note"', 'class="note reveal d5" style="position:absolute;left:0;top:520px;width:1100px"', 1))
    )
    return slide("Steady state", content + footer("Inflation", [18]))


SLIDES = [cover, device, provenance, warmup_a, warmup_b, warmup_c, inflation,
          inflation_wigner, convergence, tolerance, steady]
