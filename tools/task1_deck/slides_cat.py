"""Slides 12-22: parity and lifetime 1.3(a), preparation methods 1.3(b), costs 1.3(c)."""

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

KA, K2, G2 = "κ<sub>a</sub>", "κ₂", "g₂"
CODEWORD = "|+α⟩"
KA_OFF = setting(f"{KA} = 0")


def observables(c: Claims) -> str:
    rows = [[setting(f"{t} µs"), c.v(f"cat_t.{t}.parity"), c.v(f"cat_t.{t}.coherence"),
             c.v(f"cat_t.{t}.fid_cat"), c.v(f"cat_t.{t}.fid_plus"), c.v(f"cat_t.{t}.purity")]
            for t in (1, 3, 6)]
    left = figure("1_3a_observables", "Parity, coherence, fidelities and purity versus time",
                  f"Parity, cat coherence, fidelity with the cat and with {CODEWORD}, purity; "
                  f"plus the {KA_OFF} parity control.", 700, (775, 542))
    right = stack(
        lede(f"Pair processes conserve parity, so only memory loss {KA} can flip it. Vacuum starts at "
             f"⟨P⟩ = +{setting('1')}, and the cat it grows is even."),
        table(["t", "⟨P⟩", "coherence", "F(cat)", f"F({CODEWORD})", "purity"], rows, compact=True),
        stats([stat(c.v("parity_control"), f"largest drift of ⟨P⟩ from +{setting('1')} with {KA_OFF}", hot=True)],
              delay=4),
    )
    content = header("1.3(a) <span class='dim'>· parity</span>",
                     "Parity holds without memory loss and <em>fades</em> with it") + body(cols("700px 1fr", left, right))
    return slide("Parity observables", content + footer("Cat lifetime", [22]))


def lifetime(c: Claims) -> str:
    top = figure("1_3a_fringes", "Wigner fringes fading over time",
                 "The fringes between the blobs, the mark of a superposition, fade as parity decays.",
                 1600, (1789, 545), delay=2)
    row = cols("1fr 1fr 1fr 1fr",
               stats([stat(c.v("coh_decay"), "fitted decay of the cat coherence", "/µs")], delay=3),
               stats([stat(c.v("gamma_pred"), f"predicted phase-flip rate {math('2')}{KA}α²", "/µs")], delay=3),
               stats([stat(c.v("liou_lifetime"), "slowest even-parity Liouvillian mode, no fit", "µs")], delay=4),
               stats([stat(c.v("lifetime_us"), f"superposition lifetime; fit and theory agree to "
                                               f"{c.d('gamma_agree_pct', 2)}%", "µs", hot=True)], delay=4))
    content = (header("1.3(a) <span class='dim'>· lifetime</span>",
                      f"The cat stays a superposition for <em>{c.v('lifetime_us')} µs</em>")
               + body(top + f'<div style="margin-top:34px">{row}</div>'))
    return slide("Cat lifetime", content + footer("Cat lifetime", [22]))


def superposition(c: Claims) -> str:
    rows = [
        ["Our cat at " + setting("1 µs"), c.v("prep_fid"), c.v("prep_parity"), c.v("prep_coh"), c.v("prep_purity")],
        ["Equal mixture of |±α⟩", c.v("mix_fid"), c.v("mix_parity"), c.v("mix_coh"), c.v("mix_purity")],
    ]
    top = figure("1_3a_superposition_vs_mixture", "Wigner functions of the cat and of a mixture, with a cut",
                 "Left: our state. Middle: a classical mixture. Right: a cut through the middle; only the cat has fringes.",
                 1300, (1766, 524), delay=2)
    content = (header("1.3(a) <span class='dim'>· superposition or mixture?</span>",
                      f"Same F({CODEWORD}); <em>only the coherence</em> tells them apart")
               + body(top + '<div style="margin-top:22px;width:1300px">'
                      + table(["State", f"F({CODEWORD})", "⟨P⟩", "coherence", "purity"], rows, highlight=[0], compact=True)
                      + "</div>"))
    return slide("Superposition vs mixture", content + footer("Cat lifetime", [24]))


def method_a(c: Claims) -> str:
    betas = ("0.00", "0.50", "1.00", "1.50", "2.00")
    rows = [[setting(f"β = {b}"), c.v(f"a_beta.{b}.fid"), c.v(f"a_beta.{b}.basin")] for b in betas]
    phases = [[setting(f"{p}°"), c.v(f"a_phase.{p}.fid"), c.v(f"a_phase.{p}.basin")]
              for p in ("0.0", "22.5", "45.0", "67.5", "90.0")]
    left = figure("1_3b_fidelity_vs_beta", "Fidelity versus displacement and versus phase error",
                  f"Left: F({CODEWORD}) against displacement β, simulation and basin model. Right: phase error at |β| = "
                  f"{setting('1.5')}.", 900, (1269, 518))
    right = stack(
        lede(f"Displace the vacuum by β, then switch the stabilization on: the state rolls into the "
             f"nearer well. Basin model, no free parameters: {math('F = 1 − ½ erfc(√2 β)')}."),
        cols("1fr 1fr",
             table(["Push", "sim", "model"], rows, highlight=[3], compact=True),
             table(["Phase", "sim", "model"], phases, compact=True, delay=4)),
    )
    content = header("1.3(b) <span class='dim'>· method A</span>",
                     "Displaced start: <em>a one-line model</em> predicts the fidelity") + body(cols("900px 1fr", left, right))
    return slide("Method A", content + footer("Preparation", [26], f" · truncation monitor {c.v('a_monitor')}"))


def method_b(c: Claims) -> str:
    areas = ("0.25", "0.5", "1.0", "1.5", "2.5", "4.0")
    rows = [[setting(f"{t} ns")] + [c.v(f"b_grid.{t}.{a}") for a in areas] + [c.v(f"b_grid.{t}.max_mhz")]
            for t in ("10", "50", "100")]
    left = figure("1_3b_pulse_area", "Fidelity versus pulse area for three pulse lengths",
                  "F against pulse area for three pulse lengths, with the basin model in the short-pulse limit.",
                  620, (746, 542))
    right = stack(
        lede("Keep the stabilization on and push the memory with a short resonant pulse while the cat inflates."),
        table(["Pulse", *[setting(f"A = {a}") for a in areas], "peak MHz"], rows, highlight=[1], compact=True),
        stats([stat(c.v("b_fid"), f"chosen: area {c.v('b_area')}, {c.v('b_T_ns')} ns, {c.v('b_mhz')} MHz", hot=True),
               stat(c.d("b_grid_overdrive_drop", 3), f"lost at {setting('50 ns')} by overdriving to the largest area")],
              columns=2, delay=4),
    )
    content = header("1.3(b) <span class='dim'>· method B</span>",
                     "Bias pulse: <em>no switching</em>, but a strong drive") + body(cols("620px 1fr", left, right))
    return slide("Method B", content + footer("Preparation", [28]))


def overdrive(c: Claims) -> str:
    left = figure("1_3b_prepared_state", "Wigner function of the state prepared by method B",
                  "State after the chosen method B pulse: one blob at +α.", 560, (798, 673))
    right = stack(
        lede("Pushing harder makes it <b>worse</b>. We checked this is physics, not the box:"),
        stats([
            stat(c.v("big_pulse_fid"), f"largest pulse in a bigger memory box ({setting('N<sub>a</sub> = 40')})"),
            stat(c.v("big_beta_fid"), "method A overdriven the same way: same drop"),
            stat(c.v("eff_beta4_fid"), "one-mode model, same push: no drop", hot=True),
        ], delay=3),
        note(f"So the drop comes from the buffer, which the one-mode model leaves out. Final Re⟨a⟩ = "
             f"{c.v('b_re_a')}. Truncation monitor {c.v('b_monitor')}."),
    )
    content = header("1.3(b) <span class='dim'>· method B, check</span>",
                     "Overdriving loses fidelity, and <em>the buffer</em> is why") + body(cols("560px 1fr", left, right))
    return slide("Method B overdrive", content + footer("Preparation", [28, 31]))


def amplifier(c: Claims) -> str:
    keys = [f"{t} ns, A = {a}" for t in ("10", "50", "100") for a in ("0.5", "1.0")]
    rows = [[setting(k), c.v(f"beff.{k}.beta_eff"), c.v(f"beff.{k}.model"), c.v(f"beff.{k}.sim")] for k in keys]
    left = stack(
        lede(f"With the stabilization on, noise in the memory is amplified at "
             f"λ = {K2}α² = <b>{c.v('lambda')} /µs</b>. A push is worth its area only if it lands before "
             f"that growth: a late push is discounted."),
        lede(f"That is why method B needs a drive of {c.d('b_drive_over_g2', 1)} {G2}, and why a weak push held from the "
             f"start fails."),
        stats([stat(c.v("weak005_sim"), f"simulated F for a weak push held {setting('1 µs')} "
                                         f"(model {c.v('weak005_model')})"),
               stat(c.v("weak02_sim"), f"a {setting('4')}× stronger weak push (model {c.v('weak02_model')})")],
              columns=2, delay=4),
    )
    right = table(["Pulse", "β<sub>eff</sub>", "model F", "simulated F"], rows, delay=3)
    content = header("1.3(b) <span class='dim'>· why the drive is strong</span>",
                     "An amplifier model with <em>no free parameters</em> fits small pushes") + body(
        cols("1fr 1fr", left, right))
    return slide("Amplifier model", content + footer("Preparation", [36]))


def method_c(c: Claims) -> str:
    ramps = ("0.25", "0.50", "1.00", "2.00")
    rows = [[setting(f"{r} µs"), c.v(f"c_ramp.{r}.push"), c.v(f"c_ramp.{r}.push_g2"),
             c.v(f"c_ramp.{r}.push_mhz"), c.v(f"c_ramp.{r}.err")] for r in ramps]
    left = figure("1_3b_method_c", "Preparation error versus ramp time for method C",
                  "Error against ramp time, with methods A and B as reference lines.", 640, (797, 518))
    right = stack(
        lede(f"Ramp the stabilization up instead of switching it on. While λ is small a weak push is worth "
             f"almost its full area, so the model sets the push for β<sub>eff</sub> = {setting('1.5')}; nothing is fitted."),
        table(["Ramp", "push (rad/µs)", f"push / {G2}", "MHz", "error"], rows, highlight=[1, 2], compact=True),
        stats([stat(c.d("c_err_best", 1, sci=True), "best error of A, B and C", hot=True),
               stat(c.v("c_ramp.1.00.push_mhz"), f"drive needed, {c.d('b_over_c_drive', 0)}× below method B's {c.v('b_mhz')} MHz", "MHz")],
              columns=2, delay=4),
    )
    content = header("1.3(b) <span class='dim'>· method C</span>",
                     "A weak push <em>while the stabilization ramps</em> beats A and B") + body(cols("640px 1fr", left, right))
    return slide("Method C", content + footer("Preparation", [38], f" · truncation monitor {c.v('c_monitor')}"))


def method_d(c: Claims) -> str:
    rows = [[setting(f"β = {b}"), c.v(f"d_beta.{b}.err"), c.v(f"d_beta.{b}.re_a")] for b in ("1.80", "2.10", "2.30", "2.50")]
    left = stack(
        lede(f"Method A's push, set to exactly β = α: displace with the pump off, then switch the pump and buffer "
             f"drive on. "
             f"On {math('|α⟩|0⟩')} the pair exchange gives {math('g₂α²')}, which the drive "
             f"{math('ε<sub>d</sub> = α²g₂')} cancels, and the buffer is empty:"),
        f'<p class="lede reveal d2">{math("H|α,0⟩ = 0")}, {math("b|α,0⟩ = 0")}</p>',
        lede("So the codeword is a dark state of the full two-mode model. There is no transient and no basin to "
             "roll into; only memory loss acts."),
        table(["Start", f"error at {setting('2 µs')}", "Re⟨a⟩"], rows, highlight=[1], compact=True, delay=3),
    )
    right = stack(
        stats([stat(c.v("d_err"), "error at β = α, two-mode model", hot=True),
               stat(c.v("d_err_circuit"), "with the circuit terms (memory Kerr)"),
               stat(c.v("d_err_lossless"), f"with {KA_OFF}: the residual is memory loss"),
               stat(c.v("d_dark_h"), f"{math('‖H|α,0⟩‖')} in the {setting('28 × 12')} box")], delay=3),
        note(f"With circuit terms, {c.v('d_vs_c_circuit')}× below method C ({c.v('d_vs_c')}× in the two-mode "
             f"model). "
             f"Amplitude ±{setting('10%')}: {c.v('d_amp.0.9.err')} and {c.v('d_amp.1.1.err')}. "
             f"Phase {setting('22.5°')}: {c.v('d_phase.22.5.err')}, {setting('45°')}: {c.v('d_phase.45.0.err')}."),
    )
    content = header("1.3(b) <span class='dim'>· method D</span>",
                     "Push <em>exactly onto the codeword</em>: a dark state") + body(
        cols("1fr 1fr", left, right))
    return slide("Method D", content + footer("Preparation", [43]))


def compare(c: Claims) -> str:
    rows = [
        ["A · displaced start", c.v("phase_err.0.0.err_a"), c.v("phase_err.22.5.err_a"), c.v("phase_err.45.0.err_a"),
         c.v("robust_a_after")],
        ["B · bias pulse", c.v("phase_err.0.0.err_b"), c.v("phase_err.22.5.err_b"), c.v("phase_err.45.0.err_b"),
         c.v("robust_b_after")],
        ["C · push during ramp", c.v("c_phase.0.0.err"), c.v("c_phase.22.5.err"), c.v("c_phase.45.0.err"),
         c.v("robust_c_after")],
        ["D · displace onto the codeword", c.v("d_err"), c.v("d_phase.22.5.err"), c.v("d_phase.45.0.err"),
         c.v("d_err_circuit")],
    ]
    head = ["Method", "error, no phase error", f"at {setting('22.5°')}", f"at {setting('45°')}", "with circuit terms"]
    costs = cols("1fr 1fr 1fr 1fr",
                 stack(kicker("A"), note("Stabilization must be <b>off</b> during the push, then switched on: "
                                         "extra sequencing and a turn-on calibration."), delay=4),
                 stack(kicker("B"), note(f"Stays on, but needs a drive of {c.v('b_mhz')} MHz, far above "
                                         f"{G2}. Most tolerant of phase error."), delay=4),
                 stack(kicker("C"), note("Weak drive, lower error than A and B, but adds a buffer-drive ramp of about a "
                                         "microsecond and needs amplitude control."), delay=5),
                 stack(kicker("D"), note("Same drive line and pump-off push as A, aimed at β = α. "
                                         "Needs a phase lock; amplitude is forgiving."), delay=5))
    content = (header("1.3(c) <span class='dim'>· what each method costs</span>",
                      f"With circuit terms, D has <em>{c.v('d_vs_c_circuit')}× less error</em> than C")
               + body(table(head, rows, highlight=[3], delay=2)
                      + f'<div style="margin-top:44px">{costs}</div>'))
    return slide("Method comparison", content + footer("Costs", [34, 38, 40, 43]))


def verification(c: Claims) -> str:
    content = (
        header("Verification <span class='dim'>· why you can trust the numbers</span>",
               "Controls that <em>switch one thing off</em>, and a test suite")
        + body(cols("1fr 1fr 1fr",
                    stats([stat(c.tests(), "tests pass in the package test suite", hot=True),
                           stat(c.count(), "printed results traced into claims.json")], delay=2),
                    stats([stat(c.v("a0_max_err"), "1.0(a): simulation against the exact solution"),
                           stat(c.v("parity_control"), f"parity drift with {KA} switched off")], delay=3),
                    stats([stat(c.v("liou_gamma"), "lifetime rate from the Liouvillian, same in three boxes", "/µs"),
                           stat(c.v("robust_c_before"), f"method C error before circuit terms, "
                                                        f"{c.v('robust_c_after')} after")], delay=4))
               .replace("1.0(a):", setting("1.0(a)") + ":"))
        + footer("Verification", [3, 22, 40], " · tests: results/task1/pytest_summary.txt")
    )
    return slide("Verification", content)


def takeaways(c: Claims) -> str:
    items = [
        ("01", f"The cat inflates to <b>{c.v('n_3us')}</b> photons after an overshoot to {c.v('peak_n')}; "
               f"the box must be sized for the overshoot, not the final value."),
        ("02", f"Pair processes protect parity exactly; memory loss alone limits the superposition to "
               f"<b>{c.v('lifetime_us')} µs</b>, as theory predicts."),
        ("03", f"Displacing exactly onto the codeword, a dark state of the two-mode model, prepares {CODEWORD} "
               f"with error <b>{c.v('d_err')}</b>; with circuit terms {c.v('d_err_circuit')}, "
               f"{c.v('d_vs_c_circuit')}× below method C."),
    ]
    blocks = "".join(f'<div class="stack reveal d{k + 2}"><div class="kicker idx" style="color:var(--red)">{n}</div>'
                     f"{lede(t)}</div>" for k, (n, t) in enumerate(items))
    content = (header("Takeaways", "What we would tell <em>the hardware team</em>")
               + body(f'<div class="cols" style="grid-template-columns:1fr 1fr 1fr">{blocks}</div>')
               + footer("Takeaways", [20, 22, 43]))
    return slide("Takeaways", content, extra_class="takeaways")


SLIDES = [observables, lifetime, superposition, method_a, method_b, overdrive, amplifier,
          method_c, method_d, compare, verification, takeaways]
