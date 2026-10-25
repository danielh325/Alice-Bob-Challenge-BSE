# Cat qubits: Introductory and Standard Tracks (Alice & Bob challenge, DuQuantum 2026)

We build up a dissipatively stabilized cat qubit one ingredient at a time (Task 1.0), simulate the
full memory + buffer model (1.1), prove which numerical truncation the results need (1.2), and
show what state inflation prepares and how to prepare a definite codeword instead (1.3), including a
third method of our own that needs no strong drive.
The [Standard Track](#standard-track-section-2-bit-flips-phase-flips-and-readout) then measures the
bit-flip and phase-flip times (2.1, 2.2), reads the qubit out through the buffer (2.3), derives the
circuit terms and tests three ways to read out faster without flipping the cat (2.4).

## Reproduce

```bash
python3.13 -m venv .venv313
.venv313/bin/pip install dynamiqs pytest pytest-cov scipy matplotlib nbformat nbconvert ipykernel
.venv313/bin/python -m pytest -m "not slow"         # 430 tests, ~2.5 min (drop -m to add 19 slow ones)
.venv313/bin/jupyter nbconvert --to notebook --execute --inplace intro_track.ipynb   # ~13 min
```

- `intro_track.ipynb` produces every number and figure below. It is generated from
  `tools/build_notebook.py`, so edit that file rather than the notebook.
- `figures/` holds every plot.
- Slides: `pitch13_deck.html` (8-slide Task 1.3 pitch) and `task1_deck.html` (23-slide Track 1
  deck). Open them in a browser; keep each `-assets/` folder next to its deck. Every number on a
  slide is read from `results/task1/claims.json`, which `tools/extract_task1_claims.py` builds from
  the executed notebook. Rebuild with `tools/build_pitch13_deck.py` and `tools/build_task1_deck.py`;
  `tools/verify_task1_deck.py [deck]` fails if a number has no source or does not match it.
- `studies/task13_*.py` hold the method D studies (finite push, hardware effects, memory Kerr) and the
  two animations. They write `results/task1/*.json`.
- `catqubit/` holds the physics: parameters, states, models, observables, the truncation monitor and the
  convergence helpers. Every piece is tested against an analytic result.

Conventions: $\hbar = 1$, time in µs, rates in rad/µs, double precision. Frequencies are converted
from MHz exactly once, in `catqubit.params.to_angular`. Wigner plots show $x = \mathrm{Re}\,\alpha$,
$p = \mathrm{Im}\,\alpha$.

## Results

| Task | What we found | How we checked it |
|---|---|---|
| 1.0(a) | ⟨n⟩ follows 4e^{−κt}; the state stays coherent at every time | Max deviation 1.2e-8 photons; min fidelity with \|2e^{−iΔt−κt/2}⟩ is 0.999998 |
| 1.0(b) | Two-photon loss stalls at **0.49983**, not 0 | Equals the initial odd population (1−e⁻⁸)/2. Parity conservation sends even population to \|0⟩ and odd to \|1⟩ |
| 1.0(c) | Monotonic rise to ≈α²; κ₂ changes the speed, not the destination | With κ_a = 0, t₉₀κ₂ = 0.850 / 0.850 / 0.851 for κ₂ ×¼ / ×1 / ×4: exact 1/κ₂ scaling |
| 1.1 | ⟨a†a⟩ **overshoots to 6.89** at 230 ns, rings, and stays within 1% of α² after 1.07 µs. The buffer peaks at 2.0 photons, then empties | The effective model does not overshoot. Adiabatic condition: g₂/κ_b = 0.29, and **ε_d/κ_b = 1.29, which violates it** |
| 1.2 | Steady state converged at any N_a ≥ 16, N_b ≥ 4. The **transient peak needs N_a = 28, N_b = 12** (relative tolerance 1e-3). The suggested N_b = 6 underestimates the peak by 11.5% | Separate N_a and N_b sweeps; truncation monitor on both modes; solver-tolerance check (below) |
| 1.2 | Steady state is 0.17% below α², and the offset is **physical** | κ_a = 0 gives α²tanh α² = 4.40870 to 5 digits. The κ_a shift (−0.0061) matches the mean-field −κ_a/(2κ₂) = −0.0052 |
| 1.3(a) | Inflation prepares the **even cat \|C₊⟩, a superposition**, which lives **1.95 µs** | Memory parity commutes with H_2ph and the buffer loss; the κ_a = 0 control holds ⟨P⟩ = 1 to 2e-15. F(\|+α⟩) = 0.500 throughout. The cat coherence 2Re⟨α\|ρ\|−α⟩ (0 for a mixture) and the parity both decay at 0.513/µs, against a predicted 2κ_aα² = 0.515/µs. With no fit at all, the slowest even-parity Liouvillian eigenvalue gives 0.514559/µs (**1.943 µs**), the same to 1e-7 for (N_a, N_b) = (24, 6) to (32, 10): the fit is within 0.3% |
| 1.3(a) | The prepared state is a superposition, not a mixture, even though both give F(\|+α⟩) = ½ | Side-by-side Wigner functions: the prepared state's fringes go negative, the mixture's cut is flat. W(0,0) = (2/π)⟨P⟩ |
| 1.3(b) A | **Displaced start** \|β⟩, then inflate: **F = 0.978 at β = 1, 0.9988 at β = 1.5** | A zero-parameter basin model, F ≈ 1 − ½erfc(√2β), matches the β sweep to within 0.003 |
| 1.3(b) B | **Bias pulse during inflation**, with the stabilization left on: **F = 0.9985** for area 2.5 in 50 ns | Pulses ≤ 10 ns sit on the basin curve (pulse area plays the role of β). Longer pulses at the same area do worse, because the push arrives after the anti-squeezing has amplified vacuum noise |
| 1.3(b) | **Why B needs ~10 g₂: a linear-amplifier model.** Near vacuum the stabilization amplifies the memory's x-quadrature at λ(t) = 2g₂b(t) − κ_a/2, reaching κ₂α² = 24.8/µs. A push at time s is worth ε(s)e^{−∫λ} of initial displacement; feeding that β_eff to the basin model has **no free parameters** | Matches all 6 simulated B pulses with area ≤ 1 (10–100 ns) to within 0.003, and also a weak push with the stabilization on (0.05 g₂: 0.520 vs 0.522). Explains why a weak push fails (F < 0.6) once λ is large |
| 1.3(b) C | **Our method: a weak push while the buffer drive ramps up.** Ramp ε_d linearly over 1 µs and hold a push of **0.91 g₂ (0.70 MHz, 11× weaker than B)**, sized by the amplifier model for β_eff = 1.5: **error 5.7e-4**, half of A (1.2e-3) and B (1.5e-3). The stabilization is never switched off | Converged: 4.67e-4 vs 4.69e-4 at (28,12) vs (36,16) for a target of 2. Ramp sweep: 0.25 µs needs 1.7 g₂ (9.2e-4), 0.5 µs 1.2 g₂ (5.7e-4), 2 µs 0.69 g₂ (1.2e-3) |
| 1.3(b) | **With the real circuit's leftover terms** (buffer Kerr −2.67 MHz, cross-Kerr, memory Kerr from Task 2.4a), C stays best: errors **A 1.3e-3, B 1.3e-3, C 7.8e-4** | Buffer Kerr is the term that matters. It punishes larger pushes: aiming C at β = 2 goes from 4.7e-4 to 1.7e-3, and changing the drive phase does not help |
| 1.3(b) D | **Best method: method A's push set to exactly β = α, then the pump and buffer drive on together.** \|α⟩_a\|0⟩_b is an **exact dark state of the full two-mode model**: the pump term g₂a²b† and the buffer drive ε_d b† cancel on it (ε_d = g₂α²), and b\|α,0⟩ = 0. Error at 2 µs (`intro_track.ipynb` cell 43, double precision): **8.4e-6 (two-mode model), 5.4e-5 (circuit terms)**, against C 5.75e-4 / 7.78e-4: **68× / 14× better. Quote the 14×**: real devices have the circuit terms. With κ_a = 0 the error is 9.3e-12, so the residual is memory loss | β sweep: 1.8 → 1.3e-4, α → 8.4e-6, 2.3 → 4.6e-5, 2.5 → 2.0e-4, so the minimum is at β = α. Amplitude ±10%: 5.9e-5 / 5.1e-5; phase 22.5° → 1.7e-3, 45° → 2.7e-2. With circuit terms the residual is the memory Kerr (−4.9 kHz), the only term that does not vanish on \|α⟩\|0⟩. Method A's own sweep already gives F = 1.0000 at β = 2.0, so D is A tuned to the dark state, not a new mechanism. `studies/task13_dark_prep.py` (figure `figures/1_3b_dark_state.png`) ran in single precision and gives 6.7e-6; errors near 1e-6 are at the edge of float32, so use the notebook values. Real devices already displace first (Réglade 2024, Marquet 2024, Putterman 2025), but none states that β = α is dark, or gives a fidelity |
| 1.3(b) | **Combining A and C does not help.** A small displacement plus the C ramp-and-push: β₀ = 0.5 with C aimed at 1.5 gives 2.6e-4 (model) but 6.1e-4 (circuit); aimed at 2.0, 7.8e-4 / 2.2e-3; β₀ = 1.0: 3.3e-3 | The push and the displacement add (amplifier model), so the hybrid overshoots, and the buffer Kerr punishes the larger buffer excursion. Going straight to β = α avoids the buffer transient entirely |
| 1.3(b) | **Both methods overdrive:** a push of 4 gives F ≈ 0.955 | Unchanged at N_a = 40 (so not truncation), absent in the effective model (F = 1.000). It is the non-adiabatic buffer again. The useful push is about α |
| 1.3(c) | Costs: a memory drive line, phase locked to the pump frame (error at 22.5° / 45°: A 0.005 / 0.043, **B 0.004 / 0.013**, C 0.002 / 0.037, **D 0.0017 / 0.027 model, 0.0014 / 0.015 circuit**), and either switching the stabilization off (A, D), a drive about 10× g₂ (B), or a 1 µs ramp with an amplitude-controlled buffer drive (C). **D is the most forgiving in amplitude:** ±10% in \|β\| gives only 5–6e-5 (model) and 0.9–1.0e-4 (circuit), because \|β\| ≠ α relaxes back along the stable direction. D's real cost is the same as A's: the pump must be off (or the displacement fast compared with 1/κ₂ = 0.18 µs) while the memory is displaced, and the phase lock | The basin model is only an upper bound for phase errors: the p part of the push flows back to the saddle at the origin (seen in the effective model too) |

### The finding we would most like judges to see: solver tolerance

Our first convergence sweep used dynamiqs' default ODE tolerance (1e-6). At N_a = 32, the largest
truncation and the reference everything else is compared against, it showed **more** population in
the top memory levels than N_a = 28 (1e-3 vs 2e-4), and some of that population was **negative**.
Truncation error cannot do that. At a tolerance of 1e-9 the spurious population drops to 1e-5, the
steady state moves by 2.5e-5 relative, and the runtime stays the same. A convergence study has to
converge the solver as well as the Hilbert space. The package now defaults to 1e-9, and a regression
test pins this behavior. The notebook shows both tolerances side by side.

## How we verified things

- **At the end of the Introductory Track: 114 unit and integration tests, with 100% line and branch coverage of `catqubit`.** Each physical claim
  is pinned to an analytic result: cat-state parity, α²tanh α², dark states of the effective dissipator,
  Hamiltonian matrix elements, the e^{−κt} decay, the 1.0(b) stall, the phase-flip rate, and the codeword
  preparation. `docs/tdd/catqubit-foundation.tdd.md` records the test-first evidence for the first part of
  the package.
- **Controls:** each "is this physical?" question gets a run with one ingredient switched off
  (κ_a = 0 for the parity argument and the semiclassical offset).
- **Convergence in three directions:** N_a, N_b and solver tolerance, each with a stated tolerance.
- **What we got wrong along the way and fixed:**
  - A dark-state test tolerance was set below the real truncation error.
  - t₉₀ was snapped to the time grid, a 12% error at 4κ₂.
  - The default solver tolerance, described above.
  - We first claimed the basin model also predicts phase errors; the sweep showed it is only an upper bound.
  - We first claimed method A cannot overdrive; a β = 4 start showed it can.
  - Our first try at a weak-drive method (a weak square push with the stabilization on) failed at F ≈ 0.55. The amplifier model explained why, and the ramp in method C came out of it.

## Known limitations

- The fitted superposition lifetime does not model the transient before 2 µs, where the photon number
  is larger than α². The Liouvillian eigenvalue confirms the late-time rate, not the transient.
- Method C uses a linear ramp; we did not optimise the ramp shape. The amplifier model is linear, so it
  does not predict the overdrive floor at large pushes (about 1e-3 for B's best pulse).
- The circuit check adds the static Kerr and cross-Kerr terms from Task 2.4a. It does not include how a
  strong memory drive itself changes the ATS terms, which matters most for method B.
- There are no thermal photons (n_th = 0) anywhere, matching the challenge model.

# Standard Track (Section 2): bit flips, phase flips and readout

Parameters for the whole track are Réglade et al. 2024, Table S2:
g₂ = 4.794, κ_b = 16.34, κ_a = 0.0584, κ₂ = 4g₂²/κ_b = 5.63 (all rad/µs).
**Convention:** Γ_X is the decay rate of Re⟨a⟩. That is twice the one-way jump rate, and every
number below uses it.

## Reproduce (Track 2)

**Live demo:** `track2_demo.ipynb` recomputes one headline number per task live (Γ_X and Γ_Z at
α = 1.4–2.1, the α = 2.1 buffer response, c₂ at α = 1.4) and shows the full-sweep figures. It runs in
about 30 s. Regenerate it with `python tools/build_track2_notebook.py`, then
`jupyter nbconvert --to notebook --execute --inplace track2_demo.ipynb`.

Heavy runs share the machine, so prefix them with `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2` and
run from the repository root with `PYTHONPATH=.:studies`. Each study writes `results/<task>/*.json`
and `figures/track2/<task>_*.png`. Plot-only modes reread the JSON and finish in seconds.

| Task | Command | Plot only |
|---|---|---|
| 2.1 | `python studies/task21_bitflip.py all` (parts: `sweep large timedomain effective thermal thermalcheck adiabatic buffer analyze figures`) | `... analyze figures` |
| 2.2 | `python studies/task22_phaseflip.py` (~25 min) | — |
| 2.3 | `python studies/task23_readout.py response map errors errorcheck plots` | `... plots` |
| 2.4(a) | `python studies/task24a_symbolic.py` (~11 s), `python studies/task24a_circuit.py` | `... figures` |
| 2.4(b) | `python studies/task24_tradeoff.py`, `... --r10` (Stark / cross-Kerr) | `... --plot` |
| 2.4(c) | `python studies/task24c_protocol.py`, `python studies/task24_optimize.py`, `python studies/task24_squeezing.py` | `--plot-only`, `--stage plot`, `--plot` |
| 2.4(c–d) | `python studies/task24d_readout_upgrades.py` (~15 min), `python studies/task24e_time_domain.py` (trajectories ~40 min, read cycles ~5 min) | `... plot` |

## Results (Track 2)

### 2.1 Bit-flip time and the exponent γ

| What we found | How we checked it |
|---|---|
| **Γ_X(α = 2.1) = 3.85e-7 /µs** (T_X ≈ 2.6 s with n_th = 0). Rates computed for α = 0.8–2.5 | Rate = slowest nonzero Liouvillian eigenvalue, −λ₁. Converged to ±0.23% in all four truncation knobs (memory and buffer levels, extra buffer levels, memory taper). α = 2.3 and 2.5 are converged in two knobs only (±0.9%, ±4%). Time-domain fits of Re⟨a⟩(t) for α = 0.8–1.4 agree with the eigenvalue to 2e-12 |
| **γ depends on the window.** Plain fit Γ_X ∝ e^{−γα²}: γ = 3.03 (α 0.8–1.4), 2.33 (1.4–2.1), **1.89 (1.8–2.5)**. With a Putterman α² prefactor: 3.86 / 2.66 / 2.11 | `figures/track2/task21_local_exponent.png` plots the local exponent −d ln Γ_X / d(α²): it peaks at 3.1 near α = 1.2, then falls to 1.90 at α = 2.1 and 1.77 at α = 2.5. Single-mode effective model: 2.23 at 2.1, approaching 2.01 by α = 2.7 |
| **Consistent with the deck's γ ≲ 2 only at large α.** The asymptotic exponent is reached from above, so a fit on the small-α window overstates γ | The fits disagree by more than their residuals, so we report the window with every γ rather than a single number |
| **Fermi's golden rule fails at α = 2.1, by a factor of 34.** The first-order term dominates at small α (alone it fits γ = 3.22, which is why small-α fits give γ ≈ 3); a second-order path dominates at large α. It carries 16% of the rate at α = 1.4 and **90% at α = 2.1**. The two orders are equal at α = 1.75 (asymptotic estimate 1.79) | Perturbative rates computed term by term against the exact eigenvalue (`catqubit/bitflip_theory.py`, 100% covered) |
| **The two-mode model flips faster than the effective model:** ×1.12 at α = 1.4, ×3.29 at α = 2.1, ×5.19 at α = 2.5 | Scan g₂ → s g₂, κ_b → s² κ_b, which keeps κ₂ and κ_a fixed. At α = 2.1 the excess shrinks as 1/s² (2.29, 0.67, 0.21, 0.059 for s = 1, 2, 4, 8). The extra channel is non-adiabatic, ∝ κ₂/κ_b. *We tested and rejected one explanation (excess ∝ 0.34α⁴): the data do not collapse onto it* |
| **Thermal photons dominate in practice.** With n_th = (0.10 memory, 0.011 buffer), Γ_X rises 1014× at α = 2.1, and γ fitted over 1.4–2.1 is **1.45**. Réglade measured ×4.2 per photon, i.e. γ = ln 4.2 = **1.44** | Buffer heat alone gives ×970 and memory heat alone ×29, so the buffer channel dominates. Γ_X needs about 11 buffer levels to converge (`task21_buffer.png`) |

### 2.2 Phase-flip time

| What we found | How we checked it |
|---|---|
| **Γ_Z = 2κ_aα² holds: slope (0.99987 ± 0.0001) × 2κ_a** for α ≥ 1.4 (0.9962 over the full range 1.0–2.6). Γ_Z(2.1) = 0.5146 /µs | Three independent methods agree to 3e-7 for ten α values: a free-offset fit after 1 µs, the exact identity d⟨P⟩/dt = −2κ_a⟨a†aP⟩, and the even-block Liouvillian eigenvalue |
| Small α deviates by +3.2% (α = 1.0) | The coth / 1/cosh(2α²) correction to the formula holds only for κ_a/κ₂ → 0. At finite κ_a/κ₂ there is an extra −0.47 κ_a/κ₂ (α = 1) |
| The instantaneous rate starts at 2κ_aα² tanh α² = 0.5152, rings by 0.3%, and settles within 1 µs | `figures/track2/task22_instantaneous.png` |
| **Thermal photons explain most of Réglade's excess.** With n_th = 10% / 1.1%, Γ_Z = 2κ_a[α²(1 + 2n_th) + n_th] to 0.1%: slope +20%, against Réglade's reported +25% | Intercept 0.0113 against 0.0117 predicted. *That this explains 20 of the 25 points is our hypothesis; the remaining 5 points are not explained* |

### 2.3 Readout through the buffer

| What we found | How we checked it |
|---|---|
| **Plateau ⟨b⟩ss = ∓ε_Z/(2αg₂) holds to 0.1%** (−0.011895 against −0.011905 at α = 2.1, ε_Z = 0.05g₂) | The codeword difference matches linear theory; the imaginary part is a codeword-independent κ_a offset (`catqubit/readout_dynamics.py`, 53 tests) |
| **Ringing at ν = 2αg₂ = 3.14 MHz**, decaying at about 1.15 × κ_b/4. **1% settling takes 0.97 µs** (linear theory: 1.13 µs) | Liouvillian spectrum: two near-degenerate clusters at Im λ = 20.0 rad/µs (3.19 MHz), at ν itself rather than ν√(1−ζ²). The extra damping fits κ₂/(2α²) *(empirical, a hypothesis)*. Linear theory is off by at most 8.3% of \|b_ss\|, at the first peak (t = 0.18 µs) |
| **Readout error against integration time** (ε_Z = 0.3g₂, ideal homodyne): optimum error 0.105 at α = 1.2 (T = 3.5 µs), 0.0139 at α = 1.8 (T = 20), **2.9e-3 at α = 2.1 (T = 30)** | Exact full-counting statistics: tilted Liouvillian plus Fourier inversion, checked against Gaussian and Chernoff bounds. A matched filter beats the boxcar (0.060 vs 0.105 at α = 1.2, T = 4). At α = 1.2, stochastic master-equation trajectories (8000 per codeword) reproduce the mean record to 0.3% |
| **The readout drive itself causes bit flips:** at α = 2.1, Γ_X rises 500–700× when ε_Z = 0.3g₂ (to 2–3e-4 /µs). These flips, not the signal, set the optimum T | The 2.3 solver gives 2.9e-4 (2.5e-4 with more levels); the converged 2.4(b) quadratic law gives 2.0e-4. See Known limitations |

### 2.4(a) Circuit terms from the ATS Hamiltonian

| What we found | How we checked it |
|---|---|
| **g₂ = 0.728 MHz** (prompt's 0.763 MHz times the zero-point factor e^{−Φ²/2} = 0.954) | Normal ordering done symbolically (`catqubit/normal_order.py`, 28 tests) and checked against an all-orders Fock-space computation: agreement to 7e-8 (g₂) and 8.6e-6 (Kerr) |
| **Unwanted terms:** pump Stark shifts Δ_a = 0.929, Δ_b = 23.06 MHz; Kerr K_a = −4.9 kHz, **K_b = −2.67 MHz (3.7 \|g₂\|)**; cross-Kerr χ_ab = −0.200 MHz; photon-number-dependent g₂ corrections (+0.03 MHz per buffer photon) | No first-order number-conserving terms survive, because of the 5:6:4 frequency ratio and the sine parity of the ATS. Floquet check: `figures/track2/task24a_floquet_check.png` |
| **The pump frequency must be retuned:** 2Δ_a − Δ_b = −21.2 MHz, about 29 \|g₂\| off resonance | Follows from the Stark shifts above |
| The readout tone gives ε_Z [a† − (φ_a²/2) a†²a − φ_b² a†b†b]: **each buffer photon weakens the drive by 9%**. Pump × tone adds a b† term of +0.0775 ε_Z | Same symbolic expansion |

### 2.4(b) Contrast against bit flips

| What we found | How we checked it |
|---|---|
| **Γ_X(ε_Z) = Γ_X(0) + c₂ε_Z²**, with c₂ = 0.045, 0.016, 4.7e-3, 1.1e-3, 9.9e-5 µs for α = 1.2, 1.4, 1.6, 1.8, 2.1. The quadratic law holds to 10% up to ε_Z ≈ 0.32g₂ | c₂ from linear response of the Liouvillian eigenvalue, checked against full eigenvalues at finite ε_Z (deviation < 0.1% at 0.02g₂) and stable to 6e-4 under N_a + 4, N_b + 1 and extra buffer levels |
| **The price of information is fixed by α, not by ε_Z.** Contrast \|Δb\| = ε_Z/(αg₂) is linear in ε_Z and flips grow as ε_Z², so drive-induced flips per unit of measurement rate (κ_b\|Δb\|²/2) are the same at any ε_Z: **0.18, 0.089, 0.034, 0.010, 0.0012** for α = 1.2 → 2.1, falling as e^{−1.69α²}. Once c₂ε_Z² ≫ Γ_X(0), driving harder reads faster but buys no better error per shot | `figures/track2/task24_tradeoff_ratio.png`. A single-mode Liouvillian reproduces Dubovitskii's exact Eq. 52 (PRA 111, 012617) to 1–2% for α ≥ 1.6, so the code and the factor-2 convention are right; the **two-mode c₂ is 1.5–3.5× the single-mode value** (α = 1.2 → 2.1): the non-adiabatic buffer amplifies drive-induced flips, more so at large α (`studies/task24d_readout_upgrades.py single`). Contrast stays within 2% of the adiabatic value |
| **Stark shift and cross-Kerr do not change this in the operating range.** The ε_Z⁴ term overtakes ε_Z² only at ε_Z/g₂ = 10–32 (circuit χ) or 2.6–8 (Lescanne-2020 χ); χ raises c₂ by 1.9% (circuit) at α = 1.4 | `figures/track2/task24_epsz4.png` |

### 2.4(c–d) Strategies to improve readout

| Strategy | Result | Why it works, and the caveat |
|---|---|---|
| **Adaptive stopping with an erasure flag** (hidden-Markov filter + sequential test, stop when confident, flag ambiguous shots) | **Rerun with the computed Γ_X** (2.05e-4 /µs at α = 2.1, ε_Z = 0.3g₂, η = 0.6): **error 1.3e-3 with 1.3% of shots flagged, mean 20.9 µs**; without the flag the floor is 2.1e-3 at 31 µs. **This readout cannot reach 1e-3 at α = 2.1.** (Our earlier 15.6 µs / 1.9e-4 used a Γ_X 4× too low.) At α = 1.4: ×2.9 faster than a boxcar at equal error. D'Anjou's Gaussian-latching reference gives ×2.77 | Most shots are decided early; the slow tail is flagged as an erasure. The adaptive stop helps, but it cannot beat the price of information below. The record model is a classical surrogate, validated against stochastic master-equation trajectories at α = 1.2. Adapting ε_Z(t) to the filter's belief does not help either: for two hypotheses with rate ∝ ε² and flips c₂ε², a constant maximal drive then stop is optimal (Chernoff; Nitinawarat arXiv:1205.0858) |
| **Inflate-then-read** (control-level): ramp α up only for the readout, then back | Read at α_r = 2.5 with the same contrast (ε_Z = 0.36g₂): Γ_X falls 29× to 7.1e-6. **Error 4.9e-4 at 19 µs** (no flag), or **6.0e-5 + 5.3e-4 flagged at 37 µs** | The price of information falls as e^{−2α_r²}: computed flips per unit measurement rate 1.2e-3 / 2.5e-4 / 4.5e-5 / 8.4e-6 at α = 2.1 / 2.3 / 2.5 / 2.7 (`results/task24d/large.json`, three truncations each, 6% spread at 2.5). Costs: phase flips 2κ_aα_r² during the read (irrelevant for an X readout), ~1 µs ramps each way (not included above; ringing at 2αg₂ is cancelled by ramp lengths of n·318 ns), and real devices gain only up to ~7 photons (Réglade 2024), so 2.5 is a conservative choice |
| **Conversion readout** (hardware-level, our main 2.4(c) proposal): a beam-splitter g_cr(ac† + h.c.) into a separate lossy readout mode c, read with heterodyne | **κ_ro = 0.1κ₂: error 8.6e-4 at 0.59 µs mean** (99% within 1.7 µs), or **1.5e-5 + 2.4e-4 flagged at 1.2 µs**. κ_ro = 0.03κ₂: 8.8e-4 at 2.0 µs, 4.5e-5 at 4.6 µs. **17–52× faster than the ε_Z readout, at errors it cannot reach** | Eliminating c gives the jump √κ_ro a with κ_ro = 4g_cr²/κ_c. Since a\|±α⟩ = ±α\|±α⟩, the codewords are its eigenstates: the meter is **not** the stabilizing channel L = a² − α², so the e^{−2α²} price no longer applies. Flips it adds are single-photon-loss flips (Dubovitskii: κ_ro α² e^{−4α²} + (κ_ro²/2κ₂) e^{−2α²}). Computed (two-mode gap with extra loss): **flips per unit measurement rate 2.3e-6 – 5.1e-5 at α = 2.1, 24–544× below ε_Z readout** for κ_ro/κ₂ = 0.3 – 0.01 (`figures/track2/task24d_price.png`). An explicit three-mode model at α = 1.4 agrees with this to 6% (`results/task24d/threemode.json`). Do **not** convert into the buffer: a² + (g_cr/g₂)a − α² still has coherent dark states, so ⟨b⟩ = 0 for both codewords. Costs: an extra mode and coupler (or the ATS biased off its saddle point for a bilinear term), switched off between readouts. Kerr-cat conversion readouts already reach 99.4% (Grimm 2020, Hajr 2024, arXiv:2511.01027); we found none for a dissipative cat |
| **Time-domain proof of both proposals** (no surrogate, no Liouvillian rate): full memory + buffer (+ readout mode) stochastic trajectories with an η = 0.6 homodyne record, 2000–4000 per codeword, and master-equation read cycles including the switch-on/off of the readout drive or coupler and the α ramps | **α = 2.1, simple boxcar filter:** conversion readout reaches **1e-3 at 2.0 µs and 0 errors in 8000 shots at 3 µs** (Gaussian fit 3.7e-5); the ε_Z readout is still at **4.0e-3 after 40 µs**; inflate-then-read, **ramps included**, 2.0e-3 at 40 µs. **Flips caused by one full read cycle:** ε_Z (31 µs) 4.1e-3; inflate-then-read (31 µs, ramps included) **2.8e-4, 14× fewer**; conversion (1.2 µs) **3.3e-5, 124× fewer**. The α ramps alone, without the read drive, cause no measurable flips (\|p\| < 2e-6). At α = 1.4: conversion 4.4e-3 in 2 µs vs ε_Z 0.18 in 12 µs | `studies/task24e_time_domain.py`, `figures/track2/task24e_trajectories.png`, `results/task24e/summary.md`. Cross-checks: the trajectory SNR growth gives μ² = 5.1 (conversion) and 0.21 (ε_Z, inflate) per µs against the record model's 6.0 and 0.20; the read-cycle flip probabilities match Γ_X·T/2 from the Liouvillian gaps (conversion 3.4e-5 predicted vs 3.3e-5; ε_Z 3.2e-3 vs 4.1e-3, the rest from the edges and the settling buffer; inflate 1.1e-4 for the hold at α = 2.5, the rest from reading during the ramps, where α is smaller and flips are faster). Read cycles run in double precision: in float32 the inflate number was off by 2×. Inflate keeps the buffer signal constant by scaling ε_Z ∝ α(t), so the ramps also carry signal |
| **Shaped readout pulse ε_Z(t)** (2 µs window, 100 pieces, maximise SNR² under a flip budget) | Bit-flip probability needed for error 1e-3: **×2.7 lower at α = 1.4** (0.187 vs 0.498 for a square pulse), ×1.8 lower at α = 1.6 | The flip cost uses the exact drive kernel K(τ), not a quasi-static model. **A quasi-static flip model underestimates the cost of shaped pulses by 54%** (0.0495 vs 0.0762 at α = 1.6). Time-domain Lindblad agrees with the kernel to 5%. Linear-response signal is 7% optimistic against full nonlinear simulation |
| **Squeezed buffer bath** (hypothesis test) | **The naive idea fails:** squeezing the signal quadrature X_b shortens the readout (T × 0.74 at r = 0.15) but heats the cat (Γ_X(0) × 3.4), so net flips get worse (× 1.19). **Squeezing the other quadrature helps a little:** net flip probability × 0.92 at α = 1.4 and × 0.81 at α = 1.8 | It lowers the drive-induced c₂ more than it slows the readout. Idealised: broadband squeezing, η = 1, Gaussian error, small r only. Treat as a direction, not a result |

**What we would recommend (2.4d):** the ε_Z readout pays e^{−2α²} flips per unit of information
because its meter is the stabilizing channel itself (2.4b), so no pulse shape or stopping rule fixes it:
at α = 2.1 it cannot reach 1e-3. Change the meter instead. **Hardware-level:** a conversion readout into
a separate mode reads the codeword through a, whose eigenstates are the codewords: 17–52× faster at lower
error, with all flips as erasure-flagged or below 1e-4. **Control-level, no new hardware:** inflate to
α_r ≈ 2.5 only while reading; this reaches 4.9e-4 with the existing drive line. On top of either, keep
the adaptive stop with an erasure flag (classical processing only). Shaped pulses help only on the ε_Z
readout; the squeezing result is too idealised to recommend.

## How we verified things (Track 2)

- **All 449 tests pass (430 fast, 19 slow), 99% line coverage of `catqubit`.** New Track 2
  modules: `bitflip_theory`, `phaseflip`, `readout_dynamics`, `readout_stats`, `readout_protocol`,
  `normal_order`, `circuit`, `driven_rates`, `optimize`, `eig_gradients`.
- **Every rate is computed two ways:** Liouvillian eigenvalue against time-domain fits (2.1, 2.2),
  symbolic normal ordering against all-orders Fock space (2.4a), exact kernel against time-domain
  Lindblad (2.4c), surrogate record against stochastic trajectories (2.4c).
- **Claims we tested and dropped:** the α⁴ explanation of the two-mode excess (2.1), and squeezing the
  signal quadrature (2.4c). Results marked *hypothesis* above are untested explanations.

## Known limitations (Track 2)

- Driven Γ_X at α = 2.1 in Task 2.3 (ε_Z = 0.3g₂) is converged only to about 15% (2.92e-4 against
  2.48e-4 with more levels; the 2.4(b) value is 2.05e-4). The α = 2.1 readout error is therefore
  uncertain by a similar amount.
- The 2.4(c–d) Monte Carlo uses computed Γ_X and a classical record surrogate. The full-model
  trajectories (`task24e_time_domain.py`) confirm it with a boxcar filter; we have not run the HMM/SPRT
  filter on the trajectories themselves. The conversion-readout rates eliminate the readout mode
  (κ_c = κ_b ≫ g_cr). An explicit memory + buffer + readout-mode Liouvillian at α = 1.4 agrees: flips per
  unit measurement rate 1.11e-3 against 1.05e-3 (κ_ro = 0.03κ₂, 3 readout levels), and 1.80e-3 against
  2.01e-3 (0.1κ₂, only 2 levels, not converged). We did not run the explicit model at α = 2.1 (too large).
  The inflate-then-read Monte Carlo numbers exclude the ramps; the time-domain read cycle includes them
  and halves the gain (29× → 14× fewer flips), because the drive stays on during the ramps.
- Adaptive ε_Z(t) driven by the filter's belief was argued to give no gain (Chernoff), not simulated.
- The squeezing study is idealised (see above).
- We did not verify these references against the original papers: Gautier et al. Eq. 15b and Eq. 32,
  Chamberland et al. 2022. (PRA 111, 012617 Eq. 52 is now checked numerically, above.) Where we use them, we re-derived the result ourselves.

## AI use

> **Team: please review and complete this section before submitting.** The policy (Section 0.3)
> requires stating where and how AI tools were used, and every member must be able to defend
> every result.

We used **Claude Code (Anthropic, Claude Opus)** as a coding assistant. Under our direction it wrote:
- the `catqubit` package and its tests, test-first;
- the notebook builder `tools/build_notebook.py`;
- the first draft of this README and of the notebook's explanations;
- for the Standard Track, the studies in `studies/`, the new `catqubit` modules and their tests. Here it
  worked as several parallel agents, one per task, following `docs/track2_plan.md` and
  `docs/track2_agent_rules.md`. Each agent had to check its numbers two independent ways and label
  untested explanations as hypotheses.

It also ran the simulations. It caught and fixed the solver-tolerance artifact above when it noticed
non-physical (negative) populations. We checked the physics arguments (parity conservation,
mean-field offset, basin model, adiabatic condition) against the challenge text and the tutorials.
*TODO (team): say who reviewed which parts and what you changed.*
