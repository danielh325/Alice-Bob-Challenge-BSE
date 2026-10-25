# Track 2 (Standard) winning plan

*Version 2, 2026-10-24. Sources: 14 literature reviews (citations checked on arXiv abstract or full-text pages)
and 4 numerical prototypes run on our own model. Status: **plan only. No project code is written until the
team approves it.***

---

## 1. Summary

**What we will build.** We will build one numerical framework around the **Liouvillian**. The Liouvillian is
the matrix that moves the two-mode density matrix forward in time. We use four forms of it:

1. Its **slowest eigenvalue** gives the bit-flip rate Γ_X (Task 2.1).
2. Its **next eigenvalue** gives the phase-flip rate Γ_Z (Task 2.2).
3. Its **"tilted" version** gives the exact probability distribution of the readout signal. From that we get
   the exact readout error, including bit flips that happen during the readout (Task 2.3).
4. Its **derivative** gives the sensitivity of the bit-flip rate to every control knob. That lets us optimize
   the readout against bit flips directly (Task 2.4).

**Why this can win.** Most teams will fit noisy curves at small cat sizes. They will judge readout by the gap
between the two average signals. We will give exact rates at the physical cat size, an exact readout error,
and a readout protocol that is about 3× faster. Every step will be checked against an independent method
and tied to a published paper.

**What we must not claim.** Each mathematical tool is published. Our originality is in applying them to this
system and in combining them. Section 9 lists what is new and what is not.

---

## 2. What the prototypes already showed

These results come from scratch prototypes (scratchpad folders `proto_gap/` and `proto_tilted/`). They are
confirmed numbers, but they are not yet project code.

| # | Result | Evidence |
|---|---|---|
| C1 | The parity-odd block of the two-mode Liouvillian gives Γ_X exactly | Matches a full time simulation (`mesolve`) to 1e-12 relative at α = 1.0 and 1.2 |
| C2 | Γ_X is computable at the physical size α = 2.1 (±0.2%) and at α = 2.5 (±3%) | One LU factorization and two linear solves: 1–40 s at α ≤ 2.1, about 14 min at α = 2.5 |
| C3 | The local exponent d lnΓ_X/d(α²) falls from 3.1 to 2.0 and then to **1.86** | Two-mode model with no thermal photons |
| C4 | The buffer truncation N_b, not the memory truncation N_a, limits large α | Near the barrier the buffer holds about 0.34α⁴ photons (13 at α = 2.5) |
| C5 | The tilted Liouvillian gives the exact readout-signal distribution | Agrees with 2000 stochastic trajectories per codeword; KS p-values 0.06–0.93 |
| C6 | The usual Gaussian (erfc) estimate **underestimates** the readout error by 30–55% | Bit flips during readout create heavy tails (wrong-side tail 0.085 vs Gaussian 0.0014) |
| C7 | Readout error has an optimum window length | At α = 1.2 the error is minimal near 3.5 µs, then rises as flips accumulate |
| C8 | At linear order, drive-induced bit flips grow in proportion to measurement rate, independent of ε_Z | The ratio stays at 0.181–0.190 for ε_Z = 0.05–0.2 g₂ (but N_b = 6 was too small; this must be redone) |
| C9 | Broadband squeezed light on the buffer raises Γ_X by 7–47× | Quick check at an exaggerated κ_a, not converged. It shows the direction only |

---

## 3. Physics results we aim to present

Each result below has a status label:
- **Confirmed** means a prototype or a cross-check supports it.
- **Expected** means the literature predicts it and we will test it.
- **Hypothesis** means it is our idea and the test may fail.

### Task 2.1: bit flips

- **R1 (Confirmed).** We compute Γ_X directly at α = 2.1 and 2.5. The prompt says this is only practical for
  α ≲ 1.4 with time simulation. We also give the time-simulation result for α ≤ 1.4, and both methods agree there.
- **R2 (Confirmed).** A fit at small α gives γ ≈ 3, and this is correct physics. Two bit-flip paths compete:
  - A direct jump between the wells. It scales as e^{−4α²} and is what a naive "Fermi golden rule" estimate gives.
  - A two-step path. Single-photon loss first pushes the state out of the cat manifold; the stabilization
    then pulls it into the wrong well. It scales as e^{−2α²}.

  For our parameters the two paths cross at α ≈ 1.5 (Dubovitskii 2025, PRA 111, 012617). This answers the
  prompt's discussion question with numbers.
- **R3 (Confirmed, new).** In the two-mode model the exponent falls *below* 2 at large α (1.86), even with no
  thermal photons. The single-mode theory never goes below 2.

  Our explanation is that the buffer fills up near the barrier, so the effective model fails exactly where bit
  flips happen. This fits Ferrari et al. 2026 (arXiv:2605.24100), who show that leaving the adiabatic regime
  always raises Γ_X. We will show the effect with a plot of the buffer photon number along the switching path.
- **R4 (Expected).** The challenge dropped thermal photons. The source paper (Réglade 2024, Table S2) lists
  10% in the memory and 1.1% in the buffer. With those values we expect γ to move toward the measured value of
  about 1.44. This answers the reviewer note in the prompt.

### Task 2.2: phase flips

- **R5 (Expected).** Γ_Z = 2κ_aα² holds closely. We measure it three ways:
  1. a fit of the parity decay;
  2. the exact identity d⟨P⟩/dt = −2κ_a⟨a†aP⟩, which needs no fit;
  3. the eigenvalue of the even block.
- **R6 (Hypothesis).** Réglade measured a slope 25% above 2κ_a. Thermal photons raise the slope by
  (1 + 2n_th) = 1.2, which could explain most of that. Réglade attributes the excess to other effects, so we
  present this only as a test.

### Task 2.3: readout

- **R7 (Expected).** The buffer readout rings. It behaves like a damped oscillator with natural frequency
  ν = 2αg₂ and damping ratio ζ = κ_b/(8g₂α) ≈ 0.20. Prediction: about 3.1 MHz ringing and about 1.1 µs to settle
  within 1% (Gautier 2023, Eq. 15b). The eigenvalues of the full Liouvillian give the same numbers independently.
- **R8 (Confirmed method).** We give the exact readout error versus window length, including bit flips during
  the window. We also show how much the Gaussian estimate is off.

### Task 2.4: the readout trade-off

- **R9 (Partly confirmed).** At linear order there is no built-in trade-off. The bit flips caused by the readout
  drive grow at the same rate as the information it gains. So the real trade-off must come from effects beyond
  linear order, from the circuit, or from transients. We will redo this test with a converged N_b.
- **R10 (Expected).** At circuit level the readout fills the buffer. The memory–buffer cross-Kerr χ_ab then
  shifts the memory frequency by Δ = χ_ab⟨b†b⟩. That shift adds bit flips that grow as ε_Z⁴ (Dubovitskii;
  Putterman 2025). Retuning the drive frequency during readout cancels it.
- **R11 (Hypothesis).** Squeezed light makes the readout faster but can damage the cat. Squeezing the readout
  quadrature of the buffer anti-squeezes the memory along the line between the wells, which raises Γ_X (C9).
  Narrowband squeezing, or squeezing switched on only during readout, may avoid this. No published paper
  covers this conflict for the two-photon buffer.

---

## 4. The readout protocol we will propose (Task 2.4c)

The protocol combines four parts. Each part has a separate, checked benefit.

1. **Shaped drive.** A two-step "input-shaped" ε_Z(t) removes the ringing. A linear second-order system can be
   brought to rest in half a period, about 160 ns instead of 1.1 µs (McClure 2016 for the readout analogue).
   The gradient optimizer refines the shape on the full nonlinear model. Expected: shorter dead time at the start of readout.
2. **Frequency compensation.** The drive and pump frequencies follow the readout-induced Stark shift, which
   removes the ε_Z⁴ bit-flip term (R10). Expected: a higher usable ε_Z, so a faster readout.
3. **Bayesian filter with early stopping.** A four-state forward filter tracks the initial codeword and the
   current codeword. It therefore gives the correct probability even if a bit flip happens during readout.
   A sequential probability ratio test (SPRT, Wald 1945) stops the readout as soon as the evidence is strong enough.
   - Expected: about 2.8× shorter average readout at an error of 1e-3. This is the D'Anjou et al. 2016 bound
     for Gaussian readout; for example, 258 µs becomes about 92 µs.
4. **Erasure flag.** If the filter sees a bit flip during readout, or reaches the time limit without a decision,
   it reports "erasure" instead of guessing. An error-correcting code repairs erasures more cheaply than hidden
   errors (Kubica 2023; Wu 2022). Bit flips create a visible photon burst in the buffer (Ferrari 2026, arXiv:2607.01375).

**Target.** Vuillot et al. 2026 (arXiv:2605.24262) state that this readout's error must be as small as the
bit-flip probability, or the noise bias is lost. We will report our readout error and time against that requirement.

---

## 5. The optimization method (beyond plain gradient descent)

Plain gradient descent through the simulator already won iQuHACK 2025, so we go further in three ways.

1. **Exact gradient of the bit-flip rate.** For an isolated eigenvalue, dλ/dθ = ⟨l|∂L/∂θ|r⟩ / ⟨l|r⟩. Here r and
   l are the right and left eigenvectors.
   - Every control knob enters L linearly, so ∂L/∂θ is one known matrix.
   - The left eigenvector reuses the same LU factorization (`solve(trans='H')`), so the gradient costs almost nothing.
   - We wrap it as a `jax.custom_jvp` so JAX can use it.
   - The literature review found no published use of this to optimize a cat-qubit bit-flip time.
2. **An exact objective.** The objective is the exact readout error from the tilted Liouvillian (Section 3,
   R8). It replaces contrast or a Gaussian estimate.
3. **A Pareto front.** We do not report one compromise point. We trace the full curve of readout error against
   bit-flip probability, using an augmented Lagrangian method: minimize readout error subject to
   p_X = ∫Γ_X dt ≤ B, for a sweep of budgets B.
   - Check: one point is repeated with a gradient-free optimizer (CMA-ES).
   - Stretch: a robust version optimizes the average over an ensemble of uncertain circuit parameters (g₂, κ_b, χ_ab, detunings).

**Validation of the optimizer**
- Check the eigenvalue gradient against finite differences.
- Check optimized points with stochastic trajectories (`dq.dssesolve`).
- Check that the quasi-static rate assumption, Γ_X(ε_Z(t)), still holds for the optimized pulse. Use a full time simulation at small α.

---

## 6. Software modules (built test-first)

Each module gets an analytic or cross-method test **before** its production code, as in the Intro Track. All
modules go in the existing `catqubit/` package.

| Module | Purpose | First tests (written before the code) |
|---|---|---|
| `liouvillian.py` | Sparse two-mode L (column stacking), parity blocks, non-rectangular truncation | Matches `dq.lindbladian` on random ρ; the cross blocks are exactly zero; trace is preserved |
| `rates.py` | Γ_X (odd block, ratio method), Γ_Z (even block), with residual and convergence checks | Matches `mesolve` at α = 1.0 within 1e-6; matches the Dubovitskii formula in the effective model; detects the zero mode |
| `gradients.py` | `custom_jvp` eigenvalue derivative | Matches finite differences to 1e-6 for g₂, ε_Z and detuning |
| `readout_stats.py` | Tilted Liouvillian, characteristic function, Fourier inversion, exact error ε(T) | Cumulants match finite differences; the Gaussian limit matches erfc; matches trajectories within error bars |
| `readout_protocol.py` | Telegraph-plus-noise record generator, Bayesian filter, SPRT, erasure flag | SPRT on a pure Gaussian record matches Wald's formula; the filter recovers a planted flip |
| `circuit.py` | ATS expansion, second-order terms, Floquet check over the 1.25 ns period | Recovers g₂ = ½E_Jε_pφ_a²φ_b; Stark shift matches the analytic estimate to the order of magnitude |
| `optimize.py` | Augmented Lagrangian, Pareto sweep, CMA-ES baseline | Converges on a toy problem with a known optimum |

---

## 7. Numerical rules for every run

- Double precision; ODE tolerance 1e-9; dynamiqs 0.3.7 or newer.
- Build L with `scipy.sparse` in CSC format. Never build it densely.
- Use COLAMD ordering. One LU per parameter point serves the right eigenvector, the left eigenvector and the gradient.
- Converge N_a, N_b, the non-rectangular buffer window and the solver settings separately, each with a stated
  tolerance. N_b is the critical one.
- Check the eigen-residual ‖Lr − λr‖ ≪ |λ|. Double precision is safe down to Γ_X ≈ 1e-12 rad/µs (prototype C2).
- State the factor-2 convention in every figure. Re⟨a⟩ decays at |λ|; the one-way jump rate is |λ|/2.
- Use homodyne and heterodyne conventions consistently. Heterodyne halves the efficiency η.
- Correct the Intro write-up: the regime number is 8g₂α/κ_b = 4.93, not ε_d/κ_b = 1.29.

---

## 8. Order of work

The steps depend on each other in this order. Times assume one person; with four people, work runs in parallel
as shown in the last column.

| Step | Work | Depends on | Hours | Owner (4 people) |
|---|---|---|---|---|
| 1 | `liouvillian.py` + `rates.py` from the prototypes, with tests | – | 3 | A |
| 2 | Task 2.1: time fits (α ≤ 1.4) + spectral rates to α = 2.5; Dubovitskii overlay; R3 buffer-filling plot | 1 | 3 | A |
| 3 | Task 2.2: three Γ_Z methods; thermal ablation (R4, R6) | 1 | 2 | A |
| 4 | Task 2.3: ⟨b⟩±(t), oscillator overlay, spectral settling (R7) | 1 | 2 | B |
| 5 | `readout_stats.py`: exact error ε(T) at α = 2.1 (R8, C6–C7) | 1 | 3 | B |
| 6 | Redo the linear trade-off test with converged N_b (R9) | 1 | 1 | A |
| 7 | `circuit.py`: Task 2.4a terms and magnitudes; cross-Kerr shift (R10) | – | 4 | C |
| 8 | `gradients.py` + `optimize.py`: Pareto front with shaped pulse + frequency compensation | 1, 5, 7 | 4 | D |
| 9 | `readout_protocol.py`: filter + SPRT + erasure; gains vs a fixed window | 5 | 3 | B |
| 10 | Squeezing test, narrowband vs broadband (R11) | 1, 5 | 2 | C |
| 11 | Validation pass: trajectories, CMA-ES baseline, convergence tables | 8, 9 | 2 | all |
| 12 | Notebook, README, slides, rehearsal of the defence | all | 3 | all |

**Cut line.** If time runs short, finish steps 1–6, 8 and 12 first. Steps 7, 9 and 10 add depth but are not
needed for a complete submission.

---

## 9. Novelty: what is ours and what is not

| Element | Status |
|---|---|
| Liouvillian gap = bit-flip rate | **Published** (Gravina 2023; Ferrari 2026, at these parameters). We cite it. |
| Exponent crossover e^{−4α²} → e^{−2α²} | **Published theory** (Dubovitskii 2025). Our contribution is the two-mode numerical check at large α. |
| Exponent below 2 in the two-mode model (R3) | **Our result**; consistent with Ferrari 2026 |
| Tilted Liouvillian for readout statistics | **Published method** (Landi et al., PRX Quantum 5, 020201, 2024). No published use for cat-qubit readout was found |
| Bayesian filter, SPRT, erasure flagging | **Published separately** (Gambetta 2007; D'Anjou 2016; Ferrari 2026). The combination for this readout appears new |
| Eigenvalue-gradient optimization of a bit-flip rate | **No precedent found** in our searches |
| Squeezing conflict for the two-photon buffer (R11) | **No precedent found.** It is a hypothesis to test |

The searches were broad but not exhaustive. We will say "we found no prior work", not "this is the first".

---

## 10. Risks

| Risk | Effect | Response |
|---|---|---|
| α = 2.5 takes about 14 min per point | Sweeps are slow | Use α ≤ 2.1 for sweeps; keep α = 2.5 for single points |
| R9 or R10 fail when re-tested | The trade-off story changes | Report what we find. A failed hypothesis with a clear test still shows depth |
| Bit flips during readout are too rare at α = 2.1 to resolve by Fourier inversion (~1e-7) | The heavy-tail effect is not visible | Use Chernoff bounds from the tilted generator at real s, and show the effect at α ≈ 1.2–1.8 |
| Circuit-level simulation takes too long | Task 2.4a is shallow | Use the exact 1.25 ns period with one-period propagators, or effective terms only |
| Errors in AI-written physics | Lost credibility | Every claim has a test or a cross-check. The team rehearses defending R1–R11 |

---

## 11. Claims we will not make

- That γ "should be 2". It depends on the α window, the model and n_th.
- That the 1/(1−ν²) signal gain in Vuillot 2026 applies to our readout. It comes from a different coupling and
  is zero in our model.
- That Didier, Bourassa & Blais 2015 show "Heisenberg-limited" readout. That claim belongs to Didier et al., PRL 115, 093604.
- That a second-order-corrected effective model is accurate here. The correction series diverges, since 4ε²(1+2α²) ≈ 3.4.

---

## 12. Main sources

All of these were checked on arXiv abstract or full-text pages.

**Experiments**
- Réglade et al., *Nature* 629, 778 (2024), arXiv:2307.06617
- Lescanne et al., arXiv:1907.11729
- Berdou et al., arXiv:2204.09128
- Marquet et al., arXiv:2307.06761
- Putterman et al., arXiv:2409.17556
- Rousseau et al., arXiv:2502.07892

**Bit-flip theory**
- Dubovitskii, arXiv:2407.17299
- Gautier, Sarlette & Mirrahimi, arXiv:2112.05545
- Carde et al., arXiv:2507.18714
- Thompson & Kamenev, arXiv:2111.10433
- Mirrahimi et al., arXiv:1312.2017
- Guillaud & Mirrahimi, arXiv:1904.09474

**Spectral methods**
- Minganti et al., arXiv:1804.11293
- Gravina, Minganti & Savona, arXiv:2208.04928
- Ferrari et al., arXiv:2605.24100 and arXiv:2607.01375
- Albert & Jiang, arXiv:1310.1523
- Chamberland et al., arXiv:2012.04108

**Readout and statistics**
- Gautier, Mirrahimi & Sarlette, arXiv:2303.00760
- Landi et al., arXiv:2303.04270
- Gambetta et al., arXiv:cond-mat/0701078
- D'Anjou & Coish, arXiv:1311.2979
- D'Anjou et al., arXiv:1507.06846
- Ryan et al., arXiv:1310.6448
- McClure et al., arXiv:1503.01456
- Vuillot et al., arXiv:2605.24262
- Didier, Bourassa & Blais, arXiv:1504.04002

**Erasures**
- Kubica et al., arXiv:2208.05461
- Wu et al., arXiv:2201.03540

**Circuit level**
- Venkatraman et al., arXiv:2108.02861
- Xiao et al., arXiv:2304.13656
- Petrescu et al., arXiv:2107.02343

**Optimization**
- Sener & Koltun, arXiv:1810.04650
- Koswara et al., arXiv:2102.11813

**Past hackathons** (for positioning)
- github.com/iQuHACK/2025-Alice-and-Bob
- github.com/brightlikethelight/iQuHACK_2025
- github.com/Alice-Bob-SW/alice-bob-eth-quantum-hackathon-2026

**Prototype code** (scratch, to port with tests)
- `scratchpad/proto_gap/`: Γ_X solver and convergence logs
- `scratchpad/proto_tilted/`: exact readout statistics and trajectory checks
- `scratchpad/sqz_check.py`: squeezing check
