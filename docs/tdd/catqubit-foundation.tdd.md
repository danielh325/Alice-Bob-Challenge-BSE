# TDD evidence: `catqubit` foundation (Tasks 1.0 and 1.1)

**Source plan:** no `*.plan.md`. Scope chosen from `duquantum_2026_challenge_prompt-v4.md`.
Tasks 1.0(c) and 1.1 are the code both tracks reuse (Section 0.1). The journeys were written during this run.

**Environment:** `.venv313` (miniconda Python 3.13.11), dynamiqs 0.3.7, jax 0.12.0.1, pytest 9.1.1.
The Homebrew Python 3.14 behind `.venv`–`.venv-3` is broken: `pyexpat` fails to load against
`/usr/lib/libexpat.1.dylib`, so pip cannot bootstrap. Those venvs were left untouched.

**Git:** the directory is not a git repository, so there are no checkpoint commits. This report is the record.

## User journeys

1. As a participant, I want the provided parameters converted with 2π handled once, so I never ship the "off by ~6" bug.
2. As a participant, I want correct coherent and cat states (normalized, definite parity) as a ground truth for parity and fidelity checks.
3. As a participant, I want model builders for Tasks 1.0(a/b/c) and the two-mode model, with batched κ₂, so each task is one `dq.mesolve` call.
4. As a participant, I want the population in each mode's top Fock levels over time, so I can show a simulation is converged (B.9, Task 1.2).
5. As a judge, I want simulations to match the analytic predictions in the prompt.

## RED → GREEN

| Stage | Command | Result |
|---|---|---|
| RED | `.venv313/bin/python -m pytest -q` against stubs raising `NotImplementedError` | `37 failed, 1 passed, 21 errors`. Every failure is `NotImplementedError` or missing validation (`DID NOT RAISE`). The 21 errors are the `params` fixture calling the stub `CatParams.reglade()`. The 1 pass is `test_zero_memory_loss_allowed`, a guard against over-validation that the bare dataclass already satisfies. |
| GREEN (1st) | `pytest -q -m "not slow"` | `2 failed, 56 passed`. The dark-state residual ‖L₂\|α⟩‖ was 2.51e-6, just over a guessed threshold of 2.37e-6. |
| Investigation | residual vs N | N=20: 1.1e-2, N=30: 2.5e-6, N=40: 9.8e-11, N=50: 1.2e-14. This is pure truncation error, not a model bug. The test now asserts the real property: tiny compared with the vacuum residual, and shrinking with N. |
| GREEN | `pytest -q` (fast + slow) | `59 passed` |
| Refactor | `ruff check`, `ruff format`, conftest precision moved into `pytest_configure` | `All checks passed!`, `59 passed` |
| Coverage | `pytest --cov=catqubit --cov-branch` | **100%** of lines and branches (102 statements, 22 branches) |

## Test specification

| # | What is guaranteed | Test | Type | Result |
|---|---|---|---|---|
| 1 | `to_angular(f) = 2πf`; provided params are in rad/μs | `test_params.py::test_to_angular_*`, `test_reglade_defaults_*` | unit | PASS |
| 2 | ε_d = α²g₂ (so ε_d/g₂ = 4.41); κ₂ = 4g₂²/κ_b = 2π·0.8956 MHz | `test_buffer_drive_*`, `test_kappa2_*` | unit | PASS |
| 3 | g₂/κ_b = 0.293 and ε_d/κ_b = 1.294 > 1: the adiabatic condition fails | `test_adiabatic_ratios_*` | unit | PASS |
| 4 | Params are frozen; `with_alpha` returns a copy; invalid rates or α raise `ValueError` | `test_params_are_immutable`, `test_with_alpha_*`, validation tests | unit | PASS |
| 5 | Cat states are normalized, have parity ±1 to 1e-10, and populate only even/odd Fock states | `test_states.py` | unit | PASS |
| 6 | ⟨n⟩ = α²tanh α² (even) and α²coth α² (odd); bad parity or dim raises | `test_states.py` | unit | PASS |
| 7 | \|±α⟩ are dark states of √κ₂(a²−α²) up to a truncation error that vanishes with N | `test_models.py::test_coherent_states_*` | unit | PASS |
| 8 | Two-mode H is Hermitian, has ⟨0,1\|H\|0,0⟩ = −ε_d and ⟨0,1\|H\|2,0⟩ = √2·g₂; jumps are [√κ_b b, √κ_a a] | `test_models.py::test_two_mode_*` | unit | PASS |
| 9 | Batched κ₂ gives batched jump operators; too-small truncations raise | `test_models.py` | unit | PASS |
| 10 | Top-level population is resolved per mode, keeps the batch shape, is real, and validates `mode`/`n_levels` | `test_truncation.py` | unit | PASS |
| 11 | **1.0(a)** ⟨n⟩ = 4e^{−κt} to rtol 1e-4 | `test_dynamics.py::test_task_1_0a_*` | integration | PASS |
| 12 | **1.0(b)** two-photon loss stalls at (1−e⁻⁸)/2 = 0.499832 (simulated 0.499832) | `test_task_1_0b_*` | integration | PASS |
| 13 | **1.0(c)** from vacuum: monotonic rise to α² within 1% (4.404 at 2 μs); lossless case reaches α²tanh α² | `test_task_1_0c_*` | integration | PASS |
| 14 | **1.0(c)** κ₂ ∈ {¼, 1, 4}×κ₂ in one batched call: same steady state, faster for larger κ₂ | `test_task_1_0c_kappa2_sweep_*` | integration | PASS |
| 15 | **1.1** two-mode (N_a=24, N_b=6): overshoot > 1.1α², settles within 3% of α², buffer peaks above 1 photon then empties | `test_task_1_1_*` (`slow`) | integration | PASS |

## Measured values (Task 1.1, N_a=24, N_b=6)

- Memory ⟨n⟩ peaks at **6.12** at t = 0.23 μs, then settles to **4.402** at 2 μs (α² = 4.41).
- Buffer ⟨n⟩ peaks at **1.89**, then falls to 1e-4.
- Maximum population in the memory's top 2 levels: **9.4e-5**. Maximum in the buffer's top level: **4.6e-2**.
  **N_b = 6 is too small for the transient.** This is the starting point for the Task 1.2 convergence study.

## Known gaps / next steps

- Section 2 needs fitting helpers (exponential decay → Γ_X, Γ_Z). These are not built yet (YAGNI until Task 2.1 starts).
- No convergence-study helper yet (Task 1.2). `top_level_population` is its building block.
- The `slow` marker is a category only. The two-mode test takes about 4 s.

---

# Round 2: Tasks 1.2 and 1.3 (observables, convergence, preparation)

**Journeys:**
1. As a participant, I want memory parity, purity, fidelity and a superposition-vs-mixture
   measure, so Task 1.3(a) can be shown quantitatively.
2. As a participant, I want a helper that finds where a convergence sweep stops changing.
3. As a participant, I want large truncations that don't save every state.
4. As a participant, I want a displaced start that prepares |+α⟩.

| Stage | Command | Result |
|---|---|---|
| RED | `pytest tests/test_observables.py tests/test_convergence.py tests/test_preparation.py` against stubs | `29 failed, 4 errors`. All 33 are `NotImplementedError`. |
| GREEN (1st) | same | `8 failed, 25 passed`: a **real bug**. Observables read the Hilbert dimension from `shape[-1]`, which is 1 for a ket. Fixed to `shape[-2]`. |
| GREEN | same | `33 passed` |
| RED | new `top_level_projector` and `inflate(extra_exp_ops=, save_states=)` tests | `5 failed` (`NotImplementedError` / unexpected keyword) |
| GREEN | full suite | `97 passed` |
| Bug found in the notebook | N_a = 32 runs showed more top-level population than N_a = 28, some of it negative | Traced to the ODE tolerance (dynamiqs default 1e-6), not truncation |
| RED | `test_inflate_does_not_leak_population_into_top_levels_through_solver_error` (N_a=32, N_b=4, 1 µs) | Failed: top-2 memory population max **2.1e-4** (with negative entries); the physical value is 6.4e-8 |
| GREEN | `inflate(solver_tol=1e-9)` default | `99 passed`. 100% line and branch coverage (177 statements, 44 branches). `ruff check` clean |

| # | What is guaranteed | Test | Type |
|---|---|---|---|
| 16 | ⟨P⟩ = ±1 for cats and e^{−2α²} for \|α⟩; purity 1 / 0.5; fidelity 1 / 0 / 0.5 | `test_observables.py` | unit |
| 17 | Cat coherence 2Re⟨α\|ρ\|−α⟩ is +1 / −1 for even / odd cats and ≈ 0 for the mixture and for a codeword | `test_observables.py` | unit |
| 18 | `smallest_converged` needs every larger size to agree, and rejects bad input | `test_convergence.py` | unit |
| 19 | `fit_decay_rate` recovers the rate, independent of amplitude, and rejects y ≤ 0 | `test_convergence.py` | unit |
| 20 | The top-level projector's expectation equals `top_level_population` | `test_truncation.py` | unit |
| 21 | **1.3(a)** inflating from vacuum gives F(C₊) > 0.8 and F(\|+α⟩) = 0.5 at 1 µs; the cat coherence decays at 2κ_aα² (±5%); the long-time purity is 0.5 | `test_preparation.py` | integration |
| 22 | **1.3(b)** starting from \|β = 1.5⟩ gives F(\|+α⟩) > 0.99 | `test_preparation.py` | integration |
| 23 | No solver-error leakage into the top Fock levels at N_a = 32 | `test_preparation.py` | regression |

**Notebook-level fixes (not library code):**
- t₉₀ was snapped to the time grid, a 12% error at 4κ₂. It is now interpolated on a 1 ns grid.
- The truncation monitor now plots |value| with the solver noise floor marked.
- Two markdown claims were wrong ("lossless products identical", "fidelity saturates"). Both were
  rewritten from the printed numbers.

---

# Round 3: Track 2, Step 1 (`liouvillian.py`, `rates.py`)

**Journey:** As a Standard Track team, we want exact bit- and phase-flip rates at the physical
cat size (α = 2.1), so Task 2.1 is not limited to α ≲ 1.4.

| Stage | Result |
|---|---|
| RED | 21 new tests against stubs: `21 failed`, all `NotImplementedError`. |
| GREEN (1st) | `1 failed, 18 passed`. The test expected `bit_flip_rate` to refuse α = 0.3. That assumption was wrong: inverse iteration converges whenever the slowest odd mode is simple. The test now exercises the real guard (too few iterations → `ValueError`). |
| Bug found after GREEN | At α = 2.1 with a tapered buffer, the port gave 3.95e-7 / 3.76e-7, where the prototype gave 3.8495e-7 / 3.8539e-7. **Root cause:** H was built from already-projected a and b. Products such as a†²b pass through states a tapered buffer drops, so real matrix elements of H were lost. |
| RED | New test `test_tapered_hamiltonian_is_projection_of_full_hamiltonian` → failed. |
| GREEN | H is built on the full grid, then projected (P H P). α = 2.1 now matches the prototype to 7 digits. Full suite: **137 passed, 100% line and branch coverage**. |

| # | What is guaranteed | Test |
|---|---|---|
| 24 | The sparse L matches `dq.lindbladian` and the dynamiqs two-mode model; L preserves trace | `test_liouvillian.py` |
| 25 | Memory-parity odd and even sectors are exactly decoupled | `test_memory_parity_blocks_decouple` |
| 26 | A tapered truncation projects finished operators (P H P), not factors | `test_tapered_hamiltonian_is_projection_of_full_hamiltonian` |
| 27 | Codewords are exact steady states only when κ_a = 0 | `test_codeword_is_steady_state_only_without_memory_loss` |
| 28 | Inverse iteration matches the slowest odd eigenvalue; the bit-flip mode is >100× slower than the next odd mode | `test_rates.py` |
| 29 | **Γ_X matches a full time simulation of Re⟨a⟩ within 0.1%** (α = 0.8) | `test_bit_flip_rate_matches_time_domain_decay_of_re_a` (slow) |
| 30 | **Γ_Z matches the simulated parity decay within 1%** (α = 1.4) and is within 5% of 2κ_aα² | `test_phase_flip_rate_*` |
| 31 | A tapered buffer reproduces a wide rectangular buffer within 0.1% | `test_tapered_truncation_matches_wide_rectangular_buffer` |
