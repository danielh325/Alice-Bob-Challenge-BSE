# Rules for every Track 2 agent and subagent

Read this file and `docs/track2_plan.md` before you start. These rules apply to subagents too: pass this
file's path to every subagent you start.

## 1. File ownership

Several agents work in this folder at the same time. It is not a git repository, so overwriting another
agent's file cannot be undone.

- **Edit only files you own** (listed in your brief). Create new files only under the names your brief gives.
- **Do not edit shared files**:
  - `catqubit/liouvillian.py`, `catqubit/rates.py` and every other existing `catqubit/*.py`;
  - existing tests;
  - `pyproject.toml`, `README.md`, `GUIDE.md`, the notebook and `tools/`.
- If you need a change in a shared file, stop and report it in your final message. Do not work around it by copying the file.
- **Read-only inputs from other agents**: read their `results/<task>/*.json` files if they exist. Never write to them.

## 2. Shared API you can use (tested, 100% coverage)

```python
from catqubit.params import CatParams          # CatParams.reglade(); .with_alpha(a); .g2 .kappa_b .kappa_a .eps_d .kappa2
from catqubit.liouvillian import (
    FockTruncation,         # FockTruncation(na, nb, nb_extra=0, na_taper=1.0); tapered buffer near empty memory
    two_mode_operators,     # (a, b) sparse, projected
    two_mode_hamiltonian,   # (params, trunc, eps_z=0.0)
    two_mode_liouvillian,   # (params, trunc, eps_z=0.0, n_th_a=0.0, n_th_b=0.0), column-stacking vec
    lindblad_superoperator, parity_block, vec,
)
from catqubit.rates import bit_flip_rate, phase_flip_rate, slowest_eigenvalues
# bit_flip_rate / phase_flip_rate accept n_th_a, n_th_b. They use parity blocks, so they are
# only valid when eps_z = 0. Rates are observable decay rates (Re<a>, <P>); one-way jump = rate / 2.
from catqubit.simulate import inflate            # dynamiqs two-mode time evolution, tol 1e-9
from catqubit.observables import memory_state, parity_expectation, purity, fidelity_with, cat_coherence
from catqubit.convergence import fit_decay_rate, smallest_converged
from catqubit.truncation import top_level_population, top_level_projector
```

Known numbers:
- At α = 2.1, `FockTruncation(na=31, nb=5, nb_extra=6, na_taper=13.2)` gives Γ_X = 3.8495e-7 rad/µs in about 15 s.
  The converged value is about 3.849e-7 ± 0.1%.
- Γ_Z = 0.5146 rad/µs at α = 2.1.
- The buffer near the barrier holds about 0.34α⁴ photons. **N_b is the critical truncation.**

## 3. How to work

- **Test first.** For every function, write the test, run it, and see it fail for the intended reason (RED).
  Then implement and see it pass (GREEN). Keep the RED and GREEN evidence for your report.
- **Tests:** `tests/test_<your_module>.py`. Mark tests that take over about 20 s with `@pytest.mark.slow`.
- **Environment:**
  - Use `.venv313/bin/python` from the repository root.
  - Lint with `.venv313/bin/ruff check <your files>` and format with `.venv313/bin/ruff format <your files>`.
  - Run only your own tests: `.venv313/bin/python -m pytest -p no:cacheprovider tests/test_<yours>.py`.
- **Numerics:**
  - Use double precision (`dq.set_precision("double")`), ODE tolerance 1e-9, and `scipy.sparse`.
  - Never build a dense Liouvillian above dimension 2000.
  - Converge every reported number in N_a and N_b, and state the tolerance.
  - Check eigen-residuals.
  - State the factor-2 convention in each figure.
- **Shared machine (10 cores, about 32 GB).** Other agents run at the same time.
  - Run at most **one** heavy job (over 2 GB or over 5 min) at a time.
  - Prefix runs with `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`.
  - Use `python -u`, and run long jobs in the background.
  - Use α ≤ 2.1 for sweeps; save α = 2.5 for single points.
- **Outputs:**
  - Scripts go in `studies/<task>_*.py`. Each must regenerate its outputs from scratch.
  - Numbers go in `results/<task>/*.json`, with the parameters and truncations used.
  - Figures go in `figures/track2/<task>_*.png` (dpi 150, labelled axes with units, a legend).
- **Honesty:**
  - Report what you measured, including failures and hypotheses that did not hold.
  - Never quote a number you did not compute or read from a cited source.
  - Mark a citation UNVERIFIED unless you checked it.
- **Style:** match the existing `catqubit` code. That means type hints, short functions, named constants
  instead of magic numbers, immutable data (frozen dataclasses), and docstrings that state the physics and
  conventions.

## 4. Subagents

You may start subagents (subagent_type "general-purpose") for **independent** pieces of your task. Rules:

- Give each subagent its own files. Two subagents must never edit the same file.
- Give it this rules file, the plan section, and the exact file names it owns.
- Prefer one or two subagents for clearly separable work, such as "derive and test the closed-form
  theory" versus "run the convergence sweep". Do not start a subagent for work that takes a few minutes.
- You are responsible for checking a subagent's output before you report it.

## 5. Final report (your last message)

Keep it to about 600 words or fewer:
1. Files created.
2. Test evidence: RED and GREEN counts, coverage of your modules.
3. Key numbers, with truncations and convergence.
4. Figures produced.
5. Failures, open questions, and any shared-file changes you need.
