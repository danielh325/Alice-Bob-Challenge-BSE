"""Task 1.3(b) animation: three starts under the same stabilization, from real runs.

    OMP_NUM_THREADS=2 PYTHONPATH=.:studies .venv313/bin/python studies/task13_idea_animation.py

Every frame is a saved state of the two-mode model (28 x 12, double precision):

- vacuum: the plain Task 1.1 inflation, grows the even cat;
- method A: memory displaced to beta = 1.5, then stabilized;
- method D: a 50 ns push to beta = alpha with the pump off, then stabilized.

Writes results/task1/idea_animation.npz (Wigner frames, 1 - F(|+alpha>) per frame)
and figures/pitch13_idea.gif. The final errors are checked against the saved
studies (baseline_compare.json, finite_push.json) before anything is drawn.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np

dq.set_precision("double")  # challenge Section 0.5

from task13_dark_prep import NA, NB, P  # noqa: E402
from task13_finite_push import MODEL, memory_drive  # noqa: E402

from catqubit.observables import fidelity_with, memory_state  # noqa: E402
from catqubit.simulate import DEFAULT_SOLVER_TOL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.task1_deck.claims import study_row  # noqa: E402

DATA = ROOT / "results" / "task1" / "idea_animation.npz"
GIF = ROOT / "figures" / "pitch13_idea.gif"
BETA_A = 1.5
T_PUSH = 0.05  # us, the shortest push in finite_push.json
PUSH_FRAMES = 11
HOLD_TIMES = np.concatenate([np.linspace(0.0, 0.4, 33), np.linspace(0.45, 2.0, 32)])
XMAX = 3.6
NPIXELS = 121
MATCH_RTOL = 1e-2  # frame grid differs from the studies' grid; adaptive solver, same tolerance

STARTS = (
    ("vacuum", "Start in vacuum"),
    ("A", f"Push to β = {BETA_A} (method A)"),
    ("D", "Push to β = α (method D)"),
)


def evolve(h, rho0, tsave):
    return dq.mesolve(
        h,
        MODEL.jump_ops,
        rho0,
        jnp.asarray(tsave),
        method=dq.method.Tsit5(rtol=DEFAULT_SOLVER_TOL, atol=DEFAULT_SOLVER_TOL),
        progress_meter=False,
    ).states


def push_states():
    """finite_push.py's pump_off push: constant drive, only memory loss acts."""
    ka = P.kappa_a
    u0 = P.alpha * ka / 2 / (1 - math.exp(-ka * T_PUSH / 2))
    vacuum = dq.tensor(dq.fock(NA, 0), dq.fock(NB, 0))
    tsave = np.linspace(0.0, T_PUSH, PUSH_FRAMES)
    return evolve(memory_drive(lambda t: u0, 0.0, 1.0), vacuum, tsave)


def frames(states):
    plus = dq.coherent(NA, P.alpha)
    walls, errors = [], []
    for k in range(states.shape[0]):
        rho = memory_state(states[k], NA, NB)
        walls.append(np.asarray(dq.wigner(rho, xmax=XMAX, ymax=XMAX, npixels=NPIXELS)[2]))
        errors.append(1.0 - float(fidelity_with(rho, plus)))
    return np.stack(walls), np.asarray(errors)


def simulate():
    push = push_states()
    starts = {
        "vacuum": dq.tensor(dq.fock(NA, 0), dq.fock(NB, 0)),
        "A": dq.tensor(dq.coherent(NA, BETA_A), dq.fock(NB, 0)),
        "D": push[-1],
    }
    out = {}
    for key, _ in STARTS:
        w, e = frames(evolve(MODEL.H, starts[key], HOLD_TIMES))
        out[f"w_{key}"], out[f"e_{key}"] = w, e
        print(f"{key:6s}: 1 - F(+alpha) at 2 us = {e[-1]:.4e}")
    out["w_push"], out["e_push"] = frames(push)
    out["t_push"], out["t_hold"] = np.linspace(0.0, T_PUSH, PUSH_FRAMES), HOLD_TIMES
    return out


def check(out):
    def base(start, beta):
        row = {"start": start, "beta": beta, "model": "model"}
        return study_row("baseline_compare", row)["error"]

    expected = {
        "vacuum": base("vacuum", 0.0),
        "A": base(f"A_{BETA_A}", BETA_A),
        "D": study_row("finite_push", {"kind": "pump_off", "duration": T_PUSH})["error"],
    }
    for key, ref in expected.items():
        got = out[f"e_{key}"][-1]
        status = "ok" if abs(got - ref) <= MATCH_RTOL * ref else "MISMATCH"
        print(f"check {key:6s}: {got:.4e} vs saved {ref:.4e}  {status}")
        if status != "ok":
            raise SystemExit(f"{key} does not reproduce the saved study")


def main() -> None:
    if DATA.exists() and "--rerun" not in sys.argv:
        out = dict(np.load(DATA))
    else:
        out = simulate()
        np.savez_compressed(DATA, **out)
    check(out)
    from task13_idea_render import render  # noqa: PLC0415

    render(out, STARTS, GIF, XMAX)
    print(f"wrote {GIF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
