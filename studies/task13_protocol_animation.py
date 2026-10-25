"""Task 1.3(b) protocol animation: method D step by step, from real runs.

    OMP_NUM_THREADS=2 PYTHONPATH=.:studies .venv313/bin/python studies/task13_protocol_animation.py

Steps (two-mode model, 28 x 12, double precision):

1. stabilization off: pump and buffer drive off, memory and buffer empty;
2. microwave push on the memory drive line for 50 ns (finite_push.py's pump_off push);
3. pump and buffer drive switched on together;
4. stabilization holds |+alpha> for 2 us.

Writes results/task1/protocol_animation.npz (memory Wigner, 1 - F(|+alpha>), <a>,
<a^dag a>, <b^dag b> and the memory photon-number distribution per frame) and
figures/pitch13_protocol.gif. The final error is checked against finite_push.json
before anything is drawn. Add --rerun to simulate
again instead of reusing the saved frames.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import dynamiqs as dq
import numpy as np

dq.set_precision("double")  # challenge Section 0.5

from task13_dark_prep import NA, NB, P  # noqa: E402
from task13_finite_push import MODEL, memory_drive  # noqa: E402
from task13_idea_animation import MATCH_RTOL, NPIXELS, XMAX, evolve  # noqa: E402

from catqubit.observables import fidelity_with, memory_state  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.task1_deck.claims import study_row  # noqa: E402

DATA = ROOT / "results" / "task1" / "protocol_animation.npz"
GIF = ROOT / "figures" / "pitch13_protocol.gif"
T_PUSH = 0.05  # us, the shortest push in finite_push.json
PUSH_TIMES = np.linspace(0.0, T_PUSH, 31)
HOLD_TIMES = np.concatenate([np.linspace(0.0, 0.4, 33), np.linspace(0.45, 2.0, 32)])


def push_amplitude() -> float:
    """Constant drive (rad/us) that lands the memory on alpha at T_PUSH despite kappa_a."""
    ka = P.kappa_a
    return P.alpha * ka / 2 / (1 - math.exp(-ka * T_PUSH / 2))


def observe(states):
    a, b = dq.destroy(NA, NB)
    plus = dq.coherent(NA, P.alpha)
    out = {"w": [], "err": [], "n_a": [], "n_b": [], "a": [], "pn": []}
    for k in range(states.shape[0]):
        rho = memory_state(states[k], NA, NB)
        out["w"].append(np.asarray(dq.wigner(rho, xmax=XMAX, ymax=XMAX, npixels=NPIXELS)[2]))
        out["err"].append(1.0 - float(fidelity_with(rho, plus)))
        out["n_a"].append(float(dq.expect(a.dag() @ a, states[k]).real))
        out["n_b"].append(float(dq.expect(b.dag() @ b, states[k]).real))
        out["a"].append(complex(dq.expect(a, states[k])))
        out["pn"].append(np.real(np.diag(dq.to_numpy(rho))))
    return {key: np.asarray(val) for key, val in out.items()}


def simulate():
    vacuum = dq.tensor(dq.fock(NA, 0), dq.fock(NB, 0))
    u0 = push_amplitude()
    push = evolve(memory_drive(lambda t: u0, 0.0, 1.0), vacuum, PUSH_TIMES)
    hold = evolve(MODEL.H, push[-1], HOLD_TIMES)
    out = {f"push_{k}": v for k, v in observe(push).items()}
    out |= {f"hold_{k}": v for k, v in observe(hold).items()}
    out |= {"t_push": PUSH_TIMES, "t_hold": HOLD_TIMES, "u0": u0}
    print(f"after push: 1 - F = {out['push_err'][-1]:.3e}, <a^dag a> = {out['push_n_a'][-1]:.4f}")
    err, n_b = out["hold_err"][-1], out["hold_n_b"].max()
    print(f"after 2 us: 1 - F = {err:.4e}, max <b^dag b> = {n_b:.2e}")
    return out


def check(out):
    ref = study_row("finite_push", {"kind": "pump_off", "duration": T_PUSH})["error"]
    got = out["hold_err"][-1]
    status = "ok" if abs(got - ref) <= MATCH_RTOL * ref else "MISMATCH"
    print(f"check D: {got:.4e} vs saved {ref:.4e}  {status}")
    if status != "ok":
        raise SystemExit("method D does not reproduce finite_push.json")


def main() -> None:
    saved = dict(np.load(DATA)) if DATA.exists() else {}
    if "hold_pn" in saved and "--rerun" not in sys.argv:
        out = saved
    else:
        out = simulate()
        np.savez_compressed(DATA, **out)
    check(out)
    from task13_protocol_render import render  # noqa: PLC0415

    render(out, P, GIF, XMAX)
    print(f"wrote {GIF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
