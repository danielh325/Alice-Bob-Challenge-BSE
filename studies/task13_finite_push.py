"""Task 1.3(b): method D with a real, finite push instead of starting in |alpha>.

Every run prepares for T, then holds 2 us under the standard stabilization (two-mode model,
28 x 12 box, double precision) and reports 1 - F(|+alpha>) at the end of the hold.

Preparations (memory drive H = i u(t) (e^{i phi} a^dag - h.c.), so d<a>/dt = u - kappa_a <a>/2):

- ideal:    start in |alpha>|0> (the notebook's method D);
- pump_off: stabilization off during T, constant u sized so <a>(T) = alpha despite memory loss;
- pump_on:  same pulse, stabilization left on from t = 0;
- moving:   moving target alpha(t) = alpha sin^2(pi t / 2T), buffer drive eps_d(t) = g2 alpha(t)^2
            and u(t) = alpha'(t) + kappa_a alpha(t) / 2, so |alpha(t)>|0> stays dark throughout.

With the Task 2.4(a) circuit terms, a Kerr-cat correction -(K_a/2) alpha(t)^2 (a^dag^2 + a^2)
cancels the memory Kerr on |alpha(t)> during preparation and/or hold. The circuit terms are kept on
during a pump-off push too, although they are pump-induced (conservative). The correction is added
to the simulator only. Writes results/task1/finite_push.json.
Run: PYTHONPATH=.:studies .venv313/bin/python studies/task13_finite_push.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import dynamiqs as dq
import jax.numpy as jnp

dq.set_precision("double")  # challenge Section 0.5; float32 cannot resolve errors near 1e-6

from task13_dark_prep import NA, NB, P, circuit_hamiltonian  # noqa: E402

from catqubit.models import two_mode_model  # noqa: E402
from catqubit.observables import fidelity_with, memory_state  # noqa: E402
from catqubit.simulate import DEFAULT_SOLVER_TOL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "task1" / "finite_push.json"
T_HOLD = 2.0
HOLD_GRID = jnp.linspace(0.0, T_HOLD, 21)  # same grid as the notebook's prepare()
PREP_STEPS = 11
DURATIONS = (0.05, 0.2, 1.0)  # us
ERROR_DURATION = 0.2
PHASE_ERROR_DEG = 22.5
AMPLITUDE_ERROR = 0.9

MODEL = two_mode_model(NA, NB, P)
A, B = dq.destroy(NA, NB)
AD, BD = A.dag(), B.dag()
H_PAIR = P.g2 * (A @ A @ BD + AD @ AD @ B)
TWO_PHOTON = AD @ AD + A @ A
K_A = json.loads((ROOT / "results/task24a/effective_terms.json").read_text())["terms"]["kerr_a"][
    "value_rad_per_us"
]


def evolve(h, rho0, tsave):
    res = dq.mesolve(
        h,
        MODEL.jump_ops,
        rho0,
        tsave,
        method=dq.method.Tsit5(rtol=DEFAULT_SOLVER_TOL, atol=DEFAULT_SOLVER_TOL),
        save_states=False,
        progress_meter=False,
    )
    return res.final_state


def memory_drive(u, phase_deg, amplitude):
    """Time-dependent i s u(t) (e^{i phi} a^dag - h.c.) for a real envelope u(t)."""
    phase = math.radians(phase_deg)
    op = 1j * (math.e ** (1j * phase) * AD - math.e ** (-1j * phase) * A)
    return dq.modulated(lambda t: amplitude * u(t), op)


def prepare(kind, duration, circuit, correct, phase_deg, amplitude):
    vacuum = dq.tensor(dq.fock(NA, 0), dq.fock(NB, 0))
    if kind == "ideal":
        return dq.tensor(dq.coherent(NA, P.alpha), dq.fock(NB, 0))
    tsave = jnp.linspace(0.0, duration, PREP_STEPS)
    extra = circuit_hamiltonian() if circuit else None
    if kind in ("pump_off", "pump_on"):
        ka = P.kappa_a
        u0 = P.alpha * ka / 2 / (1 - math.exp(-ka * duration / 2))
        h = memory_drive(lambda t: u0, phase_deg, amplitude)
        if kind == "pump_on":
            h = h + MODEL.H
    else:  # moving target

        def target(t):
            return P.alpha * jnp.sin(jnp.pi * t / (2 * duration)) ** 2

        def u(t):
            slope = P.alpha * jnp.pi / (2 * duration) * jnp.sin(jnp.pi * t / duration)
            return slope + P.kappa_a * target(t) / 2

        h = H_PAIR + dq.modulated(lambda t: -P.g2 * target(t) ** 2, B + BD)
        h = h + memory_drive(u, phase_deg, amplitude)
        if correct:
            h = h + dq.modulated(lambda t: -(K_A / 2) * target(t) ** 2, TWO_PHOTON)
    if extra is not None:
        h = h + extra
    return evolve(h, vacuum, tsave)


def run(
    kind,
    duration=None,
    circuit=False,
    correct_prep=False,
    correct_hold=False,
    phase_deg=0.0,
    amplitude=1.0,
):
    rho = prepare(kind, duration, circuit, correct_prep, phase_deg, amplitude)
    h = MODEL.H
    if circuit:
        h = h + circuit_hamiltonian()
    if correct_hold:
        h = h - (K_A / 2) * P.alpha**2 * TWO_PHOTON
    final = evolve(h, rho, HOLD_GRID)
    return 1.0 - float(fidelity_with(memory_state(final, NA, NB), dq.coherent(NA, P.alpha)))


def cases():
    yield {"kind": "ideal"}
    yield {"kind": "ideal", "circuit": True}
    yield {"kind": "ideal", "circuit": True, "correct_hold": True}
    for kind in ("pump_off", "pump_on", "moving"):
        for duration in DURATIONS:
            yield {"kind": kind, "duration": duration}
            yield {"kind": kind, "duration": duration, "circuit": True}
    for duration in DURATIONS:
        base = {"kind": "moving", "duration": duration, "circuit": True, "correct_prep": True}
        yield base
        yield {**base, "correct_hold": True}
    yield {"kind": "pump_off", "duration": ERROR_DURATION, "circuit": True, "correct_hold": True}
    for err in ({"phase_deg": PHASE_ERROR_DEG}, {"amplitude": AMPLITUDE_ERROR}):
        yield {
            "kind": "pump_off",
            "duration": ERROR_DURATION,
            "circuit": True,
            "correct_hold": True,
            **err,
        }
        yield {
            "kind": "moving",
            "duration": ERROR_DURATION,
            "circuit": True,
            "correct_prep": True,
            "correct_hold": True,
            **err,
        }


def main() -> None:
    rows = []
    for case in cases():
        t0 = time.time()
        err = run(**case)
        rows.append({**case, "error": err})
        print(f"{json.dumps(case):125s} error {err:.3e}  ({time.time() - t0:.0f} s)", flush=True)
        OUT.write_text(
            json.dumps(
                {
                    "truncation": [NA, NB],
                    "t_hold_us": T_HOLD,
                    "kerr_a_rad_per_us": K_A,
                    "precision": "double",
                    "rows": rows,
                },
                indent=1,
            )
            + "\n"
        )
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
