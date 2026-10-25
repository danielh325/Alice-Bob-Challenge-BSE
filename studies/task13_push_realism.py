"""Task 1.3(b): method D's push with three hardware effects the ideal model leaves out.

Same protocol as task13_finite_push.py: push the memory to |alpha> with the pump off for T, switch
the stabilization on at T_on = T + guard, hold 2 us (two-mode model, 28 x 12 box, double
precision) and report 1 - F(|+alpha>).

1. Memory frequency shift. The pump Stark-shifts the memory (Task 2.4(a): 2nd order 929 kHz,
   exact Floquet 883 kHz), and the frame is set by the pump-on frequency. With the pump off the
   memory is therefore detuned by delta = -stark_a. The true shift is the Floquet value; the push
   is calibrated with no shift, the 2nd-order shift, or the exact one.
2. Heat and loss from the drive port: memory loss kappa_a (1 + extra) and thermal occupation n_th,
   jump operators sqrt(k (1 + n_th)) a and sqrt(k n_th) a^dag, during push and hold. The same
   noise is applied to method A (instant start at beta = 1.5) for comparison.
3. Pulse distortion in the wiring: first-order low-pass (time constant tau) and a reflected echo
   (relative amplitude r, delay d). The drive tail that arrives after T_on overlaps the pump.

Calibration: the complex drive c solves <a>(T_on) = alpha for a rectangular pulse on [0, T] with
lambda = i delta_cal + kappa / 2, i.e. c = alpha lambda e^{lambda guard} / (1 - e^{-lambda T});
the echo amplitude is divided out (1 + r), as an amplitude calibration would. The filter is not
pre-compensated (conservative).
Writes results/task1/push_realism.json.
Run: PYTHONPATH=.:studies .venv313/bin/python studies/task13_push_realism.py
"""

from __future__ import annotations

import cmath
import json
import math
import time
from pathlib import Path

import dynamiqs as dq
import jax.numpy as jnp

dq.set_precision("double")  # challenge Section 0.5; float32 cannot resolve errors near 1e-6

from task13_dark_prep import NA, NB, P, circuit_hamiltonian  # noqa: E402
from task13_finite_push import AD, HOLD_GRID, MODEL, T_HOLD, A, B  # noqa: E402

from catqubit.observables import fidelity_with, memory_state  # noqa: E402
from catqubit.simulate import DEFAULT_SOLVER_TOL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "task1" / "push_realism.json"
TERMS = json.loads((ROOT / "results/task24a/effective_terms.json").read_text())["terms"]
STARK_2ND = TERMS["stark_a"]["value_rad_per_us"]
STARK_FLOQUET = TERMS["stark_a"]["floquet_fitted_rad_per_us"]
PREP_STEPS = 11
SHORT, LONG = 0.05, 0.2  # us
BETA_A = 1.5
CALIBRATIONS = {"none": 0.0, "second_order": -STARK_2ND, "floquet": -STARK_FLOQUET}
EXTRA_LOSS = (0.0, 0.1, 1.0)  # fraction of kappa_a added by the drive port
N_TH = (0.0, 1e-3, 1e-2)
TAUS = (0.002, 0.005, 0.010)  # us
ECHO = (0.1, 0.02)  # 10% reflection arriving 20 ns later


def jump_ops(extra, n_th):
    kappa = P.kappa_a * (1 + extra)
    ops = [math.sqrt(P.kappa_b) * B, math.sqrt(kappa * (1 + n_th)) * A]
    return ops + ([math.sqrt(kappa * n_th) * AD] if n_th else [])


def evolve(h, jumps, rho0, tsave):
    res = dq.mesolve(
        h,
        jumps,
        rho0,
        tsave,
        method=dq.method.Tsit5(rtol=DEFAULT_SOLVER_TOL, atol=DEFAULT_SOLVER_TOL),
        save_states=False,
        progress_meter=False,
    )
    return res.final_state


def envelope(duration, tau, echo):
    """Rectangular pulse on [0, T] after a low-pass filter, plus an optional echo; area T."""

    def rect(t):
        if tau == 0:
            return jnp.where((t >= 0) & (t < duration), 1.0, 0.0)
        rise = 1 - jnp.exp(-jnp.clip(t, 0, None) / tau)
        tail = (1 - math.exp(-duration / tau)) * jnp.exp(-(t - duration) / tau)
        return jnp.where(t < 0, 0.0, jnp.where(t < duration, rise, tail))

    r, delay = echo
    return lambda t: (rect(t) + r * rect(t - delay)) / (1 + r)


def drive_amplitude(duration, guard, delta_cal, extra):
    lam = 1j * delta_cal + P.kappa_a * (1 + extra) / 2
    return P.alpha * lam * cmath.exp(lam * guard) / (1 - cmath.exp(-lam * duration))


def drive(c, shape):
    op = 1j * (c / abs(c) * AD - (c / abs(c)).conjugate() * A)
    return dq.modulated(lambda t: abs(c) * shape(t), op)


def run(
    start="push",
    duration=SHORT,
    detuning=0.0,
    calibration="none",
    extra=0.0,
    n_th=0.0,
    tau=0.0,
    echo=(0.0, 0.0),
    guard=0.0,
    circuit=False,
):
    jumps = jump_ops(extra, n_th)
    circ = circuit_hamiltonian() if circuit else None
    t_on = 0.0
    if start == "instant_A":
        rho = dq.tensor(dq.coherent(NA, BETA_A), dq.fock(NB, 0))
        tail = None
    else:
        t_on = duration + guard
        c = drive_amplitude(duration, guard, CALIBRATIONS[calibration], extra)
        shape = envelope(duration, tau, tuple(echo))
        h = drive(c, shape) + detuning * (AD @ A)
        if circ is not None:
            h = h + circ  # kept on during the push, as in task13_finite_push.py (conservative)
        vacuum = dq.tensor(dq.fock(NA, 0), dq.fock(NB, 0))
        rho = evolve(h, jumps, vacuum, jnp.linspace(0.0, t_on, PREP_STEPS))
        tail = drive(c, shape) if (tau or echo[0]) else None
    h = MODEL.H
    if circ is not None:
        h = h + circ
    if tail is not None:
        h = h + tail  # the part of the filtered pulse that arrives after the pump is on
    final = evolve(h, jumps, rho, t_on + HOLD_GRID)
    return 1.0 - float(fidelity_with(memory_state(final, NA, NB), dq.coherent(NA, P.alpha)))


def cases():
    yield {"effect": "reference"}
    for duration in (SHORT, LONG):
        for calibration in CALIBRATIONS:
            yield {"effect": "detuning", "duration": duration, "detuning": -STARK_FLOQUET,
                   "calibration": calibration}
    for extra in EXTRA_LOSS:
        for n_th in N_TH:
            if extra == 0 and n_th == 0:
                continue
            for start in ("push", "instant_A"):
                yield {"effect": "heat_loss", "start": start, "extra": extra, "n_th": n_th}
    for tau in TAUS:
        for guard in (0.0, 5 * tau):
            yield {"effect": "lowpass", "tau": tau, "guard": guard}
    yield {"effect": "lowpass", "duration": LONG, "tau": TAUS[-1], "guard": 0.0}
    for guard in (0.0, ECHO[1]):
        yield {"effect": "echo", "echo": list(ECHO), "guard": guard}
    worst = {"extra": 0.1, "n_th": 1e-3, "circuit": True}
    yield {"effect": "combined", **worst, "detuning": -STARK_FLOQUET,
           "calibration": "second_order", "tau": 0.005, "guard": 0.025}
    yield {"effect": "combined", "start": "instant_A", **worst}


def main() -> None:
    rows = []
    for case in cases():
        t0 = time.time()
        params = {k: v for k, v in case.items() if k != "effect"}
        err = run(**params)
        rows.append({**case, "error": err})
        print(f"{json.dumps(case):150s} error {err:.3e}  ({time.time() - t0:.0f} s)", flush=True)
        OUT.write_text(json.dumps({
            "truncation": [NA, NB],
            "t_hold_us": T_HOLD,
            "stark_a_rad_per_us": {"second_order": STARK_2ND, "floquet": STARK_FLOQUET},
            "defaults": {"start": "push", "duration": SHORT, "detuning": 0.0,
                         "calibration": "none", "extra": 0.0, "n_th": 0.0, "tau": 0.0,
                         "echo": [0.0, 0.0], "guard": 0.0, "circuit": False},
            "precision": "double",
            "rows": rows,
        }, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
