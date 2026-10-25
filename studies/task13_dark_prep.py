"""Task 1.3(b-c): prepare |+alpha> by displacing the memory onto the codeword (beta = alpha).

|alpha>_a |0>_b is an exact dark state of the full two-mode model: the pump term g2 a^2 b^dag and
the buffer drive eps_d b^dag cancel on it (eps_d = g2 alpha^2), and the buffer Kerr and
cross-Kerr terms vanish on buffer vacuum. Only the memory Kerr and kappa_a act, so there is no
inflation transient.

Compares, at t = 2 us (2.5 us for the ramped methods), with and without the circuit terms of 2.4(a):
  A(beta)  displaced start for beta = 1.5, 2.0, alpha, 2.3
  A + C    small displacement plus the method-C ramp-and-push (does combining help?)
  C        method C reference (1 us linear eps_d ramp, push sized for beta_eff = 1.5)
and the robustness of beta = alpha to amplitude (+-10 %) and phase (22.5, 45 deg) errors.

Writes results/task1/prep_dark.json and figures/1_3b_dark_state.png.

Run:  .venv313/bin/python -u studies/task13_dark_prep.py   (~15 min; --plot-only to redraw)
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

import dynamiqs as dq  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from catqubit.params import CatParams  # noqa: E402
from catqubit.preparation import ramp_push_strength, ramped_preparation  # noqa: E402
from catqubit.simulate import inflate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "task1" / "prep_dark.json"
FIGURE = ROOT / "figures" / "1_3b_dark_state.png"
NA, NB = 28, 12  # transient-converged truncation from Task 1.2
T_HOLD, T_RAMPED, T_RAMP = 2.0, 2.5, 1.0
BETAS = (1.5, 2.0, None, 2.3)  # None -> alpha
HYBRIDS = ((0.5, 1.5), (0.5, 2.0), (1.0, 1.5))  # (initial displacement, method-C target)
AMPLITUDE_ERRORS = (1.0, 0.9, 1.1)
PHASE_ERRORS_DEG = (0.0, 22.5, 45.0)

P = CatParams.reglade()


def circuit_hamiltonian() -> dq.QArray:
    terms = json.loads((ROOT / "results/task24a/effective_terms.json").read_text())["terms"]
    k_a, k_b, chi = (terms[k]["value_rad_per_us"] for k in ("kerr_a", "kerr_b", "chi_ab"))
    a, b = dq.destroy(NA, NB)
    ad, bd = a.dag(), b.dag()
    return (k_a / 2) * ad @ ad @ a @ a + (k_b / 2) * bd @ bd @ b @ b + chi * ad @ a @ bd @ b


def error(beta: complex, extra_h, t_end: float) -> float:
    """1 - F(rho_a, |+alpha>) after stabilizing from |beta>_a |0>_b."""
    res = inflate(NA, NB, P, jnp.linspace(0.0, t_end, 11), memory0=dq.coherent(NA, beta),
                  extra_H=extra_h, save_states=False)
    rho_a = dq.ptrace(res.final_state, 0, (NA, NB))
    return 1.0 - float(dq.fidelity(rho_a, dq.coherent(NA, P.alpha)).real)


def with_circuit(h, h_circ):
    if h_circ is None:
        return h
    return h_circ if h is None else h + h_circ


def method_rows(h_circ, tag: str) -> list[dict]:
    rows = []
    for beta in BETAS:
        b = P.alpha if beta is None else beta
        rows.append(dict(model=tag, method="A", beta=b, error=error(b, h_circ, T_HOLD)))
        print(rows[-1], flush=True)
    for beta0, target in HYBRIDS:
        eps = ramp_push_strength(P, T_RAMP, target)
        h = with_circuit(ramped_preparation(NA, NB, P, T_RAMP, eps), h_circ)
        rows.append(dict(model=tag, method="A+C", beta=beta0, target=target,
                         push_over_g2=eps / P.g2, error=error(beta0, h, T_RAMPED)))
        print(rows[-1], flush=True)
    eps = ramp_push_strength(P, T_RAMP, 1.5)
    h = with_circuit(ramped_preparation(NA, NB, P, T_RAMP, eps), h_circ)
    rows.append(dict(model=tag, method="C", beta=0.0, target=1.5, push_over_g2=eps / P.g2,
                     error=error(0.0, h, T_RAMPED)))
    print(rows[-1], flush=True)
    return rows


def robustness_rows(h_circ, tag: str) -> list[dict]:
    rows = []
    for amp in AMPLITUDE_ERRORS:
        for deg in PHASE_ERRORS_DEG:
            beta = amp * P.alpha * np.exp(1j * math.radians(deg))
            rows.append(dict(model=tag, amplitude=amp, phase_deg=deg,
                             error=error(beta, h_circ, T_HOLD)))
            print(rows[-1], flush=True)
    return rows


def plot(data: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for tag, color in (("model", "tab:blue"), ("circuit", "tab:red")):
        rows = [r for r in data["methods"] if r["model"] == tag and r["method"] == "A"]
        ax1.semilogy([r["beta"] for r in rows], [r["error"] for r in rows], "o-", color=color,
                     label=f"displaced start, {tag}")
        c = next(r for r in data["methods"] if r["model"] == tag and r["method"] == "C")
        ax1.axhline(c["error"], color=color, ls="--", lw=1, label=f"method C, {tag}")
    ax1.axvline(P.alpha, color="k", lw=0.8, ls=":")
    ax1.set(xlabel="initial displacement β", ylabel="1 − F(|+α⟩)",
            title="β = α is an exact dark state")
    ax1.legend(fontsize=8)
    for tag, marker in (("model", "o"), ("circuit", "s")):
        for amp, color in zip(AMPLITUDE_ERRORS, ("k", "tab:orange", "tab:green"), strict=True):
            rows = [r for r in data["robustness"] if r["model"] == tag and r["amplitude"] == amp]
            ax2.semilogy([r["phase_deg"] for r in rows], [r["error"] for r in rows],
                         marker=marker, color=color, ls="-" if tag == "model" else "--",
                         label=f"|β| = {amp:.1f}α, {tag}")
    ax2.set(xlabel="displacement phase error (deg)", ylabel="1 − F(|+α⟩)",
            title="Robustness of β = α")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=150)
    print(f"wrote {FIGURE}")


def main() -> None:
    if "--plot-only" not in sys.argv:
        t0 = time.time()
        h_circ = circuit_hamiltonian()
        data = {"truncation": [NA, NB], "t_hold_us": T_HOLD, "t_ramped_us": T_RAMPED,
                "methods": method_rows(None, "model") + method_rows(h_circ, "circuit"),
                "robustness": robustness_rows(None, "model") + robustness_rows(h_circ, "circuit")}
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(data, indent=1))
        print(f"wrote {OUT} in {time.time() - t0:.0f} s")
    plot(json.loads(OUT.read_text()))


if __name__ == "__main__":
    main()
