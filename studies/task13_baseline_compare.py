"""Task 1.3(b): baselines against the best preparation, in double precision.

Same simulation as intro_track.ipynb method A/D (two-mode model, 28 x 12 box, 2 us with the
stabilization on, 21 save points), run for:

- beta = 0: no preparation, inflate from vacuum (the baseline the prompt starts from);
- beta = 1.5: method A as the notebook compared it;
- beta = 2.0: the closest competitor (the notebook prints only F = 1.0000);
- beta = alpha: method D.

Each with and without the circuit terms of Task 2.4(a). Writes results/task1/baseline_compare.json.
Run: PYTHONPATH=.:studies .venv313/bin/python studies/task13_baseline_compare.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import dynamiqs as dq
import jax.numpy as jnp

dq.set_precision("double")  # challenge Section 0.5; float32 cannot resolve errors near 1e-6

from task13_dark_prep import NA, NB, P, circuit_hamiltonian  # noqa: E402

from catqubit.observables import fidelity_with, memory_state  # noqa: E402
from catqubit.simulate import inflate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "task1" / "baseline_compare.json"
T_P = jnp.linspace(0.0, 2.0, 21)  # same grid as the notebook's prepare()
STARTS = {"vacuum": 0.0, "A_1.5": 1.5, "A_2.0": 2.0, "D_alpha": P.alpha}


def error(beta: float, extra_h) -> float:
    res = inflate(NA, NB, P, T_P, memory0=dq.coherent(NA, beta), extra_H=extra_h, save_states=False)
    rho = memory_state(res.final_state, NA, NB)
    return 1.0 - float(fidelity_with(rho, dq.coherent(NA, P.alpha)))


def main() -> None:
    h_circuit = circuit_hamiltonian()
    rows = []
    for name, beta in STARTS.items():
        for model, extra in (("model", None), ("circuit", h_circuit)):
            t0 = time.time()
            err = error(beta, extra)
            rows.append({"start": name, "beta": beta, "model": model, "error": err})
            elapsed = time.time() - t0
            print(f"{name:8s} beta = {beta:4.2f} {model:7s}: error {err:.3e}  ({elapsed:.0f} s)",
                  flush=True)
    summary = {"truncation": [NA, NB], "t_us": 2.0, "precision": "double", "rows": rows}
    OUT.write_text(json.dumps(summary, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
