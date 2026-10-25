"""Task 1.3(b): which circuit term sets method D's error floor, and can a Kerr-cat term remove it?

Method D (start in |alpha>|0>, 2 us with the stabilization on, 28 x 12 box, double precision) with:

- each Task 2.4(a) circuit term alone (memory Kerr, buffer Kerr, cross-Kerr), then all three;
- the memory Kerr at its Floquet-fitted value (the 2nd-order value is not verified numerically);
- a two-photon correction -(K_a/2)(alpha^2 a^dag^2 + alpha^2 a^2), which turns the memory Kerr into
  (K_a/2)(a^dag^2 - alpha^2)(a^2 - alpha^2) (+ const), exactly zero on |+-alpha>.

The correction is added to the simulator only; whether hardware can apply it is a separate question.
Writes results/task1/kerr_floor.json.
Run: PYTHONPATH=.:studies .venv313/bin/python studies/task13_kerr_floor.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import dynamiqs as dq

dq.set_precision("double")  # challenge Section 0.5

from task13_baseline_compare import error  # noqa: E402
from task13_dark_prep import NA, NB, P  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "task1" / "kerr_floor.json"


def terms() -> dict:
    data = json.loads((ROOT / "results/task24a/effective_terms.json").read_text())["terms"]
    return {
        "kerr_a": data["kerr_a"]["value_rad_per_us"],
        "kerr_a_floquet": data["kerr_a"]["floquet_fitted_rad_per_us"],
        "kerr_b": data["kerr_b"]["value_rad_per_us"],
        "chi_ab": data["chi_ab"]["value_rad_per_us"],
    }


def hamiltonians(t: dict) -> dict:
    a, b = dq.destroy(NA, NB)
    ad, bd = a.dag(), b.dag()
    alpha2 = P.alpha**2

    def kerr_a(k):
        return (k / 2) * ad @ ad @ a @ a

    def correction(k):
        return -(k / 2) * alpha2 * (ad @ ad + a @ a)

    h_b = (t["kerr_b"] / 2) * bd @ bd @ b @ b
    h_chi = t["chi_ab"] * ad @ a @ bd @ b
    h_all = kerr_a(t["kerr_a"]) + h_b + h_chi
    h_all_floquet = kerr_a(t["kerr_a_floquet"]) + h_b + h_chi
    return {
        "none": None,
        "kerr_a only": kerr_a(t["kerr_a"]),
        "kerr_b only": h_b,
        "chi_ab only": h_chi,
        "all (2.4a)": h_all,
        "all, kerr_a Floquet": h_all_floquet,
        "kerr_a only + correction": kerr_a(t["kerr_a"]) + correction(t["kerr_a"]),
        "all + correction": h_all + correction(t["kerr_a"]),
        "all, kerr_a Floquet + correction": h_all_floquet + correction(t["kerr_a_floquet"]),
        # miscalibrated correction: the 2nd-order and Floquet values of K_a differ by 2.6x
        "all, kerr_a Floquet + correction for 2nd order": h_all_floquet + correction(t["kerr_a"]),
        "all + correction for Floquet": h_all + correction(t["kerr_a_floquet"]),
    }


def main() -> None:
    t = terms()
    rows = []
    for name, h in hamiltonians(t).items():
        t0 = time.time()
        err = error(P.alpha, h)
        rows.append({"case": name, "error": err})
        print(f"{name:46s}: error {err:.3e}  ({time.time() - t0:.0f} s)", flush=True)
    OUT.write_text(json.dumps({"terms_rad_per_us": t, "rows": rows}, indent=1) + "\n")
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
