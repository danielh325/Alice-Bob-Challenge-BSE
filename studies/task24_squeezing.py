"""Task 2.4 / R11 (HYPOTHESIS TEST): does a broadband squeezed buffer bath help the readout
or hurt the cat?

Model (catqubit.driven_rates). Two-mode Hamiltonian H = g2 (a^2 b^dag + h.c.) - eps_d (b + b^dag)
+ eps_z (a + a^dag), memory loss sqrt(kappa_a) a, and a broadband squeezed bath on the buffer,
jump sqrt(kappa_b) (cosh r b + e^{i phi} sinh r b^dag). phi = 0 squeezes X_b = b + b^dag, the
quadrature that carries the readout signal (the codeword contrast <b>_+ - <b>_- is real);
phi = pi squeezes P_b and anti-squeezes X_b. The bath holds sinh^2 r photons, so the buffer
truncation grows with r.

Computed per (alpha, phi, r):
  * rate0 = Gamma_X(eps_z = 0) and the exact linear-response coefficient c2, with
    Gamma_X(eps_z) = rate0 + c2 eps_z^2 + O(eps_z^4) (drive_rate_coefficient, parity blocks).
    Converged in N_a and N_b: one step up in each (na + NA_STEP, nb + NB_STEP) changes rate0
    and c2 by <= CONVERGENCE_RTOL; otherwise the base truncation is enlarged and re-checked.
  * the metastable codeword contrast <b>_+ - <b>_- at eps_z in CONTRAST_EPS (full driven
    Liouvillian, codeword_contrast), at a small and a larger truncation (contrast is an O(1)
    observable and converges much faster than the exponentially small rate).

  * cross-check (cheap, larger r): the effective single-mode model obtained by eliminating the
    buffer, kappa2 D[cosh r A - e^{i phi} sinh r A^dag] + kappa_a D[a], A = a^2 - alpha^2,
    converged in the memory size. The squeezed bath gives the memory a heavy photon tail
    (truncation at too few levels overestimates Gamma_X by up to orders of magnitude), so the
    converged two-mode sweep is limited by MAX_DIM to small r; the effective model shows the
    trend beyond. Its p_X estimate assumes an r-independent contrast.

Readout model (ideal-case HYPOTHESIS, eta = 1): homodyne of the buffer output along X_b,
    SNR^2 = 4 eta kappa_b T (Re Delta<b>)^2 / V,   V = cosh 2r - sinh 2r cos phi
(V = e^{-2r} for phi = 0, e^{+2r} for phi = pi, 1 for vacuum), Gaussian error
0.5 erfc(SNR / (2 sqrt 2)). The time T for a target error, and the bit-flip probability per
readout p_X = Gamma_X(eps_z) T with Gamma_X = rate0 + c2 eps_z^2.
Caveats: the reflected squeezing is frequency dependent near omega ~ kappa_b, the buffer is
nonlinear and eta < 1 in practice, so e^{2r} is an UPPER BOUND on the SNR^2 gain. The Gaussian
error also ignores flips during the readout (heavy tails, C6).

Convention: Gamma_X = -lambda_1 = decay rate of Re<a> = sum of both one-way rates; one-way
jump rate = Gamma_X / 2. p_X = Gamma_X T is in the same (eigenvalue) convention; the
probability that the codeword has flipped is (1 - exp(-Gamma_X T)) / 2 ~ p_X / 2.

Run: PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
         .venv313/bin/python -u studies/task24_squeezing.py      (about 1-2 h; --plot redraws)
"""

import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.special import erfcinv  # noqa: E402

from catqubit.driven_rates import (  # noqa: E402
    Dissipation,
    HamiltonianTerms,
    codeword_contrast,
    deflate_steady_state,
    drive_rate_coefficient,
    driven_bit_flip_rate,
)
from catqubit.eig_gradients import isolated_eigenpair  # noqa: E402
from catqubit.liouvillian import (  # noqa: E402
    FockTruncation,
    lindblad_superoperator,
    parity_block,
    two_mode_operators,
    vec,
)
from catqubit.params import CatParams  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "task24" / "squeezing.json"
FIGURE = ROOT / "figures" / "track2" / "task24_squeezing.png"
DPI = 150

ALPHAS = (1.4, 1.8)
R_VALUES = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)
PHIS = {"X_b squeezed (phi=0)": 0.0, "P_b squeezed (phi=pi)": math.pi}
# Base truncations at r = 0 (alpha = 1.4 from results/task24/tradeoff_r9.json, rel. change
# < 1e-5 for na+4, nb+1; alpha = 1.8 is a starting guess, converged by the loop below).
START_TRUNCATION = {
    1.4: FockTruncation(na=20, nb=5, nb_extra=2, na_taper=6.0),
    1.8: FockTruncation(na=24, nb=5, nb_extra=4, na_taper=10.0),
}
CONVERGENCE_RTOL = 5e-3
NA_STEP = 2
NB_STEP = 1
# Largest Hilbert dimension for a parity-block LU (block size dim^2 / 2; keeps each LU well
# under 2 GB on the shared machine). The squeezed bath gives the memory a heavy photon tail, so
# the two-mode sweep stops at the first r that needs more (r ~ 0.15-0.2 at alpha = 1.4).
MAX_DIM = 200
# Contrast truncation: memory levels (the buffer follows the converged base truncation).
# Probe at alpha=1.4, r=0.3: (14,5)->(14,6) changes the contrast by 2%, (14,6)->(16,6) by 0.7%.
CONTRAST_NA = {1.4: 14, 1.8: 18}
CONTRAST_RTOL = 1e-2
MAX_CONTRAST_DIM = 150
CONTRAST_EPS = (0.05, 0.1, 0.2)  # eps_z / g2
READOUT_EPS = (0.1, 0.2)  # eps_z / g2 used for the readout trade-off
TARGET_ERROR = 1e-3
ETA = 1.0
# Effective single-mode cross-check (adiabatic elimination of the buffer): memory sizes tried
# in order until two successive sizes agree to CONVERGENCE_RTOL.
EFFECTIVE_SIZES = (40, 60, 80, 100, 120)
CONVENTION = "Gamma_X = -lambda_1 = decay rate of Re<a>; one-way = Gamma_X/2"


@dataclass(frozen=True)
class LinearResponse:
    trunc: dict
    dim: int
    rate0: float | None
    c2: float | None
    residual: float | None
    seconds: float
    error: str | None = None


# --------------------------------------------------------------------- truncations


def trunc_dict(t: FockTruncation) -> dict:
    return {"na": t.na, "nb": t.nb, "nb_extra": t.nb_extra, "na_taper": t.na_taper}


def na_step(t: FockTruncation) -> FockTruncation:
    return replace(t, na=t.na + NA_STEP)


def nb_step(t: FockTruncation) -> FockTruncation:
    return replace(t, nb=t.nb + NB_STEP)


def relative_change(a: LinearResponse, b: LinearResponse) -> float:
    """max over (rate0, c2) of |b - a| / |b|; inf if either solve failed."""
    if a.rate0 is None or b.rate0 is None:
        return math.inf
    return max(abs(b.rate0 - a.rate0) / abs(b.rate0), abs(b.c2 - a.c2) / abs(b.c2))


# --------------------------------------------------------------------- physics solves


def dissipation(params: CatParams, r: float, phi: float) -> Dissipation:
    return Dissipation.from_params(params, squeeze_r=r, squeeze_phi=phi)


def linear_response(params: CatParams, r: float, phi: float, t: FockTruncation) -> LinearResponse:
    start = time.time()
    try:
        resp = drive_rate_coefficient(
            HamiltonianTerms.from_params(params), dissipation(params, r, phi), t
        )
    except ValueError as exc:  # inverse iteration failed: no isolated slow mode at this size
        return LinearResponse(trunc_dict(t), t.dim, None, None, None, time.time() - start, str(exc))
    return LinearResponse(
        trunc_dict(t), t.dim, resp.rate0, resp.coefficient, resp.residual, time.time() - start
    )


class Cache:
    """Memoized linear-response solves and contrast blocks, keyed by (alpha, r, phi, truncation)."""

    def __init__(self) -> None:
        self._store: dict[tuple, LinearResponse] = {}
        self._contrast: dict[tuple, dict] = {}

    def get(self, params: CatParams, r: float, phi: float, t: FockTruncation) -> LinearResponse:
        key = (params.alpha, r, phi, t)
        if key not in self._store:
            res = linear_response(params, r, phi, t)
            print(
                f"    LR alpha={params.alpha} r={r} phi={phi:.3f} {trunc_dict(t)} dim={t.dim} "
                f"rate0={res.rate0} c2={res.c2} ({res.seconds:.1f}s) {res.error or ''}",
                flush=True,
            )
            self._store[key] = res
        return self._store[key]

    def contrast(self, params: CatParams, r: float, phi: float, base: dict) -> dict:
        key = (params.alpha, r, phi, tuple(base.values()))
        if key not in self._contrast:
            self._contrast[key] = contrast_block(self, params, r, phi, base)
        return self._contrast[key]


def converge_point(
    cache: Cache, params: CatParams, r: float, phi: float, start: FockTruncation
) -> dict:
    """Enlarge the base truncation until one step in na and in nb each changes rate0, c2 by
    <= CONVERGENCE_RTOL. Returns the base result, both neighbours and the history."""
    base, history = start, []
    while nb_step(base).dim <= MAX_DIM and na_step(base).dim <= MAX_DIM:
        b = cache.get(params, r, phi, base)
        n_a, n_b = (
            cache.get(params, r, phi, na_step(base)),
            cache.get(params, r, phi, nb_step(base)),
        )
        d_na, d_nb = relative_change(b, n_a), relative_change(b, n_b)
        history.append({"base": trunc_dict(base), "rel_change_na": d_na, "rel_change_nb": d_nb})
        if d_na <= CONVERGENCE_RTOL and d_nb <= CONVERGENCE_RTOL:
            return {
                "converged": True,
                "base": asdict(b),
                "na_step": asdict(n_a),
                "nb_step": asdict(n_b),
                "rel_change_na": d_na,
                "rel_change_nb": d_nb,
                "history": history,
            }
        base = replace(
            base,
            na=base.na + (NA_STEP if d_na > CONVERGENCE_RTOL else 0),
            nb=base.nb + (NB_STEP if d_nb > CONVERGENCE_RTOL else 0),
        )
    return {"converged": False, "last_base": trunc_dict(base), "history": history}


def contrast_truncations(alpha: float, base: dict) -> tuple[FockTruncation, FockTruncation]:
    """Contrast grid: memory CONTRAST_NA + NA_STEP with the converged buffer (nb, nb_extra), and
    a check one step smaller in both modes."""
    large = FockTruncation(
        CONTRAST_NA[alpha] + NA_STEP, base["nb"], base["nb_extra"], base["na_taper"]
    )
    return replace(large, na=CONTRAST_NA[alpha], nb=large.nb - NB_STEP), large


def full_point(
    params: CatParams, r: float, phi: float, t: FockTruncation, eps_ratio: float
) -> dict:
    start = time.time()
    terms = HamiltonianTerms.from_params(params, eps_z=eps_ratio * params.g2)
    res = driven_bit_flip_rate(terms, dissipation(params, r, phi), t)
    contrast = codeword_contrast(res, params, t)
    out = {
        "eps_z_over_g2": eps_ratio,
        "trunc": trunc_dict(t),
        "dim": t.dim,
        "contrast_re": contrast.real,
        "contrast_im": contrast.imag,
        "rate_full_small_trunc": res.rate,
        "residual": res.residual,
        "seconds": time.time() - start,
    }
    print(
        f"    full eps={eps_ratio} {trunc_dict(t)} contrast={contrast:.6f} ({out['seconds']:.1f}s)",
        flush=True,
    )
    return out


def _with_quadratic_check(point: dict, lr: LinearResponse, g2: float) -> dict:
    """Add rate0 + c2 eps^2 on the same grid, and the ratio full / quadratic (O(eps^4) size)."""
    if lr.rate0 is None:
        return point
    quad = lr.rate0 + lr.c2 * (point["eps_z_over_g2"] * g2) ** 2
    full_over_quad = point["rate_full_small_trunc"] / quad
    return {**point, "rate_quadratic_same_trunc": quad, "rate_full_over_quadratic": full_over_quad}


def contrast_block(cache: Cache, params: CatParams, r: float, phi: float, base: dict) -> dict:
    """Contrast at every CONTRAST_EPS on the larger truncation; eps = 0.1 g2 also on the smaller
    one (truncation check). The linear-response rate at the larger truncation lets us compare
    rate0 + c2 eps^2 with the full Gamma_X(eps) on the same grid (O(eps^4) check)."""
    small, large = contrast_truncations(params.alpha, base)
    if large.dim > MAX_CONTRAST_DIM:
        return {"skipped": f"contrast truncation dim {large.dim} > {MAX_CONTRAST_DIM}"}
    try:
        points = [full_point(params, r, phi, large, e) for e in CONTRAST_EPS]
        check = full_point(params, r, phi, small, 0.1)
    except ValueError as exc:  # no isolated slow mode of the driven Liouvillian
        return {"skipped": f"full driven solve failed: {exc}"}
    lr = cache.get(params, r, phi, large)
    points = [_with_quadratic_check(p, lr, params.g2) for p in points]
    ref = next(p for p in points if p["eps_z_over_g2"] == 0.1)
    slope = {p["eps_z_over_g2"]: p["contrast_re"] / p["eps_z_over_g2"] for p in points}
    return {
        "points": points,
        "truncation_check": check,
        "rel_change_truncation": abs(check["contrast_re"] - ref["contrast_re"])
        / abs(ref["contrast_re"]),
        "linearity_0.05_vs_0.1": abs(slope[0.05] - slope[0.1]) / abs(slope[0.1]),
        "linearity_0.2_vs_0.1": abs(slope[0.2] - slope[0.1]) / abs(slope[0.1]),
    }


# --------------------------------------------------------------------- effective model


def effective_liouvillian(
    params: CatParams, r: float, phi: float, n: int, eps_z: float
) -> tuple[sp.csr_matrix, FockTruncation, sp.csr_matrix]:
    """Single-mode model: H = eps_z (a + a^dag), jumps sqrt(kappa2) (cosh r A - e^{i phi} sinh r
    A^dag) and sqrt(kappa_a) a, with A = a^2 - alpha^2. It follows from b ~ -2i g2 A / kappa_b in
    the squeezed buffer jump (our adiabatic elimination; a trend cross-check only, UNVERIFIED)."""
    trunc = FockTruncation(na=n, nb=1)  # buffer frozen in vacuum: a pure memory space
    a = two_mode_operators(trunc)[0].astype(complex)
    big_a = a @ a - params.alpha**2 * sp.identity(n, dtype=complex, format="csr")
    phase = complex(math.cos(phi), math.sin(phi))
    jump = math.cosh(r) * big_a - phase * math.sinh(r) * big_a.T
    h = eps_z * (a + a.T)
    jumps = [math.sqrt(params.kappa2) * jump, math.sqrt(params.kappa_a) * a]
    return lindblad_superoperator(h, jumps), trunc, a


def effective_rate(params: CatParams, r: float, phi: float, n: int, eps_z: float) -> float:
    """-lambda_1 of the single-mode model; parity block at eps_z = 0, deflated full L otherwise."""
    big_l, trunc, a = effective_liouvillian(params, r, phi, n, eps_z)
    start = vec((a + a.T).toarray())
    if eps_z == 0:
        odd, idx = parity_block(big_l, trunc, odd=True)
        return -isolated_eigenpair(odd, start[idx]).value.real
    deflated = deflate_steady_state(big_l, trunc, -params.kappa2)
    return -isolated_eigenpair(deflated, start).value.real


def effective_point(params: CatParams, r: float, phi: float) -> dict:
    """rate0 and Gamma_X(eps_z) at READOUT_EPS, memory size increased until two successive
    sizes agree to CONVERGENCE_RTOL for all of them."""
    drives = (0.0, *READOUT_EPS)
    history: list[tuple[int, tuple[float, ...]]] = []
    for n in EFFECTIVE_SIZES:
        rates = tuple(effective_rate(params, r, phi, n, e * params.g2) for e in drives)
        history = history + [(n, rates)]
        if len(history) > 1 and _max_rel(history[-2][1], rates) <= CONVERGENCE_RTOL:
            break
    rel = _max_rel(history[-2][1], history[-1][1])
    rates = history[-1][1]
    return {
        "rate0": rates[0],
        "gamma_x": {str(e): g for e, g in zip(READOUT_EPS, rates[1:], strict=True)},
        "n": history[-1][0],
        "rel_change_vs_previous_n": rel,
        "converged": rel <= CONVERGENCE_RTOL,
    }


def _max_rel(old: tuple[float, ...], new: tuple[float, ...]) -> float:
    return max(abs(b - a) / abs(b) for a, b in zip(old, new, strict=True))


def effective_sweep() -> list[dict]:
    rows = []
    for alpha in ALPHAS:
        params = CatParams.reglade().with_alpha(alpha)
        for label, phi in PHIS.items():
            for r in R_VALUES:
                point = effective_point(params, r, phi)
                print(f"effective alpha={alpha} {label} r={r}: {point}", flush=True)
                rows = rows + [{"alpha": alpha, "phi_label": label, "phi": phi, "r": r, **point}]
    return rows


# --------------------------------------------------------------------- readout model


def noise_variance(r: float, phi: float) -> float:
    """Variance of X_b noise relative to vacuum for ideal broadband squeezing."""
    return math.cosh(2 * r) - math.sinh(2 * r) * math.cos(phi)


def required_snr(error: float) -> float:
    """SNR with 0.5 erfc(SNR / (2 sqrt 2)) = error."""
    return 2 * math.sqrt(2) * float(erfcinv(2 * error))


def readout_time(contrast: float, r: float, phi: float, kappa_b: float) -> float:
    """T (us) for TARGET_ERROR: SNR^2 = 4 eta kappa_b T contrast^2 / V."""
    return (
        required_snr(TARGET_ERROR) ** 2 * noise_variance(r, phi) / (4 * ETA * kappa_b * contrast**2)
    )


def readout_entry(params: CatParams, r: float, phi: float, lr: dict, contrast: dict) -> dict:
    out = {}
    for eps_ratio in READOUT_EPS:
        c = next(p for p in contrast["points"] if p["eps_z_over_g2"] == eps_ratio)["contrast_re"]
        t = readout_time(c, r, phi, params.kappa_b)
        gamma = lr["rate0"] + lr["c2"] * (eps_ratio * params.g2) ** 2
        out[str(eps_ratio)] = {
            "contrast_re": c,
            "noise_variance": noise_variance(r, phi),
            "T_us": t,
            "gamma_x": gamma,
            "p_x": gamma * t,
            "p_flip": (1 - math.exp(-gamma * t)) / 2,
        }
    return out


# --------------------------------------------------------------------- sweep


def point_row(
    cache: Cache, params: CatParams, label: str, phi: float, r: float, start: FockTruncation
) -> dict:
    """Converged linear response, contrast and readout numbers for one (alpha, phi, r)."""
    phi_eff = 0.0 if r == 0 else phi  # r = 0 is the same bath for both phases
    print(f"alpha={params.alpha} {label} r={r}", flush=True)
    conv = converge_point(cache, params, r, phi_eff, start)
    head = {"alpha": params.alpha, "phi_label": label, "phi": phi, "r": r, "linear_response": conv}
    if not conv["converged"]:
        return head
    contrast = cache.contrast(params, r, phi_eff, conv["base"]["trunc"])
    if "points" not in contrast:
        return {**head, "contrast": contrast}
    readout = readout_entry(params, r, phi, conv["base"], contrast)
    return {**head, "contrast": contrast, "readout": readout}


def sweep_alpha(cache: Cache, alpha: float, save: Callable[[list[dict]], None]) -> list[dict]:
    """r ascending for each phase; each r starts from the previous converged truncation, and
    a branch stops at the first r that cannot be converged within MAX_DIM."""
    params = CatParams.reglade().with_alpha(alpha)
    rows: list[dict] = []
    for label, phi in PHIS.items():
        start = START_TRUNCATION[alpha]
        for r in R_VALUES:
            row = point_row(cache, params, label, phi, r, start)
            rows = rows + [row]
            save(rows)
            if not row["linear_response"]["converged"]:
                print(f"  not converged within MAX_DIM={MAX_DIM}; stopping this branch", flush=True)
                break
            start = FockTruncation(**row["linear_response"]["base"]["trunc"])
    return rows


def _readout_ratios(row: dict, ref: dict) -> dict:
    out = {}
    for k, v in row.get("readout", {}).items():
        v0 = ref.get("readout", {}).get(k)
        if v0 is None:
            continue
        out[f"eps{k}"] = {
            "contrast": v["contrast_re"] / v0["contrast_re"],
            "T": v["T_us"] / v0["T_us"],
            "gamma_x": v["gamma_x"] / v0["gamma_x"],
            "p_x": v["p_x"] / v0["p_x"],
            "p_flip": v["p_flip"] / v0["p_flip"],
        }
    return out


def with_ratios(rows: list[dict]) -> list[dict]:
    """New rows with ratios to r = 0 of the same alpha added (converged rows only)."""
    out = []
    for row in rows:
        ref = next(x for x in rows if x["alpha"] == row["alpha"] and x["r"] == 0.0)
        if not row["linear_response"]["converged"]:
            out.append(row)
            continue
        lr, lr0 = row["linear_response"]["base"], ref["linear_response"]["base"]
        ratios = {"rate0": lr["rate0"] / lr0["rate0"], "c2": lr["c2"] / lr0["c2"]}
        out.append({**row, "ratios": {**ratios, **_readout_ratios(row, ref)}})
    return out


def effective_with_ratios(rows: list[dict]) -> list[dict]:
    """Ratios to r = 0 in the effective model. The per-readout estimate assumes the contrast
    does not change with r (the two-mode contrast changes by <~1% where both are computed):
    p_X(r) / p_X(0) ~ [Gamma_X(eps; r) / Gamma_X(eps; 0)] V(r)."""
    out = []
    for row in rows:
        ref = next(x for x in rows if x["alpha"] == row["alpha"] and x["r"] == 0.0)
        v = noise_variance(row["r"], row["phi"])
        gamma = {k: g / ref["gamma_x"][k] for k, g in row["gamma_x"].items()}
        ratios = {
            "rate0": row["rate0"] / ref["rate0"],
            "gamma_x": gamma,
            "p_x_estimate": {k: g * v for k, g in gamma.items()},
        }
        out.append({**row, "ratios": ratios})
    return out


def metadata() -> dict:
    p = CatParams.reglade()
    return {
        "description": __doc__,
        "status": "HYPOTHESIS TEST (R11)",
        "params": {
            "g2": p.g2,
            "kappa_b": p.kappa_b,
            "kappa_a": p.kappa_a,
            "kappa2": p.kappa2,
            "units": "rad/us",
        },
        "convention": CONVENTION
        + "; p_x = Gamma_X T (same convention); p_flip = (1-exp(-Gamma_X T))/2",
        "alphas": ALPHAS,
        "r_values": R_VALUES,
        "phis": PHIS,
        "convergence": {
            "rtol": CONVERGENCE_RTOL,
            "na_step": NA_STEP,
            "nb_step": NB_STEP,
            "max_dim": MAX_DIM,
            "rule": "rate0, c2 change <= rtol for na+na_step and nb+nb_step; base reported",
        },
        "contrast": {"eps_z_over_g2": CONTRAST_EPS, "rtol": CONTRAST_RTOL, "na": CONTRAST_NA},
        "readout_model": {
            "eta": ETA,
            "target_error": TARGET_ERROR,
            "snr_required": required_snr(TARGET_ERROR),
            "formula": "SNR^2 = 4 eta kappa_b T (Re Delta<b>)^2 / V, V = cosh2r - sinh2r cos(phi)",
            "caveats": [
                "ideal broadband squeezing: the reflected squeezing is frequency dependent"
                " near omega ~ kappa_b",
                "eta = 1 assumed; finite efficiency reduces the gain (V -> 1 - eta + eta V)",
                "the buffer is nonlinear, so e^{2r} is an upper bound on the SNR^2 gain",
                "Gaussian error ignores flips during readout (heavy tails, C6)",
            ],
        },
    }


def run() -> None:
    cache = Cache()
    done: list[dict] = []
    meta = metadata()
    meta = {**meta, "effective_model": None}

    def save(rows_alpha: list[dict]) -> None:
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        payload = {**meta, "partial": True, "rows": done + rows_alpha}
        RESULTS.write_text(json.dumps(payload, indent=1, default=str))

    effective = effective_with_ratios(effective_sweep())
    meta = {**meta, "effective_model": effective}
    for alpha in ALPHAS:
        done = done + sweep_alpha(cache, alpha, save)
    final = {**meta, "partial": False, "rows": with_ratios(done)}
    RESULTS.write_text(json.dumps(final, indent=1, default=str))
    plot(json.loads(RESULTS.read_text()))


# --------------------------------------------------------------------- figure


CURVES = {(1.4, 0.0): "C0", (1.4, math.pi): "C1", (1.8, 0.0): "C2", (1.8, math.pi): "C3"}
BOX = {"boxstyle": "round", "fc": "white", "alpha": 0.85}


def _curve_rows(rows: list[dict], alpha: float, phi: float) -> list[dict]:
    """Rows of one (alpha, phi) curve with ratios, r ascending; r = 0 is shared by both phases."""
    keep = [
        x
        for x in rows
        if x["alpha"] == alpha and (x["phi"] == phi or x["r"] == 0.0) and "ratios" in x
    ]
    return sorted({x["r"]: x for x in keep}.values(), key=lambda x: x["r"])


def _plot_two_mode(ax_a: plt.Axes, ax_b: plt.Axes, rows: list[dict]) -> None:
    for (alpha, phi), color in CURVES.items():
        curve = _curve_rows(rows, alpha, phi)
        name = f"alpha={alpha}, phi={'0' if phi == 0 else 'pi'}"
        rs = [x["r"] for x in curve]
        ax_a.semilogy(rs, [x["ratios"]["rate0"] for x in curve], "o-", color=color, label=name)
        for eps, fill in (("0.2", color), ("0.1", "none")):
            pts = [
                (x["r"], x["ratios"][f"eps{eps}"]["p_x"])
                for x in curve
                if f"eps{eps}" in x["ratios"]
            ]
            if pts:
                ax_b.plot(
                    *zip(*pts, strict=True),
                    "o-",
                    color=color,
                    mfc=fill,
                    ms=5,
                    label=f"{name}, eps_z={eps} g2 (two-mode)",
                )


def _plot_effective(ax_a: plt.Axes, ax_b: plt.Axes, rows: list[dict]) -> None:
    for (alpha, phi), color in CURVES.items():
        curve = [x for x in _curve_rows(rows, alpha, phi) if x["converged"]]
        rs = [x["r"] for x in curve]
        ax_a.semilogy(rs, [x["ratios"]["rate0"] for x in curve], "--", color=color, lw=1)
        ax_b.plot(rs, [x["ratios"]["p_x_estimate"]["0.2"] for x in curve], "--", color=color, lw=1)


def plot(data: dict) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 4.8))
    _plot_two_mode(ax_a, ax_b, data["rows"])
    if data.get("effective_model"):
        _plot_effective(ax_a, ax_b, data["effective_model"])
    ax_a.plot([], [], "k--", lw=1, label="effective 1-mode model (dashed)")
    ax_b.plot([], [], "k--", lw=1, label="effective model, eps_z=0.2 g2, contrast fixed")
    ax_a.set(
        xlabel="squeezing parameter r (dimensionless)",
        ylabel="Gamma_X0(r) / Gamma_X0(0)  (idle, eps_z = 0)",
        title="(a) idle bit-flip rate with squeezed buffer bath",
    )
    ax_a.text(
        0.03,
        0.97,
        CONVENTION.replace("; ", "\n"),
        transform=ax_a.transAxes,
        va="top",
        fontsize=8,
        bbox=BOX,
    )
    ax_b.axhline(1.0, color="k", lw=0.8)
    ax_b.set_yscale("log")
    ax_b.set(
        xlabel="squeezing parameter r (dimensionless)",
        ylabel="p_X per readout / p_X at r = 0",
        title="(b) bit flips per readout at error 1e-3 (ideal, eta = 1)",
    )
    ax_b.text(
        0.03,
        0.97,
        "p_X = (rate0 + c2 eps_z^2) T(r), eigenvalue convention\n"
        "T from SNR^2 = 4 eta kappa_b T contrast^2 / V(r)\n"
        "V = e^{-2r} (phi=0), e^{+2r} (phi=pi): upper bound on the gain",
        transform=ax_b.transAxes,
        va="top",
        fontsize=7.5,
        bbox=BOX,
    )
    for ax in (ax_a, ax_b):
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6.3, loc="lower left" if ax is ax_b else "lower right")
    fig.suptitle(
        "R11 hypothesis test: broadband squeezed bath on the buffer "
        "(Reglade parameters; phi=0 squeezes the readout quadrature X_b)",
        fontsize=10,
    )
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=DPI)
    print(f"wrote {FIGURE}")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot(json.loads(RESULTS.read_text()))
    else:
        run()
