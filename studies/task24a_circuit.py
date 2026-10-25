"""Task 2.4(a): which terms the ATS circuit Hamiltonian adds, and an exact Floquet check.

Usage: python -u studies/task24a_circuit.py [terms] [floquet] [figures] (default: all)

  terms    first-order (RWA) and second-order (time-averaged) effective Hamiltonian, all orders in
           phi, pad-converged; photon-number dressing; readout flux tone at w_a
  floquet  exact one-period (1.25 ns) Floquet propagator of the circuit Hamiltonian:
           truncation convergence, eps_p scan, Floquet-fitted coefficients, resonance-restored
           g2 and memory-buffer conversion (avoided-crossing scans)
  figures  figures/track2/task24a_terms.png, task24a_floquet_check.png

Conventions: rates angular, rad/us (value / 2 pi = MHz). Effective Hamiltonian in the frame rotating
at (5u, 6u), u / 2 pi = 0.8 GHz:
  H_eff = stark_a a^dag a + stark_b b^dag b + (kerr_a/2) a^dag^2 a^2 + (kerr_b/2) b^dag^2 b^2
          + chi_ab a^dag a b^dag b + [g2(n_a, n_b) a^2 b^dag + c_a4b2 a^4 b^dag^2 + h.c.] + const.
"""

import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from catqubit.circuit import (
    ATSParams,
    FluxTone,
    FockSpace,
    diagonal_coefficients,
    displacement_operator,
    dressed_g2,
    effective_hamiltonian,
    first_order_hamiltonian,
    match_floquet_states,
    pump_displacement,
    rotating_frame_floquet,
    two_mode_ladders,
)
from catqubit.params import CatParams

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "task24a"
FIGURES = ROOT / "figures" / "track2"
TWO_PI = 2 * math.pi
KHZ = TWO_PI * 1e-3  # rad/us per kHz

SPACE = FockSpace(9, 8)  # Floquet-converged truncation (see floquet convergence table)
SPACES_CONV = (FockSpace(7, 6), FockSpace(9, 8), FockSpace(11, 10))
PADS = (4, 8, 14)
INTERIOR_NA, INTERIOR_NB = 4, 2  # states used for Floquet comparisons (far from the edge)
EPS_SCALES = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4)
DRESS_NA, DRESS_NB = 12, 4
READOUT_EPS_Z_MHZ = 0.2  # resulting memory drive eps_Z / 2 pi for the flux-tone example
CROSSING_DETUNING_MHZ = 20.0  # placement of the degenerate doublet for avoided-crossing scans
SCAN_POINTS = 21
FIT_ITERATIONS = 4
CALIBRATION_STEPS = 2
VERIFY_NOTE = (
    "2nd-order value matches the independent normal-ordered expansion (symbolic_terms.json) to"
    " ~1e-5; exact Floquet differs by the eps_p^4 error shown in floquet_fitted / eigenvalue_check"
)
NO_FOLD = 1e12  # zone width that disables folding for ordinary eigenvalues
LABELS = ("stark_a", "stark_b", "kerr_a", "kerr_b", "chi_ab")


def pump(ats: ATSParams) -> tuple[FluxTone, ...]:
    return (FluxTone(ats.eps_p, ats.harmonic_p),)


def to_khz(x: float) -> float:
    return float(np.real(x)) / KHZ


def finite_differences(energies: dict[tuple[int, int], float]) -> dict[str, float]:
    """Same finite differences as diagonal_coefficients, applied to labelled eigenvalues."""
    e = energies
    z = e[(0, 0)]
    return {
        "stark_a": e[(1, 0)] - z,
        "stark_b": e[(0, 1)] - z,
        "kerr_a": e[(2, 0)] - 2 * e[(1, 0)] + z,
        "kerr_b": e[(0, 2)] - 2 * e[(0, 1)] + z,
        "chi_ab": e[(1, 1)] - e[(1, 0)] - e[(0, 1)] + z,
    }


def interior(labels: tuple[tuple[int, int], ...]) -> np.ndarray:
    return np.array([n_a <= INTERIOR_NA and n_b <= INTERIOR_NB for n_a, n_b in labels])


# --- terms ---------------------------------------------------------------------------------


def g2_dressing(ats: ATSParams) -> dict:
    space = FockSpace(DRESS_NA, DRESS_NB + 2)
    h1 = first_order_hamiltonian(ats, space, pump(ats))
    table = {
        f"{n_a},{n_b}": abs(dressed_g2(h1, space, n_a, n_b))
        for n_a in range(2, DRESS_NA)
        for n_b in range(DRESS_NB)
    }
    g0 = table["2,0"]
    slope_a = (table["3,0"] - g0) / g0  # per memory photon (exact Laguerre is linear at O(phi^2))
    slope_b = (table["2,1"] - g0) / g0
    return {"table_rad_per_us": table, "g2_vacuum": g0, "slope_a": slope_a, "slope_b": slope_b}


def pad_convergence(ats: ATSParams) -> dict:
    space = FockSpace(6, 6)
    out = {}
    for pad in PADS:
        coeffs = diagonal_coefficients(effective_hamiltonian(ats, space, pump(ats), pad=pad), space)
        out[str(pad)] = {k: to_khz(v) for k, v in coeffs.items()}
    return out


def memory_ladder(h: np.ndarray, space: FockSpace, n_max: int) -> list[float]:
    """Memory transition frequencies E(n+1, 0) - E(n, 0) from the operator diagonal (kHz)."""
    diag = np.real(np.diag(h))
    return [to_khz(diag[space.index(n + 1, 0)] - diag[space.index(n, 0)]) for n in range(n_max)]


def readout_tone_terms(ats: ATSParams) -> dict:
    """Flux tone at w_a: first-order drive, its dressing, and second-order pump x tone terms."""
    space = FockSpace(6, 5)
    i = space.index
    probe = FluxTone(1e-4, ats.harmonic_a)
    h_tone = first_order_hamiltonian(ats, space, (probe,))
    eps_z = READOUT_EPS_Z_MHZ * TWO_PI / abs(h_tone[i(1, 0), i(0, 0)]) * probe.amplitude
    tones = (*pump(ats), FluxTone(eps_z, ats.harmonic_a))
    h = effective_hamiltonian(ats, space, tones)
    h0 = effective_hamiltonian(ats, space, pump(ats))
    drive = h[i(1, 0), i(0, 0)]
    c, c0 = diagonal_coefficients(h, space), diagonal_coefficients(h0, space)
    return {
        "flux_tone_amplitude_rad": eps_z,
        "eps_Z": float(np.real(drive)),
        "drive_dressing_per_memory_photon": float(
            np.real(h[i(2, 0), i(1, 0)] / math.sqrt(2) / drive)
        )
        - 1,
        "drive_dressing_per_buffer_photon": float(np.real(h[i(1, 1), i(0, 1)] / drive)) - 1,
        "resonant_buffer_drive": float(abs(h[i(0, 1), i(0, 0)])),
        "beam_splitter_ab_dag": float(np.real(h[i(0, 1), i(1, 0)])),
        "a_dag3_b": float(np.real(h[i(3, 0), i(0, 1)] / math.sqrt(6))),
        "a_dag2_squeezing": float(np.real(h[i(2, 0), i(0, 0)] / math.sqrt(2))),
        "stark_changes": {k: c[k] - c0[k] for k in c},
        "off_resonant_buffer_displacement": float(
            abs(h_tone[i(1, 0), i(0, 0)])
            / probe.amplitude
            * eps_z
            * ats.phi_b
            / ats.phi_a
            / (ats.omega_b - ats.omega_a)
        ),
    }


def run_terms(ats: ATSParams) -> dict:
    space = FockSpace(12, 6)
    h = effective_hamiltonian(ats, space, pump(ats))
    coeffs = diagonal_coefficients(h, space)
    i = space.index
    a4b2 = float(abs(h[i(0, 2), i(4, 0)]) / math.sqrt(math.factorial(4) * 2))
    beta_a, beta_b = pump_displacement(ats)
    out = {
        "second_order": coeffs,
        "pad_convergence_khz": pad_convergence(ats),
        "a4b2": a4b2,
        "c_number": float(np.real(h[0, 0])),
        "memory_ladder_khz": memory_ladder(h, space, 10),
        "g2": g2_dressing(ats),
        "pump_displacement": {"beta_a": beta_a, "beta_b": beta_b},
        "readout_flux_tone": readout_tone_terms(ats),
    }
    print("second order (kHz):", {k: round(to_khz(v), 3) for k, v in coeffs.items()})
    print("g2 vacuum (MHz):", out["g2"]["g2_vacuum"] / TWO_PI, " a4b2 (Hz):", a4b2 / KHZ * 1e3)
    print("readout tone:", out["readout_flux_tone"])
    return out


# --- Floquet -------------------------------------------------------------------------------


def floquet_vs_effective(ats: ATSParams, space: FockSpace, pad: int = 8) -> dict:
    u = rotating_frame_floquet(ats, space, pump(ats))
    h = effective_hamiltonian(ats, space, pump(ats), pad=pad)
    m = match_floquet_states(u, h, space, ats)
    keep = interior(m.labels)
    return {
        "match": m,
        "h_eff": h,
        "floquet": finite_differences(dict(zip(m.labels, m.floquet, strict=True))),
        "effective": finite_differences(dict(zip(m.labels, m.effective, strict=True))),
        "min_overlap_interior": float(m.overlaps[keep].min()),
        "max_residual_interior": float(
            np.abs((m.floquet - m.floquet[0] - m.effective + m.effective[0])[keep]).max()
        ),
    }


def fit_operators(space: FockSpace) -> dict[str, np.ndarray]:
    a, b = two_mode_ladders(space)
    na, nb = a.T @ a, b.T @ b
    return {
        "stark_a": na,
        "stark_b": nb,
        "kerr_a": a.T @ a.T @ a @ a / 2,
        "kerr_b": b.T @ b.T @ b @ b / 2,
        "chi_ab": na @ nb,
        "g2": a @ a @ b.T + b @ a.T @ a.T,
    }


def floquet_fit(h_eff: np.ndarray, match, space: FockSpace, g2_delta: float) -> dict:
    """Corrections dc to the diagonal H_eff coefficients so its eigenvalues reproduce Floquet.

    g2 is not fitted (in the dispersive regime only g2^2 / detuning is constrained, degenerate
    with the Kerrs); its correction g2_delta comes from the avoided-crossing measurement. Gauss-
    Newton with Hellmann-Feynman derivatives on the interior states, energies relative to the
    vacuum-like state. Sextic and higher terms are not in the model; they set the residual.
    """
    ops = fit_operators(space)
    h_eff = h_eff + g2_delta * ops.pop("g2")
    keep = np.flatnonzero(interior(match.labels))
    target = (match.floquet - match.floquet[0])[keep]
    order = [space.index(*lab) for lab in match.labels]
    deltas = dict.fromkeys(ops, 0.0)
    for _ in range(FIT_ITERATIONS):
        h = h_eff + sum(d * ops[k] for k, d in deltas.items())
        e, w = np.linalg.eigh(h)
        dominant = np.argmax(np.abs(w) ** 2, axis=0)
        pos = {int(dk): j for j, dk in enumerate(dominant)}
        cols = [pos[order[k]] for k in keep]
        ref = pos[order[0]]
        model = e[cols] - e[ref]
        jac = np.array(
            [
                [
                    np.real(w[:, c].conj() @ op @ w[:, c] - w[:, ref].conj() @ op @ w[:, ref])
                    for op in ops.values()
                ]
                for c in cols
            ]
        )
        step, *_ = np.linalg.lstsq(jac, target - model, rcond=None)
        deltas = {k: deltas[k] + s for k, s in zip(ops, step, strict=True)}
    return {"deltas": deltas, "residual": float(np.abs(target - model).max())}


def _top_pair_gap(eigvals: np.ndarray, vecs: np.ndarray, basis: np.ndarray, zone: float) -> float:
    weight = np.sum(np.abs(basis.conj().T @ vecs) ** 2, axis=0)
    top = np.argsort(weight)[-2:]
    gap = (eigvals[top[0]] - eigvals[top[1]] + zone / 2) % zone - zone / 2
    return float(abs(gap))


def floquet_splitting(ats: ATSParams, space: FockSpace, tones, pair) -> float:
    """Gap of the two Floquet modes with most weight on the pump-kicked `pair` (exact only)."""
    u = rotating_frame_floquet(ats, space, tones)
    lam, v = np.linalg.eig(u)
    kick = displacement_operator(space, *pump_displacement(ats))
    basis = kick[:, [space.index(*pair[0]), space.index(*pair[1])]]
    return _top_pair_gap(-np.angle(lam) / ats.period, v, basis, ats.unit)


def effective_splitting(ats: ATSParams, space: FockSpace, tones, pair) -> float:
    """Same gap from the eigenvalues of H1 + H2."""
    e, w = np.linalg.eigh(effective_hamiltonian(ats, space, tones))
    basis = np.eye(space.dim)[:, [space.index(*pair[0]), space.index(*pair[1])]]
    return _top_pair_gap(e, w, basis, NO_FOLD)


def avoided_crossing(splitter, base: ATSParams, space, tones, pair, centre, width) -> dict:
    """Scan detuning_b across the resonance; fit s^2 = (k (d - d0))^2 + s_min^2."""
    grid = centre + np.linspace(-width, width, SCAN_POINTS)
    split = np.array([splitter(replace(base, detuning_b=d), space, tones, pair) for d in grid])
    coef = np.polyfit(grid, split**2, 2)
    d0 = -coef[1] / (2 * coef[0])
    s_min = math.sqrt(np.polyval(coef, d0))
    return {"detuning_b_grid": grid.tolist(), "splitting": split.tolist(), "d0": d0, "s_min": s_min}


def crossing_pair(base, space, tones, pair, centre, width) -> dict:
    return {
        name: avoided_crossing(fn, base, space, tones, pair, centre, width)
        for name, fn in (("floquet", floquet_splitting), ("effective", effective_splitting))
    }


def g2_from_crossing(ats: ATSParams, second: dict) -> dict:
    """Restore 2w_a' = w_b' + w_p with frame detunings and read |g2| off the (2,0)/(0,1) gap.

    Minimum gap = 2 sqrt(2) |g2|. The memory sits CROSSING_DETUNING above the frame so that the
    doublet is isolated from (0,0) and (1,0).
    """
    space = FockSpace(7, 6)
    d_a = CROSSING_DETUNING_MHZ * TWO_PI - second["stark_a"]
    base = replace(ats, detuning_a=d_a)
    centre = 2 * (d_a + second["stark_a"]) + second["kerr_a"] - second["stark_b"]
    scans = crossing_pair(base, space, pump(ats), ((2, 0), (0, 1)), centre, 2 * TWO_PI)
    return {k: {**v, "g2": v["s_min"] / (2 * math.sqrt(2))} for k, v in scans.items()}


def conversion_from_crossing(ats: ATSParams, second: dict, tone_terms: dict) -> dict:
    """(1,0) and (0,1) made degenerate, 20 MHz from the readout tone: gap = 2 |g_bs,total|.

    In this detuned frame the tone also displaces the memory by zeta_a = eps_Z / 20 MHz, so the
    gap includes 2 g2 zeta_a from a^2 b^dag besides the pump x tone term; both models contain it.
    """
    space = FockSpace(7, 6)
    d_a = CROSSING_DETUNING_MHZ * TWO_PI - second["stark_a"]
    base = replace(ats, detuning_a=d_a)
    tones = (*pump(ats), FluxTone(tone_terms["flux_tone_amplitude_rad"], ats.harmonic_a))
    centre = d_a + second["stark_a"] - second["stark_b"]
    width = 0.3 * TWO_PI
    scans = crossing_pair(base, space, tones, ((1, 0), (0, 1)), centre, width)
    return {k: {**v, "g_bs_total": v["s_min"] / 2} for k, v in scans.items()}


def floquet_g2(ats: ATSParams) -> float:
    """Exact |g2| (vacuum doublet) from the Floquet avoided crossing at this pump."""
    space = FockSpace(7, 6)
    second = diagonal_coefficients(effective_hamiltonian(ats, space, pump(ats)), space)
    d_a = CROSSING_DETUNING_MHZ * TWO_PI - second["stark_a"]
    base = replace(ats, detuning_a=d_a)
    centre = 2 * (d_a + second["stark_a"]) + second["kerr_a"] - second["stark_b"]
    scan = avoided_crossing(
        floquet_splitting, base, space, pump(ats), ((2, 0), (0, 1)), centre, 2 * TWO_PI
    )
    return scan["s_min"] / (2 * math.sqrt(2))


def calibrated_pump(ats: ATSParams, g2_nominal: float, iterations: int = CALIBRATION_STEPS) -> dict:
    """Secant search for the eps_p at which the exact (Floquet) |g2| equals the prompt's g2."""
    target = ats.g2_formula
    x0, f0 = ats.eps_p, g2_nominal - target
    x1 = ats.eps_p * target / g2_nominal
    f1 = floquet_g2(ats.with_pump(x1)) - target
    for _ in range(iterations):
        x0, f0, x1 = x1, f1, x1 - f1 * (x1 - x0) / (f1 - f0)
        f1 = floquet_g2(ats.with_pump(x1)) - target
    cal = ats.with_pump(x1)
    res = floquet_vs_effective(cal, SPACE)
    return {
        "eps_p": x1,
        "g2_floquet": f1 + target,
        "ratio_to_prompt_eps_p": x1 / ats.eps_p,
        "floquet_khz": {k: to_khz(v) for k, v in res["floquet"].items()},
        "effective_khz": {k: to_khz(v) for k, v in res["effective"].items()},
        "second_order_operator_khz": {
            k: to_khz(v)
            for k, v in diagonal_coefficients(
                effective_hamiltonian(cal, FockSpace(6, 6), pump(cal)), FockSpace(6, 6)
            ).items()
        },
    }


def run_floquet(ats: ATSParams, terms: dict) -> dict:
    conv = {}
    for space in SPACES_CONV:
        t0 = time.time()
        res = floquet_vs_effective(ats, space)
        conv[f"{space.na}x{space.nb}"] = {
            "floquet_khz": {k: to_khz(v) for k, v in res["floquet"].items()},
            "effective_khz": {k: to_khz(v) for k, v in res["effective"].items()},
            "min_overlap_interior": res["min_overlap_interior"],
            "seconds": time.time() - t0,
        }
        print("floquet", space, conv[f"{space.na}x{space.nb}"])
    main = floquet_vs_effective(ats, SPACE)
    scan = []
    for s in EPS_SCALES:
        p = ats.with_pump(ats.eps_p * s)
        r = floquet_vs_effective(p, SPACE)
        scan.append(
            {
                "eps_p": p.eps_p,
                "floquet": r["floquet"],
                "effective": r["effective"],
                "max_residual_interior": r["max_residual_interior"],
            }
        )
    second = terms["second_order"]
    g2_scan = g2_from_crossing(ats, second)
    print("g2 crossing (MHz):", {k: v["g2"] / TWO_PI for k, v in g2_scan.items()})
    g2_delta = g2_scan["effective"]["g2"] - g2_scan["floquet"]["g2"]  # coefficient is negative
    fit = floquet_fit(main["h_eff"], main["match"], SPACE, g2_delta)
    fit["deltas"]["g2"] = g2_delta
    print("floquet fit deltas (kHz):", {k: to_khz(v) for k, v in fit["deltas"].items()})
    print("fit residual (kHz):", to_khz(fit["residual"]))
    bs_scan = conversion_from_crossing(ats, second, terms["readout_flux_tone"])
    print("g_bs crossing (kHz):", {k: to_khz(v["g_bs_total"]) for k, v in bs_scan.items()})
    cal = calibrated_pump(ats, g2_scan["floquet"]["g2"])
    print("calibrated pump:", cal)
    return {
        "calibrated_pump": cal,
        "convergence": conv,
        "fit": fit,
        "eps_scan": scan,
        "g2_crossing": g2_scan,
        "bs_crossing": bs_scan,
        "residual_full_pump": main["max_residual_interior"],
    }


# --- outputs -------------------------------------------------------------------------------


def load_symbolic() -> dict | None:
    path = RESULTS / "symbolic_terms.json"
    return json.loads(path.read_text()) if path.exists() else None


def term(value: float, operator: str, formula: str, verified: bool, **extra) -> dict:
    return {
        "value_rad_per_us": float(value),
        "value_khz": to_khz(value),
        "operator": operator,
        "formula": formula,
        "verified_numerically": verified,
        **extra,
    }


def build_effective_terms(ats: ATSParams, terms: dict, floq: dict) -> dict:
    s = terms["second_order"]
    fl = floq["convergence"][f"{SPACE.na}x{SPACE.nb}"]
    fit = floq["fit"]["deltas"]
    g2 = terms["g2"]
    tone = terms["readout_flux_tone"]
    kappa_b = CatParams.reglade().kappa_b
    second = "time-averaged 2nd order in eps_p, all orders in phi (Fock-matrix sin phi)"
    beta = terms["pump_displacement"]
    delta0 = 2 * (ats.phi_a * beta["beta_a"] + ats.phi_b * beta["beta_b"])

    def diag_term(key: str, op: str, formula: str, verified: bool = True, note: str = "") -> dict:
        return term(
            s[key],
            op,
            formula,
            verified,
            note=note or VERIFY_NOTE,
            method=second,
            floquet_fitted_rad_per_us=s[key] + fit[key],
            eigenvalue_check_khz={
                "floquet": fl["floquet_khz"][key],
                "effective": fl["effective_khz"][key],
            },
        )

    disp = (
        "E_J' = (A^2 E/2) sum_k phi_k^2 S_k; A = 4 E_J J1(eps_p) ~ 2 E_J eps_p, E = exp(-Phi^2/2),"
        " S_k = 1/(w_k - w_p) + 1/(w_k + w_p); full formulas: symbolic_terms.json"
    )
    out = {
        "stark_a": diag_term(
            "stark_a",
            "a^dag a",
            f"phi^4: (A^2 E^2 phi_a^2/2)(phi_a^2 S_a + phi_b^2 S_b); {disp}",
        ),
        "stark_b": diag_term(
            "stark_b",
            "b^dag b",
            f"phi^4: (A^2 E^2 phi_b^2/2)(phi_a^2 S_a + phi_b^2 S_b); {disp}",
        ),
        "kerr_a": diag_term(
            "kerr_a",
            "(K/2) a^dag^2 a^2",
            "A^2 E^2 phi_a^4 {phi_a^2[-(5/8)S_a - (1/8)S_3a] + phi_b^2[-(3/4)S_b"
            " + 1/(8(2w_a-w_b+w_p)) - 1/(8(2w_a+w_b-w_p)) - 1/(8(2w_a+w_b+w_p))]}, S_3a with 3w_a",
            False,
            "2nd-order value confirmed by the independent normal-ordered expansion (1e-5), but "
            "NOT resolved by Floquet: on |2,0> it is masked by the g2^2/detuning shift (~50 kHz) "
            "and the Floquet fit residual (38 kHz) exceeds it; treat |K_a| <~ 15 kHz",
        ),
        "kerr_b": diag_term(
            "kerr_b", "(K/2) b^dag^2 b^2", "displacement part -E_J' phi_b^4/2 + quantum part"
        ),
        "chi_ab": diag_term(
            "chi_ab", "a^dag a b^dag b", "displacement part -E_J' phi_a^2 phi_b^2 + quantum part"
        ),
        "g2_vacuum": term(
            -g2["g2_vacuum"],
            "a^2 b^dag + h.c. (coefficient on |2,0>)",
            "-(1/2) E_J [2 J1(eps_p)] phi_a^2 phi_b exp(-(phi_a^2+phi_b^2)/2)",
            True,
            prompt_formula_rad_per_us=ats.g2_formula,
            ratio_to_prompt_formula=g2["g2_vacuum"] / ats.g2_formula,
            floquet_avoided_crossing_abs=floq["g2_crossing"]["floquet"]["g2"],
            effective_avoided_crossing_abs=floq["g2_crossing"]["effective"]["g2"],
            third_order_relative_correction=floq["g2_crossing"]["floquet"]["g2"]
            / floq["g2_crossing"]["effective"]["g2"]
            - 1,
            classical_displacement_estimate=-3 * delta0**2 / 8,
        ),
        "g2_dressing": {
            "operator": "g2(n_a, n_b) a^2 b^dag, n_a, n_b = photons before the exchange",
            "formula": "g2(n_a,n_b) = g2_vacuum L_{n_a-2}^{(2)}(phi_a^2)/C(n_a,2)"
            " L_{n_b}^{(1)}(phi_b^2)/(n_b+1)"
            " ~ g2_vacuum [1 - (n_a-2) phi_a^2/3 - n_b phi_b^2/2]",
            "slope_per_memory_photon": g2["slope_a"],
            "slope_per_buffer_photon": g2["slope_b"],
            "closed_form_slopes": [-(ats.phi_a**2) / 3, -(ats.phi_b**2) / 2],
            "table_abs_rad_per_us": g2["table_rad_per_us"],
            "verified_numerically": True,
        },
        "a4b2": term(
            terms["a4b2"], "a^4 b^dag^2 + h.c.", "second order, 2 w_p resonance (abs value)", False
        ),
        "two_photon_detuning": term(
            2 * s["stark_a"] - s["stark_b"],
            "2 stark_a - stark_b (+ kerr_a on |2,0>)",
            "pump must be retuned to w_p' = w_p + 2 stark_a - stark_b",
            True,
            floquet_rad_per_us=(2 * fl["floquet_khz"]["stark_a"] - fl["floquet_khz"]["stark_b"])
            * KHZ,
            ratio_to_kappa_b=abs(2 * s["stark_a"] - s["stark_b"]) / kappa_b,
        ),
        "pump_displacement": terms["pump_displacement"],
        "calibrated_pump": {
            "note": "eps_p at which the exact Floquet |g2| equals 2pi x 0.763 MHz; Stark/Kerr "
            "there (kHz): Floquet eigenvalue differences and 2nd-order operator coefficients",
            **floq["calibrated_pump"],
        },
        "readout_flux_tone": {
            "description": "phi_Sigma += eps_z cos(w_a t); values for eps_Z/2pi = "
            f"{READOUT_EPS_Z_MHZ} MHz; 2nd-order terms scale as eps_Z (pump x tone) or eps_Z^2",
            "eps_Z": term(
                tone["eps_Z"], "a + a^dag", "E_J [2 J1(eps_z)] phi_a exp(-Phi^2/2)", True
            ),
            "drive_dressing_per_memory_photon": tone["drive_dressing_per_memory_photon"],
            "drive_dressing_per_buffer_photon": tone["drive_dressing_per_buffer_photon"],
            "beam_splitter": term(
                tone["beam_splitter_ab_dag"],
                "a b^dag + h.c.",
                "pump x tone second order; g_bs / eps_Z = "
                f"{tone['beam_splitter_ab_dag'] / tone['eps_Z']:.4f}",
                True,
                crossing_total_abs={k: v["g_bs_total"] for k, v in floq["bs_crossing"].items()},
                induced_memory_loss_rate=4 * tone["beam_splitter_ab_dag"] ** 2 / kappa_b,
            ),
            "a_dag3_b": term(
                tone["a_dag3_b"], "a^dag^3 b + h.c.", "pump x tone second order", False
            ),
            "a_dag2": term(
                tone["a_dag2_squeezing"], "a^dag^2 + h.c.", "tone x tone second order", False
            ),
            "stark_changes_rad_per_us": tone["stark_changes"],
            "off_resonant_buffer_displacement": tone["off_resonant_buffer_displacement"],
        },
        "capacitive_memory_drive": {
            "operator": "eps_Z (a + a^dag)",
            "note": "linear drive: no new resonant term at first order; cross terms with the pump "
            "need counter-rotating components and are suppressed by w_a and high phi order",
            "verified_numerically": False,
        },
    }
    return out


def run_outputs(ats: ATSParams, terms: dict, floq: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "conventions": {
            "units": "angular rad/us (divide by 2 pi for MHz); value_khz = value / (2 pi 1e-3)",
            "frame": "rotating at (5u, 6u), u/2pi = 0.8 GHz; bare pump w_p = 4u exactly",
            "hamiltonian": __doc__.split("Conventions:")[1].strip(),
            "floquet_fitted": "2nd-order coefficient + correction fitted to exact quasi-energies",
        },
        "parameters": {
            "omega_a": ats.omega_a,
            "omega_b": ats.omega_b,
            "omega_p": ats.omega_p,
            "e_j": ats.e_j,
            "phi_a": ats.phi_a,
            "phi_b": ats.phi_b,
            "eps_p": ats.eps_p,
            "g2_formula": ats.g2_formula,
            "floquet_truncation": [SPACE.na, SPACE.nb],
        },
        "terms": build_effective_terms(ats, terms, floq),
        "symbolic_cross_check": load_symbolic(),
    }
    (RESULTS / "effective_terms.json").write_text(json.dumps(payload, indent=2, default=float))
    raw = {"terms": terms, "floquet": {k: v for k, v in floq.items()}}
    (RESULTS / "raw.json").write_text(json.dumps(raw, indent=2, default=float))


# --- figures -------------------------------------------------------------------------------

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#0b0b0b", "#52514e"
FACTOR2_NOTE = "angular rates shown as f = value / 2pi"


def _bar_rows(eff: dict) -> list[tuple[str, float, float | None]]:
    t = eff["terms"]
    tone = t["readout_flux_tone"]
    g2 = t["g2_vacuum"]

    def khz(entry: dict) -> float:
        return entry["value_khz"]

    def fl(entry: dict) -> float | None:
        v = entry.get("floquet_fitted_rad_per_us")
        return None if v is None else to_khz(v)

    return [
        (
            "buffer Stark shift  stark_b",
            khz(t["stark_b"]),
            to_khz(t["stark_b"]["eigenvalue_check_khz"]["floquet"] * KHZ),
        ),
        (
            "2-photon detuning  2stark_a - stark_b",
            khz(t["two_photon_detuning"]),
            to_khz(t["two_photon_detuning"]["floquet_rad_per_us"]),
        ),
        ("buffer Kerr  K_b", khz(t["kerr_b"]), None),
        (
            "memory Stark shift  stark_a",
            khz(t["stark_a"]),
            to_khz(t["stark_a"]["eigenvalue_check_khz"]["floquet"] * KHZ),
        ),
        ("|g2| (vacuum, 1st order)", abs(khz(g2)), to_khz(g2["floquet_avoided_crossing_abs"])),
        ("cross-Kerr  chi_ab", khz(t["chi_ab"]), None),
        (
            "g2 shift per buffer photon",
            t["g2_dressing"]["slope_per_buffer_photon"] * abs(khz(g2)),
            None,
        ),
        (
            "g2, 3rd-order pump correction",
            g2["third_order_relative_correction"] * abs(khz(g2)),
            None,
        ),
        ("memory Kerr  K_a", khz(t["kerr_a"]), None),
        (
            "g2 shift per memory photon",
            t["g2_dressing"]["slope_per_memory_photon"] * abs(khz(g2)),
            None,
        ),
        ("a^4 b^dag^2", khz(t["a4b2"]), None),
        (
            f"flux readout: a b^dag  (eps_Z/2pi={READOUT_EPS_Z_MHZ} MHz)",
            khz(tone["beam_splitter"]),
            None,
        ),
        ("flux readout: a^dag^3 b", khz(tone["a_dag3_b"]), None),
    ]


def plot_terms(eff: dict, path: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rows = _bar_rows(eff)
    fig, ax = plt.subplots(figsize=(9, 6.2))
    y = np.arange(len(rows))[::-1]
    for yi, (_, v, fv) in zip(y, rows, strict=True):
        ax.barh(yi, abs(v), color=BLUE if v > 0 else ORANGE, height=0.6)
        if fv is not None:
            ax.plot(abs(fv), yi, "D", color=INK, ms=7, mfc="white", mew=1.5)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("|coefficient| / 2pi  (kHz)")
    refs = (("kappa_b", 2600.0), ("kappa_2", 4 * 0.763**2 / 2.6 * 1e3), ("kappa_a", 9.3))
    for name, val in refs:
        ax.axvline(val, color=MUTED, lw=1, ls="--")
        ax.text(val, len(rows) - 0.4, f" {name}", color=MUTED, fontsize=8, rotation=90, va="top")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE, label="positive coefficient"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE, label="negative coefficient"),
        plt.Line2D(
            [],
            [],
            marker="D",
            ls="",
            color=INK,
            mfc="white",
            label="exact Floquet (eigenvalue-level)",
        ),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    ax.set_title(
        "ATS effective terms at the prompt parameters (bars: 2nd-order expansion)", fontsize=10
    )
    ax.text(
        0.0,
        -0.13,
        FACTOR2_NOTE + "\nbars: 2nd-order operator coefficients; diamonds: exact Floquet "
        "quasi-energy differences (Stark) or avoided crossing (g2)",
        transform=ax.transAxes,
        fontsize=7,
        color=MUTED,
    )
    ax.grid(axis="x", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_floquet(raw: dict, path: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    scan = raw["floquet"]["eps_scan"]
    eps = np.array([r["eps_p"] for r in scan])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    # chi_ab is left out: its eigenvalue-level value includes a g2^2/detuning part that does
    # not scale as eps_p^2 and crosses zero near eps_p = 0.025 (operator chi_ab is in the JSON).
    keys = (("stark_b", BLUE), ("stark_a", ORANGE), ("kerr_b", AQUA))
    for key, col in keys:
        fq = np.array([abs(r["floquet"][key]) for r in scan]) / KHZ
        ef = np.array([abs(r["effective"][key]) for r in scan]) / KHZ
        axes[0].plot(eps, ef, "-", color=col, lw=2, label=f"{key} (H1+H2)")
        axes[0].plot(eps, fq, "o", color=col, ms=8, mfc="white", mew=1.8, label=f"{key} (Floquet)")
        axes[1].plot(eps, np.abs(fq - ef) / fq, "o-", color=col, lw=2, ms=8, label=key)
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="pump amplitude eps_p (rad)",
        ylabel="|quasi-energy difference| / 2pi (kHz)",
        title="(a) shifts vs pump",
    )
    guide = (eps / eps[-2]) ** 2 * 0.02
    axes[1].plot(eps, guide, "--", color=MUTED, lw=1, label="slope 2 (eps_p^4 error)")
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="pump amplitude eps_p (rad)",
        ylabel="|Floquet - effective| / |Floquet|",
        title="(b) relative residual",
    )
    for ax in axes[:2]:
        ax.axvline(eps[-2], color=MUTED, lw=1, ls=":")
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    axes[1].legend(fontsize=8, frameon=False)
    cross = raw["floquet"]["g2_crossing"]
    for name, col, style in (("floquet", BLUE, "o"), ("effective", ORANGE, "-")):
        c = cross[name]
        d = (np.array(c["detuning_b_grid"]) - cross["effective"]["d0"]) / KHZ / 1e3
        sp_ = np.array(c["splitting"]) / KHZ / 1e3
        lab = f"{name}: |g2|/2pi = {c['g2'] / TWO_PI:.4f} MHz"
        if style == "o":
            axes[2].plot(d, sp_, "o", color=col, ms=7, mfc="white", mew=1.8, label=lab)
        else:
            axes[2].plot(d, sp_, "-", color=col, lw=2, label=lab)
    axes[2].set(
        xlabel="buffer detuning from the H1+H2 resonance (MHz)",
        ylabel="(2,0)/(0,1) splitting / 2pi (MHz)",
        title="(c) resonance-restored g2 (min = 2 sqrt2 |g2|)",
    )
    axes[2].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(color="#e4e3df", lw=0.6)
    fig.text(
        0.01,
        0.01,
        f"Exact 1.25 ns Floquet propagator, truncation {SPACE.na}x{SPACE.nb} "
        f"(a,b) / 7x6 (c); interior states n_a<={INTERIOR_NA}, n_b<={INTERIOR_NB}; "
        f"dotted line = prompt eps_p; {FACTOR2_NOTE}",
        fontsize=7,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    eff = json.loads((RESULTS / "effective_terms.json").read_text())
    raw = json.loads((RESULTS / "raw.json").read_text())
    plot_terms(eff, FIGURES / "task24a_terms.png")
    plot_floquet(raw, FIGURES / "task24a_floquet_check.png")


def main(parts: list[str]) -> None:
    ats = ATSParams.prompt()
    if {"terms", "floquet"} & set(parts):
        terms = run_terms(ats)
        floq = run_floquet(ats, terms)
        run_outputs(ats, terms, floq)
    if "figures" in parts:
        make_figures()


if __name__ == "__main__":
    main(sys.argv[1:] or ["terms", "floquet", "figures"])
