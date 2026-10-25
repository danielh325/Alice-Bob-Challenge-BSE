"""Task 2.2: phase-flip rate Gamma_Z three ways, its alpha^2 slope, and a thermal test (R5, R6).

Methods, all from |C+> (x) |0>:
  (i)   free-offset exponential fit of <P>(t) for t >= T_FIT_START;
  (ii)  the exact identity d<P>/dt = -2 kappa_a [(1 + n) <a^dag a P> + n <a a^dag P>] (no fit);
        Gamma_Z(t) = -(d<P>/dt)/<P>, reported after dividing out (1 - P_inf / P(t)) with the
        exact steady-state parity P_inf (zero mode of the Liouvillian), and also raw;
  (iii) the slowest nonzero eigenvalue of the parity-even Liouvillian block.

Rates are decay rates of <P> in rad/us (one-way flip rate = Gamma_Z / 2).
Regenerates results/task22/task22_phaseflip.json and figures/track2/task22_*.png.
Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv313/bin/python -u studies/task22_phaseflip.py
"""

import json
import math
import sys
import time
from pathlib import Path

import dynamiqs as dq
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from catqubit.liouvillian import FockTruncation  # noqa: E402
from catqubit.params import CatParams  # noqa: E402
from catqubit.phaseflip import (  # noqa: E402
    ParityTrace,
    cat_manifold_offset,
    cat_manifold_rate,
    even_cat_rate,
    evolve_parity,
    fit_line,
    fit_parity_decay,
    instantaneous_phase_flip_rate,
    large_alpha_rate,
    steady_state_parity,
    thermal_rate,
)
from catqubit.rates import phase_flip_rate  # noqa: E402

ALPHAS = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.1, 2.2, 2.4, 2.6)
TARGET_ALPHA = 2.1
THERMAL_TIME_ALPHAS = (1.4, 2.1)
N_TH_A, N_TH_B = 0.10, 0.011  # Réglade et al. 2024, Table S2 (checked in the arXiv PDF)
REGLADE_SLOPE_EXCESS = 0.25  # Réglade 2024, Fig. 3c caption: slope 25% above 2 kappa_a
SLOPE_MIN_ALPHA = 1.4  # also fit the slope without the small-cat points
NB = 5
NA_MARGIN = 8  # na = ceil(alpha^2 + 6 alpha) + NA_MARGIN
CONV_DNA, CONV_DNB = 4, 2  # convergence check: na + 4 and nb + 2 separately
CONV_RTOL = 1e-4
T_FIT_START = 1.0  # us; |C+> (x) |0> is not the two-mode steady state
T_FIT_LATE = 2.0  # us; sensitivity of the fit to the start of the window
T_IDENTITY_START = 2.0  # us; window for the identity plateau
DECAY_TIMES = 3.0  # simulate T_FIT_START + 3 / Gamma_Z
DT = 0.01  # us
RESULTS = ROOT / "results" / "task22"
FIGURES = ROOT / "figures" / "track2"
DPI = 150
CONVENTION = "Gamma_Z = decay rate of <P> (rad/us); one-way flip rate = Gamma_Z / 2"


def truncation(alpha: float) -> FockTruncation:
    return FockTruncation(na=math.ceil(alpha**2 + 6 * alpha) + NA_MARGIN, nb=NB)


def spectral_with_convergence(p: CatParams, n_th_a: float = 0.0, n_th_b: float = 0.0) -> dict:
    """Gamma_Z from the even block at the chosen truncation, with na and nb checked separately."""
    base = truncation(p.alpha)
    variants = {
        "base": base,
        "na_plus": FockTruncation(na=base.na + CONV_DNA, nb=base.nb),
        "nb_plus": FockTruncation(na=base.na, nb=base.nb + CONV_DNB),
    }
    rates = {k: phase_flip_rate(p, tr, n_th_a, n_th_b) for k, tr in variants.items()}
    rel = {k: abs(rates[k] / rates["base"] - 1) for k in ("na_plus", "nb_plus")}
    if max(rel.values()) > CONV_RTOL:
        raise RuntimeError(f"alpha={p.alpha}: spectral Gamma_Z not converged, {rel}")
    return {
        "rate": rates["base"],
        "na": base.na,
        "nb": base.nb,
        "rel_change_na_plus_4": rel["na_plus"],
        "rel_change_nb_plus_2": rel["nb_plus"],
    }


def time_grid(gamma: float) -> jnp.ndarray:
    t_max = T_FIT_START + DECAY_TIMES / gamma
    return jnp.linspace(0.0, t_max, int(round(t_max / DT)) + 1)


def fit_from(trace: ParityTrace, t_start: float) -> dict:
    late = trace.t >= t_start
    fit = fit_parity_decay(trace.t[late], trace.parity[late])
    return {"rate": fit.rate, "rate_stderr": fit.rate_stderr, "offset": fit.offset}


def identity_summary(
    trace: ParityTrace, p: CatParams, p_inf: float, n_th: float = 0.0
) -> tuple[dict, np.ndarray, np.ndarray]:
    raw = instantaneous_phase_flip_rate(
        p.kappa_a, trace.parity, trace.n_parity, trace.m_parity, n_th
    )
    corrected = raw / (1 - p_inf / trace.parity)
    window = trace.t >= T_IDENTITY_START
    summary = {
        "rate_median": float(np.median(corrected[window])),
        "rate_min": float(np.min(corrected[window])),
        "rate_max": float(np.max(corrected[window])),
        "raw_at_t0": float(raw[0]),
        "raw_at_window_start": float(raw[window][0]),
        "window_start_us": T_IDENTITY_START,
    }
    return summary, raw, corrected


def time_domain(p: CatParams, gamma: float, n_th_a: float = 0.0, n_th_b: float = 0.0) -> dict:
    """Methods (i) and (ii) at the chosen truncation, plus a larger one for convergence."""
    base = truncation(p.alpha)
    tsave = time_grid(gamma)
    trace = evolve_parity(p, base.na, base.nb, tsave, n_th_a, n_th_b)
    big = evolve_parity(p, base.na + CONV_DNA, base.nb + CONV_DNB, tsave, n_th_a, n_th_b)
    p_inf = steady_state_parity(p, base, n_th_a, n_th_b)
    ident, raw, corrected = identity_summary(trace, p, p_inf, n_th_a)
    ident_big, _, _ = identity_summary(big, p, p_inf, n_th_a)
    fit, fit_big = fit_from(trace, T_FIT_START), fit_from(big, T_FIT_START)
    return {
        "na": base.na,
        "nb": base.nb,
        "t_max_us": float(tsave[-1]),
        "fit": fit,
        "fit_window_start_us": T_FIT_START,
        "fit_late": fit_from(trace, T_FIT_LATE),
        "fit_late_window_start_us": T_FIT_LATE,
        "fit_rel_change_bigger_trunc": abs(fit_big["rate"] / fit["rate"] - 1),
        "identity": ident,
        "identity_rel_change_bigger_trunc": abs(
            ident_big["rate_median"] / ident["rate_median"] - 1
        ),
        "steady_state_parity": p_inf,
        "_trace": trace,
        "_raw": raw,
        "_corrected": corrected,
    }


def analytic(p: CatParams) -> dict:
    return {
        "two_kappa_alpha2": large_alpha_rate(p.kappa_a, p.alpha),
        "even_cat_initial": even_cat_rate(p.kappa_a, p.alpha),
        "cat_manifold_coth": cat_manifold_rate(p.kappa_a, p.alpha),
        "cat_manifold_offset": cat_manifold_offset(p.alpha),
        "thermal": thermal_rate(p.kappa_a, p.alpha, N_TH_A),
    }


def run_alpha(p0: CatParams, alpha: float) -> dict:
    start = time.time()
    p = p0.with_alpha(alpha)
    spectral = spectral_with_convergence(p)
    td = time_domain(p, spectral["rate"])
    thermal = spectral_with_convergence(p, N_TH_A, N_TH_B)
    row = {
        "alpha": alpha,
        "alpha2": alpha**2,
        "spectral": spectral,
        "time_domain": td,
        "analytic": analytic(p),
        "thermal_spectral": thermal,
    }
    print(
        f"alpha={alpha:.1f} spec={spectral['rate']:.6f} fit={td['fit']['rate']:.6f} "
        f"identity={td['identity']['rate_median']:.6f} thermal={thermal['rate']:.6f} "
        f"({time.time() - start:.0f}s)",
        flush=True,
    )
    return row


def slope_summary(rows: list[dict], key, kappa_a: float, min_alpha: float = 0.0) -> dict:
    sel = [r for r in rows if r["alpha"] >= min_alpha - 1e-9]
    fit = fit_line([r["alpha2"] for r in sel], [key(r) for r in sel])
    return {
        "slope": fit.slope,
        "slope_stderr": fit.slope_stderr,
        "intercept": fit.intercept,
        "slope_over_2kappa_a": fit.slope / (2 * kappa_a),
        "alpha_range": [sel[0]["alpha"], sel[-1]["alpha"]],
    }


def all_slopes(rows: list[dict], kappa_a: float) -> dict:
    keys = {
        "spectral": lambda r: r["spectral"]["rate"],
        "fit": lambda r: r["time_domain"]["fit"]["rate"],
        "identity": lambda r: r["time_domain"]["identity"]["rate_median"],
        "thermal_spectral": lambda r: r["thermal_spectral"]["rate"],
    }
    out = {}
    for name, key in keys.items():
        out[name] = slope_summary(rows, key, kappa_a)
        out[f"{name}_alpha_ge_{SLOPE_MIN_ALPHA}"] = slope_summary(
            rows, key, kappa_a, SLOPE_MIN_ALPHA
        )
    return out


def thermal_time_points(p0: CatParams, rows: list[dict]) -> list[dict]:
    points = []
    for alpha in THERMAL_TIME_ALPHAS:
        row = next(r for r in rows if abs(r["alpha"] - alpha) < 1e-9)
        p = p0.with_alpha(alpha)
        td = time_domain(p, row["thermal_spectral"]["rate"], N_TH_A, N_TH_B)
        points.append({"alpha": alpha, **td})
        print(
            f"thermal alpha={alpha}: fit={td['fit']['rate']:.6f} "
            f"identity={td['identity']['rate_median']:.6f}",
            flush=True,
        )
    return points


def plot_rates(rows: list[dict], thermal_points: list[dict], kappa_a: float) -> None:
    a2 = np.array([r["alpha2"] for r in rows])
    grid = np.linspace(0.8, 7.0, 200)
    spec = np.array([r["spectral"]["rate"] for r in rows])
    fit = np.array([r["time_domain"]["fit"]["rate"] for r in rows])
    ident = np.array([r["time_domain"]["identity"]["rate_median"] for r in rows])
    therm = np.array([r["thermal_spectral"]["rate"] for r in rows])
    coth = np.array([cat_manifold_rate(kappa_a, math.sqrt(x)) for x in grid])
    th_a2 = np.array([p["alpha"] ** 2 for p in thermal_points])
    th_fit = np.array([p["fit"]["rate"] for p in thermal_points])
    th_id = np.array([p["identity"]["rate_median"] for p in thermal_points])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.8))
    ax.plot(grid, 2 * kappa_a * grid, "k--", label=r"$2\kappa_a\alpha^2$")
    thermal_line = [thermal_rate(kappa_a, math.sqrt(x), N_TH_A) for x in grid]
    ax.plot(
        grid, thermal_line, ":", color="C3", label=rf"$2\kappa_a[\alpha^2(1+2n)+n]$, $n={N_TH_A}$"
    )
    ax.plot(a2, spec, "o", mfc="none", ms=9, label="(iii) spectral, even block")
    ax.plot(a2, fit, "x", ms=7, label=r"(i) offset fit of $\langle P\rangle(t)$, $t\geq1\,\mu$s")
    ax.plot(a2, ident, "+", ms=9, label="(ii) exact identity (no fit)")
    ax.plot(a2, therm, "s", color="C3", mfc="none", label="thermal, spectral")
    ax.plot(th_a2, th_fit, "x", color="C3", ms=7, label="thermal, fit")
    ax.plot(th_a2, th_id, "+", color="C3", ms=9, label="thermal, identity")
    ax.set_xlabel(r"$\alpha^2$ (= $\epsilon_d/g_2$)")
    ax.set_ylabel(r"$\Gamma_Z$ (rad/$\mu$s)")
    ax.set_title("Phase-flip rate vs cat size")
    ax.legend(fontsize=8)

    norm = 2 * kappa_a * a2
    bx.axhline(1.0, color="k", ls="--", lw=1, label=r"$2\kappa_a\alpha^2$")
    bx.plot(
        grid,
        coth / (2 * kappa_a * grid),
        "-",
        color="grey",
        label=r"cat manifold: $\coth(2\alpha^2)$",
    )
    bx.plot(a2, spec / norm, "o", mfc="none", ms=9, label="(iii) spectral")
    bx.plot(a2, fit / norm, "x", ms=7, label="(i) offset fit")
    bx.plot(a2, ident / norm, "+", ms=9, label="(ii) identity")
    bx.set_xlabel(r"$\alpha^2$")
    bx.set_ylabel(r"$\Gamma_Z\,/\,2\kappa_a\alpha^2$")
    bx.set_title("Deviation from the linear law (no thermal photons)")
    bx.legend(fontsize=8)
    fig.suptitle(CONVENTION, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "task22_rates.png", dpi=DPI)
    plt.close(fig)


def plot_instantaneous(row: dict, thermal_point: dict, p: CatParams) -> None:
    td = row["time_domain"]
    trace, raw, corrected = td["_trace"], td["_raw"], td["_corrected"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax.plot(trace.t, raw, label=r"$-\dot{\langle P\rangle}/\langle P\rangle$ (identity, raw)")
    ax.plot(trace.t, corrected, "--", label=r"raw $/(1-P_\infty/\langle P\rangle)$")
    ax.axhline(row["spectral"]["rate"], color="k", lw=1, label="(iii) spectral")
    ax.axhline(
        large_alpha_rate(p.kappa_a, p.alpha), color="grey", ls=":", label=r"$2\kappa_a\alpha^2$"
    )
    ax.axhline(
        even_cat_rate(p.kappa_a, p.alpha),
        color="C2",
        ls="-.",
        label=r"$2\kappa_a\alpha^2\tanh\alpha^2$ ($t=0$)",
    )
    ax.axvline(T_FIT_START, color="grey", lw=0.5)
    ax.set_xlim(0, 4)
    ax.set_xlabel(r"$t$ ($\mu$s)")
    ax.set_ylabel(r"$\Gamma_Z(t)$ (rad/$\mu$s)")
    ax.set_title(rf"Instantaneous phase-flip rate, $\alpha={p.alpha}$, no thermal photons")
    ax.legend(fontsize=8)

    th_trace = thermal_point["_trace"]
    bx.semilogy(trace.t, trace.parity, label=r"$\langle P\rangle$, $n_{th}=0$")
    bx.semilogy(
        th_trace.t,
        th_trace.parity,
        color="C3",
        label=rf"$\langle P\rangle$, $n_{{th,a}}={N_TH_A}$, $n_{{th,b}}={N_TH_B}$",
    )
    f = td["fit"]
    t = trace.t[trace.t >= T_FIT_START]
    amp = (trace.parity[trace.t >= T_FIT_START][0] - f["offset"]) * np.exp(f["rate"] * t[0])
    bx.semilogy(
        t,
        f["offset"] + amp * np.exp(-f["rate"] * t),
        "k--",
        lw=1,
        label=rf"fit $t\geq{T_FIT_START}\,\mu$s: $\Gamma_Z$={f['rate']:.4f}",
    )
    bx.set_xlabel(r"$t$ ($\mu$s)")
    bx.set_ylabel(r"$\langle P\rangle$")
    bx.set_title(r"Parity decay from $|C_+\rangle\otimes|0\rangle$")
    bx.legend(fontsize=8)
    fig.suptitle(CONVENTION, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "task22_instantaneous.png", dpi=DPI)
    plt.close(fig)


def strip_private(obj):
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    return obj


def main() -> None:
    dq.set_precision("double")
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    p0 = CatParams.reglade()
    rows = [run_alpha(p0, alpha) for alpha in ALPHAS]
    thermal_points = thermal_time_points(p0, rows)
    slopes = all_slopes(rows, p0.kappa_a)
    target = next(r for r in rows if abs(r["alpha"] - TARGET_ALPHA) < 1e-9)
    plot_rates(rows, thermal_points, p0.kappa_a)
    plot_instantaneous(target, thermal_points[-1], p0.with_alpha(TARGET_ALPHA))
    excess = slopes["thermal_spectral"]["slope_over_2kappa_a"] - 1
    out = {
        "params_rad_per_us": {"g2": p0.g2, "kappa_b": p0.kappa_b, "kappa_a": p0.kappa_a},
        "convention": CONVENTION,
        "solver": "dynamiqs mesolve Tsit5 rtol=atol=1e-9, double precision; spectral: "
        "scipy splu (COLAMD) + eigs shift-invert on the parity-even block",
        "convergence_rule": f"na + {CONV_DNA} and nb + {CONV_DNB} separately, rtol {CONV_RTOL}",
        "thermal": {"n_th_a": N_TH_A, "n_th_b": N_TH_B, "source": "Réglade 2024 Table S2"},
        "rows": rows,
        "thermal_time_domain": thermal_points,
        "slopes": slopes,
        "thermal_hypothesis_R6": {
            "predicted_slope_excess_1_plus_2n": 2 * N_TH_A,
            "measured_slope_excess_spectral": excess,
            "reglade_reported_excess": REGLADE_SLOPE_EXCESS,
            "reglade_attribution": "interplay of the strong two-photon pump and uncontrolled "
            "parametric processes (Réglade 2024, main text)",
        },
    }
    path = RESULTS / "task22_phaseflip.json"
    path.write_text(json.dumps(strip_private(out), indent=2))
    print(f"wrote {path}")
    print(json.dumps(slopes, indent=1))


if __name__ == "__main__":
    main()
