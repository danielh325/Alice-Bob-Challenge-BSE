"""Task 2.4b / R9: is there a built-in trade-off between readout speed and bit flips?

For each cat size alpha and readout drive eps_z we compute, from the full two-mode
Liouvillian (catqubit.driven_rates):
  * Gamma_X(eps_z) = -lambda_1 (sum of both one-way rates; one-way = Gamma_X / 2),
  * the drive-induced extra rate Gamma_X(eps_z) - Gamma_X(0),
  * the measurement rate Gamma_meas = (kappa_b / 2) |eps_z / (alpha g2)|^2
    (Réglade et al. 2024, Eq. S18; kappa_b |Delta beta|^2 / 2 with Delta beta = eps_z/(alpha g2)),
  * the exact linear-order coefficient c2 (Gamma_X = Gamma_X(0) + c2 eps_z^2 + O(eps_z^4))
    from second-order eigenvalue perturbation theory on the parity blocks,
  * the metastable readout contrast <b>_+ - <b>_- (codeword_contrast).
The ratio R = extra / Gamma_meas is compared with Dubovitskii's single-mode prediction
Gamma_X^{eps_Z} = (2 eps_Z^2 / (kappa_2 alpha^2)) e^{-2 alpha^2}, which is a ONE-WAY rate:
in our (eigenvalue) convention it is doubled, which gives R_Dub = 2 e^{-2 alpha^2}.

Run: PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u studies/task24_tradeoff.py
     (add --plot to only redraw the figure from results/task24/tradeoff_r9.json).
"""

import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import dynamiqs as dq
import matplotlib.pyplot as plt
import numpy as np

from catqubit.driven_rates import (
    Dissipation,
    HamiltonianTerms,
    codeword_contrast,
    drive_rate_coefficient,
    driven_bit_flip_rate,
)
from catqubit.liouvillian import FockTruncation
from catqubit.params import CatParams

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "task24" / "tradeoff_r9.json"
FIGURE = ROOT / "figures" / "track2" / "task24_tradeoff_ratio.png"
DPI = 150

ALPHAS = (1.2, 1.4, 1.6, 1.8, 2.1)
# eps_z / g2: the R9 range [0.02, 0.5], plus two larger drives for the optimizer's rate table.
EPS_FRACTIONS = tuple(float(x) for x in np.geomspace(0.02, 0.5, 8)) + (0.7, 1.0)
# Converged at eps_z = 0 by the Task 2.1 work (alpha = 1.6 and 2.1 quoted in the rules file).
BASE_TRUNCATIONS = {
    1.2: FockTruncation(na=18, nb=5),
    1.4: FockTruncation(na=20, nb=5, nb_extra=2, na_taper=6.0),
    1.6: FockTruncation(na=23, nb=5, nb_extra=4, na_taper=10.0),
    1.8: FockTruncation(na=26, nb=5, nb_extra=5, na_taper=11.5),
    2.1: FockTruncation(na=31, nb=5, nb_extra=6, na_taper=13.2),
}
EXTRA_MEMORY = 4
EXTRA_BUFFER = 3
CONVERGENCE_RTOL = 2e-3
# The linear-order ratio is "broken" once the full ratio deviates from it by this much.
NONLINEAR_THRESHOLD = 0.10


def ladder(alpha: float) -> dict[str, FockTruncation]:
    base = BASE_TRUNCATIONS[alpha]
    return {
        "base": base,
        "na+4": FockTruncation(base.na + EXTRA_MEMORY, base.nb, base.nb_extra, base.na_taper),
        "nb_extra+3": FockTruncation(
            base.na, base.nb, base.nb_extra + EXTRA_BUFFER, base.na_taper + 2
        ),
        "nb+1": FockTruncation(base.na, base.nb + 1, base.nb_extra, base.na_taper),
    }


def measurement_rate(params: CatParams, eps_z: float) -> float:
    return 0.5 * params.kappa_b * (eps_z / (params.alpha * params.g2)) ** 2


def model(params: CatParams, eps_z: float = 0.0) -> tuple[HamiltonianTerms, Dissipation]:
    return HamiltonianTerms.from_params(params, eps_z=eps_z), Dissipation.from_params(params)


def linear_response(params: CatParams) -> dict:
    """c2 and Gamma_X(0) on every truncation of the ladder, with relative changes."""
    rows = {}
    for name, trunc in ladder(params.alpha).items():
        t0 = time.time()
        lin = drive_rate_coefficient(*model(params), trunc)
        rows[name] = {
            "trunc": asdict(trunc),
            "rate0": lin.rate0,
            "c2": lin.coefficient,
            "residual": lin.residual,
            "seconds": time.time() - t0,
        }
        print(f"  c2 {name:10s} dim={trunc.dim} rate0={lin.rate0:.6e} c2={lin.coefficient:.6e}")
    base = rows["base"]
    deltas = {
        k: {q: abs(v[q] / base[q] - 1) for q in ("rate0", "c2")}
        for k, v in rows.items()
        if k != "base"
    }
    return {"truncations": rows, "relative_change": deltas}


def full_point(params: CatParams, trunc: FockTruncation, eps_z: float) -> dict:
    t0 = time.time()
    res = driven_bit_flip_rate(*model(params, eps_z), trunc, with_gradient=True)
    contrast = codeword_contrast(res, params, trunc)
    return {
        "eps_z": eps_z,
        "rate": res.rate,
        "gradient": dict(res.gradient),
        "residual": res.residual,
        "contrast_re": contrast.real,
        "contrast_im": contrast.imag,
        "seconds": time.time() - t0,
    }


def sweep(params: CatParams) -> dict:
    trunc = BASE_TRUNCATIONS[params.alpha]
    points = []
    for frac in EPS_FRACTIONS:
        point = full_point(params, trunc, frac * params.g2)
        points.append(point)
        print(f"  full eps={frac:.3f} g2 rate={point['rate']:.6e} ({point['seconds']:.0f} s)")
    return {"trunc": asdict(trunc), "points": points}


def full_convergence(params: CatParams, rate0: float) -> dict:
    """Extra rate at the largest drive on enlarged truncations, relative to the base one."""
    eps_z = EPS_FRACTIONS[-1] * params.g2
    out = {}
    for name, trunc in ladder(params.alpha).items():
        if name == "nb+1":
            continue
        point = full_point(params, trunc, eps_z)
        out[name] = {"trunc": asdict(trunc), "rate": point["rate"], "extra": point["rate"] - rate0}
        print(f"  conv {name:10s} rate={point['rate']:.6e}")
    base = out["base"]["extra"]
    for v in out.values():
        v["relative_change_extra"] = abs(v["extra"] / base - 1)
    return out


def summarize(params: CatParams, lin: dict, full: dict) -> dict:
    alpha, g2 = params.alpha, params.g2
    rate0, c2 = lin["truncations"]["base"]["rate0"], lin["truncations"]["base"]["c2"]
    ratio_linear = c2 * g2**2 * alpha**2 / (0.5 * params.kappa_b)
    ratios = [(p["rate"] - rate0) / measurement_rate(params, p["eps_z"]) for p in full["points"]]
    deviation = [r / ratio_linear - 1 for r in ratios]
    onset = None
    for frac, dev in zip(EPS_FRACTIONS, deviation, strict=True):
        if abs(dev) > NONLINEAR_THRESHOLD:
            onset = frac
            break
    linear_contrast = [-p["eps_z"] / (alpha * g2) for p in full["points"]]
    return {
        "rate0": rate0,
        "c2": c2,
        "ratio_linear": ratio_linear,
        "ratio_dubovitskii_eigen": 2 * math.exp(-2 * alpha**2),
        "prefactor_vs_dubovitskii": ratio_linear / (2 * math.exp(-2 * alpha**2)),
        "ratios_full": ratios,
        "relative_deviation_from_linear": deviation,
        "first_eps_fraction_beyond_10pct": onset,
        "contrast_over_adiabatic": [
            p["contrast_re"] / lc for p, lc in zip(full["points"], linear_contrast, strict=True)
        ],
    }


def fit_alpha_scaling(per_alpha: dict) -> dict:
    """Fit ratio_linear = A exp(-gamma alpha^2) over all alphas (least squares on log)."""
    a2 = np.array([float(a) ** 2 for a in per_alpha])
    ratio = np.array([v["summary"]["ratio_linear"] for v in per_alpha.values()])
    slope, intercept = np.polyfit(a2, np.log(ratio), 1)
    local = np.diff(np.log(ratio)) / np.diff(a2)
    return {
        "gamma": float(-slope),
        "A": float(np.exp(intercept)),
        "local_exponents": (-local).tolist(),
    }


def run() -> dict:
    out = {
        "description": __doc__,
        "params": asdict(CatParams.reglade()),
        "eps_fractions": list(EPS_FRACTIONS),
        "convention": "Gamma_X = -lambda_1 = Gamma_{+->-} + Gamma_{-->+}; one-way = Gamma_X/2",
        "per_alpha": {},
    }
    for alpha in ALPHAS:
        params = CatParams.reglade().with_alpha(alpha)
        print(f"alpha={alpha}")
        lin = linear_response(params)
        full = sweep(params)
        conv = full_convergence(params, lin["truncations"]["base"]["rate0"])
        out["per_alpha"][str(alpha)] = {
            "linear_response": lin,
            "full": full,
            "full_convergence": conv,
            "summary": summarize(params, lin, full),
        }
        save(out)
    out["alpha_scaling"] = fit_alpha_scaling(out["per_alpha"])
    save(out)
    return out


def save(out: dict) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))


def plot(out: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    fracs = np.array(out["eps_fractions"])
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(out["per_alpha"])))
    for color, (alpha, data) in zip(colors, out["per_alpha"].items(), strict=True):
        s = data["summary"]
        ax1.loglog(fracs, s["ratios_full"], "o-", color=color, label=f"α = {alpha} (full L)")
        ax1.axhline(s["ratio_linear"], color=color, ls="--", lw=0.8)
        ax1.axhline(s["ratio_dubovitskii_eigen"], color=color, ls=":", lw=0.8)
    ax1.set_xlabel(r"readout drive $\epsilon_Z / g_2$")
    ax1.set_ylabel(r"$[\Gamma_X(\epsilon_Z)-\Gamma_X(0)]\,/\,\Gamma_{\rm meas}$")
    ax1.set_title("(a) drive-induced flips per unit measurement rate")
    ax1.plot([], [], "k--", lw=0.8, label="exact linear order (c₂)")
    ax1.plot([], [], "k:", lw=0.8, label="Dubovitskii 2e^{-2α²} (×2, eigen conv.)")
    ax1.legend(fontsize=7, loc="lower right", ncol=2)
    alphas = np.array([float(a) for a in out["per_alpha"]])
    lin = [v["summary"]["ratio_linear"] for v in out["per_alpha"].values()]
    dub = [v["summary"]["ratio_dubovitskii_eigen"] for v in out["per_alpha"].values()]
    ax2.semilogy(alphas**2, lin, "o-", label="two-mode, exact linear order")
    ax2.semilogy(alphas**2, dub, "s:", label="Dubovitskii single-mode ×2")
    fit = out.get("alpha_scaling")
    if fit:
        a2 = np.linspace(alphas.min() ** 2, alphas.max() ** 2, 50)
        ax2.semilogy(
            a2,
            fit["A"] * np.exp(-fit["gamma"] * a2),
            "k-",
            lw=0.7,
            label=f"fit A e^(-γα²), γ = {fit['gamma']:.2f}",
        )
    ax2.set_xlabel(r"$\alpha^2$")
    ax2.set_ylabel(r"$R = c_2\,\epsilon_Z^2 / \Gamma_{\rm meas}$")
    ax2.set_title("(b) linear-order ratio vs cat size")
    ax2.legend(fontsize=8)
    fig.text(
        0.01,
        0.01,
        r"$\Gamma_X=-\lambda_1$ = decay rate of Re$\langle a\rangle$ "
        r"(sum of both one-way rates); one-way = $\Gamma_X/2$. "
        r"$\Gamma_{\rm meas}=(\kappa_b/2)|\epsilon_Z/\alpha g_2|^2$.",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=DPI)
    print(f"wrote {FIGURE}")


# ---------------------------------------------------------------- R10: readout Stark shift

R10_JSON = ROOT / "results" / "task24" / "stark_r10.json"
R10_FIGURE = ROOT / "figures" / "track2" / "task24_epsz4.png"
R10_ALPHA = 1.4
R10_EPS_FRACTIONS = tuple(float(x) for x in np.geomspace(0.1, 1.0, 6))
R10_D2_ALPHAS = (1.4, 1.6, 2.1)  # d2 only (two gradient solves each)
TASK24A_JSON = ROOT / "results" / "task24a" / "effective_terms.json"
LESCANNE_CHI_MHZ = 0.79  # Lescanne et al. 2020 (arXiv:1907.11729), measured chi_ab / 2 pi
DETUNING_STEP = 1e-3  # rad/us, finite-difference step on the exact gradient


def chi_values() -> dict[str, float]:
    """chi_ab (rad/us, our sign: H contains -chi n_a n_b) from Task 2.4a and the literature."""
    terms = json.loads(TASK24A_JSON.read_text())["terms"]
    return {
        "circuit (Task 2.4a)": -terms["chi_ab"]["value_rad_per_us"],
        "Lescanne 2020 magnitude": 2 * math.pi * LESCANNE_CHI_MHZ,
    }


def buffer_photons(params: CatParams, eps_z: float) -> float:
    """Mean-field <b^dag b> = beta_ss^2 = eps_z^2 / (4 alpha^2 g2^2) during readout."""
    return (eps_z / (2 * params.alpha * params.g2)) ** 2


def stark_terms(params: CatParams, eps_z: float, chi: float, compensate: bool) -> HamiltonianTerms:
    """Cross-Kerr with the static buffer shift -chi alpha^2 removed by the buffer frame; with
    compensate, the memory frame also follows the readout shift +chi <n_b>."""
    delta_a = chi * buffer_photons(params, eps_z) if compensate else 0.0
    return HamiltonianTerms.from_params(
        params, eps_z=eps_z, chi_ab=chi, delta_b=chi * params.alpha**2, delta_a=delta_a
    )


def detuning_curvature(params: CatParams, trunc: FockTruncation) -> float:
    """d2 in Gamma_X(Delta) = Gamma_X(0) + d2 Delta^2, by a central difference of the
    exact adjoint gradient d Gamma_X / d delta_a."""
    grads = []
    for sign in (1, -1):
        terms = HamiltonianTerms.from_params(params, delta_a=sign * DETUNING_STEP)
        res = driven_bit_flip_rate(
            terms, Dissipation.from_params(params), trunc, with_gradient=True
        )
        grads.append(res.gradient["delta_a"])
    return (grads[0] - grads[1]) / (4 * DETUNING_STEP)


def run_r10() -> dict:
    params = CatParams.reglade().with_alpha(R10_ALPHA)
    trunc = BASE_TRUNCATIONS[R10_ALPHA]
    diss = Dissipation.from_params(params)
    d2 = detuning_curvature(params, trunc)
    eps_list = [f * params.g2 for f in R10_EPS_FRACTIONS]
    base = [driven_bit_flip_rate(*model(params, e), trunc).rate for e in eps_list]
    idle0 = driven_bit_flip_rate(*model(params, 0.0), trunc).rate
    out = {
        "alpha": R10_ALPHA,
        "trunc": asdict(trunc),
        "eps_fractions": list(R10_EPS_FRACTIONS),
        "d2": d2,
        "baseline": base,
        "convention": "H contains -chi n_a n_b + chi alpha^2 n_b (+ chi <n_b> n_a if compensated)",
        "cases": {},
    }
    for label, chi in chi_values().items():
        for compensate in (False, True):
            rates = [
                driven_bit_flip_rate(stark_terms(params, e, chi, compensate), diss, trunc).rate
                for e in eps_list
            ]
            idle = driven_bit_flip_rate(stark_terms(params, 0.0, chi, False), diss, trunc).rate
            predicted = [d2 * (chi * buffer_photons(params, e)) ** 2 for e in eps_list]
            key = f"{label} | {'compensated' if compensate else 'uncompensated'}"
            out["cases"][key] = {
                "chi": chi,
                "rates": rates,
                "idle_rate": idle,
                "excess_over_chi0": [r - b for r, b in zip(rates, base, strict=True)],
                "excess_minus_idle_shift": [
                    r - b - (idle - idle0) for r, b in zip(rates, base, strict=True)
                ],
                "predicted_d2_delta2": predicted,
            }
            print(f"  {key}: idle {idle:.4e}, excess {out['cases'][key]['excess_over_chi0']}")
    out["idle_rate_chi0"] = idle0
    out["d2_by_alpha"] = {}
    for alpha in R10_D2_ALPHAS:
        p_alpha = CatParams.reglade().with_alpha(alpha)
        lin = drive_rate_coefficient(*model(p_alpha), BASE_TRUNCATIONS[alpha])
        d2_alpha = detuning_curvature(p_alpha, BASE_TRUNCATIONS[alpha])
        out["d2_by_alpha"][str(alpha)] = {
            "d2": d2_alpha,
            "c2": lin.coefficient,
            # eps_z at which d2 (chi beta^2)^2 equals c2 eps^2, for each chi
            "crossover_eps_over_g2": {
                label: 4 * alpha**2 * p_alpha.g2 * math.sqrt(lin.coefficient / d2_alpha) / chi
                for label, chi in chi_values().items()
            },
        }
        print(f"  alpha={alpha}: d2={d2_alpha:.4e}, c2={lin.coefficient:.4e}")
    R10_JSON.write_text(json.dumps(out, indent=2))
    return out


R10_C2_ALPHAS = (1.4, 1.6, 2.1)


def run_r10_c2() -> dict:
    """Exact c2 with the cross-Kerr present (parity blocks; compensation is O(eps^2) in
    delta_a, so it cannot change c2). Merged into the R10 JSON."""
    out = json.loads(R10_JSON.read_text())
    table = {}
    for alpha in R10_C2_ALPHAS:
        params = CatParams.reglade().with_alpha(alpha)
        trunc, diss = BASE_TRUNCATIONS[alpha], Dissipation.from_params(params)
        c2_0 = drive_rate_coefficient(*model(params), trunc).coefficient
        row = {"c2_chi0": c2_0}
        for label, chi in chi_values().items():
            lin = drive_rate_coefficient(stark_terms(params, 0.0, chi, False), diss, trunc)
            row[label] = {
                "c2": lin.coefficient,
                "relative_increase": lin.coefficient / c2_0 - 1,
                "rate0": lin.rate0,
            }
            print(f"  alpha={alpha} {label}: c2 x {lin.coefficient / c2_0:.4f}")
        table[str(alpha)] = row
    out["c2_with_chi"] = table
    R10_JSON.write_text(json.dumps(out, indent=2))
    return out


def plot_r10(out: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    fracs = np.array(out["eps_fractions"])
    base_extra = np.array(out["baseline"]) - out["idle_rate_chi0"]
    labels = list(chi_values())
    for i, label in enumerate(labels):
        unc = out["cases"][f"{label} | uncompensated"]
        com = out["cases"][f"{label} | compensated"]
        stark = np.array(unc["rates"]) - np.array(com["rates"])
        residual = np.array(com["excess_minus_idle_shift"])
        ax1.loglog(fracs, stark, "o-", color=f"C{i}", label=f"removed by compensation, {label}")
        ax1.loglog(
            fracs,
            unc["predicted_d2_delta2"],
            ":",
            color=f"C{i}",
            label=r"mean-field $d_2(\chi\beta_{ss}^2)^2$, " + label,
        )
        ax1.loglog(fracs, residual, "s--", color=f"C{i}", label=f"left after compensation, {label}")
    ax1.loglog(fracs, base_extra, "k-", lw=0.8, label=r"drive-induced $\Gamma_X$ at $\chi=0$")
    ref = stark[2] * (fracs / fracs[2]) ** 4
    ax1.loglog(fracs, ref, color="gray", lw=0.6, ls="-.", label=r"$\propto\epsilon_Z^4$ guide")
    ax1.set_xlabel(r"$\epsilon_Z/g_2$")
    ax1.set_ylabel(r"$\Delta\Gamma_X$ (rad/µs)")
    ax1.set_title(f"(a) cross-Kerr during readout, α = {out['alpha']} (full Liouvillian)")
    ax1.legend(fontsize=6)
    c2 = out.get("c2_with_chi", {})
    if c2:
        alphas = [float(a) for a in c2]
        for i, label in enumerate(labels):
            ax2.plot(
                alphas,
                [100 * c2[a][label]["relative_increase"] for a in c2],
                "o-",
                color=f"C{i}",
                label=label,
            )
        ax2.set_xlabel(r"$\alpha$")
        ax2.set_ylabel(r"increase of $c_2$ by $\chi_{ab}$ (%)")
        ax2.set_title("(b) non-compensable part: c₂(χ)/c₂(0) − 1")
        ax2.legend(fontsize=7)
    fig.text(
        0.01,
        0.005,
        r"$\Gamma_X=-\lambda_1$ (sum of both one-way rates; one-way = $\Gamma_X/2$)."
        r" $H\ni-\chi n_an_b+\chi\alpha^2 n_b$; compensated: $+\chi\langle n_b\rangle n_a$.",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(R10_FIGURE, dpi=DPI)
    print(f"wrote {R10_FIGURE}")


if __name__ == "__main__":
    dq.set_precision("double")
    if "--r10c2" in sys.argv:
        plot_r10(run_r10_c2())
    elif "--r10" in sys.argv:
        plot_r10(json.loads(R10_JSON.read_text()) if "--plot" in sys.argv else run_r10())
    else:
        plot(json.loads(RESULTS.read_text()) if "--plot" in sys.argv else run())
