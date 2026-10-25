"""Task 2.4(c-d) follow-up: readouts that change the flips-per-information price, with real Gamma_X.

The eps_Z readout pays e^{-2 alpha^2} flips per unit of information because the meter is the
stabilizing channel L = a^2 - alpha^2 itself (Dubovitskii, PRA 111, 012617). Two ways out:

  inflate-then-read  raise alpha only while reading (control-level): the price falls ~e^{-2 alpha_r^2}.
  conversion readout g_cr (a c^dag + h.c.) into a separate lossy readout mode c (hardware-level).
                     Eliminating c gives a jump sqrt(kappa_ro) a with kappa_ro = 4 g_cr^2 / kappa_c.
                     a|+-alpha> = +-alpha|+-alpha>, so the jump reads the codeword without leaving
                     the code space: Gamma_meas = 2 kappa_ro alpha^2, and the flips it adds are
                     single-photon-loss flips, ~kappa_ro alpha^2 e^{-4 alpha^2} + O(kappa_ro^2).
                     Do NOT convert into the buffer: a^2 + (g_cr/g2) a - alpha^2 still has coherent
                     dark states, so <b> = 0 for both codewords.

Stages (each writes results/task24d/<stage>.json):
  single      single-mode c2 against Dubovitskii Eq. 52 (validates code and factor-2 convention)
  large       two-mode c2 at alpha = 2.3, 2.5, 2.7 (truncation ladders)
  conversion  two-mode Liouvillian gap with an extra sqrt(kappa_ro) a jump
  threemode   explicit memory + buffer + readout-mode Liouvillian at alpha = 1.4 (checks elimination)
  mc          Monte Carlo of the 2.4(c) HMM + SPRT + erasure protocol with the computed rates
  plot        figures/track2/task24d_*.png and results/task24d/summary.md

Rates follow catqubit.driven_rates: Gamma_X is the decay rate of Re<a> (each one-way jump at Gamma/2).
Flips per unit measurement rate R = (drive-induced Gamma_X) / Gamma_meas with Gamma_meas = mu^2 / 2
for eta = 1, i.e. kappa_b |b_ss|^2 * 2 for eps_Z and 2 kappa_ro alpha^2 for conversion.

Run:  .venv313/bin/python -u studies/task24d_readout_upgrades.py [stage ...]   (all: ~1.5 h)
"""

import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as sla  # noqa: E402
from scipy.special import shichi  # noqa: E402

from catqubit.driven_rates import Dissipation, HamiltonianTerms, drive_rate_coefficient  # noqa: E402
from catqubit.liouvillian import FockTruncation, lindblad_superoperator  # noqa: E402
from catqubit.params import CatParams  # noqa: E402
from catqubit.readout_protocol import RecordModel  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "task24d"
FIGURES = ROOT / "figures" / "track2"
DPI = 150

# Two-mode c2 from 2.4(b) (results/task24, converged < 0.1 %), plus rate0 at alpha = 2.1.
C2_TASK24B = {1.2: 0.045, 1.4: 0.016, 1.6: 4.7e-3, 1.8: 1.1e-3, 2.1: 9.909e-5}
RATE0_21 = 3.85e-7
LARGE_LADDER = {  # extrapolated from studies/task24_tradeoff.py BASE_TRUNCATIONS
    2.3: [FockTruncation(34, 5, 7, 14.5), FockTruncation(38, 5, 7, 14.5), FockTruncation(34, 6, 7, 14.5)],
    2.5: [FockTruncation(37, 5, 8, 16.0), FockTruncation(41, 5, 8, 16.0), FockTruncation(37, 6, 8, 16.0)],
    2.7: [FockTruncation(41, 5, 9, 17.5), FockTruncation(45, 5, 9, 17.5)],
}
CONVERSION_TRUNC = {1.4: FockTruncation(20, 5, 2, 6), 1.8: FockTruncation(26, 5, 5, 11.5),
                    2.1: FockTruncation(31, 5, 6, 13.2)}
KAPPA_RO_OVER_K2 = (0.01, 0.03, 0.1, 0.3)
THREE_MODE = dict(alpha=1.4, na=20, nb=5, runs=((0.03, 2), (0.03, 3), (0.1, 2)))  # (ratio, nc); nc=3 ~15 min
MC_ETA = 0.6
MC_EPS_Z_OVER_G2 = 0.3
MC_ALPHA, MC_ALPHA_INFLATED = 2.1, 2.5


def kappa2(p: CatParams) -> float:
    return 4 * p.g2**2 / p.kappa_b


def save(stage: str, data) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{stage}.json").write_text(json.dumps(data, indent=1, default=float))


def load(stage: str):
    return json.loads((RESULTS / f"{stage}.json").read_text())


# --------------------------------------------------------------------- Liouvillian gaps


def slowest_rate(liouvillian: sp.spmatrix, guess: float = 1e-9) -> float:
    """-Re of the slowest nonzero eigenvalue (the bit-flip rate Gamma_X in our convention).

    Shift-invert near -guess; a guess close to the answer keeps ARPACK fast on large Liouvillians.
    """
    w = sla.eigs(liouvillian.tocsc(), k=2, sigma=-guess, which="LM", return_eigenvectors=False, tol=1e-8)
    return -sorted(w, key=abs)[1].real


def destroy(n: int) -> sp.csr_matrix:
    return sp.diags(np.sqrt(np.arange(1, n)), 1, format="csr", dtype=complex)


def eq52_abs_lambda(alpha: float, k2: float) -> float:
    """|lambda| per eps_Z^2 from Dubovitskii Eq. 52 (his one-way rate is |lambda| / 2)."""
    z = alpha**2
    shi4, chi4 = shichi(4 * z)
    shi2, _ = shichi(2 * z)
    return abs(2 * (shi4 - chi4 - 2 * shi2) / (k2 * np.sinh(2 * z) ** 2))


def stage_single() -> None:
    rows = []
    for alpha, c2_two in C2_TASK24B.items():
        p = CatParams.reglade().with_alpha(alpha)
        k2 = kappa2(p)
        n = int(np.ceil(alpha**2 + 6 * alpha + 12))
        a, eye = destroy(n), sp.identity(n, format="csr", dtype=complex)
        jumps = [np.sqrt(k2) * (a @ a - alpha**2 * eye), np.sqrt(p.kappa_a) * a]
        eps = np.array([0.0, 0.05, 0.1]) * p.g2
        gaps = [slowest_rate(lindblad_superoperator(e * (a + a.T), jumps)) for e in eps]
        c2_single = float(np.polyfit(eps**2, gaps, 1)[0])
        rows.append(dict(alpha=alpha, n=n, c2_single=c2_single, c2_eq52=eq52_abs_lambda(alpha, k2),
                         c2_two_mode=c2_two, two_over_single=c2_two / c2_single))
        print(rows[-1], flush=True)
    save("single", rows)


def stage_large() -> None:
    rows = []
    for alpha, truncs in LARGE_LADDER.items():
        p = CatParams.reglade().with_alpha(alpha)
        for tr in truncs:
            r = drive_rate_coefficient(HamiltonianTerms.from_params(p), Dissipation.from_params(p), tr)
            rows.append(dict(alpha=alpha, trunc=asdict(tr), rate0=r.rate0, c2=r.coefficient,
                             R=r.coefficient * 2 * alpha**2 * p.g2**2 / p.kappa_b, residual=r.residual))
            print(rows[-1], flush=True)
            save("large", rows)


def stage_conversion() -> None:
    rows = []
    for alpha, tr in CONVERSION_TRUNC.items():
        p = CatParams.reglade().with_alpha(alpha)
        terms, d0 = HamiltonianTerms.from_params(p), Dissipation.from_params(p)
        base = drive_rate_coefficient(terms, d0, tr)
        r_eps_z = base.coefficient * 2 * alpha**2 * p.g2**2 / p.kappa_b
        for ratio in KAPPA_RO_OVER_K2:
            k_ro = ratio * kappa2(p)
            r = drive_rate_coefficient(terms, replace(d0, kappa_a=d0.kappa_a + k_ro), tr)
            meas = 2 * k_ro * alpha**2
            rows.append(dict(alpha=alpha, k_ro_over_k2=ratio, k_ro=k_ro, rate0_base=base.rate0,
                             rate0=r.rate0, R_conv=(r.rate0 - base.rate0) / meas, R_epsZ=r_eps_z,
                             meas_rate=meas))
            print(rows[-1], flush=True)
            save("conversion", rows)


def three_mode_rate(alpha: float, na: int, nb: int, nc: int, ratio: float) -> float:
    p = CatParams.reglade().with_alpha(alpha)
    ia, ib = (sp.identity(n, format="csr", dtype=complex) for n in (na, nb))
    ic = sp.identity(nc, format="csr", dtype=complex)
    a = sp.kron(sp.kron(destroy(na), ib), ic)
    b = sp.kron(sp.kron(ia, destroy(nb)), ic)
    c = sp.kron(sp.kron(ia, ib), destroy(nc))
    ad, bd, cd = a.conj().T, b.conj().T, c.conj().T
    h = p.g2 * (a @ a @ bd + ad @ ad @ b) - p.g2 * alpha**2 * (b + bd)
    jumps = [np.sqrt(p.kappa_b) * b, np.sqrt(p.kappa_a) * a]
    if ratio > 0:
        kappa_c = p.kappa_b  # readout mode as lossy as the buffer
        g_cr = math.sqrt(ratio * kappa2(p) * kappa_c / 4)
        h = h + g_cr * (a @ cd + ad @ c)
        jumps.append(np.sqrt(kappa_c) * c)
    guess = 1.2e-4 + 3e-3 * ratio * kappa2(p) * alpha**2  # rough two-mode estimate
    return slowest_rate(lindblad_superoperator(h.tocsr(), jumps), guess)


def stage_threemode() -> None:
    cfg = THREE_MODE
    p = CatParams.reglade().with_alpha(cfg["alpha"])
    base = three_mode_rate(cfg["alpha"], cfg["na"], cfg["nb"], 1, 0.0)
    rows = []
    for ratio, nc in cfg["runs"]:
        rate = three_mode_rate(cfg["alpha"], cfg["na"], cfg["nb"], nc, ratio)
        meas = 2 * ratio * kappa2(p) * cfg["alpha"] ** 2
        rows.append(dict(alpha=cfg["alpha"], k_ro_over_k2=ratio, nc=nc, rate0_base=base,
                         rate0=rate, R_three_mode=(rate - base) / meas, meas_rate=meas))
        print(rows[-1], flush=True)
        save("threemode", rows)


# --------------------------------------------------------------------- Monte Carlo


def mc_configs():
    from task24c_protocol import SNR_PER_STEP, Config  # heavy import, only for this stage

    def config(label, alpha, mu2, gamma, eps_ratio):
        model = RecordModel(mu=-math.sqrt(mu2), flip_rate=gamma, dt=SNR_PER_STEP / mu2)
        return Config(label, alpha, eps_ratio, MC_ETA, gamma, "computed (Liouvillian gap)", model)

    p = CatParams.reglade()
    b_ss = MC_EPS_Z_OVER_G2 / (2 * MC_ALPHA)  # buffer response, same contrast for both eps_Z runs
    mu2_eps_z = MC_ETA * 4 * p.kappa_b * b_ss**2
    large = {r["alpha"]: r for r in load("large")}
    configs = {"eps_z": config(f"eps_Z readout, alpha={MC_ALPHA}", MC_ALPHA, mu2_eps_z,
                               RATE0_21 + C2_TASK24B[MC_ALPHA] * (MC_EPS_Z_OVER_G2 * p.g2) ** 2,
                               MC_EPS_Z_OVER_G2)}
    eps_r = b_ss * 2 * MC_ALPHA_INFLATED * p.g2
    lr = large[MC_ALPHA_INFLATED]
    configs["inflate"] = config(f"inflate-then-read, alpha_r={MC_ALPHA_INFLATED}", MC_ALPHA_INFLATED,
                                mu2_eps_z, lr["rate0"] + lr["c2"] * eps_r**2, eps_r / p.g2)
    for row in load("conversion"):
        if row["alpha"] == MC_ALPHA and row["k_ro_over_k2"] in (0.03, 0.1):
            gamma = row["rate0"]  # includes Gamma_X(0) and the conversion-induced flips
            configs[f"conv_{row['k_ro_over_k2']}"] = config(
                f"conversion, kappa_ro={row['k_ro_over_k2']} kappa2", MC_ALPHA,
                MC_ETA * 2 * row["meas_rate"], gamma, 0.0)
    return configs


def stage_mc(only: str | None = None) -> None:
    from task24c_protocol import analyse

    out = load("mc") if (RESULTS / "mc.json").exists() else {}
    for key, cfg in mc_configs().items():
        if only and key != only:
            continue
        res = analyse(cfg, None)
        out[key] = {k: res[k] for k in ("label", "alpha", "gamma_x_rad_per_us",
                                        "measurement_rate_mu2_per_us", "gamma_over_rate",
                                        "operating_point", "fixed_1e-3_point", "dt_us", "t_max_us")}
        save("mc", out)


# --------------------------------------------------------------------- figures and summary


def plot_price() -> None:
    single, conv, large = load("single"), load("conversion"), load("large")
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    p = CatParams.reglade()
    r_two = {a: c * 2 * a**2 * p.g2**2 / p.kappa_b for a, c in C2_TASK24B.items()}
    for r in large:
        r_two.setdefault(r["alpha"], r["R"])
    alphas = sorted(r_two)
    ax.semilogy(alphas, [r_two[a] for a in alphas], "o-", color="k", label="ε_Z readout, two-mode")
    ax.semilogy([r["alpha"] for r in single],
                [r["c2_single"] * 2 * r["alpha"] ** 2 * p.g2**2 / p.kappa_b for r in single],
                "s--", color="gray", label="ε_Z readout, single-mode (= Eq. 52)")
    for ratio, color in zip(KAPPA_RO_OVER_K2, ("tab:green", "tab:blue", "tab:purple", "tab:red"), strict=True):
        rows = [r for r in conv if r["k_ro_over_k2"] == ratio]
        ax.semilogy([r["alpha"] for r in rows], [r["R_conv"] for r in rows], "^-", color=color,
                    label=f"conversion, κ_ro = {ratio} κ₂")
    if (RESULTS / "threemode.json").exists():
        tm = load("threemode")
        ax.semilogy([r["alpha"] for r in tm], [r["R_three_mode"] for r in tm], "x", color="k",
                    ms=9, label="explicit readout mode (3-mode)")
    ax.set(xlabel="α (readout)", ylabel="drive-induced flips per unit measurement rate",
           title="Price of information: ε_Z vs conversion readout")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "task24d_price.png", dpi=DPI)


def mc_rows() -> list[tuple[str, dict, dict]]:
    out = []
    for _key, d in load("mc").items():
        for point in ("fixed_1e-3_point", "operating_point"):
            if d.get(point):
                out.append((d["label"], d, d[point]))
    return out


def write_summary() -> None:
    lines = ["# Task 2.4(c-d) follow-up: readouts with computed flip rates", "",
             f"eta = {MC_ETA}, HMM + SPRT + freeze + erasure protocol of studies/task24c_protocol.py.",
             "Gamma_X from Liouvillian gaps (no illustrative rates).", "",
             "| Readout | Gamma_X (rad/us) | mu^2 (1/us) | Gamma/mu^2 | target | error | erasure | mean T (us) | 99% T (us) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for label, d, pt in mc_rows():
        e = pt["erasure"]
        lines.append(f"| {label} | {d['gamma_x_rad_per_us']:.2e} | {d['measurement_rate_mu2_per_us']:.3f} | "
                     f"{d['gamma_over_rate']:.1e} | {pt['target']:.1e} | {e['p_error']:.1e} | "
                     f"{e['p_erasure']:.1e} | {e['mean_time_us']:.2f} | {e['time_quantiles_us']['0.99']:.2f} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


STAGES = {"single": stage_single, "large": stage_large, "conversion": stage_conversion,
          "threemode": stage_threemode, "mc": stage_mc,
          "plot": lambda: (plot_price(), write_summary())}


def main() -> None:
    stages = sys.argv[1:] or list(STAGES)
    for name in stages:
        t0 = time.time()
        STAGES[name]()
        print(f"stage {name}: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
