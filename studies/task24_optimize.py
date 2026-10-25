"""Task 2.4c/d: optimize the readout drive eps_Z(t) under a bit-flip budget.

Pipeline (all rates rad/us, times us; Gamma_X = -lambda_1 = both one-way rates; one-way = /2):
  1. kernel   - exact second-order memory kernel K(tau) of drive-induced flips
                (catqubit.driven_rates.drive_kernel) and its spectrum c2(omega), checked against
                exact frequency-domain solves; compared with the signal transfer |H_beta(omega)|^2
                of the linearized buffer (catqubit.optimize.LinearReadout).
  2. pareto   - at alpha = 2.1, maximize SNR^2 (Gaussian error 1/2 erfc(SNR / 2 sqrt 2), eta = 1)
                subject to p_X <= B with the augmented Lagrangian (catqubit.optimize), over a
                piecewise-constant eps_Z(t). Cost: p_X = Gamma_X(0) T + x^T M x (exact kernel)
                + quasi-static remainder of the converged rate table
                (results/task24/tradeoff_r9.json).
                Baselines: square drive, zero-vibration two-step drive, and the optimum of a
                quasi-static cost (re-scored with the exact kernel).
  3. validate - Nelder-Mead on a 4-parameter pulse family at one budget; all-orders time-domain
                p_X (sparse expm_multiply of the full driven Liouvillian) for optimized pulses;
                dynamiqs mesolve of <b>(t) for the optimized pulse at alpha = 1.6 vs the
                linear model.
  4. inflate  - same front at alpha = 1.6 (operate small, read large: compare with alpha = 2.1).
Run: PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -u studies/task24_optimize.py
     [--stage kernel|pareto|validate|plot]  (default: all, in order).
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import erfcinv

from catqubit.driven_rates import (
    Dissipation,
    DriveKernel,
    HamiltonianTerms,
    ResponseKernel,
    buffer_response,
    drive_kernel,
    drive_susceptibility,
    pulse_bit_flip_probability,
)
from catqubit.liouvillian import FockTruncation
from catqubit.optimize import (
    HermiteTable,
    LinearReadout,
    ReadoutProblem,
    deadbeat_two_step,
    gaussian_error,
)
from catqubit.params import CatParams

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "task24"
R9_JSON = OUT / "tradeoff_r9.json"
FIGURE = ROOT / "figures" / "track2" / "task24_pareto.png"
DPI = 150

KERNEL_ALPHAS = (1.6, 2.1)
TRUNCATIONS = {
    1.4: FockTruncation(na=20, nb=5, nb_extra=2, na_taper=6.0),
    1.6: FockTruncation(na=23, nb=5, nb_extra=4, na_taper=10.0),
    2.1: FockTruncation(na=31, nb=5, nb_extra=6, na_taper=13.2),
}
KERNEL_SPAN, KERNEL_STEP = 6.0, 1e-3  # us; K(tau) has decayed to ~1e-14 of K(0) by 6 us
OMEGAS = np.linspace(0.0, 80.0, 161)  # rad/us, for the spectrum
EXACT_OMEGAS = (0.0, 15.0, 30.0, 45.0)  # checked by direct sparse solves

DURATION, PIECES = 2.0, 100  # readout window (us) and pulse pieces (20 ns each)
SUBSTEPS = 4
EPS_MAX_FRACTION = 1.0  # |eps_Z| <= g2 (range of the converged rate table)
# Budgets: idle cost Gamma_X(0) T plus these multiples of the second-order flip cost a square
# drive needs for the target error (SNR^2_target / gain_square).
BUDGET_MULTIPLES = tuple(float(b) for b in np.geomspace(0.03, 30.0, 7))
TARGET_ERROR = 1e-3
TARGET_SNR2 = 8 * float(erfcinv(2 * TARGET_ERROR)) ** 2  # 1/2 erfc(SNR / 2 sqrt 2) = target
NM_BUDGET_INDEX = 2
SETTLE = 1.5  # us of idle after a pulse before reading the slow amplitude
SMALL_EPS_FRACTION = 0.01  # extra table knot from the exact c2


# ---------------------------------------------------------------- inputs


def selected_alphas() -> tuple[float, ...]:
    if "--alphas" in sys.argv:
        return tuple(float(a) for a in sys.argv[sys.argv.index("--alphas") + 1].split(","))
    return KERNEL_ALPHAS


def params_at(alpha: float) -> CatParams:
    return CatParams.reglade().with_alpha(alpha)


def model(alpha: float) -> tuple[HamiltonianTerms, Dissipation]:
    p = params_at(alpha)
    return HamiltonianTerms.from_params(p), Dissipation.from_params(p)


def rate_table(alpha: float) -> tuple[HermiteTable, float, float]:
    """Hermite table of Gamma_X(|eps|) from the R9 full-Liouvillian points + exact slopes."""
    data = json.loads(R9_JSON.read_text())["per_alpha"][str(alpha)]
    s = data["summary"]
    rate0, c2, g2 = s["rate0"], s["c2"], params_at(alpha).g2
    eps_small = SMALL_EPS_FRACTION * g2
    knots = [0.0, eps_small] + [p["eps_z"] for p in data["full"]["points"]]
    values = [rate0, rate0 + c2 * eps_small**2] + [p["rate"] for p in data["full"]["points"]]
    slopes = [0.0, 2 * c2 * eps_small] + [p["gradient"]["eps_z"] for p in data["full"]["points"]]
    return HermiteTable.build(np.array(knots), np.array(values), np.array(slopes)), rate0, c2


def kernel_path(alpha: float) -> Path:
    return OUT / f"kernel_alpha{alpha}.npz"


def response_path(alpha: float) -> Path:
    return OUT / f"response_alpha{alpha}.npz"


RESPONSE_SPAN = DURATION + 0.5  # us; must cover the readout window


def load_response(alpha: float) -> ResponseKernel:
    d = np.load(response_path(alpha))
    return ResponseKernel(d["taus"], d["values"])


def load_kernel(alpha: float) -> DriveKernel:
    d = np.load(kernel_path(alpha))
    return DriveKernel(d["taus"], d["values"], float(d["rate0"]))


# ---------------------------------------------------------------- stage 1: kernel


def spectrum(kernel: DriveKernel, omegas: np.ndarray) -> np.ndarray:
    """c2(omega) = -Re int_0^inf K(tau) e^{-i omega tau} d tau (trapezoid on the grid)."""
    phase = np.exp(-1j * np.outer(omegas, kernel.taus))
    return -np.trapezoid(phase * kernel.values[None, :], kernel.taus, axis=1).real


def exact_spectrum(alpha: float, omegas: tuple[float, ...]) -> list[float]:
    """c2(omega) from direct sparse solves (catqubit.driven_rates.drive_susceptibility)."""
    return drive_susceptibility(*model(alpha), TRUNCATIONS[alpha], np.array(omegas)).tolist()


def signal_transfer(alpha: float, omegas: np.ndarray) -> np.ndarray:
    """|beta(omega) / eps(omega)|^2 of the linearized buffer, normalized to omega = 0."""
    readout = LinearReadout.from_params(params_at(alpha))
    a = np.asarray(readout.drift())
    h = [np.linalg.solve(1j * w * np.eye(2) - a, np.array([-1.0, 0.0]))[1] for w in omegas]
    return np.abs(np.array(h)) ** 2 * (2 * alpha * readout.g2) ** 2


def stage_kernel() -> dict:
    out = {}
    taus = np.arange(0.0, KERNEL_SPAN + KERNEL_STEP / 2, KERNEL_STEP)
    for alpha in selected_alphas():
        t0 = time.time()
        kernel = drive_kernel(*model(alpha), TRUNCATIONS[alpha], taus)
        np.savez(kernel_path(alpha), taus=kernel.taus, values=kernel.values, rate0=kernel.rate0)
        resp_taus = np.arange(0.0, RESPONSE_SPAN + KERNEL_STEP / 2, KERNEL_STEP)
        response = buffer_response(*model(alpha), TRUNCATIONS[alpha], resp_taus)
        np.savez(response_path(alpha), taus=response.taus, values=response.values)
        g_w = np.trapezoid(
            np.exp(-1j * np.outer(OMEGAS, response.taus)) * response.values, response.taus, axis=1
        )
        exact_transfer = np.abs(g_w / g_w[0]) ** 2
        c2w = spectrum(kernel, OMEGAS)
        exact = exact_spectrum(alpha, EXACT_OMEGAS)
        from_grid = spectrum(kernel, np.array(EXACT_OMEGAS))
        transfer = signal_transfer(alpha, OMEGAS)
        out[str(alpha)] = {
            "trunc": asdict(TRUNCATIONS[alpha]),
            "rate0": kernel.rate0,
            "c2_from_kernel": float(c2w[0]),
            "kernel_tail_abs": float(abs(kernel.values[-1])),
            "kernel_seconds": time.time() - t0,
            "omegas": OMEGAS.tolist(),
            "c2_omega": c2w.tolist(),
            "signal_transfer_oscillator": transfer.tolist(),
            "signal_transfer": exact_transfer.tolist(),
            "signal_dc_response": float(g_w[0].real),
            "signal_imag_max": float(np.max(np.abs(response.values.imag))),
            "info_per_flip_relative": (exact_transfer / (c2w / c2w[0])).tolist(),
            "exact_check": {
                "omegas": list(EXACT_OMEGAS),
                "exact": exact,
                "from_kernel": from_grid.tolist(),
            },
        }
        print(f"alpha={alpha}: c2(0)={c2w[0]:.4e}, exact checks {exact} vs {from_grid}")
    path = OUT / "optimize_kernel.json"
    merged = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps({**merged, **out}, indent=2))
    return out


# ---------------------------------------------------------------- stage 2: fronts


def problem(alpha: float, quasi_static: bool = False) -> ReadoutProblem:
    """Exact-kernel problem (signal and flips); quasi_static swaps only the flip cost."""
    table, _, c2 = rate_table(alpha)
    h = DURATION / PIECES
    form = None if quasi_static else load_kernel(alpha).quadratic_form(PIECES, h)
    signal = load_response(alpha).signal_form(PIECES, h, params_at(alpha).kappa_b)
    return ReadoutProblem(
        model=LinearReadout.from_params(params_at(alpha)),
        rate_table=table,
        duration=DURATION,
        pieces=PIECES,
        substeps=SUBSTEPS,
        eps_max=EPS_MAX_FRACTION * params_at(alpha).g2,
        flip_form=form,
        c2=c2,
        signal_form=signal,
    )


def piece_times() -> np.ndarray:
    return (np.arange(PIECES) + 0.5) * DURATION / PIECES


def zv_shape(alpha: float) -> np.ndarray:
    shape = deadbeat_two_step(LinearReadout.from_params(params_at(alpha)), 1.0)
    return np.where(piece_times() < shape.switch_time, shape.first_amplitude, 1.0)


def scaled_family(prob: ReadoutProblem, shape: np.ndarray, label: str) -> list[dict]:
    """Amplitude sweep of a fixed shape: (p_X, SNR^2, error) with the exact-kernel cost."""
    top = prob.bounds()[0][1] / np.max(np.abs(shape))
    rows = []
    for s in np.geomspace(0.01, 1.0, 25) * top:
        x = s * shape
        snr2 = prob.snr_squared(x)
        rows.append(
            {
                "label": label,
                "scale": float(s),
                "p_x": prob.bit_flip_probability(x)[0],
                "snr2": snr2,
                "error": float(gaussian_error(jnp.asarray(snr2))),
            }
        )
    return rows


START_HARMONICS = (1, 2, 3)  # AC starts at k * omega_0, omega_0 = 2 alpha g2
BISECTION_STEPS = 60


def smooth_envelope() -> np.ndarray:
    t = piece_times()
    ramp = np.clip(np.minimum(t, DURATION - t) / (0.15 * DURATION), 0.0, 1.0)
    return np.sin(0.5 * np.pi * ramp) ** 2


def start_shapes(alpha: float) -> dict[str, np.ndarray]:
    omega0 = 2 * alpha * params_at(alpha).g2
    shapes = {"zero-vibration DC": zv_shape(alpha)}
    for k in START_HARMONICS:
        shapes[f"AC {k} omega_0"] = smooth_envelope() * np.cos(k * omega0 * piece_times())
    return shapes


def scale_to_budget(prob: ReadoutProblem, shape: np.ndarray, budget: float) -> np.ndarray:
    """Largest multiple of shape (within the amplitude bound) with p_X <= budget."""
    top = prob.bounds()[0][1] / np.max(np.abs(shape))
    if prob.bit_flip_probability(top * shape)[0] <= budget:
        return top * shape
    lo, hi = 0.0, top
    for _ in range(BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if prob.bit_flip_probability(mid * shape)[0] <= budget else (lo, mid)
    return lo * shape


def budgets(prob: ReadoutProblem, rate0: float) -> list[float]:
    needed = TARGET_SNR2 / _gain(prob, np.ones(PIECES))
    return [rate0 * DURATION + m * needed for m in BUDGET_MULTIPLES]


def optimized_front(
    prob: ReadoutProblem, rate0: float, rescore: ReadoutProblem | None, grid: list[float]
) -> list[dict]:
    rows = []
    for budget in grid:
        t0 = time.time()
        starts = {
            name: scale_to_budget(prob, shape, budget)
            for name, shape in start_shapes(prob.model.alpha).items()
        }
        solved = {name: prob.solve(budget, x0) for name, x0 in starts.items()}
        feasible = {
            k: v for k, v in solved.items() if v.bit_flip_probability <= budget * (1 + 1e-3)
        } or solved
        best_start = max(feasible, key=lambda k: feasible[k].snr_squared)
        pt = feasible[best_start]
        row = {
            "budget": budget,
            "pulse": pt.pulse.tolist(),
            "p_x_model": pt.bit_flip_probability,
            "snr2": pt.snr_squared,
            "error": pt.error,
            "multiplier": pt.multiplier,
            "converged": pt.converged,
            "best_start": best_start,
            "snr2_by_start": {k: v.snr_squared for k, v in solved.items()},
            "seconds": time.time() - t0,
        }
        if rescore is not None:
            row["p_x_exact_kernel"] = rescore.bit_flip_probability(pt.pulse)[0]
        rows.append(row)
        print(
            f"  B={budget:.2e}: SNR2={pt.snr_squared:.3f} err={pt.error:.3e} "
            f"start={best_start} conv={pt.converged} ({time.time() - t0:.0f} s)"
        )
    return rows


def p_x_at_error(rows: list[dict], key: str = "p_x") -> float | None:
    """Interpolate log p_X at the target error along a front (None if not reached)."""
    pts = sorted((r[key], r["error"]) for r in rows)
    for (p0, e0), (p1, e1) in zip(pts, pts[1:], strict=False):
        if e0 >= TARGET_ERROR >= e1 and e0 > e1:
            w = (np.log(TARGET_ERROR) - np.log(e0)) / (np.log(e1) - np.log(e0))
            return float(np.exp(np.log(p0) + w * (np.log(p1) - np.log(p0))))
    return None


def stage_pareto() -> dict:
    path = OUT / "optimize_pareto.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    out.update({"duration": DURATION, "pieces": PIECES, "budget_multiples": list(BUDGET_MULTIPLES)})
    for alpha in selected_alphas():
        print(f"alpha={alpha}")
        prob = problem(alpha)
        quasi = problem(alpha, quasi_static=True)
        _, rate0, c2 = rate_table(alpha)
        res = {
            "rate0": rate0,
            "c2": c2,
            "square": scaled_family(prob, np.ones(PIECES), "square"),
            "zero_vibration": scaled_family(prob, zv_shape(alpha), "zero-vibration two-step"),
            "budgets": budgets(prob, rate0),
            "optimized": optimized_front(prob, rate0, None, budgets(prob, rate0)),
            "quasi_static_optimum": optimized_front(quasi, rate0, prob, budgets(prob, rate0)),
            # SNR^2 per unit second-order flip probability (amplitude independent)
            "linear_gain": {
                "square": _gain(prob, np.ones(PIECES)),
                **{k: _gain(prob, v) for k, v in start_shapes(alpha).items()},
            },
        }
        res["p_x_at_target"] = {
            "square": p_x_at_error(res["square"]),
            "zero_vibration": p_x_at_error(res["zero_vibration"]),
            "optimized": p_x_at_error(res["optimized"], "p_x_model"),
            "quasi_static_optimum_rescored": p_x_at_error(
                res["quasi_static_optimum"], "p_x_exact_kernel"
            ),
        }
        print("  p_X at error 1e-3:", res["p_x_at_target"])
        out[str(alpha)] = res
        path.write_text(json.dumps(out, indent=2))
    return out


def _gain(prob: ReadoutProblem, x: np.ndarray) -> float:
    return prob.snr_squared(x) / float(x @ prob.flip_form @ x)


# ---------------------------------------------------------------- stage 3: validation


def family(theta: np.ndarray, g2: float) -> np.ndarray:
    """Smooth (sin^2-edged) envelope times [dc + ac cos(omega t + phase)]."""
    dc, ac, omega, phase = theta
    t = piece_times()
    edge = 0.15 * DURATION
    env = np.clip(np.minimum(t, DURATION - t) / edge, 0.0, 1.0)
    env = np.sin(0.5 * np.pi * env) ** 2
    return g2 * env * (dc + ac * np.cos(omega * t + phase))


def nelder_mead_point(alpha: float, budget: float) -> dict:
    """Gradient-free baseline: maximize SNR^2 over the 4-parameter family with a penalty."""
    prob, g2 = problem(alpha), params_at(alpha).g2

    def cost(theta: np.ndarray) -> float:
        x = np.clip(family(theta, g2), -g2, g2)
        excess = max(prob.bit_flip_probability(x)[0] / budget - 1, 0.0)
        return -prob.snr_squared(x) + 1e4 * excess**2

    best = None
    for omega0 in (0.0, 2 * alpha * g2, 3 * alpha * g2):
        res = minimize(
            cost,
            np.array([0.2, 0.2, omega0, 0.0]),
            method="Nelder-Mead",
            options={"maxiter": 4000, "xatol": 1e-6, "fatol": 1e-9},
        )
        if best is None or res.fun < best.fun:
            best = res
    x = np.clip(family(best.x, g2), -g2, g2)
    return {
        "theta": best.x.tolist(),
        "snr2": prob.snr_squared(x),
        "p_x": prob.bit_flip_probability(x)[0],
        "budget": budget,
    }


def time_domain_flips(alpha: float, pulse: np.ndarray) -> dict:
    terms, diss = model(alpha)
    t0 = time.time()
    res = pulse_bit_flip_probability(
        terms, diss, TRUNCATIONS[alpha], pulse, DURATION / PIECES, SETTLE
    )
    return {
        "extra": res.extra,
        "total_with_idle": res.total,
        "rate0": res.rate0,
        "seconds": time.time() - t0,
    }


def mesolve_signal(alpha: float, pulse: np.ndarray) -> dict:
    """Delta<b>(t) from dynamiqs (full nonlinear model, rectangular truncation), starting in
    the settled undriven wells, vs the exact linear-response kernel and the 2x2 oscillator."""
    p = params_at(alpha)
    na, nb = 23, 9  # alpha = 1.6: matches the tapered (23, 5, 4, 10) to 1e-3 at eps = 0
    a, b = dq.destroy(na, nb)
    h_static = p.g2 * (a @ a @ b.dag() + a.dag() @ a.dag() @ b) - p.eps_d * (b + b.dag())
    edges = jnp.linspace(0.0, DURATION, PIECES + 1)
    h_drive = dq.pwc(edges, jnp.asarray(pulse), a + a.dag())
    jumps = [jnp.sqrt(p.kappa_b) * b, jnp.sqrt(p.kappa_a) * a]
    method = dq.method.Tsit5(rtol=1e-9, atol=1e-9, max_steps=10**7)
    codewords = dq.stack([dq.tensor(dq.coherent(na, s * p.alpha), dq.fock(nb, 0)) for s in (1, -1)])
    settled = dq.mesolve(
        h_static, jumps, codewords, jnp.array([0.0, SETTLE]), method=method, progress_meter=False
    ).states[:, -1]
    tsave = jnp.linspace(0.0, DURATION, 4 * PIECES + 1)
    res = dq.mesolve(
        h_static + h_drive,
        jumps,
        settled,
        tsave,
        exp_ops=[b],
        method=method,
        save_states=False,
        progress_meter=False,
    )
    b_t = np.asarray(res.expects[:, 0, :])
    delta = (b_t[0] - b_t[1]).real
    delta = delta - delta[0]  # undriven offset (constant on this time scale)
    t = np.asarray(tsave)
    kernel = (load_response(alpha).response_matrix(PIECES, DURATION / PIECES, t) @ pulse).real
    readout = LinearReadout.from_params(p)
    oscillator = 2 * np.asarray(readout.buffer_amplitude(jnp.asarray(pulse), DURATION, 2))

    def snr2(signal: np.ndarray) -> float:
        return float(4 * p.kappa_b * np.trapezoid(signal**2, t))

    return {
        "t": t.tolist(),
        "delta_b_full": delta.tolist(),
        "delta_b_kernel": kernel.tolist(),
        "delta_b_oscillator": oscillator.tolist(),
        "snr2_full": snr2(delta),
        "snr2_kernel": snr2(kernel),
        "snr2_oscillator": snr2(oscillator),
    }


def stage_validate() -> dict:
    fronts = json.loads((OUT / "optimize_pareto.json").read_text())
    path = OUT / "optimize_validate.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for alpha in selected_alphas():
        res = fronts[str(alpha)]
        budget = res["budgets"][NM_BUDGET_INDEX]
        opt = res["optimized"][NM_BUDGET_INDEX]
        pulse = np.array(opt["pulse"])
        nm = nelder_mead_point(alpha, budget)
        print(f"alpha={alpha} NM: SNR2={nm['snr2']:.3f} vs AL {opt['snr2']:.3f}")
        prob, quasi = problem(alpha), problem(alpha, quasi_static=True)
        _, rate0, _ = rate_table(alpha)
        checks = []
        qs = np.array(res["quasi_static_optimum"][NM_BUDGET_INDEX]["pulse"])
        sq = res["square"][-8]
        square = np.full(PIECES, sq["scale"])
        for label, x in (("optimized", pulse), ("quasi_static_optimum", qs), ("square", square)):
            td = time_domain_flips(alpha, x)
            checks.append(
                {
                    "label": label,
                    "time_domain_extra": td["extra"],
                    "kernel_plus_remainder_extra": prob.bit_flip_probability(x)[0]
                    - rate0 * DURATION,
                    "quasi_static_extra": quasi.bit_flip_probability(x)[0] - rate0 * DURATION,
                    "seconds": td["seconds"],
                }
            )
            print(
                f"  {label}: time-domain extra {td['extra']:.4e}, "
                f"kernel {checks[-1]['kernel_plus_remainder_extra']:.4e}, "
                f"quasi-static {checks[-1]['quasi_static_extra']:.4e}"
            )
        out[str(alpha)] = {
            "nelder_mead": nm,
            "al_snr2": opt["snr2"],
            "al_p_x": opt["p_x_model"],
            "flip_checks": checks,
        }
        if alpha == 1.6:
            out[str(alpha)]["mesolve_signal"] = mesolve_signal(alpha, pulse)
            m = out[str(alpha)]["mesolve_signal"]
            print(
                f"  mesolve SNR2 full {m['snr2_full']:.3f}, exact kernel {m['snr2_kernel']:.3f}, "
                f"oscillator {m['snr2_oscillator']:.3f}"
            )
        path.write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------- figure


def plot() -> None:
    kern = json.loads((OUT / "optimize_kernel.json").read_text())
    fronts = json.loads((OUT / "optimize_pareto.json").read_text())
    val_path = OUT / "optimize_validate.json"
    val = json.loads(val_path.read_text()) if val_path.exists() else {}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    ax = axes[0, 0]
    for alpha, style in (("1.6", "--"), ("2.1", "-")):
        k = kern[alpha]
        w = np.array(k["omegas"])
        c2w = np.array(k["c2_omega"])
        ax.semilogy(
            w,
            np.clip(c2w / c2w[0], 1e-6, None),
            "C0" + style,
            label=f"flips c₂(ω)/c₂(0), α={alpha}",
        )
        ax.semilogy(
            w, k["signal_transfer"], "C1" + style, label=f"signal |H_β(ω)|²/|H_β(0)|², α={alpha}"
        )
        ax.semilogy(
            w,
            np.clip(k["info_per_flip_relative"], 1e-6, 1e6),
            "C2" + style,
            label=f"info per flip (rel.), α={alpha}",
        )
        ex = k["exact_check"]
        ax.plot(ex["omegas"], np.array(ex["exact"]) / ex["exact"][0], "kx")
    ax.set_ylim(1e-4, 1e3)
    ax.set_xlabel(r"drive frequency offset $\omega$ (rad/µs)")
    ax.set_ylabel("relative to ω = 0")
    ax.set_title("(a) flip susceptibility vs signal transfer (×: exact solves)")
    ax.legend(fontsize=6.5)
    ax = axes[0, 1]
    f = fronts["2.1"]
    for key, style, label in (
        ("square", "C0-", "square drive"),
        ("zero_vibration", "C1-", "zero-vibration two-step"),
    ):
        rows = f[key]
        ax.loglog([r["p_x"] for r in rows], [r["error"] for r in rows], style, label=label)
    ax.loglog(
        [r["p_x_model"] for r in f["optimized"]],
        [r["error"] for r in f["optimized"]],
        "C2o-",
        label="optimized (exact kernel, AL)",
    )
    ax.loglog(
        [r["p_x_exact_kernel"] for r in f["quasi_static_optimum"]],
        [r["error"] for r in f["quasi_static_optimum"]],
        "C3s--",
        label="quasi-static optimum, re-scored",
    )
    if "2.1" in val:
        for c in val["2.1"]["flip_checks"]:
            if c["label"] == "optimized":
                opt = f["optimized"][NM_BUDGET_INDEX]
                ax.loglog(
                    c["time_domain_extra"] + f["rate0"] * DURATION,
                    opt["error"],
                    "k*",
                    ms=11,
                    label="time-domain check (all orders)",
                )
        nm = val["2.1"]["nelder_mead"]
        ax.loglog(
            nm["p_x"],
            float(gaussian_error(jnp.asarray(nm["snr2"]))),
            "m^",
            ms=9,
            label="Nelder–Mead (4-param family)",
        )
    ax.axhline(TARGET_ERROR, color="gray", lw=0.6, ls=":")
    ax.set_xlabel(r"bit-flip probability per readout $p_X=\int\Gamma_X dt$ (one-way = $p_X/2$)")
    ax.set_ylabel("Gaussian readout error (η = 1)")
    ax.set_title(f"(b) Pareto front, α = 2.1, T = {DURATION} µs")
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    t = piece_times()
    g2 = params_at(2.1).g2
    for idx, color in ((1, "C0"), (NM_BUDGET_INDEX, "C2"), (len(BUDGET_MULTIPLES) - 1, "C3")):
        r = f["optimized"][idx]
        ax.step(
            t, np.array(r["pulse"]) / g2, where="mid", color=color, label=f"B = {r['budget']:.1e}"
        )
    ax.set_xlabel("t (µs)")
    ax.set_ylabel(r"$\epsilon_Z(t)/g_2$")
    ax.set_title("(c) optimized drives, α = 2.1")
    ax.legend(fontsize=7)
    ax = axes[1, 1]
    if "1.6" in val and "mesolve_signal" in val["1.6"]:
        m = val["1.6"]["mesolve_signal"]
        ax.plot(m["t"], m["delta_b_full"], "k-", lw=1.2, label="dynamiqs mesolve (full model)")
        ax.plot(m["t"], m["delta_b_kernel"], "C2--", lw=1, label="exact linear-response kernel")
        ax.plot(m["t"], m["delta_b_oscillator"], "C1:", lw=1, label="2x2 oscillator model")
        ax.set_title(
            f"(d) ⟨b⟩₊−⟨b⟩₋, optimized drive, α = 1.6; "
            f"SNR² {m['snr2_full']:.2f} / {m['snr2_kernel']:.2f} / {m['snr2_oscillator']:.2f}"
        )
    ax.set_xlabel("t (µs)")
    ax.set_ylabel(r"$\Delta\langle b\rangle$")
    ax.legend(fontsize=7)
    fig.text(
        0.01,
        0.005,
        r"$\Gamma_X=-\lambda_1$ (decay rate of Re$\langle a\rangle$, sum of both one-way rates); "
        r"one-way flip probability = $p_X/2$. Homodyne, η = 1, Gaussian error ½erfc(SNR/2√2).",
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIGURE, dpi=DPI)
    print(f"wrote {FIGURE}")


STAGES = {"kernel": stage_kernel, "pareto": stage_pareto, "validate": stage_validate, "plot": plot}

if __name__ == "__main__":
    dq.set_precision("double")
    OUT.mkdir(parents=True, exist_ok=True)
    chosen = [sys.argv[sys.argv.index("--stage") + 1]] if "--stage" in sys.argv else list(STAGES)
    for name in chosen:
        print(f"== stage {name}")
        STAGES[name]()
