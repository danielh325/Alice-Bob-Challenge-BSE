"""Task 2.4c: adaptive readout protocol (boxcar, matched filter, HMM, HMM+SPRT, HMM+SPRT+erasure).

Regenerates results/task24c/*.json, results/task24c/summary.md and figures/track2/task24c_*.png.

Record surrogate: catqubit.readout_protocol (telegraph codeword + Gaussian noise, with the
buffer's underdamped linear response). Gamma_X(eps_Z) is read from results/task24/*.json when that
file provides it; otherwise Gamma_X(0) from catqubit.rates.bit_flip_rate plus the ILLUSTRATIVE
linear-order drive term 4 eps_Z^2 / (kappa2 alpha^2) exp(-2 alpha^2) is used and labelled as such.

Protocol: thresholds A and erasure theta are tuned on a training set and evaluated on an
independent test set (different RNG seeds). Factor-2 convention: Gamma_X is the decay rate of
Re<a>; each one-way jump happens at Gamma_X / 2, and the bit-flip probability during a readout
of length T is p_X(T) = (1 - exp(-Gamma_X T)) / 2.

Run:  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv313/bin/python -u studies/task24c_protocol.py
      (about 1 h; add --plot-only to rebuild figures and summary from the JSON).
"""

import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere

import dynamiqs as dq  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from catqubit.liouvillian import FockTruncation  # noqa: E402
from catqubit.params import CatParams  # noqa: E402
from catqubit.rates import bit_flip_rate  # noqa: E402
from catqubit.readout_protocol import (  # noqa: E402
    RecordModel,
    boxcar_statistic,
    gaussian_fixed_error,
    hmm_filter,
    matched_statistic,
    simulate_records,
    sprt_stop_steps,
    state_at,
    stats_from_arrays,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "task24c"
FIGURES = ROOT / "figures" / "track2"
DPI = 150

EPS_Z_OVER_G2 = (0.1, 0.2, 0.3, 0.5)
ETAS = (1.0, 0.6)
DEMO_ALPHAS = (1.2, 1.4)
DEMO_EPS_Z_OVER_G2 = 0.3
# Truncations for Gamma_X(0): alpha = 2.1 from the shared rules file (converged to ~0.1%);
# the small-alpha ones agree with a larger (na+4, nb+1) truncation to < 1e-4 relative.
GAMMA_TRUNCATIONS = {
    2.1: FockTruncation(na=31, nb=5, nb_extra=6, na_taper=13.2),
    1.2: FockTruncation(na=18, nb=6),
    1.4: FockTruncation(na=20, nb=6),
}
# Per-step SNR mu^2 dt and number of steps: T_max = N_STEPS * SNR_PER_STEP / mu^2 = 15 / mu^2.
SNR_PER_STEP = 0.003
N_STEPS = 5000
N_TRAIN, N_TEST, CHUNK = 100_000, 200_000, 5000
if os.environ.get("TASK24C_QUICK"):  # smoke run only; never used for reported numbers
    N_TRAIN, N_TEST = 10_000, 10_000
TRAIN_SEED, TEST_SEED = 2024, 7_000_001
THRESHOLDS = np.round(np.arange(2.0, 14.01, 0.5), 3)
THETAS = (None, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
# Stop once |P(s_k=+|s0=+,y) - P(s_k=+|s0=-,y)| <= FREEZE_GAP: later data cannot inform s0.
FREEZE_GAP = 0.01
# Erasure rule search: erase if |log-odds| < a_erase or P(flip) > theta. Erasures are allowed up
# to ERASURE_BUDGET_FACTOR x the error target (erasures are much cheaper than hidden errors for
# a decoder; the factor is a stated assumption, not a derived number), at most MAX_ERASURE.
ERASE_LEVELS = (0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
ERASURE_BUDGET_FACTOR = 10.0
MAX_ERASURE = 0.25
# Small-alpha demo: p_X during readout is ~10%, so the target is instead the best error the
# boxcar reaches at its optimal window (training data): how fast does each method get there?
DEMO_TARGET = "boxcar optimum"
FIXED_TARGET = 1e-3
N_CHECKS = 80
QUANTILES = (0.5, 0.9, 0.99, 0.999)


@dataclass(frozen=True)
class Config:
    label: str
    alpha: float
    eps_z_over_g2: float
    eta: float
    gamma_x: float
    gamma_source: str
    model: RecordModel


# --------------------------------------------------------------------- inputs


def gamma_x0(alpha: float) -> float:
    params = CatParams.reglade().with_alpha(alpha)
    return bit_flip_rate(params, GAMMA_TRUNCATIONS[alpha])


def gamma_from_task24(alpha: float, eps_z: float) -> float | None:
    """Gamma_X(eps_Z) from results/task24/*.json, if any record matches (alpha, eps_Z) to 1e-6."""
    for path in sorted(glob.glob(str(ROOT / "results" / "task24" / "*.json"))):
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"skipping unreadable {path}: {exc}")
            continue
        records = data if isinstance(data, list) else data.get("records", [])
        for rec in records if isinstance(records, list) else []:
            if not isinstance(rec, dict):
                continue
            a, e, g = rec.get("alpha"), rec.get("eps_z"), rec.get("gamma_x")
            if None not in (a, e, g) and abs(a - alpha) < 1e-6 and abs(e - eps_z) < 1e-6:
                return float(g)
    return None


def make_config(alpha: float, eps_ratio: float, eta: float, g0: float) -> Config:
    params = CatParams.reglade().with_alpha(alpha)
    eps_z = eps_ratio * params.g2
    measured = gamma_from_task24(alpha, eps_z)
    if measured is not None:
        gamma, source = measured, "results/task24 (computed)"
    else:
        drive = 4 * eps_z**2 / (params.kappa2 * alpha**2) * math.exp(-2 * alpha**2)
        gamma, source = g0 + drive, "ILLUSTRATIVE: Gamma_X(0) + linear-order drive term"
    rate = RecordModel.from_physics(params, eps_z, eta, gamma, dt=1.0).measurement_rate
    model = RecordModel.from_physics(params, eps_z, eta, gamma, dt=SNR_PER_STEP / rate)
    label = f"alpha={alpha}, eps_z={eps_ratio}g2, eta={eta}"
    return Config(label, alpha, eps_ratio, eta, gamma, source, model)


# --------------------------------------------------------------------- simulation


def check_steps() -> np.ndarray:
    return np.unique(np.geomspace(10, N_STEPS, N_CHECKS).astype(int) - 1)


def run_chunk(model: RecordModel, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Fixed-time error counts and per-shot SPRT quantities for one chunk of records.

    Two stopping rules per threshold: "cap" (|log-odds| >= A or the hard cap, as in the brief)
    and "frz" (also stop once the s0 evidence is frozen, |branch_gap| <= FREEZE_GAP).
    """
    batch = simulate_records(model, n, N_STEPS, rng)
    s0, flips = batch.s0, batch.flips
    checks = check_steps()
    wrong = {}
    for name, stat in (
        ("boxcar", boxcar_statistic(model, batch.y)),
        ("matched", matched_statistic(model, batch.y)),
    ):
        wrong[name] = np.sum(np.where(stat[checks] >= 0, 1.0, -1.0) != s0, axis=1)
    trace = hmm_filter(model, batch.y)
    del batch, stat  # free the records; the trace holds what is still needed
    wrong["hmm"] = np.sum(np.where(trace.log_odds[checks] >= 0, 1.0, -1.0) != s0, axis=1)
    out = {**{f"wrong_{k}": v for k, v in wrong.items()}, "s0": s0}
    shots = np.arange(n)
    for kind, gap in (("cap", None), ("frz", FREEZE_GAP)):
        stops = sprt_stop_steps(trace, THRESHOLDS, N_STEPS, freeze_gap=gap)
        out[f"{kind}_stop"] = stops
        out[f"{kind}_final"] = trace.log_odds[stops, shots]
        out[f"{kind}_flip"] = trace.flip_prob[stops, shots]
        out[f"{kind}_s_stop"] = np.stack([state_at(s0, flips, st) for st in stops])
    return out


def simulate(model: RecordModel, n_total: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    parts = [run_chunk(model, CHUNK, rng) for _ in range(n_total // CHUNK)]
    out = {}
    for key in parts[0]:
        if key.startswith("wrong_"):
            out[key] = sum(p[key] for p in parts)
        else:
            out[key] = np.concatenate([p[key] for p in parts], axis=-1)
    return out


# --------------------------------------------------------------------- analysis

# Protocol variants: (stopping rule, erase undecided / theta?)
VARIANTS = {"sprt": ("cap", False), "freeze": ("frz", False), "erasure": ("frz", True)}


def sprt_stats(
    sim: dict, variant: str, i_a: int, theta: float | None, dt: float, a_erase: float = 0.0
) -> dict:
    """Test statistics of one protocol; erasure: |log-odds| < a_erase or P(flip) > theta."""
    kind, erase = VARIANTS[variant]
    final, flip = sim[f"{kind}_final"][i_a], sim[f"{kind}_flip"][i_a]
    erased = np.zeros(final.size, dtype=bool)
    if erase:
        erased |= np.abs(final) < a_erase
        if theta is not None:
            erased |= flip > theta
    decision = np.where(final >= 0, 1.0, -1.0)
    stop, s_stop = sim[f"{kind}_stop"][i_a], sim[f"{kind}_s_stop"][i_a]
    s = stats_from_arrays(decision, erased, stop, sim["s0"], s_stop, dt, QUANTILES)
    return {
        "A": float(THRESHOLDS[i_a]),
        "theta": theta,
        "a_erase": a_erase if erase else None,
        "p_error": s.p_error,
        "p_erasure": s.p_erasure,
        "p_hidden": s.p_hidden,
        "mean_time_us": s.mean_time,
        "time_quantiles_us": {str(q): v for q, v in s.time_quantiles.items()},
        "frac_at_cap": float(np.mean(stop == N_STEPS - 1)),
        "n_shots": s.n_shots,
    }


def p_flip_during(gamma: float, t: float) -> float:
    """Bit-flip probability over a readout of length t: (1 - exp(-Gamma_X t)) / 2."""
    return -math.expm1(-gamma * t) / 2


def pick_threshold(sim: dict, cfg: Config, target: float | None, variant: str) -> int:
    """Smallest A whose training error meets target (None: the Vuillot target p_X(mean T))."""
    for i_a in range(THRESHOLDS.size):
        st = sprt_stats(sim, variant, i_a, None, cfg.model.dt)
        goal = p_flip_during(cfg.gamma_x, st["mean_time_us"]) if target is None else target
        if st["p_error"] <= goal:
            return i_a
    return THRESHOLDS.size - 1


def pick_erasure(sim: dict, cfg: Config, target: float | None) -> tuple[int, float, float | None]:
    """(A, a_erase, theta) minimizing the training mean time subject to p_error <= target and
    p_erasure <= ERASURE_BUDGET_FACTOR x target (at most MAX_ERASURE)."""
    dt = cfg.model.dt
    best, best_time = None, math.inf
    for i_a in range(THRESHOLDS.size):
        for a_e in (x for x in ERASE_LEVELS if x <= THRESHOLDS[i_a]):
            for theta in THETAS:
                st = sprt_stats(sim, "erasure", i_a, theta, dt, a_e)
                goal = p_flip_during(cfg.gamma_x, st["mean_time_us"]) if target is None else target
                budget = min(ERASURE_BUDGET_FACTOR * goal, MAX_ERASURE)
                feasible = st["p_error"] <= goal and st["p_erasure"] <= budget
                if feasible and st["mean_time_us"] < best_time:
                    best, best_time = (i_a, a_e, theta), st["mean_time_us"]
    return best if best is not None else (pick_threshold(sim, cfg, target, "freeze"), 0.0, None)


def fixed_curves(sim: dict, dt: float, n: int) -> dict:
    times = (check_steps() + 1) * dt
    curves = {name: (sim[f"wrong_{name}"] / n).tolist() for name in ("boxcar", "matched", "hmm")}
    return {"t_us": times.tolist(), **curves}


def fixed_window_for(curve: list[float], times: list[float], target: float) -> float | None:
    for t, e in zip(times, curve, strict=True):
        if e <= target:
            return t
    return None


def window_for_error(curve: list[float], times: list[float], eps: float) -> float | None:
    """Shortest fixed window whose (test) error is at most eps, log-interpolated."""
    for i, (t, e) in enumerate(zip(times, curve, strict=True)):
        if e <= eps:
            if i == 0 or e <= 0 or curve[i - 1] <= 0:
                return t
            t0, e0 = times[i - 1], curve[i - 1]
            frac = (math.log(e0) - math.log(eps)) / (math.log(e0) - math.log(e))
            return t0 + frac * (t - t0)
    return None


def wilson(p: float, n: int, z: float = 1.96) -> list[float]:
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return [max(0.0, centre - half), centre + half]


def operating_point(train: dict, test: dict, cfg: Config, target: float | None) -> dict:
    """Tune A (and the erasure rule) on training data; report every method on the test data."""
    dt = cfg.model.dt
    test_fixed = fixed_curves(test, dt, N_TEST)
    points = {}
    for variant in VARIANTS:
        if variant == "erasure":
            i_a, a_e, theta = pick_erasure(train, cfg, target)
            st = sprt_stats(test, variant, i_a, theta, dt, a_e)
        else:
            st = sprt_stats(test, variant, pick_threshold(train, cfg, target, variant), None, dt)
        st["p_error_ci95"] = wilson(st["p_error"], N_TEST)
        st["p_flip_during_mean_T"] = p_flip_during(cfg.gamma_x, st["mean_time_us"])
        for name in ("boxcar", "hmm"):
            win = window_for_error(test_fixed[name], test_fixed["t_us"], max(st["p_error"], 1e-12))
            st[f"{name}_window_same_error_us"] = win
            st[f"speedup_vs_{name}"] = None if win is None else win / st["mean_time_us"]
        points[variant] = st
    goal = target if target is not None else points["sprt"]["p_flip_during_mean_T"]
    train_fixed = fixed_curves(train, dt, N_TRAIN)
    fixed = {}
    for name in ("boxcar", "matched", "hmm"):
        t_win = fixed_window_for(train_fixed[name], train_fixed["t_us"], goal)
        idx = None if t_win is None else train_fixed["t_us"].index(t_win)
        p_err = None if idx is None else test_fixed[name][idx]
        fixed[name] = {"window_us": t_win, "p_error_test": p_err}
    rule = "Vuillot: p_error <= p_X(mean T)" if target is None else f"p_error <= {target}"
    return {"target_rule": rule, "target": goal, **points, "fixed": fixed}


def sprt_curve(test: dict, cfg: Config, point: dict) -> list[dict]:
    dt = cfg.model.dt
    er = point["erasure"]
    return [
        {
            "sprt": sprt_stats(test, "sprt", i, None, dt),
            "freeze": sprt_stats(test, "freeze", i, None, dt),
            "erasure": sprt_stats(test, "erasure", i, er["theta"], dt, er["a_erase"]),
        }
        for i in range(THRESHOLDS.size)
    ]


def stop_histograms(test: dict, point: dict, dt: float) -> dict:
    out = {}
    for variant in ("sprt", "freeze"):
        kind = VARIANTS[variant][0]
        stop = test[f"{kind}_stop"][THRESHOLDS.tolist().index(point[variant]["A"])]
        counts, edges = np.histogram((stop + 1) * dt, bins=60, range=(0, N_STEPS * dt))
        out[variant] = {"counts": counts.tolist(), "edges_us": edges.tolist()}
    return out


def analyse(cfg: Config, target: float | str | None) -> dict:
    """target None: Vuillot operating point; also a fixed 1e-3 point when target is None."""
    t0 = time.time()
    train = simulate(cfg.model, N_TRAIN, TRAIN_SEED)
    test = simulate(cfg.model, N_TEST, TEST_SEED)
    if target == DEMO_TARGET:
        target = float(min(fixed_curves(train, cfg.model.dt, N_TRAIN)["boxcar"]))
    point = operating_point(train, test, cfg, target)
    fixed_target = operating_point(train, test, cfg, FIXED_TARGET) if target is None else None
    m = cfg.model
    print(f"{cfg.label}: {time.time() - t0:.0f} s")
    return {
        "label": cfg.label,
        "alpha": cfg.alpha,
        "eps_z_over_g2": cfg.eps_z_over_g2,
        "eta": cfg.eta,
        "gamma_x_rad_per_us": cfg.gamma_x,
        "gamma_x_source": cfg.gamma_source,
        "mu_per_sqrt_us": m.mu,
        "measurement_rate_mu2_per_us": m.measurement_rate,
        "gamma_over_rate": cfg.gamma_x / m.measurement_rate,
        "nu_rad_per_us": m.nu,
        "damping_rad_per_us": m.damping,
        "dt_us": m.dt,
        "t_max_us": N_STEPS * m.dt,
        "fixed_time_test": fixed_curves(test, m.dt, N_TEST),
        "sprt_curve_test": sprt_curve(test, cfg, point),
        "operating_point": point,
        "fixed_1e-3_point": fixed_target,
        "stop_hist_operating": stop_histograms(test, point, m.dt),
        "runtime_s": time.time() - t0,
    }


# --------------------------------------------------------------------- figures


STYLE = {
    "boxcar": ("tab:gray", "-", "boxcar (fixed window T)"),
    "matched": ("tab:blue", "-", "matched filter (fixed window T)"),
    "hmm": ("tab:green", "-", "HMM (fixed window T)"),
    "sprt": ("tab:red", "-", "HMM + SPRT, hard cap (mean T)"),
    "freeze": ("tab:brown", "-", "HMM + SPRT + freeze stop (mean T)"),
    "erasure": ("tab:purple", "--", "HMM + SPRT + freeze + erasure: p_error"),
}


def plot_error_panel(ax: plt.Axes, res: dict) -> None:
    fixed = res["fixed_time_test"]
    t = np.array(fixed["t_us"])
    floor = 1 / N_TEST
    for name in ("boxcar", "matched", "hmm"):
        color, ls, lab = STYLE[name]
        ax.plot(t, np.maximum(fixed[name], floor), color=color, ls=ls, label=lab)
    pts = res["sprt_curve_test"]
    for key in ("sprt", "freeze", "erasure"):
        color, ls, lab = STYLE[key]
        xs = [p[key]["mean_time_us"] for p in pts]
        ax.plot(
            xs,
            [max(p[key]["p_error"], floor) for p in pts],
            color=color,
            ls=ls,
            marker=".",
            ms=3,
            label=lab,
        )
    ax.plot(
        [p["erasure"]["mean_time_us"] for p in pts],
        [max(p["erasure"]["p_erasure"], floor) for p in pts],
        color="tab:purple",
        ls=":",
        label="HMM + SPRT + freeze + erasure: p_erasure",
    )
    rate = res["measurement_rate_mu2_per_us"]
    ax.plot(
        t,
        [gaussian_fixed_error(rate, x) for x in t],
        "k:",
        lw=0.8,
        label="Gaussian latching erfc (no flips)",
    )
    ax.plot(
        t,
        [p_flip_during(res["gamma_x_rad_per_us"], x) for x in t],
        color="tab:orange",
        lw=1.2,
        label="p_X(T) = (1-exp(-Gamma_X T))/2 (Vuillot bound)",
    )
    ax.axhline(floor, color="k", lw=0.4, alpha=0.4)
    ax.set_yscale("log")
    ax.set_ylim(floor / 2, 0.5)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel("readout time T, or mean stopping time (us)")
    ax.set_ylabel("assignment error P(decision != s0)")
    ax.set_title(res["label"], fontsize=8)


def fig_error_vs_time(results: list[dict]) -> None:
    fig, axes = plt.subplots(len(ETAS), len(EPS_Z_OVER_G2), figsize=(18, 8.5), squeeze=False)
    for res in results:
        ax = axes[ETAS.index(res["eta"]), EPS_Z_OVER_G2.index(res["eps_z_over_g2"])]
        plot_error_panel(ax, res)
        ax.set_xlim(0, 0.7 * res["t_max_us"])
    axes[0, 0].legend(fontsize=6, loc="lower left")
    fig.suptitle(
        "Task 2.4c, alpha = 2.1: readout error vs time (test set, 2e5 shots; homodyne; "
        "heterodyne = eta/2). Gamma_X = Re<a> decay rate, one-way jump Gamma_X/2. "
        "Gamma_X(eps_Z) ILLUSTRATIVE.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "task24c_error_vs_time.png", dpi=DPI)
    plt.close(fig)


def fig_stopping_hist(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(EPS_Z_OVER_G2), figsize=(18, 4), squeeze=False)
    for res in (r for r in results if r["eta"] == 1.0):
        ax = axes[0, EPS_Z_OVER_G2.index(res["eps_z_over_g2"])]
        pt = res["operating_point"]
        for variant, color in (("sprt", "tab:red"), ("freeze", "tab:brown")):
            h = res["stop_hist_operating"][variant]
            ax.stairs(
                h["counts"],
                np.array(h["edges_us"]),
                color=color,
                lw=1.2,
                label=f"{STYLE[variant][2].split(' (')[0]}, A = {pt[variant]['A']}",
            )
            q = pt[variant]["time_quantiles_us"]
            for key, ls in (("0.5", "-"), ("0.99", "--"), ("0.999", ":")):
                ax.axvline(q[key], color=color, ls=ls, lw=0.8)
        window = pt["fixed"]["boxcar"]["window_us"]
        if window is not None:
            ax.axvline(window, color="k", lw=1.5, label="boxcar window, same target")
        ax.set_yscale("log")
        ax.set_xlabel("stopping time (us)")
        ax.set_ylabel("shots per bin (of 2e5)")
        ax.set_title(f"{res['label']}; lines: median, 99%, 99.9% quantiles", fontsize=8)
        ax.legend(fontsize=6)
    fig.suptitle(
        "Stopping-time distribution at the Vuillot operating point (p_error <= p_X during "
        "readout). A synchronous QEC cycle must budget for the tail (or the cap), not the mean.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "task24c_stopping_hist.png", dpi=DPI)
    plt.close(fig)


def fig_erasure(demos: list[dict], results: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(demos) + 1, figsize=(6 * (len(demos) + 1), 4.8), squeeze=False)
    for ax, res in zip(axes[0], demos, strict=False):
        plot_error_panel(ax, res)
        ax.set_xlim(0, 0.5 * res["t_max_us"])
    axes[0, 0].legend(fontsize=6, loc="upper right")
    ax = axes[0, -1]
    for res in results:
        pt = res["operating_point"]
        marker = "o" if res["eta"] == 1.0 else "s"
        for key in ("sprt", "freeze", "erasure"):
            st = pt[key]
            ax.scatter(
                st["p_error"] + 1e-6,
                st["p_erasure"] + 1e-6,
                color=STYLE[key][0],
                marker=marker,
                s=15 + 60 * res["eps_z_over_g2"],
            )
    for key in ("sprt", "freeze", "erasure"):
        ax.scatter([], [], color=STYLE[key][0], label=STYLE[key][2].split(":")[0])
    ax.scatter([], [], color="k", marker="o", label="eta = 1 (size ~ eps_Z)")
    ax.scatter([], [], color="k", marker="s", label="eta = 0.6")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("p_error (+1e-6 offset)")
    ax.set_ylabel("p_erasure (+1e-6 offset)")
    ax.set_title("alpha = 2.1: (p_error, p_erasure) at the Vuillot points (test)", fontsize=8)
    ax.legend(fontsize=6)
    fig.suptitle(
        "Small-alpha demo (flips during readout are frequent) and the erasure trade-off. "
        "Gamma_X(eps_Z) ILLUSTRATIVE; one-way jump rate = Gamma_X/2.",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "task24c_erasure.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------- summary


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3g}"


def summary_row(res: dict) -> str:
    pt = res["operating_point"]
    win = {k: pt["fixed"][k]["window_us"] for k in ("boxcar", "matched", "hmm")}
    cells = [
        f"{res['alpha']}",
        f"{res['eps_z_over_g2']}",
        f"{res['eta']}",
        f"{res['gamma_x_rad_per_us']:.3g}",
        f"{res['measurement_rate_mu2_per_us']:.3g}",
        f"{pt['target']:.2e}",
        " / ".join(_fmt(win[k]) for k in win),
    ]
    for key in ("sprt", "freeze", "erasure"):
        st = pt[key]
        q = st["time_quantiles_us"]
        speed = st["speedup_vs_boxcar"]
        text = (
            f"A={st['A']}: {st['p_error']:.1e}, T={st['mean_time_us']:.3g} "
            f"(q99 {q['0.99']:.3g}), x{_fmt(speed)}"
        )
        if key == "erasure":
            text += f", eras {st['p_erasure']:.1e} (a_e={st['a_erase']}, theta={st['theta']})"
        cells.append(text)
    return "| " + " | ".join(cells) + " |"


def write_summary(results: list[dict], demos: list[dict], meta: dict) -> None:
    head = (
        "| alpha | eps_Z/g2 | eta | Gamma_X (rad/us) | mu^2 (1/us) | target p_X(T) | "
        "window box/MF/HMM (us) | HMM+SPRT cap: p_err, mean T, speedup | +freeze | "
        "+freeze+erasure |\n|" + "---|" * 10
    )
    rows = "\n".join(summary_row(r) for r in results + demos)
    text = f"""# Task 2.4c: adaptive readout protocol (generated by studies/task24c_protocol.py)

Monte Carlo on the classical surrogate (catqubit.readout_protocol). Test set {meta["n_test"]}
shots; thresholds and theta tuned on an independent training set of {meta["n_train"]} shots
(seeds {meta["seeds"]}). Gamma_X source: {meta["gamma_source"]}.
Gamma_X(0) = {meta["gamma_x0"]} rad/us (bit_flip_rate, truncations {meta["gamma_truncations"]}).
Factor-2 convention: Gamma_X is the decay rate of Re<a>; one-way jump rate Gamma_X/2;
p_X(T) = (1 - exp(-Gamma_X T))/2. Homodyne; heterodyne halves eta and doubles every time below.
Errors below 1/{meta["n_test"]} are not resolved; 95% Wilson intervals are in the JSON.

Operating point at alpha = 2.1 ("Vuillot point"): smallest A whose training error is at most
the bit-flip probability accumulated during the readout itself, p_X(mean T). Small-alpha demo
rows use target = the best boxcar error (at its optimal window) on training data.
"window" columns: shortest fixed window meeting
the same target on training data. Speedup xN = boxcar test window reaching the SAME test error
as the protocol, divided by the protocol's mean stopping time (D'Anjou's t_f / T).
Hard cap: {N_STEPS} steps = 15 / mu^2. Freeze stop: also stop once
|P(s_k=+|s0=+,y) - P(s_k=+|s0=-,y)| <= {FREEZE_GAP} (further data cannot inform s0).
Erasure: |log-odds at stop| < a_e or P(s_k != s0 | y) > theta; (A, a_e, theta) chosen on training
data to minimize mean time subject to p_error <= target and p_erasure <= {ERASURE_BUDGET_FACTOR:g}
x target (assumed erasure budget, at most {MAX_ERASURE}).

{head}
{rows}

Gaussian-latching reference (D'Anjou et al. 2016, Eqs. 2-3): speedup 2.77 at eps = 1e-3,
tending to 4 as eps -> 0. Fixed-target (1e-3) operating points are in the JSON ("fixed_1e-3_point").
"""
    (RESULTS / "summary.md").write_text(text)


def compute() -> None:
    dq.set_precision("double")
    g0 = {a: gamma_x0(a) for a in GAMMA_TRUNCATIONS}
    print("Gamma_X(0):", g0)
    configs = [make_config(2.1, e, eta, g0[2.1]) for eta in ETAS for e in EPS_Z_OVER_G2]
    results = [analyse(c, None) for c in configs]
    demos = [
        analyse(make_config(a, DEMO_EPS_Z_OVER_G2, 1.0, g0[a]), DEMO_TARGET) for a in DEMO_ALPHAS
    ]
    truncations = {
        str(a): {"na": t.na, "nb": t.nb, "nb_extra": t.nb_extra, "na_taper": t.na_taper}
        for a, t in GAMMA_TRUNCATIONS.items()
    }
    meta = {
        "gamma_x0": {str(a): g for a, g in g0.items()},
        "gamma_truncations": truncations,
        "gamma_source": sorted({r["gamma_x_source"] for r in results + demos}),
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "seeds": [TRAIN_SEED, TEST_SEED],
        "snr_per_step": SNR_PER_STEP,
        "n_steps": N_STEPS,
        "thresholds": THRESHOLDS.tolist(),
        "thetas": list(THETAS),
        "erasure_budget_factor": ERASURE_BUDGET_FACTOR,
        "max_erasure": MAX_ERASURE,
        "freeze_gap": FREEZE_GAP,
        "demo_target": DEMO_TARGET,
    }
    for name, res in (
        ("protocol_alpha2.1.json", results),
        ("protocol_demo_small_alpha.json", demos),
    ):
        (RESULTS / name).write_text(json.dumps({"meta": meta, "results": res}, indent=1))


def make_outputs() -> None:
    main_data = json.loads((RESULTS / "protocol_alpha2.1.json").read_text())
    demo_data = json.loads((RESULTS / "protocol_demo_small_alpha.json").read_text())
    results, demos = main_data["results"], demo_data["results"]
    fig_error_vs_time(results)
    fig_stopping_hist(results)
    fig_erasure(demos, results)
    write_summary(results, demos, main_data["meta"])


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    if "--plot-only" not in sys.argv:
        compute()
    make_outputs()


if __name__ == "__main__":
    main()
