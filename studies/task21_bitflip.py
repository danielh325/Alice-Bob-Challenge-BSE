"""Task 2.1: the bit-flip rate Gamma_X, from time-domain fits to the exact Liouvillian gap.

Usage: python -u studies/task21_bitflip.py PART [PART ...], with PART one of
  sweep      exact two-mode Gamma_X for alpha = 0.8..2.1, converged in every truncation knob
  large      the same at the single points alpha = 2.3 and 2.5 (minutes each)
  timedomain fits of Re<a>(t) from |alpha>|0> at alpha = 0.8, 1.0, 1.2, 1.4
  effective  effective one-mode gap and the Dubovitskii closed form on the same grid
  thermal    two-mode Gamma_X with Réglade's thermal populations (ablation)
  thermalcheck  time-domain check of one thermal point (alpha = 1.4)
  adiabatic  fixed-kappa2 scan g2 -> s g2, kappa_b -> s^2 kappa_b (R3 evidence)
  buffer     buffer-level sensitivity and the transient from the saddle (R3 evidence)
  analyze    exponent fits and local exponents -> results/task21/analysis.json
  figures    figures/track2/task21_*.png from the JSON files
  all        everything above, in order

Convention (every figure says so): Gamma_X is the decay rate of Re<a>, i.e. -Re(lambda) of the
slowest parity-odd Liouvillian mode. The one-way jump rate |+a> -> |-a> is Gamma_X / 2.
Rates in rad/us.
"""

import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np

from catqubit import bitflip_theory as bt
from catqubit.convergence import fit_decay_rate
from catqubit.liouvillian import FockTruncation
from catqubit.params import CatParams
from catqubit.rates import bit_flip_rate
from catqubit.simulate import inflate

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "task21"
FIGURES = ROOT / "figures" / "track2"

SWEEP_ALPHAS = tuple(round(0.8 + 0.1 * k, 1) for k in range(14))  # 0.8 .. 2.1
LARGE_ALPHAS = (2.3, 2.5)
DENSE_EFFECTIVE_ALPHAS = tuple(round(0.7 + 0.05 * k, 2) for k in range(41))  # 0.7 .. 2.7
TIME_DOMAIN_ALPHAS = (0.8, 1.0, 1.2, 1.4)

# Truncation rule (base): memory alpha^2 + 8 alpha + 6; buffer 5 levels everywhere, widened
# near empty memory to the barrier occupation 4 g2^2 alpha^4 / kappa_b^2 (~0.34 alpha^4) + 5,
# tapering off by n_a = 3 alpha^2.
NA_PER_ALPHA, NA_OFFSET = 8, 6
NB_BASE = 5
NB_MARGIN = 5
TAPER_PER_ALPHA2 = 3.0
# Refinements: each knob alone, then all together.
REFINE_NA, REFINE_NB, REFINE_NB_EXTRA, REFINE_TAPER = 6, 1, 3, 4 / 3

# Time domain: fit windows start well after the ~1 us transient; the next odd mode decays at
# |lambda_2| >= 0.08 /us (alpha = 0.8), so it is damped by e^{-1.6} or more at t = 20 us.
TD_T_END = 300.0
TD_N_SAVE = 301
TD_FIT_STARTS = (20.0, 60.0, 120.0)
TD_TRUNC = {0.8: (14, 5), 1.0: (16, 5), 1.2: (18, 5), 1.4: (21, 6)}

# Réglade et al. 2024, Table S2 / Sec. S5: n_th_a = 0.10 (memory), n_th_b = 0.011 (buffer).
THERMAL_CASES = ((0.10, 0.011), (0.05, 0.011), (0.0, 0.011), (0.10, 0.0))
THERMAL_ALPHAS = (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.9, 2.0, 2.1)
ABLATION_ALPHAS = (1.4, 1.8, 2.1)  # for the single-mode-thermal cases (last two)
# Knobs refined per part (machine time): the sweep refines all, the rest the sensitive ones.
ALL_KNOBS = ("na", "nb", "nb_extra", "na_taper", "all")
LARGE_KNOBS = ("nb_extra", "na_taper")
CHECK_KNOBS = ("nb_extra",)

ADIABATIC_SCALES = (2.0, 4.0)  # s = 1 is the main sweep
ADIABATIC_ALPHAS = (1.2, 1.4, 1.6, 1.8, 2.0, 2.2)
COLLAPSE_SCALES = (1.0, 1.5, 2.0, 3.0, 4.0, 8.0)
COLLAPSE_ALPHAS = (1.4, 1.8, 2.1)

BUFFER_ALPHA = 2.1
BUFFER_NA = 28
BUFFER_EXTRA_SCAN = (0, 1, 2, 3, 4, 6, 8)
BUFFER_TAPER_SCAN = (1.0, 3.0, 6.0, 9.0, 13.2, 17.6)
SADDLE_T_END = 2.0
SADDLE_TRUNC = (28, 12)
SADDLE_SCALES = (1.0, 4.0)

FIT_WINDOWS = {"small": (0.8, 1.4), "mid": (1.4, 2.1), "large": (1.8, 2.5)}

DPI = 150
CONVENTION = r"$\Gamma_X$ = decay rate of Re$\langle a\rangle$ ($= 2\times$ one-way jump rate)"


def base_truncation(alpha: float, g2: float | None = None, kappa_b: float | None = None):
    """Base FockTruncation from the rule above (g2, kappa_b default to Réglade's)."""
    p = CatParams.reglade()
    g2 = p.g2 if g2 is None else g2
    kappa_b = p.kappa_b if kappa_b is None else kappa_b
    barrier_photons = 4 * g2**2 * alpha**4 / kappa_b**2
    nb_max = max(NB_BASE, math.ceil(barrier_photons) + NB_MARGIN)
    return FockTruncation(
        na=math.ceil(alpha**2 + NA_PER_ALPHA * alpha + NA_OFFSET),
        nb=NB_BASE,
        nb_extra=nb_max - NB_BASE,
        na_taper=max(1.0, round(TAPER_PER_ALPHA2 * alpha**2, 3)),
    )


def refinements(base: FockTruncation, knobs: tuple[str, ...]) -> dict[str, FockTruncation]:
    variants = {
        "na": replace(base, na=base.na + REFINE_NA),
        "nb": replace(base, nb=base.nb + REFINE_NB),
        "nb_extra": replace(base, nb_extra=base.nb_extra + REFINE_NB_EXTRA),
        "na_taper": replace(base, na_taper=round(base.na_taper * REFINE_TAPER, 3)),
    }
    variants["all"] = FockTruncation(
        na=base.na + REFINE_NA,
        nb=base.nb + REFINE_NB,
        nb_extra=base.nb_extra + REFINE_NB_EXTRA,
        na_taper=round(base.na_taper * REFINE_TAPER, 3),
    )
    return {k: variants[k] for k in knobs}


def timed_rate(params: CatParams, trunc: FockTruncation, **thermal: float) -> dict:
    start = time.perf_counter()
    rate = bit_flip_rate(params, trunc, **thermal)
    return {
        "gamma_x": rate,
        "trunc": asdict(trunc),
        "dim": trunc.dim,
        "wall_s": round(time.perf_counter() - start, 2),
    }


def converged_point(alpha: float, knobs: tuple[str, ...]) -> dict:
    """Gamma_X at the base truncation, plus the shift from refining each knob."""
    p = CatParams.reglade().with_alpha(alpha)
    base = base_truncation(alpha)
    point = {"alpha": alpha, "base": timed_rate(p, base)}
    g0 = point["base"]["gamma_x"]
    point["refined"] = {}
    for name, trunc in refinements(base, knobs).items():
        run = timed_rate(p, trunc)
        run["rel_shift"] = run["gamma_x"] / g0 - 1
        point["refined"][name] = run
        print(
            f"  alpha={alpha} {name:9s} G={run['gamma_x']:.6e} shift={run['rel_shift']:+.2e} "
            f"dim={run['dim']} t={run['wall_s']}s",
            flush=True,
        )
    singles = [v["rel_shift"] for k, v in point["refined"].items() if k != "all"]
    point["rel_uncertainty"] = float(sum(abs(s) for s in singles))
    print(f"alpha={alpha} G_base={g0:.6e} rel_unc={point['rel_uncertainty']:.2e}", flush=True)
    return point


def params_record() -> dict:
    p = CatParams.reglade()
    return {
        "g2": p.g2,
        "kappa_b": p.kappa_b,
        "kappa_a": p.kappa_a,
        "kappa2": p.kappa2,
        "units": "rad/us",
        "source": "Réglade et al. 2024, Table S2",
    }


def save(name: str, payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {"params": params_record(), "convention": CONVENTION, **payload}
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=1))
    print(f"wrote {RESULTS / name}.json", flush=True)


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def two_mode_points() -> list[dict]:
    """Sweep points plus the slow large-alpha points when they have been computed."""
    large = load("large")["points"] if (RESULTS / "large.json").exists() else []
    return load("sweep")["points"] + large


def truncation_rule_record() -> dict:
    return {
        "na": f"ceil(alpha^2 + {NA_PER_ALPHA} alpha + {NA_OFFSET})",
        "nb": NB_BASE,
        "nb_max": f"max({NB_BASE}, ceil(4 g2^2 alpha^4 / kappa_b^2) + {NB_MARGIN})",
        "na_taper": f"{TAPER_PER_ALPHA2} alpha^2",
        "refinements": {
            "na": f"+{REFINE_NA}",
            "nb": f"+{REFINE_NB}",
            "nb_extra": f"+{REFINE_NB_EXTRA}",
            "na_taper": f"x{REFINE_TAPER:.4f}",
        },
        "rel_uncertainty": "sum of |relative shift| over the four single-knob refinements",
    }


def part_sweep() -> None:
    points = [converged_point(a, ALL_KNOBS) for a in SWEEP_ALPHAS]
    save("sweep", {"truncation_rule": truncation_rule_record(), "points": points})


def part_large() -> None:
    """Single points at alpha = 2.3, 2.5; saved after each point (minutes to hours each)."""
    points = []
    for alpha in LARGE_ALPHAS:
        points.append(converged_point(alpha, LARGE_KNOBS))
        save(
            "large",
            {"truncation_rule": truncation_rule_record(), "knobs": LARGE_KNOBS, "points": points},
        )


def time_domain_point(alpha: float) -> dict:
    """mesolve from |alpha>|0>, fit Re<a>(t)/A = e^{-Gamma t} on late windows."""
    na, nb = TD_TRUNC[alpha]
    p = CatParams.reglade().with_alpha(alpha)
    a, _ = dq.destroy(na, nb)
    tsave = jnp.linspace(0.0, TD_T_END, TD_N_SAVE)
    start = time.perf_counter()
    res = inflate(
        na, nb, p, tsave, memory0=dq.coherent(na, alpha), extra_exp_ops=[a], save_states=False
    )
    wall = time.perf_counter() - start
    t = np.asarray(tsave)
    re_a = np.real(np.asarray(res.expects[2]))
    fits = {}
    for t0 in TD_FIT_STARTS:
        late = t >= t0
        gamma = fit_decay_rate(t[late], re_a[late])
        slope, intercept = np.polyfit(t[late], np.log(re_a[late]), 1)
        fits[str(t0)] = {"gamma_x": gamma, "plateau_over_alpha": math.exp(intercept) / alpha}
    same_trunc = bit_flip_rate(p, FockTruncation(na=na, nb=nb))
    print(
        f"alpha={alpha} td={fits[str(TD_FIT_STARTS[1])]['gamma_x']:.6e} "
        f"spectral(same trunc)={same_trunc:.6e} wall={wall:.0f}s",
        flush=True,
    )
    early = t <= 10.0
    return {
        "alpha": alpha,
        "na": na,
        "nb": nb,
        "t_end_us": TD_T_END,
        "n_save": TD_N_SAVE,
        "solver_tol": 1e-9,
        "wall_s": round(wall, 1),
        "fits": fits,
        "spectral_same_trunc": same_trunc,
        "t_early": t[early].tolist(),
        "re_a_early": re_a[early].tolist(),
        "t": t[::10].tolist(),
        "re_a": re_a[::10].tolist(),
    }


def part_timedomain() -> None:
    save(
        "timedomain",
        {
            "points": [time_domain_point(a) for a in TIME_DOMAIN_ALPHAS],
            "fit": "log-linear on t >= t0 (fit_decay_rate); plateau = e^intercept",
        },
    )


def effective_point(alpha: float, n_th: float = 0.0) -> dict:
    p = CatParams.reglade().with_alpha(alpha)
    na = bt.effective_memory_dim(alpha)
    rate = bt.effective_bit_flip_rate(p, na, n_th=n_th)
    check = bt.effective_bit_flip_rate(p, na + 10, n_th=n_th)
    return {"alpha": alpha, "na": na, "gamma_x": rate, "rel_shift_na_plus_10": check / rate - 1}


def part_effective() -> None:
    p = CatParams.reglade()
    alphas = SWEEP_ALPHAS + LARGE_ALPHAS
    a2 = np.array(alphas) ** 2
    k1, k2 = p.kappa_a, p.kappa2
    save(
        "effective",
        {
            "model": "kappa2 D[a^2 - alpha^2] + kappa1 (1+n_th) D[a] + kappa1 n_th D[a^dag]",
            "points": [effective_point(a) for a in alphas],
            "dense": [effective_point(a) for a in DENSE_EFFECTIVE_ALPHAS],
            "thermal": {str(n): [effective_point(a, n) for a in alphas] for n in (0.05, 0.10)},
            "dubovitskii": {
                "alpha": list(alphas),
                "gamma_x": bt.dubovitskii_bit_flip_rate(k1, k2, a2).tolist(),
                "first_order_x": (2 * bt.first_order_jump_rate(k1, a2)).tolist(),
                "second_order_x": (2 * bt.second_order_jump_rate(k1, k2, a2)).tolist(),
                "asymptotic_x": bt.asymptotic_bit_flip_rate(k1, k2, a2).tolist(),
            },
            "crossover_alpha": {
                "leading_log": math.sqrt(bt.crossover_alpha2_leading_log(k1, k2)),
                "asymptotic_terms_equal": math.sqrt(bt.crossover_alpha2_asymptotic(k1, k2)),
                "exact_orders_equal": math.sqrt(bt.crossover_alpha2_exact(k1, k2)),
            },
        },
    )


def part_thermal() -> None:
    cases = []
    for k, (n_a, n_b) in enumerate(THERMAL_CASES):
        pts = []
        for alpha in THERMAL_ALPHAS if k < 2 else ABLATION_ALPHAS:
            p = CatParams.reglade().with_alpha(alpha)
            run = timed_rate(p, base_truncation(alpha), n_th_a=n_a, n_th_b=n_b)
            pts.append({"alpha": alpha, **run})
            print(f"n_th=({n_a},{n_b}) alpha={alpha} G={run['gamma_x']:.5e}", flush=True)
        case = {"n_th_a": n_a, "n_th_b": n_b, "points": pts}
        if k == 0:
            p21 = CatParams.reglade().with_alpha(2.1)
            check = timed_rate(
                p21,
                refinements(base_truncation(2.1), CHECK_KNOBS)["nb_extra"],
                n_th_a=n_a,
                n_th_b=n_b,
            )
            check["rel_shift"] = check["gamma_x"] / pts[-1]["gamma_x"] - 1
            case["check_alpha_2p1_nb_extra"] = check
        cases.append(case)
    save("thermal", {"cases": cases, "source": "Réglade et al. 2024, Table S2 / Sec. S5"})


def part_thermalcheck() -> None:
    """Independent check of the thermal Liouvillian: mesolve with the thermal jumps at
    alpha = 1.4 (Gamma_X ~ 1e-2 /us, so the decay is resolved in tens of us)."""
    n_a, n_b = THERMAL_CASES[0]
    na, nb = TD_TRUNC[1.4]
    p = CatParams.reglade().with_alpha(1.4)
    a, b = dq.destroy(na, nb)
    jumps = [
        math.sqrt(p.kappa_b * (1 + n_b)) * b,
        math.sqrt(p.kappa_b * n_b) * b.dag(),
        math.sqrt(p.kappa_a * (1 + n_a)) * a,
        math.sqrt(p.kappa_a * n_a) * a.dag(),
    ]
    h = p.g2 * (a @ a @ b.dag() + a.dag() @ a.dag() @ b) - p.eps_d * (b + b.dag())
    tsave = jnp.linspace(0.0, 150.0, 151)
    psi0 = dq.tensor(dq.coherent(na, 1.4), dq.fock(nb, 0))
    res = dq.mesolve(
        h,
        jumps,
        psi0,
        tsave,
        exp_ops=[a],
        progress_meter=False,
        method=dq.method.Tsit5(rtol=1e-9, atol=1e-9),
    )
    t, re_a = np.asarray(tsave), np.real(np.asarray(res.expects[0]))
    late = t >= TD_FIT_STARTS[0]
    td = fit_decay_rate(t[late], re_a[late])
    spec = bit_flip_rate(p, FockTruncation(na=na, nb=nb), n_th_a=n_a, n_th_b=n_b)
    print(f"thermal check alpha=1.4: td={td:.6e} spectral={spec:.6e}", flush=True)
    save(
        "thermalcheck",
        {
            "alpha": 1.4,
            "n_th_a": n_a,
            "n_th_b": n_b,
            "na": na,
            "nb": nb,
            "fit_start_us": TD_FIT_STARTS[0],
            "gamma_x_time_domain": td,
            "gamma_x_spectral_same_trunc": spec,
            "rel_diff": td / spec - 1,
        },
    )


def scaled_params(alpha: float, s: float) -> CatParams:
    """g2 -> s g2, kappa_b -> s^2 kappa_b: kappa2 = 4 g2^2/kappa_b and kappa_a are unchanged."""
    p = CatParams.reglade()
    return replace(p, g2=s * p.g2, kappa_b=s**2 * p.kappa_b, alpha=alpha)


def adiabatic_point(alpha: float, s: float) -> dict:
    p = scaled_params(alpha, s)
    base = base_truncation(alpha, p.g2, p.kappa_b)
    run = timed_rate(p, base)
    check = timed_rate(p, refinements(base, CHECK_KNOBS)["nb_extra"])
    eff = bt.effective_bit_flip_rate(p, bt.effective_memory_dim(alpha))
    out = {
        "alpha": alpha,
        "s": s,
        "g2": p.g2,
        "kappa_b": p.kappa_b,
        "barrier_photons": 4 * p.g2**2 * alpha**4 / p.kappa_b**2,
        "adiabatic_parameter": p.g2 * alpha / p.kappa_b,
        **run,
        "gamma_x_refined": check["gamma_x"],
        "trunc_refined": check["trunc"],
        "gamma_x_effective": eff,
        "excess": run["gamma_x"] / eff - 1,
    }
    print(
        f"s={s} alpha={alpha} G={run['gamma_x']:.5e} refined={check['gamma_x']:.5e} "
        f"eff={eff:.5e} excess={out['excess']:.3f}",
        flush=True,
    )
    return out


def part_adiabatic() -> None:
    grid = [adiabatic_point(a, s) for s in ADIABATIC_SCALES for a in ADIABATIC_ALPHAS]
    collapse = [adiabatic_point(a, s) for a in COLLAPSE_ALPHAS for s in COLLAPSE_SCALES]
    save(
        "adiabatic",
        {
            "scan": "g2 -> s g2, kappa_b -> s^2 kappa_b (kappa2, kappa_a fixed)",
            "grid": grid,
            "collapse": collapse,
        },
    )


def saddle_transient(s: float) -> dict:
    """Start at the saddle (memory and buffer in vacuum); compare <b^dag b> with the value
    the buffer would have if it followed the memory adiabatically,
    n_ad = (4 g2^2 / kappa_b^2) <(a^dag^2 - alpha^2)(a^2 - alpha^2)>."""
    na, nb = SADDLE_TRUNC
    p = scaled_params(BUFFER_ALPHA, s)
    a, _ = dq.destroy(na, nb)
    a2 = a @ a - BUFFER_ALPHA**2 * dq.eye(na, nb)
    tsave = jnp.linspace(0.0, SADDLE_T_END, 201)
    res = inflate(na, nb, p, tsave, extra_exp_ops=[a2.dag() @ a2], save_states=False)
    n_a, n_b, dev = (np.real(np.asarray(res.expects[k])) for k in range(3))
    n_ad = 4 * p.g2**2 / p.kappa_b**2 * dev
    return {
        "s": s,
        "na": na,
        "nb": nb,
        "t": np.asarray(tsave).tolist(),
        "n_a": n_a.tolist(),
        "n_b": n_b.tolist(),
        "n_b_adiabatic": n_ad.tolist(),
        "peak_n_b": float(n_b.max()),
        "barrier_photons": float(n_ad[0]),
    }


def part_buffer() -> None:
    p = CatParams.reglade().with_alpha(BUFFER_ALPHA)
    extra = []
    for k in BUFFER_EXTRA_SCAN:
        run = timed_rate(p, FockTruncation(na=BUFFER_NA, nb=NB_BASE, nb_extra=k, na_taper=13.2))
        extra.append(run)
        print(f"nb_extra={k} G={run['gamma_x']:.6e} t={run['wall_s']}s", flush=True)
    taper = []
    for c in BUFFER_TAPER_SCAN:
        run = timed_rate(p, FockTruncation(na=BUFFER_NA, nb=NB_BASE, nb_extra=6, na_taper=c))
        taper.append(run)
        print(f"na_taper={c} G={run['gamma_x']:.6e} t={run['wall_s']}s", flush=True)
    rect = [timed_rate(p, FockTruncation(na=BUFFER_NA, nb=k)) for k in (3, 4, 5, 6)]
    saddle = [saddle_transient(s) for s in SADDLE_SCALES]
    for tr in saddle:
        print(
            f"saddle s={tr['s']}: peak n_b={tr['peak_n_b']:.2f}, "
            f"barrier n_ad={tr['barrier_photons']:.2f}",
            flush=True,
        )
    save(
        "buffer",
        {
            "alpha": BUFFER_ALPHA,
            "extra_scan": extra,
            "taper_scan": taper,
            "rect_scan": rect,
            "saddle": saddle,
        },
    )


def _fits(alpha: np.ndarray, rate: np.ndarray, window: tuple[float, float]) -> dict:
    m = (alpha >= window[0] - 1e-9) & (alpha <= window[1] + 1e-9)
    out = {"window_alpha": list(window), "n_points": int(m.sum())}
    for power in (0, 1):
        fit = bt.fit_exponent(alpha[m] ** 2, rate[m], prefactor_power=power)
        out[f"p{power}"] = asdict(fit)
    return out


def series() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    sweep = two_mode_points()
    eff = load("effective")
    two = (np.array([q["alpha"] for q in sweep]), np.array([q["base"]["gamma_x"] for q in sweep]))
    one = (
        np.array([q["alpha"] for q in eff["points"]]),
        np.array([q["gamma_x"] for q in eff["points"]]),
    )
    dub = (np.array(eff["dubovitskii"]["alpha"]), np.array(eff["dubovitskii"]["gamma_x"]))
    td = load("timedomain")["points"]
    tdx = (
        np.array([q["alpha"] for q in td]),
        np.array([q["fits"][str(TD_FIT_STARTS[1])]["gamma_x"] for q in td]),
    )
    dense = (
        np.array([q["alpha"] for q in eff["dense"]]),
        np.array([q["gamma_x"] for q in eff["dense"]]),
    )
    out = {
        "two_mode": two,
        "effective": one,
        "effective_dense": dense,
        "dubovitskii": dub,
        "time_domain": tdx,
    }
    for case in load("thermal")["cases"]:
        key = f"thermal_{case['n_th_a']}_{case['n_th_b']}"
        out[key] = (
            np.array([q["alpha"] for q in case["points"]]),
            np.array([q["gamma_x"] for q in case["points"]]),
        )
    return out


def part_analyze() -> None:
    data = series()
    fits = {}
    for name, (alpha, rate) in data.items():
        fits[name] = {
            w: _fits(alpha, rate, rng)
            for w, rng in FIT_WINDOWS.items()
            if np.sum((alpha >= rng[0] - 1e-9) & (alpha <= rng[1] + 1e-9)) >= 2
        }
    local = {
        name: {
            "alpha": alpha.tolist(),
            "local_exponent": bt.local_exponent(alpha**2, rate).tolist(),
        }
        for name, (alpha, rate) in data.items()
        if alpha.size >= 3
    }
    td = load("timedomain")["points"]
    sweep = {q["alpha"]: q for q in load("sweep")["points"]}
    agreement = [
        {
            "alpha": q["alpha"],
            "td_vs_same_trunc": q["fits"][str(TD_FIT_STARTS[1])]["gamma_x"]
            / q["spectral_same_trunc"]
            - 1,
            "td_vs_converged": q["fits"][str(TD_FIT_STARTS[1])]["gamma_x"]
            / sweep[q["alpha"]]["base"]["gamma_x"]
            - 1,
        }
        for q in td
    ]
    two_alpha, two_rate = data["two_mode"]
    eff_alpha, eff_rate = data["effective"]
    ratio = {
        "alpha": two_alpha.tolist(),
        "two_mode_over_effective": (two_rate / np.interp(two_alpha, eff_alpha, eff_rate)).tolist(),
    }
    excess = two_rate - np.interp(two_alpha, eff_alpha, eff_rate)
    ratio["nonadiabatic_excess"] = excess.tolist()
    ratio["nonadiabatic_excess_local_exponent"] = bt.local_exponent(two_alpha**2, excess).tolist()
    dub_alpha, dub_rate = data["dubovitskii"]
    ratio["dubovitskii_over_effective"] = (
        dub_rate / np.interp(dub_alpha, eff_alpha, eff_rate)
    ).tolist()
    collapse = load("adiabatic")["collapse"]
    excess_s2 = {
        str(a): [[q["s"], q["excess"] * q["s"] ** 2] for q in collapse if q["alpha"] == a]
        for a in COLLAPSE_ALPHAS
    }
    save(
        "analysis",
        {
            "fits": fits,
            "local_exponent": local,
            "time_domain_agreement": agreement,
            "two_mode_over_effective": ratio,
            "adiabatic_excess_times_s2": excess_s2,
            "reference": {
                "deck_gamma_bound": 2.0,
                "reglade_measured_gamma": math.log(4.2),
                "reglade_note": "x4.2 per added photon (Réglade Fig. 4b)",
                "putterman_gamma_p1": 2.24,
                "putterman_gamma_p0": 1.75,
            },
        },
    )


# Categorical slots of the dataviz reference palette (validated; markers/linestyles too).
COLORS = {
    "two_mode": "#2a78d6",
    "effective": "#eb6834",
    "dubovitskii": "#1baf7a",
    "thermal_hi": "#eda100",
    "thermal_lo": "#e87ba4",
    "time_domain": "#008300",
    "extra": "#6250d6",
    "ink": "#333333",
}


def _rates_axes(ax, d: dict) -> None:
    sweep = two_mode_points()
    a = np.array([q["alpha"] for q in sweep])
    g = np.array([q["base"]["gamma_x"] for q in sweep])
    err = g * np.array([q["rel_uncertainty"] for q in sweep])
    ax.errorbar(
        a**2,
        g,
        yerr=err,
        fmt="o-",
        ms=5,
        lw=2,
        color=COLORS["two_mode"],
        label="two-mode Liouvillian gap (exact)",
    )
    ea, eg = d["effective"]
    ax.plot(
        ea**2,
        eg,
        "s--",
        ms=4,
        lw=1.5,
        color=COLORS["effective"],
        label=r"effective one-mode gap ($\kappa_2=4g_2^2/\kappa_b$)",
    )
    dense = np.linspace(0.5, 6.5, 300)
    p = CatParams.reglade()
    ax.plot(
        dense,
        bt.dubovitskii_bit_flip_rate(p.kappa_a, p.kappa2, dense),
        "-",
        lw=1.5,
        color=COLORS["dubovitskii"],
        label="Dubovitskii Eqs. 6+23 (x2)",
    )
    ax.plot(
        dense,
        2 * bt.first_order_jump_rate(p.kappa_a, dense),
        ":",
        lw=1.2,
        color=COLORS["dubovitskii"],
        label=r"  1st order $\propto e^{-4\alpha^2}$",
    )
    ax.plot(
        dense,
        2 * bt.second_order_jump_rate(p.kappa_a, p.kappa2, dense),
        "-.",
        lw=1.2,
        color=COLORS["dubovitskii"],
        label=r"  2nd order $\propto e^{-2\alpha^2}$",
    )
    for key, col, lab in (
        ("thermal_0.1_0.011", "thermal_hi", "two-mode, $n_{th}$=(0.10, 0.011)"),
        ("thermal_0.05_0.011", "thermal_lo", "two-mode, $n_{th}$=(0.05, 0.011)"),
        ("thermal_0.1_0.0", "extra", "two-mode, $n_{th}$=(0.10, 0) memory only"),
    ):
        ta, tg = d[key]
        ax.plot(ta**2, tg, "^-", ms=5, lw=1.5, color=COLORS[col], label=lab)
    ax.plot(
        dense,
        bt.thermal_bit_flip_estimate(p.kappa_a, 0.10, dense),
        ":",
        lw=1.2,
        color=COLORS["extra"],
        label=r"estimate $\kappa_a n_{th,a} e^{-2\alpha^2}$, $n_{th,a}$=0.10 (Gautier 2022)",
    )
    tda, tdg = d["time_domain"]
    ax.plot(
        tda**2,
        tdg,
        "D",
        ms=9,
        mfc="none",
        mew=2,
        color=COLORS["time_domain"],
        label=r"time-domain fit of Re$\langle a\rangle(t)$",
    )
    ax.set_yscale("log")
    ax.set_ylabel(r"$\Gamma_X$ (rad/$\mu$s)")


def fig_rates(d: dict) -> None:
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(8, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]}
    )
    _rates_axes(ax, d)
    ax.set_title("Task 2.1: bit-flip rate vs cat size\n" + CONVENTION, fontsize=10)
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    ta, tg = d["two_mode"]
    ea, eg = d["effective"]
    da, dg = d["dubovitskii"]
    bx.plot(
        ta**2,
        tg / np.interp(ta, ea, eg),
        "o-",
        color=COLORS["two_mode"],
        label="two-mode / effective",
    )
    bx.plot(
        da**2,
        dg / np.interp(da, ea, eg),
        "-",
        color=COLORS["dubovitskii"],
        label="Dubovitskii / effective",
    )
    bx.axhline(1.0, color=COLORS["ink"], lw=0.8)
    bx.set_xlabel(r"$\alpha^2$ (mean photon number)")
    bx.set_ylabel("ratio")
    bx.legend(fontsize=8)
    bx.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "task21_rates.png", dpi=DPI)
    plt.close(fig)


def fig_local_exponent(d: dict) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    p = CatParams.reglade()
    dense = np.linspace(0.5, 6.5, 300)
    ax.plot(
        dense,
        bt.local_exponent(dense, bt.dubovitskii_bit_flip_rate(p.kappa_a, p.kappa2, dense)),
        "-",
        color=COLORS["dubovitskii"],
        label="Dubovitskii",
    )
    for key, col, mk, lab in (
        ("two_mode", "two_mode", "o-", "two-mode (exact gap)"),
        ("effective_dense", "effective", "--", "effective one-mode"),
        ("thermal_0.1_0.011", "thermal_hi", "^-", "two-mode, $n_{th}$=(0.10, 0.011)"),
    ):
        a, g = d[key]
        ax.plot(a**2, bt.local_exponent(a**2, g), mk, ms=4, color=COLORS[col], label=lab)
    adi = load("adiabatic")["grid"]
    for s, ls in ((2.0, "x:"), (4.0, "+:")):
        pts = [q for q in adi if q["s"] == s]
        a2 = np.array([q["alpha"] ** 2 for q in pts])
        g = np.array([q["gamma_x"] for q in pts])
        ax.plot(
            a2,
            bt.local_exponent(a2, g),
            ls,
            ms=7,
            color=COLORS["extra"],
            label=rf"two-mode, $g_2\to{s:g}g_2$, $\kappa_b\to{s * s:g}\kappa_b$ "
            r"(same $\kappa_2$)",
        )
    cross = load("effective")["crossover_alpha"]["exact_orders_equal"] ** 2
    ax.axvline(cross, color=COLORS["ink"], lw=0.8, ls="--")
    ax.text(cross + 0.05, 3.7, "1st = 2nd order", fontsize=8, color=COLORS["ink"])
    for y, lab in ((2.0, r"$\gamma=2$ (deck bound)"), (math.log(4.2), "Réglade 2024: ln 4.2")):
        ax.axhline(y, color=COLORS["ink"], lw=0.8, ls=":")
        ax.text(0.55, y + 0.04, lab, fontsize=8, color=COLORS["ink"])
    ax.set_xlabel(r"$\alpha^2$")
    ax.set_ylabel(r"local exponent $-d\ln\Gamma_X/d(\alpha^2)$")
    ax.set_ylim(0.7, 4.2)
    ax.set_title(
        "Local bit-flip exponent (central differences; end points one-sided)\n"
        + CONVENTION
        + " (exponent is convention-free)",
        fontsize=10,
    )
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "task21_local_exponent.png", dpi=DPI)
    plt.close(fig)


def fig_buffer() -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    col = load("adiabatic")["collapse"]
    for alpha, mk in zip(COLLAPSE_ALPHAS, "os^", strict=True):
        pts = [q for q in col if q["alpha"] == alpha]
        axes[0].loglog(
            [q["barrier_photons"] for q in pts],
            [q["excess"] for q in pts],
            mk + "-",
            ms=6,
            label=rf"$\alpha$={alpha}",
            color=(COLORS["two_mode"], COLORS["effective"], COLORS["dubovitskii"])[
                COLLAPSE_ALPHAS.index(alpha)
            ],
        )
    axes[0].set_xlabel(r"barrier buffer occupation $n_b^*=4g_2^2\alpha^4/\kappa_b^2$")
    axes[0].set_ylabel(r"$\Gamma_X^{2\,mode}/\Gamma_X^{eff}-1$")
    axes[0].set_title(
        "(a) fixed $\\kappa_2$: $g_2\\to sg_2$, $\\kappa_b\\to s^2\\kappa_b$", fontsize=10
    )
    buf = load("buffer")
    ex = buf["extra_scan"]
    axes[1].plot(
        [q["trunc"]["nb"] + q["trunc"]["nb_extra"] for q in ex],
        [q["gamma_x"] for q in ex],
        "o-",
        color=COLORS["two_mode"],
        label=r"buffer levels at $n_a\approx0$ (nb=5 elsewhere)",
    )
    rect = buf["rect_scan"]
    axes[1].plot(
        [q["trunc"]["nb"] for q in rect],
        [q["gamma_x"] for q in rect],
        "s--",
        color=COLORS["effective"],
        label="rectangular buffer (all $n_a$)",
    )
    axes[1].set_xlabel("buffer levels kept")
    axes[1].set_ylabel(r"$\Gamma_X$ (rad/$\mu$s), $\alpha$=2.1")
    axes[1].set_title("(b) where the bit-flip path needs buffer room", fontsize=10)
    for tr, ls in zip(buf["saddle"], ("-", "--"), strict=True):
        s = tr["s"]
        axes[2].plot(
            tr["t"],
            tr["n_b"],
            ls,
            color=COLORS["two_mode"],
            label=rf"$\langle b^\dagger b\rangle$, s={s:g}",
        )
        axes[2].plot(
            tr["t"],
            tr["n_b_adiabatic"],
            ls,
            color=COLORS["effective"],
            label=rf"adiabatic $n_b$ from memory, s={s:g}",
        )
    axes[2].set_xlabel(r"time ($\mu$s), start at saddle $|0\rangle_a|0\rangle_b$, $\alpha$=2.1")
    axes[2].set_ylabel("buffer photons")
    axes[2].set_title("(c) the buffer lags the memory near the saddle", fontsize=10)
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7.5)
    fig.suptitle("R3 evidence: two-mode model vs adiabatic elimination. " + CONVENTION, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "task21_buffer.png", dpi=DPI)
    plt.close(fig)


def part_figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    FIGURES.mkdir(parents=True, exist_ok=True)
    d = series()
    fig_rates(d)
    fig_local_exponent(d)
    fig_buffer()
    print(f"wrote {FIGURES}/task21_*.png", flush=True)


PARTS = {
    "sweep": part_sweep,
    "large": part_large,
    "timedomain": part_timedomain,
    "effective": part_effective,
    "thermal": part_thermal,
    "thermalcheck": part_thermalcheck,
    "adiabatic": part_adiabatic,
    "buffer": part_buffer,
    "analyze": part_analyze,
    "figures": part_figures,
}


def main(argv: list[str]) -> None:
    dq.set_precision("double")
    names = list(PARTS) if argv == ["all"] else argv
    unknown = [n for n in names if n not in PARTS]
    if not names or unknown:
        raise SystemExit(f"usage: task21_bitflip.py PART..., PART in {list(PARTS)} or all")
    for name in names:
        print(f"=== {name}", flush=True)
        PARTS[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
