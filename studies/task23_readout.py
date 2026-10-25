"""Task 2.3: transmon-free memory-bit readout through the buffer (plan R7, R8).

Parts (each regenerates its JSON and figure from scratch):
  response  <b>(t) from |+-alpha> at alpha = 2.1, eps_Z / g2 in {0.05, 0.1, 0.2, 0.3}: full model
            (mesolve via catqubit.simulate.inflate) vs damped oscillator vs Liouvillian modes;
            1% settling time; monotonic or ringing.
  map       Regime map of the linear oscillator (zeta = kappa_b / (8 g2 alpha)) vs alpha and
            kappa_b around zeta = 1, with full-Liouvillian spot checks.
  errors    Exact assignment error eps(T) of the boxcar window [0, T] (tilted Liouvillian) at
            alpha = 1.2, 1.8, 2.1, vs Gaussian estimates and a Chernoff bound; optimal window.

Conventions: time in us, rates in rad/us, frequencies in MHz = rate / 2 pi. Gamma_X is the
decay rate of Re<a> (sum of both one-way rates); the one-way flip rate is Gamma_X / 2.
Ideal homodyne (eta = 1) at phi = 0, i.e. of Re<b>, the quadrature that carries the codeword.

Run: PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv313/bin/python -u \
         studies/task23_readout.py [response|map|errors|plots] [--workers N]
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import erfc  # noqa: E402

import catqubit.readout_stats as rs  # noqa: E402
from catqubit.liouvillian import FockTruncation, two_mode_liouvillian  # noqa: E402
from catqubit.params import CatParams, to_angular  # noqa: E402
from catqubit.readout_dynamics import (  # noqa: E402
    SLOW_SHIFT,
    LinearBufferResponse,
    buffer_spectrum,
    liouvillian_modes,
    settling_time_from_curve,
    simulate_buffer_response,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "task23"
FIGURES = ROOT / "figures" / "track2"
DPI = 150

EPS_FRACTIONS = (0.05, 0.1, 0.2, 0.3)
SETTLING_TOL = 0.01
# Response part: converged truncations (see the convergence block in the JSON).
RESPONSE_TRUNC = FockTruncation(na=24, nb=5)
RESPONSE_CHECK_TRUNCS = (FockTruncation(na=28, nb=5), FockTruncation(na=24, nb=6))
RESPONSE_CHECK_EPS = (0.05, 0.3)  # truncation checks at the smallest and largest drive
RESPONSE_T_MAX = 4.0
RESPONSE_DT = 0.002

# Regime map.
MAP_ALPHAS = np.linspace(0.6, 3.0, 121)
MAP_KAPPA_B_MHZ = np.geomspace(0.3, 15.0, 121)
MAP_CHECKS = ((1.0, 1.0), (1.0, 2.6), (1.0, 8.0), (1.5, 2.6), (1.5, 12.0), (2.1, 2.6), (2.1, 10.0))
MAP_CHECK_EPS = 0.05
MAP_CHECK_TRUNC = {
    1.0: FockTruncation(14, 4),
    1.5: FockTruncation(20, 4),
    2.1: FockTruncation(26, 5),
}

# Readout-error part: one readout drive for all alpha.
ERROR_EPS = 0.3
# Production truncations are rectangular (cost: one LU per Fourier mode on a loaded machine);
# the check truncations widen the buffer near the barrier, where bit flips need it.
T_SWITCH = 6.0  # fast modes decay at >= kappa_b / 4: e^{-4.08 * 6} ~ 2e-11
ERROR_SETUPS = {
    1.2: dict(
        trunc=FockTruncation(18, 4),
        check=FockTruncation(22, 6),
        times=np.concatenate([np.arange(0.25, 6.01, 0.25), [7, 8, 9, 10, 12, 14, 16]]),
    ),
    1.8: dict(
        trunc=FockTruncation(22, 4),
        check=FockTruncation(22, 4, 5, 10.0),
        times=np.concatenate([np.arange(1.0, 6.01, 0.5), [8, 10, 15, 20, 25, 30, 35, 40, 50, 60]]),
    ),
    2.1: dict(
        trunc=FockTruncation(26, 4),
        check=FockTruncation(26, 4, 7, 13.2),
        times=np.concatenate(
            [np.arange(1.0, 6.01, 0.5), [8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]]
        ),
    ),
}
CHERNOFF_S = np.geomspace(0.02, 1.5, 14)
# Convergence of the characteristic function: sample points k (times: 2 us, 8 us, T_max).
CHECK_KS = (0.05, 0.3, 1.0, 2.0)
SLOW_MODES = 6


def _trunc_dict(trunc: FockTruncation) -> dict:
    return dict(asdict(trunc), dim=trunc.dim)


def _save(name: str, payload: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(payload, indent=1, default=_json_default))
    print(f"wrote {RESULTS / name}", flush=True)


def _json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, complex | np.complexfloating):
        return [float(np.real(x)), float(np.imag(x))]
    if isinstance(x, np.floating | np.integer):
        return x.item()
    raise TypeError(type(x))


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


# ================================================================== part 1: response


def _oscillator_summary(p: CatParams, eps: float) -> dict:
    osc = LinearBufferResponse.from_params(p, eps, 1)
    return dict(
        nu=osc.nu,
        zeta=osc.zeta,
        damping_rate=osc.damping_rate,
        ringing_mhz=osc.ringing_frequency_mhz,
        steady_state_plus=osc.steady_state,
        settling_time=osc.settling_time(SETTLING_TOL),
        envelope_settling_time=osc.envelope_settling_time(SETTLING_TOL),
    )


def _spectral_summary(p: CatParams, trunc: FockTruncation, eps: float) -> dict:
    t0 = time.time()
    spec = buffer_spectrum(p, trunc, eps)
    plus, minus, diff = spec.response(1), spec.response(-1), spec.differential()
    return dict(
        trunc=_trunc_dict(trunc),
        ringing_eigenvalue=diff.ringing_eigenvalue,
        ringing_mhz=diff.ringing_frequency_mhz,
        decay_rate=diff.decay_rate,
        plateau_plus=plus.plateau,
        plateau_minus=minus.plateau,
        plateau_diff=diff.plateau,
        settling_time_diff=diff.settling_time(SETTLING_TOL),
        residual=spec.modes.residual,
        seconds=time.time() - t0,
        _diff=diff,
        _plus=plus,
        _minus=minus,
    )


def _timedomain(p: CatParams, trunc: FockTruncation, eps: float, tsave: np.ndarray) -> dict:
    t0 = time.time()
    _, bp = simulate_buffer_response(p, trunc.na, trunc.nb, eps, 1, tsave)
    _, bm = simulate_buffer_response(p, trunc.na, trunc.nb, eps, -1, tsave)
    return dict(b_plus=bp, b_minus=bm, seconds=time.time() - t0)


def run_response() -> None:
    p = CatParams.reglade()
    tsave = np.round(np.arange(0.0, RESPONSE_T_MAX + 1e-9, RESPONSE_DT), 10)
    out = dict(
        description="Task 2.3 buffer response <b>(t) from |+-alpha>|0>, alpha=2.1",
        params=asdict(p),
        settling_tol=SETTLING_TOL,
        t=tsave,
        cases={},
    )
    for frac in EPS_FRACTIONS:
        eps = frac * p.g2
        osc = _oscillator_summary(p, eps)
        spec = _spectral_summary(p, RESPONSE_TRUNC, eps)
        td = _timedomain(p, RESPONSE_TRUNC, eps, tsave)
        diff_td = (td["b_plus"] - td["b_minus"]) / 2
        plateau = spec["plateau_diff"]
        settle_td = settling_time_from_curve(tsave, diff_td.real, plateau.real, SETTLING_TOL)
        checks = {}
        for tr in RESPONSE_CHECK_TRUNCS if frac in RESPONSE_CHECK_EPS else ():
            s2 = _spectral_summary(p, tr, eps)
            td2 = _timedomain(p, tr, eps, tsave)
            checks[f"{tr.na}x{tr.nb}"] = dict(
                ringing_eigenvalue=s2["ringing_eigenvalue"],
                plateau_diff=s2["plateau_diff"],
                max_abs_db=float(np.max(np.abs(td2["b_plus"] - td["b_plus"]))),
            )
        diff_spec = spec["_diff"]
        out["cases"][str(frac)] = dict(
            eps_z=eps,
            oscillator=osc,
            spectral={k: v for k, v in spec.items() if not k.startswith("_")},
            timedomain=dict(
                trunc=_trunc_dict(RESPONSE_TRUNC),
                b_plus=td["b_plus"],
                b_minus=td["b_minus"],
                settling_time_diff=settle_td,
                plateau_rel_dev_end=float(abs(diff_td[-1] - plateau) / abs(plateau)),
                seconds=td["seconds"],
                overshoot=float(np.max(-diff_td.real) / abs(plateau.real) - 1),
            ),
            oscillator_curve=LinearBufferResponse.from_params(p, eps, 1).mean_b(tsave),
            spectral_curve_diff=diff_spec.mean_b(tsave),
            max_dev_oscillator=float(
                np.max(np.abs(LinearBufferResponse.from_params(p, eps, 1).mean_b(tsave) - diff_td))
                / abs(plateau)
            ),
            max_dev_spectral_after_0p3us=float(
                np.max(np.abs(diff_spec.mean_b(tsave) - diff_td)[tsave >= 0.3]) / abs(plateau)
            ),
            convergence=checks,
        )
        c = out["cases"][str(frac)]
        print(
            f"eps={frac:.2f}g2 osc f={osc['ringing_mhz']:.3f}MHz ts={osc['settling_time']:.3f} | "
            f"spec f={spec['ringing_mhz']:.3f} rate={spec['decay_rate']:.3f} "
            f"plateau={plateau:.5f} ts={spec['settling_time_diff']:.3f} | td ts={settle_td:.3f} "
            f"overshoot={c['timedomain']['overshoot']:.3f} dev_osc={c['max_dev_oscillator']:.3e}",
            flush=True,
        )
    _save("response.json", out)


def plot_response() -> None:
    d = _load("response.json")
    t = np.array(d["t"])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (frac, c) in zip(axes.ravel(), d["cases"].items(), strict=True):
        bss = abs(c["oscillator"]["steady_state_plus"])
        bp = (
            np.array(c["timedomain"]["b_plus"])[:, 0]
            + 1j * np.array(c["timedomain"]["b_plus"])[:, 1]
        )
        bm = (
            np.array(c["timedomain"]["b_minus"])[:, 0]
            + 1j * np.array(c["timedomain"]["b_minus"])[:, 1]
        )
        osc = np.array(c["oscillator_curve"])
        spec = np.array(c["spectral_curve_diff"])
        spec = spec[:, 0] + 1j * spec[:, 1]
        ax.plot(t, bp.real / bss, color="C0", lw=2, label=r"full model, $|{+}\alpha\rangle$")
        ax.plot(t, bm.real / bss, color="C1", lw=2, label=r"full model, $|{-}\alpha\rangle$")
        ax.plot(t, osc / bss, "k--", lw=1, label="damped oscillator (Eq. 15b)")
        ax.plot(t, -osc / bss, "k--", lw=1)
        ax.plot(t, spec.real / bss, ":", color="C3", lw=1.5, label="Liouvillian modes")
        ax.plot(t, -spec.real / bss, ":", color="C3", lw=1.5)
        ts = c["timedomain"]["settling_time_diff"]
        ax.axvline(ts, color="0.5", lw=0.8)
        ax.axhspan(1 - SETTLING_TOL, 1 + SETTLING_TOL, color="0.85")
        ax.axhspan(-1 - SETTLING_TOL, -1 + SETTLING_TOL, color="0.85")
        ax.set_title(
            rf"$\epsilon_Z = {float(frac):.2f}\,g_2$: settles (1%) at {ts:.2f} µs, "
            rf"overshoot {100 * c['timedomain']['overshoot']:.0f}%",
            fontsize=9,
        )
        ax.set_ylabel(r"Re$\langle b\rangle$ / $(\epsilon_Z / 2\alpha g_2)$")
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("time t (µs)")
    axes[0, 0].legend(fontsize=7, loc="center right")
    fig.suptitle(
        r"Task 2.3: buffer response at $\alpha = 2.1$ (Réglade parameters); grey band: ±1% of "
        r"$\mp\epsilon_Z/(2\alpha g_2)$",
        fontsize=10,
    )
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "task23_response.png", dpi=DPI)
    plt.close(fig)


# ================================================================== part 2: regime map


def _osc_map(p0: CatParams) -> tuple[np.ndarray, np.ndarray]:
    zeta = np.zeros((MAP_KAPPA_B_MHZ.size, MAP_ALPHAS.size))
    settle = np.zeros_like(zeta)
    for i, kb in enumerate(MAP_KAPPA_B_MHZ):
        for j, a in enumerate(MAP_ALPHAS):
            p = CatParams(g2=p0.g2, kappa_b=to_angular(kb), kappa_a=p0.kappa_a, alpha=a)
            osc = LinearBufferResponse.from_params(p, MAP_CHECK_EPS * p.g2, 1)
            zeta[i, j], settle[i, j] = osc.zeta, osc.settling_time(SETTLING_TOL)
    return zeta, settle


def run_map() -> None:
    p0 = CatParams.reglade()
    zeta, settle = _osc_map(p0)
    checks = []
    for a, kb in MAP_CHECKS:
        p = CatParams(g2=p0.g2, kappa_b=to_angular(kb), kappa_a=p0.kappa_a, alpha=a)
        eps = MAP_CHECK_EPS * p.g2
        osc = LinearBufferResponse.from_params(p, eps, 1)
        spec = buffer_spectrum(p, MAP_CHECK_TRUNC[a], eps).differential()
        checks.append(
            dict(
                alpha=a,
                kappa_b_mhz=kb,
                zeta=osc.zeta,
                osc_settling=osc.settling_time(SETTLING_TOL),
                spec_settling=spec.settling_time(SETTLING_TOL),
                osc_eigenvalue=osc.eigenvalues[0],
                spec_slowest=complex(spec.values[np.argmax(spec.values.real)]),
                spec_dominant=spec.ringing_eigenvalue if np.any(spec.values.imag > 0) else None,
                trunc=_trunc_dict(MAP_CHECK_TRUNC[a]),
            )
        )
        print(checks[-1], flush=True)
    col = np.argmin(abs(MAP_ALPHAS - 2.1))
    best = int(np.argmin(settle[:, col]))
    _save(
        "settling_map.json",
        dict(
            description="Oscillator settling (1%) vs alpha and kappa_b; g2 fixed (Réglade)",
            g2=p0.g2,
            alphas=MAP_ALPHAS,
            kappa_b_mhz=MAP_KAPPA_B_MHZ,
            zeta=zeta,
            settling_time=settle,
            fastest_at_alpha_2p1=dict(
                kappa_b_mhz=MAP_KAPPA_B_MHZ[best], zeta=zeta[best, col], settling=settle[best, col]
            ),
            reglade=dict(alpha=2.1, kappa_b_mhz=2.6, zeta=p0.kappa_b / (8 * p0.g2 * 2.1)),
            checks=checks,
        ),
    )


def plot_map() -> None:
    d = _load("settling_map.json")
    a, kb = np.array(d["alphas"]), np.array(d["kappa_b_mhz"])
    zeta, settle = np.array(d["zeta"]), np.array(d["settling_time"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    mesh = ax.pcolormesh(a, kb, settle, shading="auto", norm=matplotlib.colors.LogNorm())
    fig.colorbar(mesh, ax=ax, label="1% settling time of <b> (µs)")
    cs = ax.contour(a, kb, zeta, levels=[0.2, 0.5, 1.0, 2.0], colors="w", linewidths=0.8)
    ax.clabel(cs, fmt=lambda z: rf"$\zeta$={z:g}", fontsize=7)
    ax.plot(2.1, 2.6, "r*", ms=12, label="Réglade (α=2.1, κ_b/2π=2.6 MHz)")
    for c in d["checks"]:
        ax.plot(c["alpha"], c["kappa_b_mhz"], "wo", mfc="none", ms=7)
    ax.set_yscale("log")
    ax.set_xlabel(r"cat amplitude $\alpha$")
    ax.set_ylabel(r"$\kappa_b / 2\pi$ (MHz)")
    ax.set_title(
        r"$\zeta = \kappa_b/(8 g_2\alpha)$; ringing for $\zeta<1$ ($g_2/2\pi$ = 0.763 MHz)",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="lower right")
    col = np.argmin(abs(a - 2.1))
    ax2.loglog(zeta[:, col], settle[:, col], "k-", label=r"oscillator, $\alpha=2.1$")
    for c in d["checks"]:
        ax2.loglog(
            c["zeta"],
            c["spec_settling"],
            "o",
            mfc="none",
            label=f"Liouvillian modes, α={c['alpha']}, κ_b/2π={c['kappa_b_mhz']} MHz",
        )
    ax2.axvline(1.0, color="0.5", lw=0.8)
    ax2.axvline(d["reglade"]["zeta"], color="r", lw=0.8, ls="--")
    ax2.set_xlabel(r"damping ratio $\zeta$")
    ax2.set_ylabel("1% settling time (µs)")
    ax2.set_title("settling vs damping ratio (spot checks: full two-mode L)", fontsize=9)
    ax2.legend(fontsize=6)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES / "task23_settling_map.png", dpi=DPI)
    plt.close(fig)


# ================================================================== part 3: readout error


def _driven_flip_rate(p: CatParams, trunc: FockTruncation, eps: float) -> float:
    """Gamma_X(eps_z): slowest nonzero rate of the full L (eigenvalue convention)."""
    modes = liouvillian_modes(two_mode_liouvillian(p, trunc, eps_z=eps), SLOW_SHIFT, 3)
    rates = -modes.values.real
    return float(np.sort(rates[np.abs(modes.values) > 1e-12])[0])


def _signal_difference(p: CatParams, trunc: FockTruncation, eps: float, t: np.ndarray):
    """(<b>+ - <b>-)(t) on a uniform grid t (starting at 0) by exact propagation of L."""
    from scipy.sparse.linalg import expm_multiply

    from catqubit.liouvillian import two_mode_operators, vec

    big_l = two_mode_liouvillian(p, trunc, eps_z=eps).tocsr()
    v0 = np.column_stack([rs.codeword_state(p, trunc, 1), rs.codeword_state(p, trunc, -1)])
    states = expm_multiply(big_l, v0, start=t[0], stop=t[-1], num=t.size, endpoint=True)
    _, b = two_mode_operators(trunc)
    row = vec(b.toarray().T)
    return states[:, :, 0] @ row - states[:, :, 1] @ row


def _matched_filter(p: CatParams, trunc: FockTruncation, eps: float, times, t_switch) -> dict:
    """Shot-noise Gaussian estimates. Delta<b> is exact up to t_switch, then held at its
    metastable plateau (the estimate ignores bit flips by construction)."""
    t = np.linspace(0.0, t_switch, 4001)
    delta = _signal_difference(p, trunc, eps, t)
    integrand = np.abs(delta) ** 2
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(t))])
    snr = np.interp(np.minimum(times, t_switch), t, cum)
    snr = snr + integrand[-1] * np.clip(times - t_switch, 0, None)
    return dict(
        error=0.5 * erfc(np.sqrt(p.kappa_b / 2 * snr)),
        delta_b_plateau=complex(delta[-1]),
        t=t[::40],
        delta_b=delta[::40],
    )


def _chi_convergence(setup: dict, p: CatParams, eps: float, workers: int) -> dict:
    """Generating function at sample (s, T): production vs larger truncation, and the
    from_start projection vs exact propagation to t_switch (which measures the dropped part)."""
    s_values = [1j * k for k in CHECK_KS] + [0.5]
    times = np.array([2.0, 8.0, float(setup["times"][-1])])
    vals = {}
    for name in ("trunc", "check"):
        tr = setup[name]
        gen = rs.cat_readout_generator(p, tr, eps)
        rhos = [rs.codeword_state(p, tr, 1), rs.codeword_state(p, tr, -1)]
        for from_start in (True, False) if name == "trunc" else (True,):
            slow = rs.SlowModes(T_SWITCH, SLOW_MODES, from_start=from_start)
            z, err = rs._generating_function_and_error(
                gen, rhos, s_values, times, slow=slow, workers=workers
            )
            vals[(name, from_start)] = (z, err)
    z, z_check, z_exact = (vals[k][0] for k in (("trunc", True), ("check", True), ("trunc", False)))
    trunc_diff = np.abs(z - z_check)
    start_diff = np.abs(z - z_exact)
    return dict(
        s_values=s_values,
        times=times,
        check_trunc=_trunc_dict(setup["check"]),
        max_abs_diff_chi_truncation=float(np.max(trunc_diff[:, :-1])),
        max_abs_diff_chi_truncation_per_time=np.max(trunc_diff[:, :-1], axis=(0, 1)),
        max_rel_diff_real_s_truncation=float(np.max(trunc_diff[:, -1] / np.abs(z_check[:, -1]))),
        max_abs_diff_chi_from_start=float(np.max(start_diff[:, :-1])),
        max_rel_diff_real_s_from_start=float(np.max(start_diff[:, -1] / np.abs(z_exact[:, -1]))),
        switch_error_exact_path=vals[("trunc", False)][1],
    )


def run_errors(workers: int, alphas: tuple[float, ...]) -> None:
    path = RESULTS / "error_vs_T.json"
    data = json.loads(path.read_text()) if path.exists() else dict(per_alpha={})
    data.update(
        description=(
            "Exact assignment error of the boxcar window [0,T] (tilted Liouvillian, Fourier "
            "inversion), Gaussian estimates and Chernoff bound. Ideal homodyne of Re<b>."
        ),
        eps_z_over_g2=ERROR_EPS,
        convention="Gamma_X = decay rate of Re<a> (sum of one-way rates); one-way = Gamma_X/2",
    )
    for alpha in alphas:
        data["per_alpha"][str(alpha)] = _errors_for_alpha(alpha, workers)
        _save("error_vs_T.json", data)


def _errors_for_alpha(alpha: float, workers: int) -> dict:
    setup = ERROR_SETUPS[alpha]
    p = CatParams.reglade().with_alpha(alpha)
    eps, trunc, times = ERROR_EPS * p.g2, setup["trunc"], np.asarray(setup["times"], float)
    gen = rs.cat_readout_generator(p, trunc, eps)
    rp, rm = rs.codeword_state(p, trunc, 1), rs.codeword_state(p, trunc, -1)
    slow = rs.SlowModes(T_SWITCH, SLOW_MODES, from_start=True)
    t0 = time.time()
    res = rs.readout_error(gen, rp, rm, times, slow=slow, workers=workers)
    t_exact = time.time() - t0
    print(
        f"alpha={alpha}: readout_error {t_exact:.0f}s switch_error={res.switch_error:.2e}",
        flush=True,
    )
    t0 = time.time()
    chern = rs.chernoff_bound(
        gen, rp, rm, times, res.thresholds, CHERNOFF_S, slow=slow, workers=workers
    )
    t_chern = time.time() - t0
    gamma = _driven_flip_rate(p, trunc, eps)
    gamma0 = _driven_flip_rate(p, trunc, 0.0)
    gamma_check = _driven_flip_rate(p, setup["check"], eps)
    mf = _matched_filter(p, trunc, eps, times, T_SWITCH)
    shot = 0.5 * erfc(np.abs(res.mean[0] - res.mean[1]) / (2 * np.sqrt(2 * times)))
    best = int(np.argmin(res.error_threshold))
    keep_t = {2.0, T_SWITCH, float(times[best]), float(times[-1])}
    dists = [
        dict(time=d.time, q=d.q[::2], p_plus=d.p_plus[::2], p_minus=d.p_minus[::2])
        for d in res.distributions
        if d.time in keep_t
    ]
    conv = _chi_convergence(setup, p, eps, workers)
    print(
        f"alpha={alpha}: chernoff {t_chern:.0f}s, Gamma_X(eps)={gamma:.3e}, "
        f"conv={conv['max_abs_diff_chi_truncation']:.2e}",
        flush=True,
    )
    for j, t in enumerate(times):
        print(
            f"  T={t:6.2f} exact={res.error_threshold[j]:.4e} bayes={res.error_bayes[j]:.4e} "
            f"gauss={res.error_gauss[j]:.4e} shot={shot[j]:.4e} matched={mf['error'][j]:.4e} "
            f"chernoff={chern[j]:.3e} var/T={res.var[0, j] / t:.3f}",
            flush=True,
        )
    return dict(
        alpha=alpha,
        params=asdict(p),
        eps_z=eps,
        trunc=_trunc_dict(trunc),
        t_switch=T_SWITCH,
        slow_modes=SLOW_MODES,
        times=times,
        mean=res.mean,
        var=res.var,
        error_threshold=res.error_threshold,
        error_bayes=res.error_bayes,
        thresholds=res.thresholds,
        error_gauss_moments=res.error_gauss,
        error_gauss_shot_noise=shot,
        error_matched_filter=mf["error"],
        delta_b_plateau=mf["delta_b_plateau"],
        chernoff=chern,
        chernoff_s=CHERNOFF_S,
        gamma_x_driven=gamma,
        gamma_x_undriven=gamma0,
        gamma_x_driven_check_trunc=gamma_check,
        telegraph_estimate=gamma * times / 4,
        optimal_window=dict(T=float(times[best]), error=float(res.error_threshold[best])),
        switch_error=res.switch_error,
        seconds=dict(exact=t_exact, chernoff=t_chern),
        convergence=conv,
        distributions=dists,
    )


def plot_errors() -> None:
    d = _load("error_vs_T.json")["per_alpha"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, (alpha, r) in enumerate(sorted(d.items())):
        t = np.array(r["times"])
        col = f"C{i}"
        ax.loglog(t, r["error_threshold"], "-", color=col, lw=2, label=rf"exact, $\alpha$={alpha}")
        ax.loglog(
            t, r["error_gauss_moments"], "--", color=col, lw=1, label="Gaussian, exact moments"
        )
        ax.loglog(
            t,
            r["error_matched_filter"],
            ":",
            color=col,
            lw=1.5,
            label=r"½erfc$\sqrt{(\kappa_b/2)\int|\Delta b|^2}$",
        )
        ax.loglog(t, r["chernoff"], "^", color=col, ms=3, label="Chernoff bound")
        ax.loglog(t, r["telegraph_estimate"], "-.", color="0.5", lw=0.8)
        opt = r["optimal_window"]
        ax.plot(opt["T"], opt["error"], "*", color=col, ms=12, mec="k")
    ax.set_ylim(1e-9, 0.6)
    ax.set_xlabel("window length T (µs)")
    ax.set_ylabel("assignment error ε(T)")
    ax.set_title(
        rf"$\epsilon_Z$ = {ERROR_EPS} $g_2$, boxcar, ideal homodyne of Re⟨b⟩; ★ optimal window; "
        r"grey: $\Gamma_X T/4$ (one-way rate $\Gamma_X/2$)",
        fontsize=8,
    )
    ax.grid(alpha=0.3, which="both")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:8], labels[:8], fontsize=6, ncol=2, loc="lower left")
    r = d[min(d, key=float)]
    for j, dist in enumerate(r["distributions"]):
        q, pp = np.array(dist["q"]), np.array(dist["p_plus"])
        k = list(r["times"]).index(dist["time"])
        mu, var = r["mean"][0][k], r["var"][0][k]
        gauss = np.exp(-((q - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
        ax2.semilogy(
            q, np.clip(pp, 1e-14, None), color=f"C{j}", label=f"exact, T = {dist['time']:g} µs"
        )
        ax2.semilogy(q, gauss, "--", color=f"C{j}", lw=0.8)
    ax2.set_ylim(1e-10, 2)
    ax2.set_xlabel("integrated record Q (√µs units: dY = ⟨c+c†⟩dt + dW)")
    ax2.set_ylabel(r"P(Q | +α)")
    ax2.set_title(
        rf"$\alpha$ = {r['alpha']}: exact P(Q|+α) (solid) vs Gaussian, same moments (dashed)",
        fontsize=8,
    )
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "task23_error_vs_T.png", dpi=DPI)
    plt.close(fig)


def run_error_check(workers: int, alphas: tuple[float, ...]) -> None:
    """eps(T) at the check truncation, at a short window and at the optimal window."""
    errors = _load("error_vs_T.json")["per_alpha"]
    out = dict(description="Direct truncation check of eps(T): production vs check truncation")
    for alpha in alphas:
        r = errors[str(alpha)]
        setup = ERROR_SETUPS[alpha]
        p = CatParams.reglade().with_alpha(alpha)
        eps, tr = ERROR_EPS * p.g2, setup["check"]
        times = np.array(sorted({3.0, r["optimal_window"]["T"]}))
        gen = rs.cat_readout_generator(p, tr, eps)
        rp, rm = rs.codeword_state(p, tr, 1), rs.codeword_state(p, tr, -1)
        slow = rs.SlowModes(T_SWITCH, SLOW_MODES, from_start=True)
        res = rs.readout_error(gen, rp, rm, times, slow=slow, workers=workers)
        prod = [r["error_threshold"][list(r["times"]).index(t)] for t in times]
        out[str(alpha)] = dict(
            check_trunc=_trunc_dict(tr),
            times=times,
            error_check=res.error_threshold,
            error_production=prod,
            rel_diff=np.abs(res.error_threshold - prod) / res.error_threshold,
        )
        print(alpha, out[str(alpha)], flush=True)
    _save("error_check.json", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["response", "map", "errors", "errorcheck", "plots"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--alphas", type=float, nargs="*", default=sorted(ERROR_SETUPS))
    args = parser.parse_args()
    if args.part == "response":
        run_response()
        plot_response()
    elif args.part == "map":
        run_map()
        plot_map()
    elif args.part == "errorcheck":
        run_error_check(args.workers, tuple(a for a in args.alphas if a < 2.0))
    elif args.part == "errors":
        run_errors(args.workers, tuple(args.alphas))
        plot_errors()
    else:
        plot_response()
        plot_map()
        plot_errors()


if __name__ == "__main__":
    main()
