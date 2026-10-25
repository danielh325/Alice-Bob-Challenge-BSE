"""Task 2.4(c-d): time-domain proof of the two proposed readouts, with no surrogate model.

Everything here is a direct simulation of the full Hamiltonian and dissipators; nothing is taken
from the Liouvillian gaps or the classical record model of task24c/task24d.

  records   Stochastic Schroedinger trajectories with a homodyne record (dynamiqs dssesolve).
            Detector efficiency eta is exact: the measured channel is split into sqrt(eta) L
            (recorded) and sqrt(1 - eta) L (recorded, then discarded), which reproduces the
            statistics of the inefficient stochastic master equation for the kept record.
            Boxcar assignment error vs integration time is counted from the trajectories.
              eps_Z   memory drive eps_Z (a + a^dag), buffer output measured (Section 2.4 baseline)
              conv    beam splitter g_cr (a c^dag + h.c.) to a readout mode c with kappa_c = kappa_b,
                      output of c measured; no eps_Z drive.
              inflate eps_Z readout while alpha is ramped 2.1 -> 2.5 -> 2.1, eps_Z scaled with alpha so
                      the buffer response is constant; ramps are part of the read window.
  flips     Master-equation runs of the full read cycle (drive on, drive off, settle). The
            drive-induced flip probability is p = (1 - X_read(T) / X_idle(T)) / 2, X = Re<a>,
            against the same schedule without the read drive. This is the "Gamma_X is not
            degraded" check, including switching the conversion coupling off and the alpha ramps.

Run:  PYTHONPATH=. .venv313/bin/python -u studies/task24e_time_domain.py [stage ...]
Writes results/task24e/<stage>.json, figures/track2/task24e_*.png, results/task24e/summary.md.
"""

import json
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import dynamiqs as dq  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import erfc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catqubit.params import CatParams  # noqa: E402

dq.set_progress_meter(False)
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "task24e"
FIGURES = ROOT / "figures" / "track2"
DPI = 150

ETA = 0.6
EPS_Z_OVER_G2 = 0.3
KAPPA_RO_OVER_K2 = 0.1
ALPHA_INFLATED = 2.5
T_RAMP = 1.0  # us, alpha ramp for inflate-then-read
T_EDGE = 0.05  # us, rise/fall of eps_Z and g_cr
DT = 1e-3  # us, Rouchon1 step (normalized: unnormalized steps corrupt the alpha = 2.1 record)
DT_SAVE = 0.02  # us, record bin
N_TRAJ = 4000  # per codeword
BATCH = 1000
SOLVER_TOL = dict(rtol=1e-9, atol=1e-9)

# (alpha, na, nb, nc): truncations from studies/task24_tradeoff.py; nc = 4 for the readout mode
# (mean photon number 4 g_cr^2 alpha^2 / kappa_c^2 <= 0.2 here).
TRUNC = {1.4: (20, 5, 4), 2.1: (31, 5, 4), 2.5: (37, 6, 4)}

# Record runs: (key, method, alpha, read time in us, trajectories per codeword).
RECORD_RUNS = (
    ("conv_1.4", "conv", 1.4, 3.0, N_TRAJ),
    ("conv_2.1", "conv", 2.1, 3.0, N_TRAJ),
    ("eps_z_1.4", "eps_z", 1.4, 12.0, N_TRAJ),
    ("eps_z_2.1", "eps_z", 2.1, 40.0, N_TRAJ // 2),
    ("inflate_2.1", "inflate", 2.1, 40.0, N_TRAJ // 2),
)
# Flip runs: (key, method, alpha, read time, settle time).
FLIP_RUNS = (
    ("eps_z_1.4", "eps_z", 1.4, 12.0, 4.0),
    ("conv_1.4", "conv", 1.4, 2.0, 4.0),
    ("eps_z_2.1", "eps_z", 2.1, 31.0, 4.0),
    ("conv_2.1", "conv", 2.1, 1.2, 4.0),
    ("inflate_2.1", "inflate", 2.1, 31.0, 4.0),
)


def kappa2(p: CatParams) -> float:
    return 4 * p.g2**2 / p.kappa_b


def g_cr(p: CatParams) -> float:
    return math.sqrt(KAPPA_RO_OVER_K2 * kappa2(p) * p.kappa_b / 4)


def smooth_window(t, t_on, t_off, edge=T_EDGE):
    """1 on [t_on, t_off] with cos^2 edges of length `edge`; 0 outside."""
    rise = jnp.clip((t - t_on) / edge, 0.0, 1.0)
    fall = jnp.clip((t_off - t) / edge, 0.0, 1.0)
    return jnp.sin(0.5 * jnp.pi * rise) ** 2 * jnp.sin(0.5 * jnp.pi * fall) ** 2


def alpha_profile(alpha0: float, t_read: float):
    """alpha(t) for inflate-then-read: cos^2 ramp up over T_RAMP, hold, ramp down, ending at t_read."""

    def alpha_t(t):
        up = jnp.sin(0.5 * jnp.pi * jnp.clip(t / T_RAMP, 0.0, 1.0)) ** 2
        down = jnp.sin(0.5 * jnp.pi * jnp.clip((t_read - t) / T_RAMP, 0.0, 1.0)) ** 2
        return alpha0 + (ALPHA_INFLATED - alpha0) * up * down

    return alpha_t


# --------------------------------------------------------------------- model


def build(method: str, alpha: float, t_read: float, read_on: bool = True):
    """Hamiltonian, jump list (measured channel first, split by eta), initial states, operators."""
    p = CatParams.reglade().with_alpha(alpha)
    na, nb, nc = TRUNC[ALPHA_INFLATED if method == "inflate" else alpha]
    if method != "conv":
        nc = 1
    a, b, c = dq.destroy(na, nb, nc)
    pump = p.g2 * (a @ a @ b.dag() + a.dag() @ a.dag() @ b)
    window = lambda t: smooth_window(t, 0.0, t_read) * (1.0 if read_on else 0.0)  # noqa: E731
    if method == "inflate":
        alpha_t = alpha_profile(alpha, t_read)
        h = pump + dq.modulated(lambda t: -p.g2 * alpha_t(t) ** 2, b + b.dag())
        h = h + dq.modulated(lambda t: EPS_Z_OVER_G2 * p.g2 * alpha_t(t) / alpha * window(t), a + a.dag())
    else:
        h = pump - p.g2 * alpha**2 * (b + b.dag())
        if method == "eps_z":
            h = h + dq.modulated(lambda t: EPS_Z_OVER_G2 * p.g2 * window(t), a + a.dag())
        else:
            h = h + dq.modulated(lambda t: g_cr(p) * window(t), a @ c.dag() + a.dag() @ c)
    # c ~ -i (2 g_cr / kappa_c) a, so i c reads +alpha as a positive signal.
    measured = 1j * c if method == "conv" else b
    jumps = [math.sqrt(ETA * p.kappa_b) * measured, math.sqrt((1 - ETA) * p.kappa_b) * measured]
    if method == "conv":
        jumps.append(math.sqrt(p.kappa_b) * b)
    jumps.append(math.sqrt(p.kappa_a) * a)
    psis = [dq.tensor(dq.coherent(na, s * alpha), dq.basis(nb, 0), dq.basis(nc, 0)) for s in (1, -1)]
    return h, jumps, psis, a


# --------------------------------------------------------------------- records


def simulate_records(method: str, alpha: float, t_read: float, n_traj: int = N_TRAJ, seed: int = 0):
    """Kept-record bins (2 codewords, n_traj, n_bins) of dY/dt for the eta-efficient channel."""
    h, jumps, psis, _ = build(method, alpha, t_read)
    tsave = tuple(float(x) for x in np.round(np.arange(int(round(t_read / DT_SAVE)) + 1) * DT_SAVE, 9))
    out = []
    for start in range(0, n_traj, BATCH):
        keys = jax.random.split(jax.random.key(seed * 1_000_003 + start), min(BATCH, n_traj - start))
        r = dq.dssesolve(h, jumps, dq.stack(psis), tsave, keys,
                         method=dq.method.Rouchon1(dt=DT, normalize=True), save_states=False)
        out.append(np.asarray(r.measurements)[:, :, 0, :])
    return np.concatenate(out, axis=1)


def boxcar_errors(records: np.ndarray, times: np.ndarray):
    """Counted and Gaussian-fit boxcar assignment error vs integration time (threshold at midpoint)."""
    cum = np.cumsum(records, axis=-1) * DT_SAVE  # (2, n, n_bins)
    rows = []
    for k, t in enumerate(times):
        plus, minus = cum[0, :, k], cum[1, :, k]
        mid = 0.5 * (plus.mean() + minus.mean())
        sign = np.sign(plus.mean() - minus.mean())
        counted = 0.5 * (np.mean(sign * (plus - mid) < 0) + np.mean(sign * (minus - mid) > 0))
        sigma = math.sqrt(0.5 * (plus.var() + minus.var()))
        sep = abs(plus.mean() - minus.mean())
        rows.append(dict(t_us=float(t), counted=float(counted), gaussian=float(0.5 * erfc(sep / (2 * sigma) / math.sqrt(2))),
                         snr2=float((sep / sigma) ** 2)))
    return rows


def predicted_mu2(method: str, alpha: float) -> float:
    """mu^2 of the record model (catqubit.readout_protocol): eta kappa (2 x_ss)^2 for the measured output."""
    p = CatParams.reglade().with_alpha(alpha)
    if method == "conv":
        return 4 * ETA * KAPPA_RO_OVER_K2 * kappa2(p) * alpha**2
    return ETA * 4 * p.kappa_b * (EPS_Z_OVER_G2 / (2 * alpha)) ** 2


def measured_mu2(run: dict) -> float:
    """mu^2 from the trajectories: (separation / sigma)^2 of the integrated record grows as 4 mu^2 t.

    Fitted on 10-30% of the window, after the ring-up and before drive-induced flips shrink the
    separation (at alpha = 1.4 the eps_Z run flips ~40% of shots by its end).
    """
    rows = [r for r in run["errors"] if 0.1 * run["t_read_us"] <= r["t_us"] <= 0.3 * run["t_read_us"]]
    return float(np.polyfit([r["t_us"] for r in rows], [r["snr2"] for r in rows], 1)[0] / 4)


def stage_records() -> None:
    out = load("records") if (RESULTS / "records.json").exists() else {}
    for key, method, alpha, t_read, n_traj in RECORD_RUNS:
        t0 = time.time()
        rec = simulate_records(method, alpha, t_read, n_traj)
        n_bins = rec.shape[-1]
        times = np.arange(1, n_bins + 1) * DT_SAVE
        pick = np.unique(np.round(np.geomspace(1, n_bins, 40)).astype(int)) - 1
        rows = boxcar_errors(rec, times)
        out[key] = dict(method=method, alpha=alpha, t_read_us=t_read, eta=ETA, n_traj=n_traj,
                        errors=[rows[i] for i in pick],
                        mean_record=[rec[0].mean(axis=0)[::5].tolist(), rec[1].mean(axis=0)[::5].tolist()],
                        example=[rec[0, 0, ::5].tolist(), rec[1, 0, ::5].tolist()])
        save("records", out)
        print(f"records {key}: mu2 {measured_mu2(out[key]):.3f}/us, {time.time() - t0:.0f} s", flush=True)


# --------------------------------------------------------------------- flips


def final_x(method: str, alpha: float, t_read: float, t_settle: float, read_on: bool) -> float:
    h, jumps, psis, a = build(method, alpha, t_read, read_on)
    tsave = (0.0, t_read + t_settle)
    r = dq.mesolve(h, jumps, psis[0], tsave, exp_ops=[a], method=dq.method.Tsit5(**SOLVER_TOL))
    return float(np.real(np.asarray(r.expects)[0, -1]))


def stage_flips() -> None:
    # p_flip ~ 1e-5 is a 1e-5 relative change of Re<a>: float32 cannot resolve it.
    dq.set_precision("double")
    out = load("flips") if (RESULTS / "flips.json").exists() else {}
    for key, method, alpha, t_read, t_settle in FLIP_RUNS:
        t0 = time.time()
        x_read = final_x(method, alpha, t_read, t_settle, True)
        x_idle = final_x(method, alpha, t_read, t_settle, False)
        x_static = final_x("eps_z", alpha, t_read, t_settle, False) if method == "inflate" else x_idle
        out[key] = dict(method=method, alpha=alpha, t_read_us=t_read, t_settle_us=t_settle,
                        x_read=x_read, x_idle=x_idle, x_no_ramp=x_static,
                        p_flip_drive=(1 - x_read / x_idle) / 2,
                        p_flip_ramps=(1 - x_idle / x_static) / 2)
        save("flips", out)
        print(f"flips {key}: p_drive {out[key]['p_flip_drive']:.2e}, p_ramps {out[key]['p_flip_ramps']:.2e}, "
              f"{time.time() - t0:.0f} s", flush=True)


# --------------------------------------------------------------------- output


def save(stage: str, data) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{stage}.json").write_text(json.dumps(data, indent=1))


def load(stage: str):
    return json.loads((RESULTS / f"{stage}.json").read_text())


LABELS = {"eps_z": "ε_Z readout", "conv": f"conversion, κ_ro={KAPPA_RO_OVER_K2}κ₂",
          "inflate": f"inflate-then-read (→{ALPHA_INFLATED})"}
COLORS = {"eps_z": "k", "conv": "tab:purple", "inflate": "tab:orange"}


def plot() -> None:
    rec = load("records")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, alpha in zip(axes, (1.4, 2.1), strict=True):
        for d in rec.values():
            if d["alpha"] != alpha:
                continue
            t = [r["t_us"] for r in d["errors"]]
            ax.loglog(t, [max(r["counted"], 0.5 / d["n_traj"]) for r in d["errors"]], "o", ms=3,
                      color=COLORS[d["method"]], label=f"{LABELS[d['method']]} (counted)")
            ax.loglog(t, [r["gaussian"] for r in d["errors"]], "-", color=COLORS[d["method"]], lw=1)
        ax.axhline(1e-3, color="gray", ls=":", lw=1)
        ax.set(xlabel="integration time (µs)", ylabel="boxcar assignment error",
               title=f"Stochastic trajectories, α = {alpha}, η = {ETA}", ylim=(1e-6, 0.6))
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "task24e_trajectories.png", dpi=DPI)


def write_summary() -> None:
    rec, flips = load("records"), load("flips")
    lines = ["# Task 2.4(c-d): time-domain proof of the proposed readouts", "",
             f"Full-model stochastic trajectories (eta = {ETA}, {N_TRAJ // 2}-{N_TRAJ} per codeword, boxcar filter) and",
             "master-equation read cycles. No surrogate model or Liouvillian rate is used here.", "",
             "## Records", "",
             "| Run | alpha | mu^2 trajectories / record model (1/us) | T for error <= 1e-3 (us) | error at T_end (counted / Gaussian) |",
             "|---|---|---|---|---|"]
    for d in rec.values():
        hit = next((r["t_us"] for r in d["errors"] if r["counted"] <= 1e-3 and r["gaussian"] <= 1e-3), None)
        last = d["errors"][-1]
        lines.append(f"| {LABELS[d['method']]} | {d['alpha']} | {measured_mu2(d):.3f} / {predicted_mu2(d['method'], d['alpha']):.3f} | "
                     f"{'-' if hit is None else f'{hit:.2f}'} | {last['counted']:.1e} / {last['gaussian']:.1e} at {last['t_us']:.1f} |")
    lines += ["", "## Drive-induced flips over a full read cycle (master equation)", "",
              "| Run | alpha | read time (us) | p_flip from read drive | p_flip from alpha ramps |", "|---|---|---|---|---|"]
    for d in flips.values():
        ramps = f"{d['p_flip_ramps']:.1e}" if d["method"] == "inflate" else "-"
        lines.append(f"| {LABELS[d['method']]} | {d['alpha']} | {d['t_read_us']} | {d['p_flip_drive']:.2e} | {ramps} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


STAGES = {"records": stage_records, "flips": stage_flips, "plot": lambda: (plot(), write_summary())}


def main() -> None:
    for name in sys.argv[1:] or list(STAGES):
        t0 = time.time()
        STAGES[name]()
        print(f"stage {name}: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
