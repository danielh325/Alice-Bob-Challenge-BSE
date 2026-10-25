"""Draw figures/pitch13_protocol.gif from results/task1/protocol_animation.npz.

Called by task13_protocol_animation.py; no new physics is simulated here.

* Phase space: the saved memory Wigner function, the saved <a> path, and the
  mean-field flow of each step's equations of motion, drawn from CatParams:
  push (pump off)  da/dt = u0 - kappa_a a / 2
  stabilization    da/dt = -kappa_2 a* (a^2 - alpha^2) - kappa_a a / 2
* Device: memory and buffer coupled through the ATS; a line wiggles while its tone
  is on. The bars in the memory are the saved photon-number distribution P(n).
* Timeline: which tone is on (on/off only) and the saved 1 - F(|+alpha>).
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from task13_idea_render import ACCENT, CMAP, INK, MUTED, RULE  # noqa: E402

FPS = 12
IDLE_FRAMES, SWITCH_FRAMES, TAIL_FRAMES = 14, 12, 36
X_PUSH, X_ON, X_END = 1.0, 3.0, 9.0  # timeline: off | push (stretched) | hold
N_BARS = 15
BAR_FULL = 0.25  # P(n) that fills the bar area
TAU = 2 * math.pi
ERR_FLOOR = 1e-6

FLOW_LAW = {
    "off": "no drive: nothing moves",
    "push": r"push: $\dot a = u_0 - \kappa_a a/2$",
    "hold": r"stabilization: $\dot a = -\kappa_2 a^*(a^2-\alpha^2) - \kappa_a a/2$",
}


def sci(x: float) -> str:
    if x < 1e-12:
        return "0"
    mantissa, exponent = f"{x:.1e}".split("e")
    return rf"{mantissa}\times10^{{{int(exponent)}}}"


def steps(out):
    """One entry per GIF frame: (step, phase, index, timeline x)."""
    t_push, t_hold = out["t_push"], out["t_hold"]
    seq = [(1, "push", 0, X_PUSH * n / (IDLE_FRAMES - 1)) for n in range(IDLE_FRAMES)]
    seq += [(2, "push", k, X_PUSH + (X_ON - X_PUSH) * t / t_push[-1]) for k, t in enumerate(t_push)]
    seq += [(3, "hold", 0, X_ON)] * SWITCH_FRAMES
    seq += [(4, "hold", k, X_ON + (X_END - X_ON) * t / t_hold[-1]) for k, t in enumerate(t_hold)]
    return seq + [(5, "hold", len(t_hold) - 1, X_END)] * TAIL_FRAMES


def captions(out):
    u0_mhz = float(out["u0"]) / TAU
    return {
        1: ("Stabilization off", "Pump and buffer drive off. Memory and buffer start empty."),
        2: ("Microwave push on the memory",
            f"Resonant drive on the memory line, {u0_mhz:.1f} MHz for 50 ns. Pump stays off."),
        3: ("Switch the pump on",
            "Pump and buffer drive on together. The memory already sits at α."),
        4: ("Stabilization holds the codeword",
            r"$g_2\alpha^2$ cancels $\varepsilon_d$: the buffer stays empty,"
            " nothing pulls the state"),
        5: ("Codeword prepared",
            rf"$1-F(|{{+}}\alpha\rangle) = {sci(float(out['hold_err'][-1]))}$"
            " after 2 µs of stabilization"),
    }


# ---- phase space -------------------------------------------------------------

def flow(p, u0, xmax, stabilized):
    """Mean-field velocity field da/dt on the Wigner plane (x = Re a, y = Im a)."""
    grid = np.linspace(-xmax, xmax, 41)
    x, y = np.meshgrid(grid, grid)
    a = x + 1j * y
    if stabilized:
        v = -p.kappa2 * np.conj(a) * (a**2 - p.alpha**2) - p.kappa_a * a / 2
    else:
        v = u0 - p.kappa_a * a / 2 + 0 * a
    return grid, v.real, v.imag


def phase_space(ax, out, p, xmax):
    vmax = max(np.abs(out["push_w"]).max(), np.abs(out["hold_w"]).max())
    image = ax.imshow(out["push_w"][0].T, origin="lower", cmap=CMAP, vmin=-vmax, vmax=vmax,
                      extent=(-xmax, xmax, -xmax, xmax), interpolation="bilinear", zorder=1)
    fields = {}
    for key, stabilized in (("push", False), ("hold", True)):
        grid, vx, vy = flow(p, float(out["u0"]), xmax, stabilized)
        before = set(ax.patches)
        sp = ax.streamplot(grid, grid, vx, vy, density=0.75 if stabilized else 0.45,
                           color=MUTED, linewidth=0.7, arrowsize=0.9, zorder=2)
        # streamplot draws each arrowhead as its own patch, outside sp.arrows
        fields[key] = (sp.lines, *(patch for patch in ax.patches if patch not in before))
    marks = ((1, INK, r"$|{+}\alpha\rangle$"), (-1, MUTED, r"$|{-}\alpha\rangle$"))
    for sign, color, label in marks:
        ax.plot([sign * p.alpha], [0], marker="o", ms=11, mfc="none", mec=color, mew=1.5, zorder=4)
        ax.text(sign * p.alpha, -0.45, label, ha="center", va="top", fontsize=14, color=color,
                zorder=4)
    (path,) = ax.plot([], [], color=INK, lw=2, zorder=5)
    (dot,) = ax.plot([], [], marker="o", ms=7, color=INK, zorder=6)
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-xmax, xmax)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(RULE)
    ax.set_title("memory phase space: Wigner function + mean-field flow", fontsize=12,
                 color=MUTED, loc="left")
    law = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=12, color=INK, zorder=7,
                  bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 3})
    return image, fields, path, dot, law


# ---- device ------------------------------------------------------------------

def box(ax, x0, y0, w, h, title):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.04,rounding_size=0.25",
                                fc="white", ec=INK, lw=2))
    ax.text(x0 + w / 2, y0 + h - 0.3, title, ha="center", va="top", fontsize=14,
            fontweight="bold", color=INK)


def coupler(ax, x0, y0, s):
    """ATS coupler: a square with the Josephson-junction cross."""
    ax.add_patch(Rectangle((x0, y0), s, s, fc="white", ec=INK, lw=2))
    ax.plot([x0, x0 + s], [y0, y0 + s], color=INK, lw=1.5)
    ax.plot([x0, x0 + s], [y0 + s, y0], color=INK, lw=1.5)
    ax.text(x0 + s / 2, y0 - 0.2, "ATS coupler", ha="center", va="top", fontsize=11, color=MUTED)


def photon_bars(ax):
    width = 3.1 / N_BARS
    bars = [Rectangle((0.55 + k * width, 0.95), width * 0.8, 0, fc=ACCENT, ec="none")
            for k in range(N_BARS)]
    for bar in bars:
        ax.add_patch(bar)
    ax.plot([0.5, 3.75], [0.95, 0.95], color=RULE, lw=1)
    ax.text(0.55, 0.85, "n = 0", fontsize=9, color=MUTED, va="top")
    ax.text(3.75, 0.85, f"{N_BARS - 1}", fontsize=9, color=MUTED, va="top", ha="right")
    ax.text(3.75, 2.95, "P(n)", fontsize=10, color=MUTED, ha="right")
    return bars


def device(ax, out, p):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    box(ax, 0.2, 0.5, 3.8, 3.6, "memory  a")
    box(ax, 6.5, 1.0, 2.6, 2.5, "buffer  b")
    coupler(ax, 4.75, 1.75, 1.0)
    ax.plot([4.0, 4.75], [2.25, 2.25], color=INK, lw=2)
    ax.plot([5.75, 6.5], [2.25, 2.25], color=INK, lw=2)
    loss = FancyArrowPatch((7.8, 0.95), (7.8, 0.1), arrowstyle="-|>", mutation_scale=16,
                           color=MUTED, lw=2)
    ax.add_patch(loss)
    ax.text(8.0, 0.35, rf"loss $\kappa_b/2\pi$ = {p.kappa_b / TAU:.1f} MHz", fontsize=10,
            color=MUTED)
    bars = photon_bars(ax)
    n_a = ax.text(0.55, 2.95, "", fontsize=12, color=INK)
    n_b = ax.text(7.8, 1.9, "", ha="center", fontsize=12, color=INK)
    drive = f"memory drive at $\\omega_a$\n{float(out['u0']) / TAU:.1f} MHz"
    pump = f"pump at $2\\omega_a-\\omega_b$\n$g_2/2\\pi$ = {p.g2 / TAU:.2f} MHz"
    eps = f"buffer drive at $\\omega_b$\n$\\varepsilon_d/2\\pi$ = {p.eps_d / TAU:.2f} MHz"
    lines = {}
    for key, x, y_end, label in (("drive", 2.1, 4.15, drive), ("pump", 5.25, 2.8, pump),
                                 ("eps", 7.8, 3.55, eps)):
        (wave,) = ax.plot([], [], lw=2.5)
        (tip,) = ax.plot([x], [y_end + 0.12], marker="v", ms=11)
        text = ax.text(x, 5.85, label, ha="center", va="bottom", fontsize=11, linespacing=1.25)
        lines[key] = (wave, tip, text, x, y_end)
    return bars, n_a, n_b, lines


def draw_tone(line, on, phase):
    wave, tip, text, x, y_end = line
    y = np.linspace(y_end + 0.25, 5.75, 120)
    amplitude = 0.14 if on else 0.0
    wave.set_data(x + amplitude * np.sin(9 * y - phase), y)
    for artist in (wave, tip):
        artist.set_color(ACCENT if on else RULE)
    text.set_color(ACCENT if on else MUTED)
    text.set_fontweight("bold" if on else "normal")


# ---- timeline ----------------------------------------------------------------

def controls(ax):
    rows = {"drive": 2.0, "pump": 1.0, "eps": 0.0}
    names = {"drive": "memory drive", "pump": "pump", "eps": "buffer drive"}
    xs = np.linspace(0, X_END, 901)
    shapes = {"drive": (xs >= X_PUSH) & (xs < X_ON), "pump": xs >= X_ON, "eps": xs >= X_ON}
    ax.set_xlim(0, X_END)
    ax.set_ylim(-0.15, 2.85)
    ax.axis("off")
    lit = {}
    for key, on in shapes.items():
        y = rows[key] + 0.55 * on
        ax.plot(xs, y, color=RULE, lw=2.5)
        lit[key] = (ax.plot([], [], color=ACCENT, lw=3.5)[0], xs, y)
        ax.text(-0.15, rows[key] + 0.1, names[key], ha="right", fontsize=11, color=INK)
    return lit


def error_curve(ax, out):
    push_x = X_PUSH + (X_ON - X_PUSH) * out["t_push"] / out["t_push"][-1]
    hold_x = X_ON + (X_END - X_ON) * out["t_hold"] / out["t_hold"][-1]
    t_x = np.concatenate([[0.0], push_x, hold_x])
    err = np.concatenate([[out["push_err"][0]], out["push_err"], out["hold_err"]])
    err = np.maximum(err, ERR_FLOOR)
    ax.set_xlim(0, X_END)
    ax.set_yscale("log")
    ax.set_ylim(ERR_FLOOR, 2)
    ax.plot(t_x, err, color=RULE, lw=2)
    (lit,) = ax.plot([], [], color=ACCENT, lw=2.5)
    ax.set_yticks([1e-6, 1e-3, 1], [r"$10^{-6}$", r"$10^{-3}$", "1"])
    ax.minorticks_off()
    ax.set_ylabel(r"$1-F$", fontsize=12, color=INK)
    ax.set_xticks([X_PUSH, X_ON, 6.0, X_END], ["0", "50 ns", "1.05 µs", "2.05 µs"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.text(X_END, 2.5, "time axis stretched during the push", ha="right", va="bottom",
            fontsize=10, color=MUTED)
    return lit, t_x, err


def timeline(control_ax, error_ax, out):
    lit = controls(control_ax)
    curve = error_curve(error_ax, out)
    cursors = []
    for ax in (control_ax, error_ax):
        for x in (X_PUSH, X_ON):
            ax.axvline(x, color=RULE, lw=1, ls="--", zorder=0)
        cursors.append(ax.axvline(0, color=INK, lw=1.3))
    return lit, curve, cursors


# ---- assemble ----------------------------------------------------------------

def render(out, p, path, xmax):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
    fig.text(0.03, 0.965, "Method D, step by step · every frame is a saved two-mode "
             "simulation state (28 × 12 levels)", fontsize=11, color=MUTED)
    number = fig.text(0.03, 0.865, "", fontsize=44, fontweight="bold", color=ACCENT)
    head = fig.text(0.085, 0.9, "", fontsize=24, fontweight="bold", color=INK)
    sub = fig.text(0.085, 0.862, "", fontsize=14, color=MUTED)
    text = captions(out)
    image, fields, path_line, dot, law = phase_space(
        fig.add_axes([0.03, 0.09, 0.39, 0.69]), out, p, xmax)
    error = fig.text(0.03, 0.025, "", fontsize=18, color=INK)
    bars, n_a, n_b, lines = device(fig.add_axes([0.47, 0.4, 0.52, 0.44]), out, p)
    lit, (err_lit, t_x, err_y), cursors = timeline(
        fig.add_axes([0.58, 0.235, 0.39, 0.15]), fig.add_axes([0.58, 0.06, 0.39, 0.15]), out)
    seq = steps(out)
    a_path = np.concatenate([out["push_a"], out["hold_a"]])

    def draw(n):
        step, phase, k, x = seq[n]
        number.set_text("✓" if step == 5 else str(step))
        head.set_text(text[step][0])
        sub.set_text(text[step][1])
        image.set_data(out[f"{phase}_w"][k].T)
        mode = "off" if step == 1 else ("push" if step == 2 else "hold")
        for key, artists in fields.items():
            for artist in artists:
                artist.set_visible(key == mode)
        law.set_text(FLOW_LAW[mode])
        upto = k + 1 if phase == "push" else len(out["push_a"]) + k + 1
        path_line.set_data(a_path[:upto].real, a_path[:upto].imag)
        dot.set_data([a_path[upto - 1].real], [a_path[upto - 1].imag])
        error.set_text(rf"$1-F(|{{+}}\alpha\rangle) = {sci(float(out[f'{phase}_err'][k]))}$")
        error.set_color(ACCENT if step >= 4 else INK)
        for bar, height in zip(bars, out[f"{phase}_pn"][k][:N_BARS], strict=True):
            bar.set_height(1.85 * min(height / BAR_FULL, 1.0))
        n_a.set_text(rf"$\langle a^\dagger a\rangle = {float(out[f'{phase}_n_a'][k]):.2f}$")
        n_b.set_text(rf"$\langle b^\dagger b\rangle = {sci(float(out[f'{phase}_n_b'][k]))}$")
        active = {"drive": step == 2, "pump": step >= 3, "eps": step >= 3}
        for key, line in lines.items():
            draw_tone(line, active[key], 0.9 * n)
            trace, xs, ys = lit[key]
            trace.set_data(xs[xs <= x], ys[xs <= x])
        err_lit.set_data(t_x[t_x <= x], err_y[t_x <= x])
        for cursor in cursors:
            cursor.set_xdata([x, x])
        return []

    FuncAnimation(fig, draw, frames=len(seq), blit=False).save(path, writer=PillowWriter(fps=FPS))
    plt.close(fig)
