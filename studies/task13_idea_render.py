"""Draw figures/pitch13_idea.gif from results/task1/idea_animation.npz (no new physics).

Called by task13_idea_animation.py. Panels show the memory Wigner function of each
saved frame; the numbers under the panels and the curves below are 1 - F(|+alpha>)
of the same frames.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

INK, MUTED, RULE, ACCENT, NEG = "#0b0b0b", "#8a8580", "#e4e0dc", "#ccd9f5", "#2f5bd3"
COLORS = {"vacuum": MUTED, "A": INK, "D": ACCENT}
FPS = 12
LEAD_FRAMES = 6
TAIL_FRAMES = 30
CMAP = LinearSegmentedColormap.from_list("wigner", [NEG, "#ffffff", ACCENT])

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 15,
    "axes.edgecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": INK,
})


def sci(x: float) -> str:
    mantissa, exponent = f"{x:.1e}".split("e")
    return rf"$1-F = {mantissa}\times10^{{{int(exponent)}}}$"


def timeline(out):
    """(phase, index) per GIF frame: lead-in, push, hold, pause on the last frame."""
    n_push, n_hold = len(out["t_push"]), len(out["t_hold"])
    steps = [("push", 0)] * LEAD_FRAMES + [("push", k) for k in range(n_push)]
    steps += [("hold", k) for k in range(n_hold)]
    return steps + [("hold", n_hold - 1)] * TAIL_FRAMES


def panel_frame(out, key, phase, k):
    if phase == "push":
        if key == "D":
            return out["w_push"][k], out["e_push"][k]
        k = 0
    return out[f"w_{key}"][k], out[f"e_{key}"][k]


def render(out, starts, path, xmax):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="white")
    lead = fig.text(0.05, 0.93, "Don't roll into the valley.", fontsize=26, fontweight="bold",
                    color=INK)
    lead_end = lead.get_window_extent(fig.canvas.get_renderer()).x1 / fig.bbox.width
    fig.text(lead_end + 0.012, 0.93, "Start at the bottom.", fontsize=26, fontweight="bold",
             color=ACCENT)
    clock = fig.text(0.95, 0.935, "", fontsize=16, ha="right", color=INK)
    vmax = max(np.abs(out[f"w_{key}"]).max() for key, _ in starts)

    images, labels = {}, {}
    for j, (key, title) in enumerate(starts):
        ax = fig.add_axes([0.05 + j * 0.31, 0.40, 0.27, 0.45])
        ax.set_title(title, fontsize=16, color=COLORS[key] if key != "vacuum" else INK,
                     fontweight="bold", loc="left")
        images[key] = ax.imshow(out[f"w_{key}"][0].T, origin="lower", cmap=CMAP,
                                vmin=-vmax, vmax=vmax, extent=(-xmax, xmax, -xmax, xmax),
                                interpolation="bilinear")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(ACCENT if key == "D" else RULE)
            spine.set_linewidth(3 if key == "D" else 1)
        labels[key] = ax.text(0.03, -0.1, "", transform=ax.transAxes, fontsize=17,
                              color=COLORS[key] if key != "vacuum" else INK)

    err = fig.add_axes([0.09, 0.12, 0.86, 0.2])
    err.set_yscale("log")
    err.set_xlim(0, out["t_hold"][-1])
    err.set_ylim(1e-8, 2)
    err.set_xlabel("time since the stabilization was switched on (µs)", color=MUTED)
    err.set_ylabel(r"$1-F(|{+}\alpha\rangle)$", color=INK)
    for side in ("top", "right"):
        err.spines[side].set_visible(False)
    curves = {key: err.plot([], [], color=COLORS[key], lw=3, label=title)[0]
              for key, title in starts}

    steps = timeline(out)

    def draw(n):
        phase, k = steps[n]
        for key, _ in starts:
            wigner, error = panel_frame(out, key, phase, k)
            images[key].set_data(wigner.T)
            labels[key].set_text(sci(error))
            upto = k + 1 if phase == "hold" else 0
            curves[key].set_data(out["t_hold"][:upto], out[f"e_{key}"][:upto])
        if phase == "push":
            clock.set_text(f"pump off · push {1e3 * out['t_push'][k]:.0f} ns")
        else:
            clock.set_text(f"stabilization on · t = {out['t_hold'][k]:.2f} µs")
        return [*images.values(), *labels.values(), *curves.values(), clock]

    FuncAnimation(fig, draw, frames=len(steps), blit=False).save(path, writer=PillowWriter(fps=FPS))
    plt.close(fig)
