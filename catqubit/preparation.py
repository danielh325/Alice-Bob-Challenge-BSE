"""Preparing a definite codeword |+alpha> (Task 1.3(b))."""

import math
from collections.abc import Callable

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
from dynamiqs import TimeQArray
from scipy.optimize import brentq

from catqubit.params import CatParams


def basin_fidelity(beta: float, theta: float = 0.0) -> float:
    """Predicted F(|+alpha>) after inflating from |beta e^{i theta}>.

    Basin model: the stabilization sends each half-plane to its nearer valley, so F is
    the probability that the x-quadrature (mean beta cos(theta), variance 1/4) is > 0.
    """
    if beta < 0:
        raise ValueError(f"beta must be non-negative, got {beta}")
    return 1 - 0.5 * math.erfc(math.sqrt(2) * beta * math.cos(theta))


def bias_pulse(na: int, nb: int, area: float, duration: float, theta: float = 0.0) -> TimeQArray:
    """Square memory drive H(t) = (area / duration) i(e^{i theta} a^dag - h.c.) for t < duration.

    On its own it displaces the memory by area * e^{i theta}, so a pulse much
    shorter than the inflation is equivalent to a displaced start.
    """
    if area < 0:
        raise ValueError(f"area must be non-negative, got {area}")
    if duration <= 0:
        raise ValueError(f"duration must be positive, got {duration}")
    a, _ = dq.destroy(na, nb)
    phase = complex(math.cos(theta), math.sin(theta))
    drive = 1j * (phase * a.dag() - phase.conjugate() * a)
    amplitude = area / duration
    return dq.modulated(
        lambda t: jnp.where(t < duration, amplitude, 0.0),
        drive,
        discontinuity_ts=jnp.array([duration]),
    )


def amplified_push(
    push: Callable[[float], float],
    buffer_drive: Callable[[float], float],
    t_end: float,
    params: CatParams,
    steps: int = 4001,
) -> float:
    """Effective displacement beta_eff of a memory push applied while the cat inflates.

    Linear-amplifier model: near vacuum the stabilization amplifies the memory
    x-quadrature at lambda(t) = 2 g2 b(t) - kappa_a / 2, where the buffer field b(t)
    relaxes to 2 eps_d(t) / kappa_b at rate kappa_b / 2. A push eps(s) at time s is
    worth eps(s) exp(-Lambda(s)) of displacement at t = 0, Lambda(s) = int_0^s lambda,
    so beta_eff = int_0^t_end eps(s) exp(-Lambda(s)) ds. Feed it to basin_fidelity.
    A pulse much shorter than 1 / lambda gives beta_eff = its area, the displaced start.
    """
    if t_end <= 0:
        raise ValueError(f"t_end must be positive, got {t_end}")
    if steps < 2:
        raise ValueError(f"steps must be at least 2, got {steps}")
    t = np.linspace(0.0, t_end, steps)
    dt = t[1] - t[0]
    decay = math.exp(-params.kappa_b / 2 * dt)
    b = np.zeros(steps)
    for i in range(1, steps):
        target = 2 * buffer_drive(t[i - 1]) / params.kappa_b
        b[i] = target + (b[i - 1] - target) * decay
    rate = 2 * params.g2 * b - params.kappa_a / 2
    gain = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * dt)])
    integrand = np.array([push(s) for s in t]) * np.exp(-gain)
    return float(np.trapezoid(integrand, t))


def ramp_push_strength(params: CatParams, t_ramp: float, beta_target: float) -> float:
    """Constant push eps (rad/us) held over a linear ramp of eps_d that reaches beta_target."""
    if beta_target <= 0:
        raise ValueError(f"beta_target must be positive, got {beta_target}")

    def ramp(s: float) -> float:
        return params.eps_d * min(s / t_ramp, 1.0)

    def push(strength: float) -> Callable[[float], float]:
        return lambda s: strength if s < t_ramp else 0.0

    return float(
        brentq(lambda e: amplified_push(push(e), ramp, t_ramp, params) - beta_target, 1e-6, 1e3)
    )


def ramped_preparation(
    na: int, nb: int, params: CatParams, t_ramp: float, strength: float, theta: float = 0.0
) -> TimeQArray:
    """Method C: ramp the buffer drive 0 -> eps_d over t_ramp while pushing the memory.

    Returned as an extra_H for inflate(): the first term cancels part of the model's
    constant drive -eps_d (b + b^dag) so the net drive is -eps_d min(t / t_ramp, 1),
    the second is a bias_pulse of strength `strength` held for the whole ramp.
    """
    if t_ramp <= 0:
        raise ValueError(f"t_ramp must be positive, got {t_ramp}")
    _, b = dq.destroy(na, nb)
    eps_d = params.eps_d
    ramp = dq.modulated(
        lambda t: eps_d * (1 - jnp.clip(t / t_ramp, 0.0, 1.0)),
        b + b.dag(),
        discontinuity_ts=jnp.array([t_ramp]),
    )
    return ramp + bias_pulse(na, nb, area=strength * t_ramp, duration=t_ramp, theta=theta)
