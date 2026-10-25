"""Readout optimization under a bit-flip budget (Task 2.4c).

Signal model. Linearizing the two-mode model around the codeword |+alpha>, |0> with a
real readout drive eps_z(t), the memory displacement delta = i y is imaginary and the buffer
amplitude beta is real:

    dy/dt = -2 g2 alpha beta - eps_z,      dbeta/dt = 2 g2 alpha y - (kappa_b / 2) beta,

a damped oscillator with omega_0 = 2 alpha g2 and zeta = kappa_b / (8 g2 alpha) (Gautier
et al. 2023, Eq. 15b; plan R7), steady state beta = -eps_z / (2 alpha g2). The -alpha
codeword gives -beta (the T symmetry of catqubit.driven_rates), so Delta beta = 2 beta.
kappa_a (about 1e-3 kappa_b) is neglected here. Homodyne of the buffer output with
efficiency eta and the optimal (matched) filter gives

    SNR^2 = eta kappa_b int (Delta <b + b^dag>)^2 dt = 16 eta kappa_b int beta^2 dt,

and, for Gaussian records, a readout error 0.5 erfc(SNR / (2 sqrt 2)).

Bit-flip cost (quasi-static). p_X = int Gamma_X(eps_z(t)) dt + d2 int Delta(t)^2 dt, with
Gamma_X the exact two-mode rate (-lambda_1, both one-way rates; one-way = p_X / 2) from a
cubic Hermite table built with the adjoint gradients of catqubit.eig_gradients. The second
term is the readout-induced Stark shift: a memory-buffer cross-Kerr -chi_ab n_a n_b detunes
the memory by Delta = -chi_ab <n_b> = -chi_ab beta^2 (codeword independent), and a memory
detuning raises Gamma_X by d2 Delta^2 at leading order (both even in eps_z and Delta).
Compensation retunes the memory frame by +chi_ab beta(t)^2 (feed-forward from this model).
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from jax.scipy.special import erfc
from scipy.linalg import eigh
from scipy.optimize import minimize

from catqubit.params import CatParams

# Augmented Lagrangian settings.
PENALTY_START = 10.0
PENALTY_GROWTH = 10.0
PENALTY_MAX = 1e8
# Grow the penalty unless the violation drops at least this much per outer step.
VIOLATION_DECREASE = 0.25
CONSTRAINT_TOL = 1e-6
MULTIPLIER_RTOL = 1e-5
MAX_OUTER = 40
INNER_GTOL = 1e-10
INNER_MAXITER = 2000
SNR_PER_ERFC_ARGUMENT = 2 * np.sqrt(2)
FORM_EIGEN_FLOOR = 1e-12


@dataclass(frozen=True)
class ALResult:
    x: np.ndarray
    fun: float
    constraint: float
    multiplier: float
    outer_iterations: int
    converged: bool


ValueGrad = Callable[[np.ndarray], tuple[float, np.ndarray]]


def augmented_lagrangian(
    fun: ValueGrad,
    constraint: ValueGrad,
    x0: np.ndarray,
    bounds: list[tuple[float, float]] | None = None,
) -> ALResult:
    """Minimize fun(x) subject to constraint(x) <= 0 (Powell-Hestenes-Rockafellar).

    Inner problems: L-BFGS-B on f + ((max(0, lam + mu g))^2 - lam^2) / (2 mu). Outer: the
    multiplier update lam <- max(0, lam + mu g); mu grows when the violation stalls.
    fun and constraint return (value, gradient). Scale g to order one (e.g. p_X / B - 1).
    """
    x, lam, mu = np.asarray(x0, dtype=float), 0.0, PENALTY_START
    previous_violation = np.inf
    for outer in range(1, MAX_OUTER + 1):

        def lagrangian(z: np.ndarray, lam: float = lam, mu: float = mu) -> tuple[float, np.ndarray]:
            f, df = fun(z)
            g, dg = constraint(z)
            s = max(0.0, lam + mu * g)
            return f + (s**2 - lam**2) / (2 * mu), df + s * dg

        res = minimize(
            lagrangian,
            x,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"gtol": INNER_GTOL, "ftol": 1e-15, "maxiter": INNER_MAXITER},
        )
        x = res.x
        g = constraint(x)[0]
        new_lam = max(0.0, lam + mu * g)
        violation = max(g, 0.0)
        settled = abs(new_lam - lam) <= MULTIPLIER_RTOL * max(1.0, abs(new_lam))
        lam = new_lam
        if violation <= CONSTRAINT_TOL and settled:
            return ALResult(x, fun(x)[0], g, lam, outer, True)
        if violation > VIOLATION_DECREASE * previous_violation:
            mu = min(mu * PENALTY_GROWTH, PENALTY_MAX)
        previous_violation = violation
    return ALResult(x, fun(x)[0], constraint(x)[0], lam, MAX_OUTER, False)


@dataclass(frozen=True)
class HermiteTable:
    """Piecewise-cubic Hermite interpolant through (knots, values, slopes); JAX-traceable."""

    knots: tuple[float, ...]
    values: tuple[float, ...]
    slopes: tuple[float, ...]

    @classmethod
    def build(cls, knots: np.ndarray, values: np.ndarray, slopes: np.ndarray) -> "HermiteTable":
        knots = np.asarray(knots, dtype=float)
        if knots.size < 2 or np.any(np.diff(knots) <= 0):
            raise ValueError("knots must be strictly increasing (at least two)")
        if not (len(values) == len(slopes) == knots.size):
            raise ValueError("knots, values and slopes must have the same length")
        return cls(tuple(knots), tuple(map(float, values)), tuple(map(float, slopes)))

    def check_range(self, x: float) -> None:
        if not self.knots[0] <= x <= self.knots[-1]:
            raise ValueError(f"{x} outside the table range [{self.knots[0]}, {self.knots[-1]}]")

    def __call__(self, x: jax.Array) -> jax.Array:
        k, v, s = (jnp.asarray(q) for q in (self.knots, self.values, self.slopes))
        i = jnp.clip(jnp.searchsorted(k, x, side="right") - 1, 0, k.size - 2)
        h = k[i + 1] - k[i]
        t = (x - k[i]) / h
        h00 = (1 + 2 * t) * (1 - t) ** 2
        h10 = t * (1 - t) ** 2
        h01 = t**2 * (3 - 2 * t)
        h11 = t**2 * (t - 1)
        return h00 * v[i] + h10 * h * s[i] + h01 * v[i + 1] + h11 * h * s[i + 1]


@dataclass(frozen=True)
class TwoStep:
    """eps_z = first_amplitude on [0, switch_time), then final_amplitude."""

    first_amplitude: float
    switch_time: float
    final_amplitude: float


@dataclass(frozen=True)
class LinearReadout:
    """Linearized readout response of the codeword |+alpha> (module docstring)."""

    alpha: float
    g2: float
    kappa_b: float

    @classmethod
    def from_params(cls, params: CatParams) -> "LinearReadout":
        return cls(params.alpha, params.g2, params.kappa_b)

    def drift(self) -> jax.Array:
        w = 2 * self.g2 * self.alpha
        return jnp.array([[0.0, -w], [w, -0.5 * self.kappa_b]])

    def steady_state(self, eps_z: float) -> np.ndarray:
        w = 2 * self.g2 * self.alpha
        beta = -eps_z / w
        return np.array([0.25 * self.kappa_b * beta / w, beta])

    def propagate_constant(self, state: np.ndarray, eps_z: float, t: float) -> np.ndarray:
        """Exact state after time t at constant drive."""
        target = self.steady_state(eps_z)
        return target + np.asarray(jsl.expm(self.drift() * t)) @ (np.asarray(state) - target)

    def _half_step_maps(self, step: float) -> tuple[jax.Array, jax.Array]:
        """(P, q) with s(t + step) = P s(t) + q eps for a constant eps over the step."""
        a = self.drift()
        p = jsl.expm(a * step)
        q = jnp.linalg.solve(a, (p - jnp.eye(2)) @ jnp.array([-1.0, 0.0]))
        return p, q

    def trajectory(self, pulse: jax.Array, duration: float, substeps: int) -> jax.Array:
        """States at 2 * pieces * substeps + 1 equally spaced times, from rest."""
        pieces = pulse.shape[0]
        n_half = 2 * pieces * substeps
        p, q = self._half_step_maps(duration / n_half)
        eps = jnp.repeat(pulse, 2 * substeps)

        def step(s: jax.Array, e: jax.Array) -> tuple[jax.Array, jax.Array]:
            nxt = p @ s + q * e
            return nxt, nxt

        _, states = jax.lax.scan(step, jnp.zeros(2), eps)
        return jnp.vstack([jnp.zeros((1, 2)), states])

    def buffer_amplitude(self, pulse: jax.Array, duration: float, substeps: int) -> jax.Array:
        return self.trajectory(pulse, duration, substeps)[:, 1]

    def snr_squared(
        self, pulse: jax.Array, duration: float, substeps: int, efficiency: float
    ) -> jax.Array:
        """16 eta kappa_b int_0^T beta^2 dt (composite Simpson on the half-step grid)."""
        beta = self.buffer_amplitude(pulse, duration, substeps)
        return 16 * efficiency * self.kappa_b * _simpson(beta**2, duration)


def _simpson(y: jax.Array, duration: float) -> jax.Array:
    h = duration / (y.shape[0] - 1)
    return h / 3 * (y[0] + y[-1] + 4 * jnp.sum(y[1:-1:2]) + 2 * jnp.sum(y[2:-1:2]))


def gaussian_error(snr_squared: jax.Array) -> jax.Array:
    """Assignment error of two equal-variance Gaussians separated by SNR standard deviations."""
    return 0.5 * erfc(jnp.sqrt(snr_squared) / SNR_PER_ERFC_ARGUMENT)


def deadbeat_two_step(model: LinearReadout, eps_z: float) -> TwoStep:
    """Zero-vibration two-step drive: the buffer reaches steady state with no ringing.

    From rest, a constant drive e1 overshoots and peaks (zero velocity) at half the damped
    period t1 = pi / omega_d with amplitude (1 + q) times its own steady state, where
    q = exp(-zeta omega_0 t1). Choosing e1 = eps_z / (1 + q) makes that peak the target, and
    switching to eps_z at t1 holds it (input shaping, cf. McClure et al. 2016).
    """
    omega0 = 2 * model.alpha * model.g2
    zeta = model.kappa_b / (8 * model.g2 * model.alpha)
    if zeta >= 1:
        raise ValueError(f"overdamped buffer (zeta = {zeta:.2f}); no ringing to cancel")
    t1 = np.pi / (omega0 * np.sqrt(1 - zeta**2))
    q = np.exp(-zeta * omega0 * t1)
    return TwoStep(eps_z / (1 + q), float(t1), eps_z)


def positive_part(form: np.ndarray, floor: float = FORM_EIGEN_FLOOR) -> np.ndarray:
    """Symmetrize and clip eigenvalues to >= floor * max: a flip cost x^T M x is physically
    non-negative; tiny negative eigenvalues are quadrature noise where c2(omega) ~ 0."""
    vals, vecs = np.linalg.eigh(0.5 * (form + form.T))
    vals = np.maximum(vals, floor * vals.max())
    return (vecs * vals) @ vecs.T


@dataclass(frozen=True)
class FrontPoint:
    budget: float
    pulse: np.ndarray
    snr_squared: float
    error: float
    bit_flip_probability: float
    multiplier: float
    converged: bool


@dataclass(frozen=True)
class ReadoutProblem:
    """Maximize SNR^2 over a piecewise-constant eps_z(t) on [0, duration] with p_X <= B.

    stark_coefficient is d2 (Gamma_X = ... + d2 Delta^2); chi_ab in rad/us. With compensate,
    the memory frame follows +chi_ab beta(t)^2, which cancels Delta at this (mean-field) order.
    flip_form / signal_form: exact second-order flip and signal quadratic forms (x^T M x,
    x^T S x) from the Liouvillian kernels; signal_form already includes the efficiency.
    """

    model: LinearReadout
    rate_table: HermiteTable
    duration: float
    pieces: int
    substeps: int = 8
    efficiency: float = 1.0
    chi_ab: float = 0.0
    stark_coefficient: float = 0.0
    compensate: bool = False
    eps_max: float | None = None
    flip_form: np.ndarray | None = field(default=None, compare=False)
    c2: float = 0.0
    signal_form: np.ndarray | None = field(default=None, compare=False)
    _jitted: dict = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.flip_form is not None:
            if np.shape(self.flip_form) != (self.pieces,) * 2:
                raise ValueError(f"flip_form must be {self.pieces} x {self.pieces}")
            object.__setattr__(self, "flip_form", positive_part(np.asarray(self.flip_form)))
        if self.signal_form is not None and np.shape(self.signal_form) != (self.pieces,) * 2:
            raise ValueError(f"signal_form must be {self.pieces} x {self.pieces}")

    def _snr2(self, x: jax.Array) -> jax.Array:
        """SNR^2 from signal_form (exact kernel, catqubit.driven_rates.ResponseKernel) if
        given, else from the linearized oscillator."""
        if self.signal_form is not None:
            return x @ jnp.asarray(self.signal_form) @ x
        return self.model.snr_squared(x, self.duration, self.substeps, self.efficiency)

    def _drive_cost(self, x: jax.Array) -> jax.Array:
        """Idle plus drive-induced flips. With flip_form (the exact second-order memory
        kernel, catqubit.driven_rates.DriveKernel), only the beyond-second-order remainder
        of the rate table is treated quasi-statically."""
        h = self.duration / self.pieces
        table = self.rate_table(jnp.abs(x))
        if self.flip_form is None:
            return jnp.sum(table) * h
        rate0 = self.rate_table(jnp.asarray(0.0))
        remainder = jnp.sum(table - rate0 - self.c2 * x**2) * h
        return rate0 * self.duration + x @ jnp.asarray(self.flip_form) @ x + remainder

    def _p_x(self, x: jax.Array) -> jax.Array:
        idle_and_drive = self._drive_cost(x)
        if self.compensate or self.stark_coefficient == 0 or self.chi_ab == 0:
            return idle_and_drive
        beta = self.model.buffer_amplitude(x, self.duration, self.substeps)
        detuning = -self.chi_ab * beta**2
        return idle_and_drive + self.stark_coefficient * _simpson(detuning**2, self.duration)

    def _value_grad(self, name: str, fn: Callable) -> ValueGrad:
        if name not in self._jitted:
            self._jitted[name] = jax.jit(jax.value_and_grad(fn))
        jitted = self._jitted[name]

        def call(x: np.ndarray) -> tuple[float, np.ndarray]:
            v, g = jitted(jnp.asarray(x, dtype=float))
            return float(v), np.asarray(g, dtype=float)

        return call

    def objective(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """-SNR^2 and its gradient."""
        return self._value_grad("snr", lambda z: -self._snr2(z))(x)

    def bit_flip_probability(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        return self._value_grad("p_x", self._p_x)(x)

    def snr_squared(self, x: np.ndarray) -> float:
        return -self.objective(x)[0]

    def linear_order_optimum(self, flip_form: np.ndarray) -> tuple[float, np.ndarray]:
        """Largest SNR^2 / (x^T M x) over pulse shapes and its shape (generalized eigenpair).

        At second order both the signal and the drive-induced flips are quadratic forms
        in the pulse, so their best ratio is the top eigenvalue of S x = g M x.
        """
        signal = 0.5 * np.asarray(jax.hessian(self._snr2)(jnp.zeros(self.pieces)))
        values, vectors = eigh(0.5 * (signal + signal.T), positive_part(flip_form))
        return float(values[-1]), vectors[:, -1]

    def bounds(self) -> list[tuple[float, float]]:
        top = self.rate_table.knots[-1] if self.eps_max is None else self.eps_max
        return [(-top, top)] * self.pieces

    def solve(self, budget: float, x0: np.ndarray | None = None) -> FrontPoint:
        """One Pareto point: the best SNR^2 with p_X <= budget."""
        top = self.bounds()[0][1]
        x0 = np.full(self.pieces, 0.1 * top) if x0 is None else x0
        scale = max(abs(self.objective(x0)[0]), 1.0)

        def scaled_objective(x: np.ndarray) -> tuple[float, np.ndarray]:
            v, g = self.objective(x)
            return v / scale, g / scale

        def scaled_constraint(x: np.ndarray) -> tuple[float, np.ndarray]:
            v, g = self.bit_flip_probability(x)
            return v / budget - 1, g / budget

        res = augmented_lagrangian(scaled_objective, scaled_constraint, x0, self.bounds())
        snr2 = self.snr_squared(res.x)
        return FrontPoint(
            budget=budget,
            pulse=res.x,
            snr_squared=snr2,
            error=float(gaussian_error(jnp.asarray(snr2))),
            bit_flip_probability=self.bit_flip_probability(res.x)[0],
            multiplier=res.multiplier,
            converged=res.converged,
        )
