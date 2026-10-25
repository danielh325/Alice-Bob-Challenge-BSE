"""Phase-flip rate Gamma_Z without fitting, plus analytic references (Task 2.2).

The exact identity. Every term of the two-mode model except memory loss and gain commutes
with the memory parity P = exp(i pi a^dag a): H_2ph moves memory photons in pairs and the
buffer jumps do not touch the memory. Since P a = -a P, the adjoint dissipators give
D^dag[a] P = a^dag P a - {a^dag a, P}/2 = -2 a^dag a P and D^dag[a^dag] P = -2 a a^dag P, so

    d<P>/dt = -2 kappa_a [(1 + n_th) <a^dag a P> + n_th <a a^dag P>].

Gamma_Z(t) = -(d<P>/dt) / <P> then follows from expectation values of one simulation. Both
relations hold exactly in the truncated Fock space when a a^dag is the truncated product
(it is zero, not n + 1, on the top memory level).

Conventions: rates are decay rates of <P> in rad/us. The one-way flip rate |C+> -> |C->
is half of Gamma_Z. kappa_a is the memory loss rate (the slides' kappa_1).
"""

import math
from typing import NamedTuple

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg as sla
from dynamiqs import QArray
from jax.typing import ArrayLike
from numpy.typing import ArrayLike as NpArrayLike
from scipy.optimize import curve_fit

from catqubit.liouvillian import FockTruncation, parity_block, two_mode_liouvillian
from catqubit.models import two_mode_model
from catqubit.params import CatParams
from catqubit.rates import ZERO_MODE_SHIFT
from catqubit.simulate import DEFAULT_SOLVER_TOL
from catqubit.states import cat_state

MIN_FIT_POINTS = 4
MIN_LINE_POINTS = 3


class ParityOperators(NamedTuple):
    """P, a^dag a P and a a^dag P on the memory (tensored with the buffer identity)."""

    parity: QArray
    n_parity: QArray
    m_parity: QArray


class ParityTrace(NamedTuple):
    """<P>, <a^dag a P> and <a a^dag P> over time (us), as real numpy arrays."""

    t: np.ndarray
    parity: np.ndarray
    n_parity: np.ndarray
    m_parity: np.ndarray


class DecayFit(NamedTuple):
    """y = offset + amplitude exp(-rate t), with the 1-sigma error of the rate."""

    rate: float
    amplitude: float
    offset: float
    rate_stderr: float


class LineFit(NamedTuple):
    """y = slope x + intercept, with the 1-sigma error of the slope."""

    slope: float
    intercept: float
    slope_stderr: float


def _check_n_th(n_th: float) -> None:
    if n_th < 0:
        raise ValueError(f"n_th must be non-negative, got {n_th}")


def parity_operators(na: int, nb: int | None = None) -> ParityOperators:
    """Memory-parity moments; single-mode when nb is None, else memory (x) buffer."""
    a = dq.destroy(na)
    p = dq.parity(na)
    ops = (p, a.dag() @ a @ p, a @ a.dag() @ p)
    if nb is not None:
        ops = tuple(dq.tensor(op, dq.eye(nb)) for op in ops)
    return ParityOperators(*ops)


def parity_derivative(
    kappa_a: float, n_parity: NpArrayLike, m_parity: NpArrayLike, n_th: float = 0.0
) -> np.ndarray:
    """d<P>/dt = -2 kappa_a [(1 + n_th) <a^dag a P> + n_th <a a^dag P>] (exact)."""
    _check_n_th(n_th)
    n_parity, m_parity = np.asarray(n_parity), np.asarray(m_parity)
    return -2 * kappa_a * ((1 + n_th) * n_parity + n_th * m_parity)


def instantaneous_phase_flip_rate(
    kappa_a: float,
    parity: NpArrayLike,
    n_parity: NpArrayLike,
    m_parity: NpArrayLike,
    n_th: float = 0.0,
) -> np.ndarray:
    """Gamma_Z(t) = -(d<P>/dt) / <P>, from the exact identity (no fit, no differencing).

    With a steady-state offset P_inf it reads Gamma_Z (1 - P_inf / P(t)) once the fast
    modes have died out, so it lies slightly below the spectral rate.
    """
    return -parity_derivative(kappa_a, n_parity, m_parity, n_th) / np.asarray(parity)


def thermal_two_mode_jumps(
    na: int, nb: int, params: CatParams, n_th_a: float = 0.0, n_th_b: float = 0.0
) -> list[QArray]:
    """sqrt(k (1 + n)) c and sqrt(k n) c^dag for c = b, a (Réglade 2024, Eq. S4)."""
    _check_n_th(n_th_a)
    _check_n_th(n_th_b)
    a, b = dq.destroy(na, nb)
    jumps = []
    for rate, op, n_th in ((params.kappa_b, b, n_th_b), (params.kappa_a, a, n_th_a)):
        jumps.append(math.sqrt(rate * (1 + n_th)) * op)
        if n_th > 0:
            jumps.append(math.sqrt(rate * n_th) * op.dag())
    return jumps


def evolve_parity(
    params: CatParams,
    na: int,
    nb: int,
    tsave: ArrayLike,
    n_th_a: float = 0.0,
    n_th_b: float = 0.0,
    solver_tol: float = DEFAULT_SOLVER_TOL,
) -> ParityTrace:
    """Evolve |C+> (x) |0> under H_2ph with thermal loss and gain; return parity moments.

    Same Hamiltonian as catqubit.simulate.inflate; the jumps add thermal gain on both modes.
    """
    model = two_mode_model(na, nb, params)
    ops = parity_operators(na, nb)
    psi0 = dq.tensor(cat_state(na, params.alpha, +1), dq.fock(nb, 0))
    res = dq.mesolve(
        model.H,
        thermal_two_mode_jumps(na, nb, params, n_th_a, n_th_b),
        psi0,
        tsave,
        exp_ops=list(ops),
        method=dq.method.Tsit5(rtol=solver_tol, atol=solver_tol),
        save_states=False,
        progress_meter=False,
    )
    expects = np.real(np.asarray(res.expects))
    return ParityTrace(np.asarray(jnp.asarray(tsave)), *expects)


def steady_state_parity(
    params: CatParams, trunc: FockTruncation, n_th_a: float = 0.0, n_th_b: float = 0.0
) -> float:
    """Memory parity of the two-mode steady state (zero mode of the even sector).

    P_inf > 0 because |C-> flips to |C+> faster than the reverse (it holds more photons).
    Diagonal entries |i><i| always lie in the even sector, so the trace needs only them.
    """
    big_l = two_mode_liouvillian(params, trunc, n_th_a=n_th_a, n_th_b=n_th_b)
    even, idx = parity_block(big_l, trunc, odd=False)
    _, vecs = sla.eigs(even, k=1, sigma=ZERO_MODE_SHIFT, which="LM")
    rho_vec = np.zeros(trunc.dim**2, dtype=complex)
    rho_vec[idx] = vecs[:, 0]
    diag = rho_vec[np.arange(trunc.dim) * (trunc.dim + 1)]
    sign = 1.0 - 2.0 * (trunc.memory_photons % 2)
    return float((sign @ diag / diag.sum()).real)


# --- analytic references ------------------------------------------------------------------


def large_alpha_rate(kappa_a: float, alpha: float) -> float:
    """Gamma_Z = 2 kappa_a alpha^2: the challenge's prediction (alpha >> 1)."""
    return 2 * kappa_a * alpha**2


def even_cat_rate(kappa_a: float, alpha: float) -> float:
    """2 kappa_a n_+ with n_+ = alpha^2 tanh(alpha^2), the photon number of |C+>.

    The initial parity decay rate from |C+>; Guillaud, Cohen & Mirrahimi (arXiv:2203.03222,
    Sec. 4.3): a|C+-> = alpha tanh(alpha^2)^(+-1/2) |C-+>.
    """
    return 2 * kappa_a * alpha**2 * math.tanh(alpha**2)


def cat_manifold_rate(kappa_a: float, alpha: float) -> float:
    """Parity decay rate of the two-level cat-manifold model: 2 kappa_a alpha^2 coth(2 alpha^2).

    The flip rates are r_+ = kappa_a alpha^2 tanh(alpha^2) (|C+> -> |C->) and
    r_- = kappa_a alpha^2 coth(alpha^2); <P> relaxes at r_+ + r_-. Derived here from
    the a|C+-> relation of arXiv:2203.03222 Sec. 4.3, which drops the exponentially small
    terms (its result is 2 n kappa_1). It is the kappa_a / kappa_2 -> 0 limit of
    kappa_2 D[a^2 - alpha^2] + kappa_a D[a]: a^dag a rho leaves the manifold, which lowers
    the rate by about 0.5 kappa_a / kappa_2 (alpha = 1) to 0.25 kappa_a / kappa_2 (alpha = 1.5).
    """
    return 2 * kappa_a * alpha**2 / math.tanh(2 * alpha**2)


def cat_manifold_offset(alpha: float) -> float:
    """Steady-state parity P_inf = (r_- - r_+) / (r_- + r_+) = 1 / cosh(2 alpha^2)."""
    return 1 / math.cosh(2 * alpha**2)


def cat_manifold_parity(t: NpArrayLike, kappa_a: float, alpha: float) -> np.ndarray:
    """<P>(t) from |C+> in the cat manifold: P_inf + (1 - P_inf) exp(-Gamma_Z t)."""
    p_inf = cat_manifold_offset(alpha)
    rate = cat_manifold_rate(kappa_a, alpha)
    return p_inf + (1 - p_inf) * np.exp(-rate * np.asarray(t, dtype=float))


def thermal_rate(kappa_a: float, alpha: float, n_th: float) -> float:
    """Gamma_Z = 2 kappa_a [alpha^2 (1 + 2 n_th) + n_th], large alpha.

    Loss jumps occur at kappa_a (1 + n) <a^dag a> and gain jumps at kappa_a n <a a^dag>
    = kappa_a n (alpha^2 + 1); each flips parity. kappa_1 -> kappa_1 (1 + 2n) (Chamberland
    et al. 2022, arXiv:2012.04108) drops the "+ n_th".
    """
    _check_n_th(n_th)
    return 2 * kappa_a * (alpha**2 * (1 + 2 * n_th) + n_th)


# --- fits ---------------------------------------------------------------------------------


def _decay(t: np.ndarray, rate: float, amplitude: float, offset: float) -> np.ndarray:
    return offset + amplitude * np.exp(-rate * t)


def fit_parity_decay(t: NpArrayLike, parity: NpArrayLike) -> DecayFit:
    """Least-squares fit of <P>(t) = offset + amplitude exp(-rate t), free offset.

    The time origin is shifted to t[0] for conditioning; the amplitude refers to t = 0.
    """
    t, y = np.asarray(t, dtype=float), np.asarray(parity, dtype=float)
    if t.size < MIN_FIT_POINTS or t.shape != y.shape:
        raise ValueError(f"need at least {MIN_FIT_POINTS} (t, y) points of matching shape")
    shifted = t - t[0]
    guess_rate = -np.polyfit(shifted, np.log(np.clip(y, 1e-300, None)), 1)[0]
    guess = (max(guess_rate, 1e-12), y[0] - y[-1], y[-1])
    popt, pcov = curve_fit(_decay, shifted, y, p0=guess, maxfev=20000)
    rate, amplitude, offset = (float(v) for v in popt)
    return DecayFit(rate, amplitude * math.exp(rate * t[0]), offset, float(np.sqrt(pcov[0, 0])))


def fit_line(x: NpArrayLike, y: NpArrayLike) -> LineFit:
    """Ordinary least squares y = slope x + intercept."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < MIN_LINE_POINTS or x.shape != y.shape:
        raise ValueError(f"need at least {MIN_LINE_POINTS} (x, y) points of matching shape")
    coeffs, cov = np.polyfit(x, y, 1, cov="unscaled")
    residual = y - np.polyval(coeffs, x)
    dof = x.size - 2
    sigma2 = float(residual @ residual) / dof
    return LineFit(float(coeffs[0]), float(coeffs[1]), math.sqrt(sigma2 * cov[0, 0]))
