"""Buffer readout dynamics <b>(t) under the memory drive eps_z (a + a^dag) (Task 2.3).

Three independent descriptions of the readout signal, starting from |+-alpha> (x) |0>:

1. ``LinearBufferResponse``: linearize a = +-alpha + da about a well. With g2 alpha^2 = eps_d,
       b'' + (kappa_b / 2) b' + nu^2 b = -(+-1) nu eps_z,   nu = 2 alpha g2,   b(0) = b'(0) = 0,
   a damped oscillator with damping ratio zeta = kappa_b / (8 g2 alpha), eigenvalues
   -kappa_b / 4 +- i nu sqrt(1 - zeta^2) and steady state <b>_ss = -(+-1) eps_z / (2 alpha g2).
   (Gautier, Mirrahimi, Sarlette, PRX Quantum 4, 040316 (2023), Eq. 15b: UNVERIFIED equation
   number; the equation itself is re-derived here and checked against the full model.)
2. ``BufferSpectrum``: shift-invert eigenmodes of the full two-mode Liouvillian with eps_z,
   <b>(t) = plateau + sum_j c_j exp(lambda_j t), c_j = Tr[b r_j] <<l_j|rho0>>.
3. ``simulate_buffer_response``: time-domain reference through ``catqubit.simulate.inflate``.

Conventions: times in us, rates in rad/us. Ringing frequencies are angular (Im lambda,
rad/us); the ``*_mhz`` values divide by 2 pi. Decay rates are amplitude decay rates of <b>
(-Re lambda). Column-stacking vec: Tr[X rho] = vec(X^T) @ vec(rho).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import dynamiqs as dq
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.optimize import brentq

from catqubit.liouvillian import FockTruncation, two_mode_liouvillian, two_mode_operators, vec
from catqubit.params import CatParams
from catqubit.simulate import inflate

# Damping ratios within this distance of 1 use the critically damped closed form.
CRITICAL_DAMPING_TOL = 1e-9
# Settling-time search grid: samples per ringing period (or per fastest decay time).
SAMPLES_PER_PERIOD = 64
DEFAULT_SETTLING_TOL = 0.01

LU_ORDERING = "COLAMD"
# A right eigenvalue lambda pairs with a left one mu if |conj(mu) - lambda| <= PAIR_RTOL *
# max(|lambda|, |lambda - sigma|), and the next-closest left eigenvalue is at least
# 1 / PAIR_SEPARATION times farther away. Near-degenerate modes (0 and -Gamma_X ~ 1e-7)
# stay unambiguous because shift-invert resolves eigenvalues to ~1e-14 relative.
PAIR_RTOL = 1e-6
PAIR_SEPARATION = 1e-2
# Extra left eigenpairs requested so that a pair split by the k-th distance is still found.
EXTRA_LEFT_MODES = 4

# Full-model spectrum (c): modes near the oscillator eigenvalue, plus modes near zero found
# from a shift slightly off the steady state (0 and the bit-flip mode -Gamma_X are both kept).
DEFAULT_RINGING_MODES = 8
DEFAULT_SLOW_MODES = 4
SLOW_SHIFT = -1e-4
# Modes with |lambda| below this (rad/us) form the metastable plateau: the steady state and
# the bit-flip mode. It holds for t << 1 / PLATEAU_RATE = 100 us.
PLATEAU_RATE = 1e-2
# Two computed modes are the same mode if their eigenvalues agree to this relative level.
DUPLICATE_RTOL = 1e-9
SIGNS = (1, -1)


# ---------------------------------------------------------------- settling-time helpers


def _crossing(t0: float, t1: float, e0: float, e1: float) -> float:
    return t0 + (t1 - t0) * e0 / (e0 - e1)


def settling_time_from_curve(
    t: np.ndarray, y: np.ndarray, y_final: complex, tol: float = DEFAULT_SETTLING_TOL
) -> float:
    """Last time |y - y_final| > tol |y_final| on a sampled curve (linear interpolation).

    Returns t[0] if the curve never leaves the band. Raises ValueError if the last sample
    is still outside it, since the curve has then not settled within the window.
    """
    if y_final == 0:
        raise ValueError("settling needs a nonzero final value")
    t = np.asarray(t, dtype=float)
    excess = np.abs(np.asarray(y) - y_final) - tol * abs(y_final)
    outside = np.flatnonzero(excess > 0)
    if outside.size == 0:
        return float(t[0])
    i = outside[-1]
    if i == len(t) - 1:
        raise ValueError(f"curve has not settled within {tol:g} by t = {t[-1]:g}")
    return float(_crossing(t[i], t[i + 1], excess[i], excess[i + 1]))


def _last_crossing(excess: Callable[[np.ndarray], np.ndarray], t_max: float, dt: float) -> float:
    """Exact last root of excess(t) on [0, t_max], given excess(t) <= 0 beyond t_max."""
    t = np.linspace(0.0, t_max, max(2, math.ceil(t_max / dt) + 1))
    values = excess(t)
    outside = np.flatnonzero(values > 0)
    if outside.size == 0:
        return 0.0
    i = outside[-1]
    if i == len(t) - 1:  # pragma: no cover - only by rounding, since excess(t_max) <= 0
        return float(t_max)
    scalar = lambda s: float(excess(np.array([s]))[0])  # noqa: E731
    return float(brentq(scalar, t[i], t[i + 1], xtol=1e-14, rtol=1e-12))


# ---------------------------------------------------------------- (a) linear oscillator


@dataclass(frozen=True)
class LinearBufferResponse:
    """Linearized buffer response b'' + 2 zeta nu b' + nu^2 b = -sign nu eps_z, b(0)=b'(0)=0.

    nu = 2 alpha g2 (rad/us), zeta = kappa_b / (8 g2 alpha), sign = +1 for |+alpha>, -1 for
    |-alpha>. b(t) is real: the linear model has no codeword-independent (kappa_a) part.
    """

    nu: float
    zeta: float
    eps_z: float
    sign: int

    def __post_init__(self) -> None:
        if self.sign not in (1, -1):
            raise ValueError(f"sign must be +1 or -1, got {self.sign}")
        if self.nu <= 0:
            raise ValueError(f"nu must be positive, got {self.nu}")
        if self.zeta <= 0:
            raise ValueError(f"zeta must be positive, got {self.zeta}")

    @classmethod
    def from_params(cls, params: CatParams, eps_z: float, sign: int) -> "LinearBufferResponse":
        return cls(
            nu=2 * params.alpha * params.g2,
            zeta=params.kappa_b / (8 * params.g2 * params.alpha),
            eps_z=eps_z,
            sign=sign,
        )

    @property
    def damping_rate(self) -> float:
        """zeta nu = kappa_b / 4: amplitude decay rate of an underdamped ringing."""
        return self.zeta * self.nu

    @property
    def _regime(self) -> str:
        if abs(self.zeta - 1.0) <= CRITICAL_DAMPING_TOL:
            return "critical"
        return "under" if self.zeta < 1.0 else "over"

    @property
    def is_underdamped(self) -> bool:
        return self._regime == "under"

    @property
    def ringing_frequency(self) -> float:
        """nu sqrt(1 - zeta^2) in rad/us; 0.0 when critically or overdamped (no ringing)."""
        return self.nu * math.sqrt(1 - self.zeta**2) if self.is_underdamped else 0.0

    @property
    def ringing_frequency_mhz(self) -> float:
        return self.ringing_frequency / (2 * math.pi)

    @property
    def steady_state(self) -> float:
        """-sign eps_z / nu = -(+-1) eps_z / (2 alpha g2)."""
        return -self.sign * self.eps_z / self.nu

    @property
    def eigenvalues(self) -> np.ndarray:
        """Roots of l^2 + 2 zeta nu l + nu^2, slowest decay first (Im > 0 first if complex)."""
        root = self.nu * np.sqrt(complex(self.zeta**2 - 1))
        if self._regime == "critical":
            root = 0j
        lam = -self.damping_rate + np.array([root, -root])
        if not self.is_underdamped:
            lam = lam.real.astype(complex)
        return lam

    def _deviation(self, t: np.ndarray) -> np.ndarray:
        """b(t) - b_ss, which solves the homogeneous equation with x(0) = -b_ss, x'(0) = 0."""
        t = np.asarray(t, dtype=float)
        x0 = -self.steady_state
        gamma = self.damping_rate
        if self._regime == "under":
            w = self.ringing_frequency
            return x0 * np.exp(-gamma * t) * (np.cos(w * t) + gamma / w * np.sin(w * t))
        if self._regime == "critical":
            return x0 * np.exp(-self.nu * t) * (1 + self.nu * t)
        r1, r2 = self.eigenvalues.real
        return x0 * (r2 * np.exp(r1 * t) - r1 * np.exp(r2 * t)) / (r2 - r1)

    def mean_b(self, t: np.ndarray) -> np.ndarray:
        """Closed-form <b>(t) (real) on the time grid t (us)."""
        return self.steady_state + self._deviation(t)

    def envelope_settling_time(self, tol: float = DEFAULT_SETTLING_TOL) -> float:
        """ln(1 / (tol sqrt(1 - zeta^2))) / (kappa_b / 4): where the envelope reaches tol."""
        if not self.is_underdamped:
            raise ValueError("the envelope estimate needs an underdamped oscillator")
        return math.log(1 / (tol * math.sqrt(1 - self.zeta**2))) / self.damping_rate

    def _settling_bound(self, tol: float) -> float:
        """A time after which |b - b_ss| <= tol |b_ss| for sure."""
        if self.is_underdamped:
            return self.envelope_settling_time(tol)
        if self._regime == "critical":
            return 2 * math.log(2 / tol) / self.nu  # (1 + x) e^-x <= 2 e^(-x/2)
        r1, r2 = self.eigenvalues.real
        return math.log((abs(r1) + abs(r2)) / ((r1 - r2) * tol)) / abs(r1)

    def settling_time(self, tol: float = DEFAULT_SETTLING_TOL) -> float:
        """Exact last time with |b(t) - b_ss| > tol |b_ss| (grid scan, then brentq)."""
        if self.steady_state == 0:
            raise ValueError("settling needs a nonzero steady state (eps_z != 0)")
        timescale = 1 / abs(self.eigenvalues[-1].real)
        if self.is_underdamped:
            timescale = min(timescale, 2 * math.pi / self.ringing_frequency)
        bound = abs(self.steady_state) * tol
        return _last_crossing(
            lambda t: np.abs(self._deviation(t)) - bound,
            self._settling_bound(tol),
            timescale / SAMPLES_PER_PERIOD,
        )


# ---------------------------------------------------------------- (b) Liouvillian modes


@dataclass(frozen=True)
class LiouvillianModes:
    """Biorthonormal eigenmodes of a (non-Hermitian) Liouvillian near a shift sigma.

    values: (k,) eigenvalues, sorted by decreasing real part. right: (n, k) right
    eigenvectors r_j (columns, column-stacking vec). left: (n, k) left eigenvectors l_j with
    L^H l_j = conj(lambda_j) l_j and l_j.conj() @ r_i = delta_ij. residual: max over j of
    ||L r_j - lambda_j r_j|| and ||L^H l_j - conj(lambda_j) l_j|| for unit vectors, relative
    to max(|lambda_j|, 1).
    """

    values: np.ndarray
    right: np.ndarray
    left: np.ndarray
    residual: float


def _pair_left_to_right(right_vals: np.ndarray, left_vals: np.ndarray, sigma: complex) -> list:
    """Index of the left eigenvalue conj(mu) matching each right eigenvalue lambda."""
    partners = []
    for lam in right_vals:
        dist = np.abs(np.conj(left_vals) - lam)
        order = np.argsort(dist)
        tol = PAIR_RTOL * max(abs(lam), abs(lam - sigma))
        if not dist[order[0]] < tol:
            raise ValueError(f"eigenvalue {lam:.6g} has no left partner within {tol:.2g}")
        if len(order) > 1 and dist[order[0]] > PAIR_SEPARATION * dist[order[1]]:
            raise ValueError(f"ambiguous left/right pairing near eigenvalue {lam:.6g}")
        partners.append(int(order[0]))
    if len(set(partners)) != len(partners):
        raise ValueError("ambiguous left/right pairing: a left mode matches two right modes")
    return partners


def _relative_residuals(op: sp.spmatrix, vecs: np.ndarray, vals: np.ndarray) -> np.ndarray:
    return np.linalg.norm(op @ vecs - vecs * vals, axis=0) / np.maximum(np.abs(vals), 1.0)


def _paired_left_vectors(
    big_lh: sp.spmatrix, solve_h: sla.LinearOperator, vals: np.ndarray, sigma: complex, k: int
) -> np.ndarray:
    """Left eigenvectors matched to vals: k modes first, k + EXTRA_LEFT_MODES on a retry.

    The retry catches a mode pair split by the k-th distance (e.g. lambda, conj(lambda) for a
    real sigma). Extra modes are not requested up front: shift-invert separates them poorly,
    so they cost many more solves.
    """

    def attempt(k_left: int) -> np.ndarray:
        lvals, left = sla.eigs(big_lh, k=k_left, sigma=np.conj(sigma), OPinv=solve_h, which="LM")
        return left[:, _pair_left_to_right(vals, lvals, sigma)]

    k_max = min(k + EXTRA_LEFT_MODES, big_lh.shape[0] - 2)
    try:
        return attempt(k)
    except ValueError:
        if k_max <= k:
            raise
    return attempt(k_max)


def liouvillian_modes(big_l: sp.spmatrix, sigma: complex, k: int) -> LiouvillianModes:
    """The k eigenmodes of big_l closest to sigma, with biorthonormal left eigenvectors.

    One sparse LU of (L - sigma I) (COLAMD) drives both shift-invert Arnoldi runs: on L with
    lu.solve, and on L^H with lu.solve(trans='H') since (L - sigma I)^H = L^H - conj(sigma) I.
    Left and right modes are paired by eigenvalue (ValueError if a pairing is missing or
    ambiguous), normalized to unit norm for the residual, then l_j is rescaled so that
    l_j^H r_j = 1. Keep sigma off any eigenvalue (e.g. slightly off 0 for a steady state).
    """
    n = big_l.shape[0]
    big_l = big_l.tocsc().astype(complex)
    big_lh = big_l.conj().T.tocsc()
    lu = sla.splu((big_l - sigma * sp.identity(n, format="csc")).tocsc(), permc_spec=LU_ORDERING)
    solve = sla.LinearOperator((n, n), matvec=lu.solve, dtype=complex)
    solve_h = sla.LinearOperator((n, n), matvec=lambda x: lu.solve(x, trans="H"), dtype=complex)
    vals, right = sla.eigs(big_l, k=k, sigma=sigma, OPinv=solve, which="LM")
    order = np.argsort(-vals.real, kind="stable")
    vals, right = vals[order], right[:, order]
    left = _paired_left_vectors(big_lh, solve_h, vals, sigma, k)
    right = right / np.linalg.norm(right, axis=0)
    left = left / np.linalg.norm(left, axis=0)
    residual = max(
        np.max(_relative_residuals(big_l, right, vals)),
        np.max(_relative_residuals(big_lh, left, np.conj(vals))),
    )
    left = left / np.conj(np.sum(left.conj() * right, axis=0))
    return LiouvillianModes(vals, right, left, float(residual))


def modal_weights(
    modes: LiouvillianModes, observable_row: np.ndarray, rho0_vec: np.ndarray
) -> np.ndarray:
    """c_j = (observable_row @ r_j) (l_j^H rho0): Tr[X e^(Lt) rho0] = sum_j c_j e^(lambda_j t).

    observable_row = vec(X^T), so that observable_row @ vec(rho) = Tr[X rho].
    """
    return (observable_row @ modes.right) * (modes.left.conj().T @ rho0_vec)


def _hermitian_conjugate(vecs: np.ndarray, dim: int) -> np.ndarray:
    mats = vecs.T.reshape(-1, dim, dim).transpose(0, 2, 1)  # column-stacked -> matrices
    return np.conj(mats).reshape(-1, dim * dim).T  # vec(X^dag) for each column


def hermitian_conjugate_modes(modes: LiouvillianModes, dim: int) -> LiouvillianModes:
    """Partner modes lambda* <-> (vec(R^dag), vec(L^dag)) of a Hermiticity-preserving L.

    A Lindbladian maps rho^dag to L(rho)^dag, so R^dag is a right eigenmatrix for conj(lambda);
    the adjoint map does the same for the left eigenmatrix. Biorthonormality is preserved.
    dim is the Hilbert-space dimension (vec length dim^2). Not valid for a general matrix.
    """
    return LiouvillianModes(
        np.conj(modes.values),
        _hermitian_conjugate(modes.right, dim),
        _hermitian_conjugate(modes.left, dim),
        modes.residual,
    )


def _merge_modes(*mode_sets: LiouvillianModes) -> LiouvillianModes:
    """Union of mode sets, dropping repeats of the same eigenvalue, by decreasing Re lambda."""
    kept: list[tuple[complex, np.ndarray, np.ndarray]] = []
    for modes in mode_sets:
        for j, lam in enumerate(modes.values):
            tol = DUPLICATE_RTOL * max(abs(lam), 1.0)
            if all(abs(lam - other) > tol for other, _, _ in kept):
                kept.append((lam, modes.right[:, j], modes.left[:, j]))
    kept.sort(key=lambda mode: -mode[0].real)
    return LiouvillianModes(
        np.array([m[0] for m in kept]),
        np.stack([m[1] for m in kept], axis=1),
        np.stack([m[2] for m in kept], axis=1),
        max(m.residual for m in mode_sets),
    )


# ---------------------------------------------------------------- (c) spectral response


def _check_sign(sign: int) -> None:
    if sign not in SIGNS:
        raise ValueError(f"sign must be +1 or -1, got {sign}")


def codeword_density(params: CatParams, trunc: FockTruncation, sign: int) -> np.ndarray:
    """vec of |sign alpha><sign alpha| (x) |0><0| on the kept states, renormalized."""
    _check_sign(sign)
    memory = np.asarray(dq.coherent(trunc.na, sign * params.alpha).to_jax()).ravel()
    buffer = np.zeros(trunc.nb_max)
    buffer[0] = 1.0
    psi = np.kron(memory, buffer)[trunc.kept_indices()]
    psi = psi / np.linalg.norm(psi)
    return vec(np.outer(psi, psi.conj()))


@dataclass(frozen=True)
class SpectralBufferResponse:
    """<b>(t) = plateau + sum_j weights_j exp(values_j t) from full-Liouvillian modes.

    plateau = sum of the weights of the modes with |lambda| < PLATEAU_RATE (steady state and
    bit-flip mode): the metastable value for 1 / (kappa_b / 4) << t << 1 / Gamma_X. It is
    complex: Re carries the codeword (-sign eps_z / (2 alpha g2)); Im is a codeword-
    independent kappa_a part. sign = +1 / -1 for |+-alpha>, 0 for the differential signal
    (<b>+ - <b>-) / 2. Only the kept modes enter, so the sum misses the fastest transients
    near t = 0. residual is the eigen-residual of the modes (see LiouvillianModes).
    """

    values: np.ndarray
    weights: np.ndarray
    plateau: complex
    plateau_values: np.ndarray
    plateau_weights: np.ndarray
    residual: float
    sign: int

    @property
    def _ringing_index(self) -> int:
        oscillating = np.flatnonzero(self.values.imag > 0)
        return int(oscillating[np.argmax(np.abs(self.weights[oscillating]))])

    @property
    def ringing_eigenvalue(self) -> complex:
        """The oscillating mode (Im lambda > 0) with the largest weight |c_j|."""
        return complex(self.values[self._ringing_index])

    @property
    def ringing_weight(self) -> complex:
        return complex(self.weights[self._ringing_index])

    @property
    def ringing_frequency(self) -> float:
        """Im lambda of the dominant ringing mode, rad/us."""
        return self.ringing_eigenvalue.imag

    @property
    def ringing_frequency_mhz(self) -> float:
        return self.ringing_frequency / (2 * math.pi)

    @property
    def decay_rate(self) -> float:
        """-Re lambda of the dominant ringing mode: amplitude decay rate of <b>, rad/us."""
        return -self.ringing_eigenvalue.real

    def _deviation(self, t: np.ndarray) -> np.ndarray:
        return np.exp(np.outer(np.asarray(t, dtype=float), self.values)) @ self.weights

    def mean_b(self, t: np.ndarray) -> np.ndarray:
        """plateau + sum_j c_j exp(lambda_j t) on the time grid t (us); complex."""
        return self.plateau + self._deviation(t)

    def settling_time(self, tol: float = DEFAULT_SETTLING_TOL) -> float:
        """Exact last time with |<b>(t) - plateau| > tol |plateau| for the modal sum."""
        if self.plateau == 0:
            raise ValueError("settling needs a nonzero plateau")
        threshold = tol * abs(self.plateau)
        magnitudes, rates = np.abs(self.weights), -self.values.real
        bound = lambda t: float(np.sum(magnitudes * np.exp(-rates * t))) - threshold  # noqa: E731
        if bound(0.0) <= 0:
            return 0.0
        t_hi = math.log(np.sum(magnitudes) / threshold) / np.min(rates)
        t_max = brentq(bound, 0.0, t_hi)
        dt = 2 * math.pi / np.max(np.abs(self.values)) / SAMPLES_PER_PERIOD
        return _last_crossing(lambda t: np.abs(self._deviation(t)) - threshold, t_max, dt)


@dataclass(frozen=True)
class BufferSpectrum:
    """Modes of the full two-mode Liouvillian with eps_z, and the readout row vec(b^T).

    modes merges the modes nearest the oscillator eigenvalue, their Hermitian-conjugate
    partners (conj(lambda)) and the modes nearest zero, all biorthonormal.
    """

    params: CatParams
    trunc: FockTruncation
    eps_z: float
    modes: LiouvillianModes
    observable_row: np.ndarray
    plateau_rate: float = PLATEAU_RATE

    def _from_weights(self, weights: np.ndarray, sign: int) -> SpectralBufferResponse:
        slow = np.abs(self.modes.values) < self.plateau_rate
        return SpectralBufferResponse(
            values=self.modes.values[~slow],
            weights=weights[~slow],
            plateau=complex(np.sum(weights[slow])),
            plateau_values=self.modes.values[slow],
            plateau_weights=weights[slow],
            residual=self.modes.residual,
            sign=sign,
        )

    def weights(self, sign: int) -> np.ndarray:
        """Modal weights c_j = Tr[b r_j] <<l_j | rho0>> for the codeword |sign alpha>."""
        rho0 = codeword_density(self.params, self.trunc, sign)
        return modal_weights(self.modes, self.observable_row, rho0)

    def response(self, sign: int) -> SpectralBufferResponse:
        return self._from_weights(self.weights(sign), sign)

    def differential(self) -> SpectralBufferResponse:
        """(<b>+ - <b>-) / 2: the codeword-independent part cancels."""
        return self._from_weights((self.weights(1) - self.weights(-1)) / 2, 0)


def buffer_spectrum(
    params: CatParams,
    trunc: FockTruncation,
    eps_z: float,
    k_ringing: int = DEFAULT_RINGING_MODES,
    k_slow: int = DEFAULT_SLOW_MODES,
) -> BufferSpectrum:
    """Shift-invert modes of the full L(eps_z) near the oscillator root and near zero.

    The ringing shift is the linear eigenvalue -kappa_b/4 + i nu sqrt(1 - zeta^2) (the slow
    real root if overdamped). Parity is broken by eps_z, so the full L is used (two LUs).
    """
    big_l = two_mode_liouvillian(params, trunc, eps_z=eps_z)
    sigma = LinearBufferResponse.from_params(params, eps_z, 1).eigenvalues[0]
    ringing = liouvillian_modes(big_l, sigma=sigma, k=k_ringing)
    slow = liouvillian_modes(big_l, sigma=SLOW_SHIFT, k=k_slow)
    modes = _merge_modes(
        ringing,
        hermitian_conjugate_modes(ringing, trunc.dim),
        slow,
        hermitian_conjugate_modes(slow, trunc.dim),
    )
    _, b = two_mode_operators(trunc)
    return BufferSpectrum(params, trunc, eps_z, modes, vec(b.toarray().T))


# ---------------------------------------------------------------- (d) time domain


def simulate_buffer_response(
    params: CatParams, na: int, nb: int, eps_z: float, sign: int, tsave: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(t, <b>(t)) from |sign alpha> (x) |0> under H_2ph + eps_z (a + a^dag), via inflate.

    Rectangular truncation na x nb, solver tolerance 1e-9. <b> is complex.
    """
    _check_sign(sign)
    a, b = dq.destroy(na, nb)
    result = inflate(
        na,
        nb,
        params,
        np.asarray(tsave, dtype=float),
        memory0=dq.coherent(na, sign * params.alpha),
        extra_exp_ops=[b],
        save_states=False,
        extra_H=eps_z * (a + a.dag()),
    )
    return np.asarray(tsave, dtype=float), np.asarray(result.expects[2])
