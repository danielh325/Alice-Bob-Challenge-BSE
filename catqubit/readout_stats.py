"""Exact statistics of the integrated homodyne readout record (Task 2.3, plan R8).

Record. The buffer output c = sqrt(kappa_b) b is measured by homodyne detection at phase phi
with efficiency eta: dY = sqrt(eta) <c e^{-i phi} + c^dag e^{i phi}> dt + dW. The filtered
charge is Q(T) = int_0^T f(t) dY(t) (boxcar f = 1 by default).

Method (Landi et al., PRX Quantum 5, 020201 (2024), arXiv:2303.04270, Eqs. 124 and 142-144):
    E[exp(s Q(T))] = exp(s^2 F2(T) / 2) Tr[T exp(int_0^T (L + s f(t) J) dt) rho0],
    J rho = sqrt(eta) (c e^{-i phi} rho + rho c^dag e^{i phi}),   F2(T) = int_0^T f^2 dt.
The white-noise factor exp(s^2 F2 / 2) is applied analytically. s = i k gives the
characteristic function, which is Fourier-inverted to P(Q | codeword); real s gives cumulants
(finite differences) and Chernoff bounds.

Long windows. Up to t_switch the tilted state is propagated exactly (expm_multiply). After it,
only the slowest modes of L + s f J survive (the two well populations, plus slow coherences),
so the state is continued with a few shift-invert eigenpairs. The neglected fast part at
t_switch is reported as `switch_error`; it only decays afterwards.

Vectors use column stacking (catqubit.liouvillian): Tr[X rho] = vec(X^T) @ vec(rho).
"""

import math
import multiprocessing as mp
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import brentq
from scipy.sparse.linalg import expm_multiply
from scipy.special import erfc

from catqubit.liouvillian import FockTruncation, two_mode_liouvillian, two_mode_operators, vec
from catqubit.params import CatParams
from catqubit.readout_dynamics import codeword_density, liouvillian_modes

# Five-point finite-difference step in real s for the record mean and variance.
FD_STEP = 1e-3
# Fourier modes whose white-noise factor exp(-k^2 F2 / 2) falls below this are dropped.
NOISE_FLOOR = 1e-16
# The Fourier window holds every codeword mean +- this many standard deviations: a Gaussian
# tail beyond 8 sigma is ~1e-15, and flip-induced tails lie between the two means.
N_SIGMA_WINDOW = 8.0
# Q samples per Nyquist interval pi / k_max of the band-limited density.
Q_OVERSAMPLE = 4
# Shift-invert point for the slow modes: near zero, off the real and imaginary axes so it never
# coincides with an eigenvalue (s = 0 makes L singular).
SLOW_SHIFT = 0.0123 + 0.0071j
DEFAULT_SLOW_MODES = 6
# Density crossings below this fraction of the peak are numerical noise, not refined.
CROSSING_FLOOR = 1e-12


# ---------------------------------------------------------------- detector and generator


@dataclass(frozen=True)
class Homodyne:
    """Homodyne detection of c e^{-i phi} + h.c. with efficiency eta (heterodyne: eta / 2)."""

    eta: float = 1.0
    phi: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.eta <= 1:
            raise ValueError(f"eta must be in (0, 1], got {self.eta}")


IDEAL_HOMODYNE = Homodyne()


@dataclass(frozen=True)
class TiltedGenerator:
    """L (Lindbladian), J (tilt superoperator) and vec(identity), all column-stacking."""

    liouvillian: sp.csr_matrix
    tilt: sp.csr_matrix
    trace_row: np.ndarray

    def at(self, s: complex) -> sp.csr_matrix:
        """L + s J."""
        return (self.liouvillian + s * self.tilt).tocsr()


def tilt_superoperator(c: sp.spmatrix, detector: Homodyne) -> sp.csr_matrix:
    """Matrix of rho -> sqrt(eta) (c e^{-i phi} rho + rho c^dag e^{i phi}) on vec(rho)."""
    ce = sp.csr_matrix(c, dtype=complex) * np.exp(-1j * detector.phi)
    eye = sp.identity(ce.shape[0], format="csr", dtype=complex)
    spre = sp.kron(eye, ce)  # vec(A rho) = (I kron A) vec(rho)
    spost = sp.kron(ce.conj(), eye)  # vec(rho B) = (B^T kron I) vec(rho), B^T = conj(ce)
    return (math.sqrt(detector.eta) * (spre + spost)).tocsr()


def homodyne_generator(
    big_l: sp.spmatrix, c: sp.spmatrix, detector: Homodyne = IDEAL_HOMODYNE
) -> TiltedGenerator:
    """Tilted generator for any Lindbladian big_l whose output field c is monitored."""
    dim = c.shape[0]
    return TiltedGenerator(
        liouvillian=sp.csr_matrix(big_l, dtype=complex),
        tilt=tilt_superoperator(c, detector),
        trace_row=vec(np.identity(dim, dtype=complex)),
    )


def cat_readout_generator(
    params: CatParams, trunc: FockTruncation, eps_z: float, detector: Homodyne = IDEAL_HOMODYNE
) -> TiltedGenerator:
    """Two-mode model with readout drive eps_z (a + a^dag); the buffer output is measured."""
    big_l = two_mode_liouvillian(params, trunc, eps_z=eps_z)
    _, b = two_mode_operators(trunc)
    return homodyne_generator(big_l, math.sqrt(params.kappa_b) * b, detector)


def codeword_state(params: CatParams, trunc: FockTruncation, sign: int) -> np.ndarray:
    """vec(|sign alpha><sign alpha| (x) |0><0|) on the kept states (same as the dynamics module)."""
    return codeword_density(params, trunc, sign).astype(complex)


# ---------------------------------------------------------------- filters


@dataclass(frozen=True)
class PiecewiseFilter:
    """f(t) = weights[i] on [edges[i], edges[i+1]); the last weight holds for all later t."""

    edges: tuple[float, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.edges) != len(self.weights) or not self.edges:
            raise ValueError("edges and weights must be non-empty and of equal length")
        if self.edges[0] != 0.0:
            raise ValueError(f"the first edge must be 0, got {self.edges[0]}")
        if np.any(np.diff(self.edges) <= 0):
            raise ValueError(f"edges must be strictly increasing, got {self.edges}")

    @property
    def last_edge(self) -> float:
        return self.edges[-1]

    def weight(self, t: float) -> float:
        return self.weights[int(np.searchsorted(self.edges, t, side="right")) - 1]

    def squared_integral(self, times: np.ndarray) -> np.ndarray:
        """F2(T) = int_0^T f(t)^2 dt."""
        starts = np.asarray(self.edges)
        ends = np.append(starts[1:], np.inf)
        w2 = np.asarray(self.weights) ** 2
        t = np.asarray(times, dtype=float)[:, None]
        overlap = np.clip(np.minimum(t, ends) - starts, 0.0, None)
        return overlap @ w2


BOXCAR = PiecewiseFilter(edges=(0.0,), weights=(1.0,))


# ---------------------------------------------------------------- tilted propagation


@dataclass(frozen=True)
class SlowModes:
    """Continue the tilted state after t_switch with its n_modes eigenmodes closest to sigma.

    from_start (constant filters only): project rho0 itself onto the slow modes, skipping the
    exact propagation to t_switch. For T >= t_switch both forms agree up to the fast modes'
    decay e^{-gap t_switch}, but the dropped part is then not measured (switch_error = nan).
    """

    t_switch: float
    n_modes: int = DEFAULT_SLOW_MODES
    sigma: complex = SLOW_SHIFT
    from_start: bool = False

    def __post_init__(self) -> None:
        if self.t_switch <= 0:
            raise ValueError(f"t_switch must be positive, got {self.t_switch}")
        if self.n_modes < 2:
            raise ValueError(f"n_modes must be at least 2, got {self.n_modes}")


def _check_times(times: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size == 0 or t[0] <= 0 or np.any(np.diff(t) <= 0):
        raise ValueError(f"times must be positive and strictly increasing, got {times}")
    return t


def _propagate(
    generator_for: Callable[[float], sp.csr_matrix],
    v0: np.ndarray,
    stops: np.ndarray,
    filt: PiecewiseFilter,
) -> list[np.ndarray]:
    """States at each time in stops (sorted), stepping through the filter edges with
    expm_multiply; generator_for(w) is the generator while the filter weight is w."""
    marks = np.union1d(stops, [e for e in filt.edges if 0 < e < stops[-1]])
    generators: dict[float, sp.csr_matrix] = {}
    v, t_prev, out = v0, 0.0, []
    for t in marks:
        w = filt.weight(t_prev)
        if w not in generators:
            generators[w] = generator_for(w)
        v = expm_multiply(generators[w] * (t - t_prev), v)
        t_prev = t
        if np.isin(t, stops):
            out.append(v)
    return out


def _slow_continuation(
    gen: TiltedGenerator,
    v_switch: np.ndarray,
    s: complex,
    times: np.ndarray,
    filt: PiecewiseFilter,
    slow: SlowModes,
    t_origin: float,
) -> tuple[np.ndarray, float]:
    """Tr[exp((L + s w J)(T - t_origin)) v_switch] from the slow modes, and the dropped part
    at t_origin (nan when t_origin = 0, i.e. from_start)."""
    tr_switch = gen.trace_row @ v_switch
    s_last = s * filt.weights[-1]
    if s_last == 0:  # L alone preserves the trace
        return np.repeat(tr_switch[:, None], len(times), axis=1), 0.0
    modes = liouvillian_modes(gen.at(s_last), slow.sigma, slow.n_modes)
    coeffs = modes.left.conj().T @ v_switch  # (n_modes, n_rho)
    mode_traces = gen.trace_row @ modes.right  # (n_modes,)
    decay = np.exp(np.outer(times - t_origin, modes.values))  # (nT, n_modes)
    values = (decay * mode_traces) @ coeffs  # (nT, n_rho)
    if t_origin == 0:
        return values.T, math.nan
    kept = mode_traces @ coeffs
    noise = np.exp(s**2 * filt.squared_integral(np.array([slow.t_switch]))[0] / 2)
    # Dropped part of E[e^{sQ}] at t_switch: absolute for |E| <= 1 (s = ik), else relative.
    remainder = np.abs((tr_switch - kept) * noise) / np.maximum(np.abs(tr_switch * noise), 1.0)
    return values.T, float(np.max(remainder))


def _tilted_trace_and_error(
    gen: TiltedGenerator,
    rho0s: Sequence[np.ndarray],
    s: complex,
    times: np.ndarray,
    filt: PiecewiseFilter = BOXCAR,
    slow: SlowModes | None = None,
) -> tuple[np.ndarray, float]:
    t = _check_times(times)
    v0 = np.column_stack(rho0s).astype(complex)
    long = slow is not None and t[-1] > slow.t_switch
    if long and filt.last_edge > slow.t_switch:
        raise ValueError(
            f"t_switch={slow.t_switch} must not precede the last filter edge {filt.last_edge}"
        )
    if long and slow.from_start and filt.last_edge > 0:
        raise ValueError("from_start needs a constant filter (a single edge at 0)")
    short_t = t[t <= slow.t_switch] if long else t
    to_switch = long and not slow.from_start
    stops = np.append(short_t, slow.t_switch) if to_switch else short_t
    stops = np.unique(stops)
    states = _propagate(lambda w: gen.at(s * w), v0, stops, filt) if stops.size else []
    values = [gen.trace_row @ v for v in states[: len(short_t)]]
    out = np.array(values).T if values else np.zeros((v0.shape[1], 0), complex)
    if not long:
        return out, 0.0
    start, origin = (v0, 0.0) if slow.from_start else (states[-1], slow.t_switch)
    tail, remainder = _slow_continuation(gen, start, s, t[t > slow.t_switch], filt, slow, origin)
    return np.concatenate([out, tail], axis=1), remainder


def tilted_trace(
    gen: TiltedGenerator,
    rho0s: Sequence[np.ndarray],
    s: complex,
    times: np.ndarray,
    filt: PiecewiseFilter = BOXCAR,
    slow: SlowModes | None = None,
) -> np.ndarray:
    """Tr[T exp(int_0^T (L + s f J) dt) rho0] for each rho0 and T: shape (n_rho, nT).

    This excludes the white-noise factor exp(s^2 F2 / 2). With slow, times after
    slow.t_switch use the slow-mode continuation (see the module docstring).
    """
    return _tilted_trace_and_error(gen, rho0s, s, times, filt, slow)[0]


_WORKER_ARGS: tuple = ()


def _init_worker(*args) -> None:
    global _WORKER_ARGS
    _WORKER_ARGS = args


def _worker_trace(s: complex) -> tuple[np.ndarray, float]:
    gen, rho0s, times, filt, slow = _WORKER_ARGS
    return _tilted_trace_and_error(gen, rho0s, s, times, filt, slow)


def _traces_for_s(
    gen: TiltedGenerator,
    rho0s: Sequence[np.ndarray],
    s_values: Sequence[complex],
    times: np.ndarray,
    filt: PiecewiseFilter,
    slow: SlowModes | None,
    workers: int,
) -> tuple[np.ndarray, float]:
    """Tilted traces (n_rho, ns, nT) without the noise factor, and the worst fast remainder."""
    args = (gen, list(rho0s), _check_times(times), filt, slow)
    if workers > 1:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            workers, mp_context=ctx, initializer=_init_worker, initargs=args
        ) as ex:
            results = list(ex.map(_worker_trace, list(s_values)))
    else:
        results = [_tilted_trace_and_error(gen, args[1], s, args[2], filt, slow) for s in s_values]
    traces = np.stack([r[0] for r in results], axis=1)
    return traces, _worst(r[1] for r in results)


def _worst(errors) -> float:
    """Largest dropped part; nan if any part was not measured (from_start)."""
    values = list(errors)
    return math.nan if any(math.isnan(e) for e in values) else max(values, default=0.0)


def _generating_function_and_error(
    gen: TiltedGenerator,
    rho0s: Sequence[np.ndarray],
    s_values: Sequence[complex],
    times: np.ndarray,
    filt: PiecewiseFilter = BOXCAR,
    slow: SlowModes | None = None,
    workers: int = 1,
) -> tuple[np.ndarray, float]:
    traces, remainder = _traces_for_s(gen, rho0s, s_values, times, filt, slow, workers)
    s = np.asarray(s_values, dtype=complex)
    noise = np.exp(np.outer(s**2, filt.squared_integral(np.asarray(times))) / 2)
    return traces * noise[None], remainder


def generating_function(
    gen: TiltedGenerator,
    rho0s: Sequence[np.ndarray],
    s_values: Sequence[complex],
    times: np.ndarray,
    filt: PiecewiseFilter = BOXCAR,
    slow: SlowModes | None = None,
    workers: int = 1,
) -> np.ndarray:
    """E[exp(s Q(T))] for each rho0, s and T: shape (n_rho, ns, nT). s = i k: characteristic fn.

    workers > 1 spreads the s values over spawned processes (one LU / expm chain each).
    """
    return _generating_function_and_error(gen, rho0s, s_values, times, filt, slow, workers)[0]


# ---------------------------------------------------------------- cumulants


def record_cumulants(
    gen: TiltedGenerator, rho0: np.ndarray, times: np.ndarray, filt: PiecewiseFilter = BOXCAR
) -> tuple[np.ndarray, np.ndarray]:
    """Exact mean and variance of Q(T), with no finite differences.

    The block generator [[L, 0, 0], [fJ, L, 0], [0, fJ, L]] propagates (v, dv/ds, d2v/ds2 / 2)
    at s = 0, so mean = Tr x1 and var = F2 + 2 Tr x2 - (Tr x1)^2. Cost: 3x the dimension.
    """
    t = _check_times(times)
    n = rho0.shape[0]
    zero = sp.csr_matrix((n, n), dtype=complex)
    lap = gen.liouvillian

    def block(w: float) -> sp.csr_matrix:
        j = w * gen.tilt
        return sp.bmat([[lap, zero, zero], [j, lap, zero], [zero, j, lap]], format="csr")

    u0 = np.concatenate([rho0, np.zeros(2 * n)]).astype(complex)
    states = _propagate(block, u0, t, filt)
    tr = gen.trace_row
    mean = np.array([(tr @ u[n : 2 * n]).real for u in states])
    second = np.array([(tr @ u[2 * n :]).real for u in states])
    return mean, filt.squared_integral(t) + 2 * second - mean**2


def fd_cumulants(
    gen: TiltedGenerator,
    rho0s: Sequence[np.ndarray],
    times: np.ndarray,
    filt: PiecewiseFilter = BOXCAR,
    slow: SlowModes | None = None,
    workers: int = 1,
    step: float = FD_STEP,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance (n_rho, nT) from five-point differences of log E[e^{sQ}] at real s."""
    s_values = [-2 * step, -step, step, 2 * step]
    z = _generating_function_and_error(gen, rho0s, s_values, times, filt, slow, workers)[0]
    lz = np.log(z.real)  # (n_rho, 4, nT); log Z(0) = 0
    m2, m1, p1, p2 = (lz[:, i] for i in range(4))
    mean = (-p2 + 8 * p1 - 8 * m1 + m2) / (12 * step)
    var = (-p2 + 16 * p1 + 16 * m1 - m2) / (12 * step**2)
    return mean, var


# ---------------------------------------------------------------- Fourier inversion


@dataclass(frozen=True)
class FourierGrid:
    """k_n = n dk, n = 0..N: the density is resolved on a periodic Q window of length 2 pi / dk."""

    dk: float
    k_max: float

    @classmethod
    def covering(
        cls, half_width: float, f2_min: float, floor: float = NOISE_FLOOR
    ) -> "FourierGrid":
        """Window >= 2 half_width; k_max where the noise factor at F2 = f2_min reaches floor."""
        dk = math.pi / half_width
        k_need = math.sqrt(2 * math.log(1 / floor) / f2_min)
        return cls(dk=dk, k_max=dk * math.ceil(k_need / dk))

    @property
    def window(self) -> float:
        return 2 * math.pi / self.dk

    @property
    def ks(self) -> np.ndarray:
        return self.dk * np.arange(round(self.k_max / self.dk) + 1)


def _fourier_density(chi: np.ndarray, ks: np.ndarray, window: float, q: np.ndarray) -> np.ndarray:
    """P(q) = (1 + 2 Re sum_{n>=1} chi_n e^{-i k_n q}) / W, chi_0 = 1."""
    phase = np.exp(-1j * np.outer(q, ks[1:]))
    return (1 + 2 * (phase @ chi[1:]).real) / window


def _fourier_cdf(chi: np.ndarray, ks: np.ndarray, window: float, q: np.ndarray, q_lo: float):
    """int_{q_lo}^{q} P: exact for the periodic (aliased) density."""
    k = ks[1:]
    diff = (np.exp(-1j * np.outer(q, k)) - np.exp(-1j * k * q_lo)) / (-1j * k)
    return ((np.asarray(q) - q_lo) + 2 * (diff @ chi[1:]).real) / window


@dataclass(frozen=True)
class RecordDistribution:
    """P(Q | +alpha) and P(Q | -alpha) at one window length, on a uniform periodic Q grid."""

    time: float
    q: np.ndarray
    p_plus: np.ndarray
    p_minus: np.ndarray


@dataclass(frozen=True)
class ReadoutError:
    """Exact assignment errors (equal priors) and the record moments, per window length T.

    error_bayes is the likelihood-ratio optimum 1/2 int min(P+, P-); error_threshold uses the
    single best threshold on Q. error_gauss is the Gaussian estimate with the exact moments.
    switch_error is the largest fast remainder dropped by the slow-mode continuation.
    """

    times: np.ndarray
    mean: np.ndarray  # (2, nT): rows +alpha, -alpha
    var: np.ndarray
    error_bayes: np.ndarray
    error_threshold: np.ndarray
    thresholds: np.ndarray
    error_gauss: np.ndarray
    distributions: tuple[RecordDistribution, ...]
    switch_error: float


@dataclass(frozen=True)
class _Inversion:
    chi: np.ndarray  # (2, nk) for rows +, -
    ks: np.ndarray
    window: float
    q_lo: float

    def density(self, q: np.ndarray, row: int) -> np.ndarray:
        return _fourier_density(self.chi[row], self.ks, self.window, np.atleast_1d(q))

    def cdf(self, q: np.ndarray, row: int) -> np.ndarray:
        return _fourier_cdf(self.chi[row], self.ks, self.window, np.atleast_1d(q), self.q_lo)


def _crossings(inv: _Inversion, q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Points where P+ = P-, refined by root finding where the densities are not negligible."""
    diff = p[0] - p[1]
    floor = CROSSING_FLOOR * p.max()
    positive = diff > 0  # a zero on the grid counts once, as the end of a positive run
    idx = np.flatnonzero(positive[:-1] != positive[1:])
    roots = []
    for i in idx:
        if max(p[:, i].max(), p[:, i + 1].max()) < floor:
            continue

        def g(x: float) -> float:
            return float(inv.density(x, 0)[0] - inv.density(x, 1)[0])

        ga, gb = g(q[i]), g(q[i + 1])
        if ga * gb > 0:  # a rounding-level zero on the grid: keep the closer endpoint
            roots.append(q[i] if abs(ga) < abs(gb) else q[i + 1])
            continue
        roots.append(brentq(g, q[i], q[i + 1], xtol=1e-13, rtol=1e-14))
    return np.array(roots)


def _assignment_errors(inv: _Inversion, q: np.ndarray, p: np.ndarray, means: np.ndarray):
    """(Bayes error, threshold error, threshold) from exact CDFs between density crossings."""
    roots = _crossings(inv, q, p)
    low = int(np.argmin(means))  # this codeword is assigned when Q < threshold
    high = 1 - low
    inner = roots[(roots > means.min()) & (roots < means.max())]
    if inner.size == 0:
        inner = np.array([means.mean()])
    thr_err = 0.5 * ((1 - inv.cdf(inner, low)) + inv.cdf(inner, high))
    best = int(np.argmin(thr_err))
    bounds = np.concatenate([[q[0]], roots, [q[-1] + (q[1] - q[0])]])
    # Between consecutive crossings one density is everywhere smaller: its integral is the min.
    cdfs = np.stack([inv.cdf(bounds, 0), inv.cdf(bounds, 1)])
    bayes = float(np.sum(np.min(np.diff(cdfs, axis=1), axis=0)))
    return 0.5 * bayes, float(thr_err[best]), float(inner[best])


def gaussian_error(
    mean_a: np.ndarray, mean_b: np.ndarray, var_a: np.ndarray, var_b: np.ndarray
) -> np.ndarray:
    """Error of two Gaussians at their best threshold: 1/2 erfc(|dmu| / (sqrt2 (sig_a + sig_b)))."""
    spread = np.sqrt(2) * (np.sqrt(var_a) + np.sqrt(var_b))
    return 0.5 * erfc(np.abs(np.asarray(mean_a) - np.asarray(mean_b)) / spread)


def matched_filter_error(
    delta_b: np.ndarray, times: np.ndarray, kappa_b: float, eta: float
) -> np.ndarray:
    """1/2 erfc(sqrt((eta kappa_b / 2) int_0^T |<b>+ - <b>-|^2 dt)), cumulative in T.

    Shot-noise-limited Gaussian estimate with the matched filter f = Delta<b>, measuring the
    quadrature along Delta<b>; it ignores excess noise and bit flips. Trapezoid rule.
    """
    integrand = np.abs(np.asarray(delta_b)) ** 2
    steps = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(times)
    snr = eta * kappa_b / 2 * np.concatenate([[0.0], np.cumsum(steps)])
    return 0.5 * erfc(np.sqrt(snr))


def _regime_errors(
    gen: TiltedGenerator,
    rhos: list[np.ndarray],
    times: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    filt: PiecewiseFilter,
    slow: SlowModes | None,
    workers: int,
):
    centers = mean.mean(axis=0)
    half = float(np.max(np.abs(mean - centers) + N_SIGMA_WINDOW * np.sqrt(var)))
    grid = FourierGrid.covering(half, float(filt.squared_integral(times[:1])[0]))
    ks = grid.ks
    chi, remainder = _generating_function_and_error(gen, rhos, 1j * ks, times, filt, slow, workers)
    n_q = math.ceil(grid.window * grid.k_max * Q_OVERSAMPLE / math.pi)
    rows = []
    for j, t in enumerate(times):
        q = centers[j] - grid.window / 2 + grid.window * np.arange(n_q) / n_q
        inv = _Inversion(chi[:, :, j], ks, grid.window, q[0])
        p = np.stack([inv.density(q, 0), inv.density(q, 1)])
        errs = _assignment_errors(inv, q, p, mean[:, j])
        rows.append((errs, RecordDistribution(float(t), q, p[0], p[1])))
    return rows, remainder


def readout_error(
    gen: TiltedGenerator,
    rho_plus: np.ndarray,
    rho_minus: np.ndarray,
    times: np.ndarray,
    filt: PiecewiseFilter = BOXCAR,
    slow: SlowModes | None = None,
    workers: int = 1,
) -> ReadoutError:
    """Exact assignment error of the window [0, T] for each T, by Fourier inversion.

    Times up to slow.t_switch (all, without slow) share one Fourier grid; later times share a
    second, finer grid sized for the largest window. Each grid's window holds every mean
    +- N_SIGMA_WINDOW sigma; k_max is set by the shortest time in it (NOISE_FLOOR).
    Accuracy limit: errors below about 1e-12 are not resolved (use chernoff_bound).
    """
    t = _check_times(times)
    rhos = [rho_plus, rho_minus]
    mean, var = fd_cumulants(gen, rhos, t, filt, slow, workers)
    split = t <= slow.t_switch if slow is not None else np.ones(t.size, bool)
    rows, remainder = [], 0.0
    for mask in (split, ~split):
        if mask.any():
            part, rem = _regime_errors(
                gen, rhos, t[mask], mean[:, mask], var[:, mask], filt, slow, workers
            )
            rows += part
            remainder = _worst([remainder, rem])
    errs = np.array([r[0] for r in rows])
    return ReadoutError(
        times=t,
        mean=mean,
        var=var,
        error_bayes=errs[:, 0],
        error_threshold=errs[:, 1],
        thresholds=errs[:, 2],
        error_gauss=gaussian_error(mean[0], mean[1], var[0], var[1]),
        distributions=tuple(r[1] for r in rows),
        switch_error=remainder,
    )


# ---------------------------------------------------------------- Chernoff bound


def chernoff_bound(
    gen: TiltedGenerator,
    rho_plus: np.ndarray,
    rho_minus: np.ndarray,
    times: np.ndarray,
    thresholds: np.ndarray,
    s_values: np.ndarray,
    filt: PiecewiseFilter = BOXCAR,
    slow: SlowModes | None = None,
    workers: int = 1,
) -> np.ndarray:
    """Upper bound on the threshold-assignment error from real-s generating functions.

    P(Q > q | low) <= min_{s>0} e^{-s q} E_low[e^{sQ}] and P(Q < q | high) <= min_{s>0}
    e^{s q} E_high[e^{-sQ}]; the bound is half their sum. Valid for any s grid (positive s);
    a finer grid only tightens it. Reaches tails far below Fourier-inversion precision.
    """
    s = np.asarray(s_values, dtype=float)
    if np.any(s <= 0):
        raise ValueError("s_values must be positive")
    t = _check_times(times)
    s_all = np.concatenate([s, -s, [FD_STEP, -FD_STEP]])
    z = _generating_function_and_error(gen, [rho_plus, rho_minus], s_all, t, filt, slow, workers)[0]
    lz = np.log(z.real)  # (2, 2 ns + 2, nT)
    ns = s.size
    mean = (lz[:, -2] - lz[:, -1]) / (2 * FD_STEP)  # (2, nT)
    q = np.asarray(thresholds, dtype=float)[None, :]
    bound = np.zeros(t.size)
    for j in range(t.size):
        low = int(np.argmin(mean[:, j]))
        up = np.min(lz[low, :ns, j] - s * q[0, j])
        down = np.min(lz[1 - low, ns : 2 * ns, j] + s * q[0, j])
        bound[j] = 0.5 * (min(math.exp(up), 1.0) + min(math.exp(down), 1.0))
    return bound
