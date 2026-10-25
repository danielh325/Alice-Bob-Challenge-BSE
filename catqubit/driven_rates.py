"""Bit-flip rate with a readout drive and circuit-level extra terms (Task 2.4b).

With eps_z != 0 the memory-parity symmetry is broken, so the parity blocks of
catqubit.rates no longer apply. We use the full two-mode Liouvillian instead. Its zero
mode (the steady state) has the trace functional vec(I)^T as exact left eigenvector, so
adding shift * e_0 vec(I)^T (Brauer's rank-one deflation) moves that eigenvalue from 0 to
`shift` and leaves every other eigenvalue unchanged. The slowest remaining eigenvalue
lambda_1 is found by inverse iteration on one LU.

Convention. In the two-well (telegraph) picture, -lambda_1 = Gamma_{+->-} + Gamma_{-->+}.
At eps_z = 0 this is the decay rate of Re<a>, i.e. catqubit.rates.bit_flip_rate. With a
real eps_z the antiunitary map T = P_a P_b K (memory and buffer parity times complex
conjugation in the Fock basis) sends H -> -H and leaves the jumps invariant, so it is a
symmetry of the Lindbladian that swaps the wells: the two one-way rates stay equal, each
-lambda_1 / 2. Detunings and Kerr terms break T; then -lambda_1 is still the sum of the
two one-way rates, but they may differ.

Hamiltonian (rotating frame, rad/us):
    H = g2 (a^2 b^dag + h.c.) - eps_d (b + b^dag) + eps_z (a + a^dag)
        + delta_a a^dag a + delta_b b^dag b
        - (kerr_a / 2) a^dag^2 a^2 - (kerr_b / 2) b^dag^2 b^2 - chi_ab a^dag a b^dag b.
Jumps: sqrt(kappa_a) a and sqrt(kappa_b) (cosh r b + e^{i phi} sinh r b^dag) (a broadband
squeezed bath on the buffer; r = 0 is plain loss; phi = 0 squeezes X_b = b + b^dag).
"""

import math
from dataclasses import dataclass, fields, replace
from types import MappingProxyType

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from numpy.typing import ArrayLike
from scipy.integrate import cumulative_trapezoid

from catqubit.eig_gradients import (
    LU_ORDERING,
    EigenPair,
    eigenvalue_derivative,
    isolated_eigenpair,
)
from catqubit.liouvillian import (
    FockTruncation,
    _full_grid_operators,
    _project,
    lindblad_superoperator,
    parity_block,
    vec,
)
from catqubit.params import CatParams

# A residual above this fraction of |lambda_1| means the eigenpair is not trustworthy.
RESIDUAL_RTOL = 1e-4


@dataclass(frozen=True)
class HamiltonianTerms:
    """Coefficients (rad/us) of the Hamiltonian generators; see the module docstring."""

    g2: float
    eps_d: float
    eps_z: float = 0.0
    delta_a: float = 0.0
    delta_b: float = 0.0
    kerr_a: float = 0.0
    kerr_b: float = 0.0
    chi_ab: float = 0.0

    @classmethod
    def from_params(
        cls, params: CatParams, eps_z: float = 0.0, **extra: float
    ) -> "HamiltonianTerms":
        """Terms of the challenge model (g2, eps_d = alpha^2 g2) plus a drive and extras."""
        return cls(g2=params.g2, eps_d=params.eps_d, eps_z=eps_z, **extra)

    def coefficients(self) -> tuple[tuple[str, float], ...]:
        return tuple((f.name, float(getattr(self, f.name))) for f in fields(self))

    def replace(self, **changes: float) -> "HamiltonianTerms":
        return replace(self, **changes)


@dataclass(frozen=True)
class Dissipation:
    """Memory loss kappa_a and (optionally squeezed) buffer loss kappa_b, in rad/us."""

    kappa_a: float
    kappa_b: float
    squeeze_r: float = 0.0
    squeeze_phi: float = 0.0

    def __post_init__(self) -> None:
        if self.kappa_a < 0:
            raise ValueError(f"kappa_a must be non-negative, got {self.kappa_a}")
        if self.kappa_b <= 0:
            raise ValueError(f"kappa_b must be positive, got {self.kappa_b}")
        if self.squeeze_r < 0:
            raise ValueError(f"squeeze_r must be non-negative, got {self.squeeze_r}")

    @classmethod
    def from_params(
        cls, params: CatParams, squeeze_r: float = 0.0, squeeze_phi: float = 0.0
    ) -> "Dissipation":
        return cls(params.kappa_a, params.kappa_b, squeeze_r, squeeze_phi)


@dataclass(frozen=True)
class DrivenRate:
    """rate = -Re lambda_1 (sum of both one-way rates); gradient = d rate / d coefficient."""

    rate: float
    eigenvalue: complex
    residual: float
    gradient: MappingProxyType | None = None
    pair: EigenPair | None = None


def hamiltonian_generators(trunc: FockTruncation) -> dict[str, sp.csr_matrix]:
    """Operator multiplying each HamiltonianTerms coefficient, built on the full grid and
    projected (products are never formed from projected factors)."""
    a, b = _full_grid_operators(trunc)
    ad, bd = a.T.tocsr(), b.T.tocsr()
    n_a, n_b = ad @ a, bd @ b
    full = {
        "g2": a @ a @ bd + ad @ ad @ b,
        "eps_d": -(b + bd),
        "eps_z": a + ad,
        "delta_a": n_a,
        "delta_b": n_b,
        "kerr_a": -0.5 * (ad @ ad @ a @ a),
        "kerr_b": -0.5 * (bd @ bd @ b @ b),
        "chi_ab": -(n_a @ n_b),
    }
    return {k: _project(v, trunc).astype(complex) for k, v in full.items()}


def buffer_jump(trunc: FockTruncation, squeeze_r: float, squeeze_phi: float) -> sp.csr_matrix:
    """cosh(r) b + e^{i phi} sinh(r) b^dag (no rate), projected."""
    _, b = _full_grid_operators(trunc)
    op = math.cosh(squeeze_r) * b + np.exp(1j * squeeze_phi) * math.sinh(squeeze_r) * b.T
    return _project(op, trunc).astype(complex)


def hamiltonian_superoperator(h: sp.spmatrix) -> sp.csr_matrix:
    """rho -> -i[h, rho] on vec(rho); equals dL/d theta for a coefficient theta of h."""
    return lindblad_superoperator(h, [])


def driven_liouvillian(
    terms: HamiltonianTerms, dissipation: Dissipation, trunc: FockTruncation
) -> sp.csr_matrix:
    """Full (all parity sectors) Liouvillian of the module-docstring model."""
    gens = hamiltonian_generators(trunc)
    h = sum(c * gens[k] for k, c in terms.coefficients() if c != 0)
    a, _ = _full_grid_operators(trunc)
    jumps = [
        math.sqrt(dissipation.kappa_b)
        * buffer_jump(trunc, dissipation.squeeze_r, dissipation.squeeze_phi)
    ]
    if dissipation.kappa_a > 0:
        jumps.append(math.sqrt(dissipation.kappa_a) * _project(a, trunc).astype(complex))
    return lindblad_superoperator(h, jumps)


def deflate_steady_state(
    big_l: ArrayLike | sp.spmatrix, trunc: FockTruncation, shift: float
) -> sp.csc_matrix:
    """L + shift e_0 vec(I)^T: the zero eigenvalue moves to `shift`, all others stay.

    e_0 is the vec index of |0,0><0,0|, so the added matrix has only dim nonzeros. Right
    eigenvectors of the traceless modes are unchanged (vec(I)^T r = Tr r = 0).
    """
    m = sp.csc_matrix(big_l)
    n = trunc.dim
    diagonal_positions = np.arange(n) * (n + 1)
    rank_one = sp.csc_matrix(
        (np.full(n, shift, dtype=complex), (np.zeros(n, dtype=int), diagonal_positions)),
        shape=m.shape,
    )
    return (m + rank_one).tocsc()


def _gradient(pair: EigenPair, trunc: FockTruncation) -> MappingProxyType:
    gens = hamiltonian_generators(trunc)
    return MappingProxyType(
        {
            k: -eigenvalue_derivative(pair, hamiltonian_superoperator(g)).real
            for k, g in gens.items()
        }
    )


def driven_bit_flip_rate(
    terms: HamiltonianTerms,
    dissipation: Dissipation,
    trunc: FockTruncation,
    with_gradient: bool = False,
) -> DrivenRate:
    """Slowest nonzero decay rate -Re lambda_1 of the full Liouvillian (any eps_z, extras).

    Inverse iteration on the deflated Liouvillian, started from vec(a + a^dag). With
    with_gradient, also returns the adjoint derivative of the rate with respect to every
    Hamiltonian coefficient (catqubit.eig_gradients), from the same LU.
    """
    big_l = driven_liouvillian(terms, dissipation, trunc)
    deflated = deflate_steady_state(big_l, trunc, -dissipation.kappa_b)
    a = _project(_full_grid_operators(trunc)[0], trunc)
    start = vec((a + a.T).toarray())
    pair = isolated_eigenpair(deflated, start)
    if pair.residual > RESIDUAL_RTOL * abs(pair.value):
        raise ValueError(f"eigen-residual {pair.residual:.2e} too large for lambda={pair.value}")
    gradient = _gradient(pair, trunc) if with_gradient else None
    return DrivenRate(-pair.value.real, pair.value, pair.residual, gradient, pair)


def codeword_density(alpha: complex, trunc: FockTruncation) -> np.ndarray:
    """|alpha><alpha| (x) |0><0| on the kept states, renormalized after truncation."""
    n = np.arange(trunc.na)
    log_fact = np.array([math.lgamma(k + 1) for k in n])
    memory = np.exp(n * np.log(complex(alpha)) - 0.5 * log_fact - abs(alpha) ** 2 / 2)
    buffer = np.zeros(trunc.nb_max)
    buffer[0] = 1.0
    psi = np.kron(memory, buffer)[trunc.kept_indices()]
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def codeword_contrast(result: DrivenRate, params: CatParams, trunc: FockTruncation) -> complex:
    """<b>_+ - <b>_- in the metastable (quasi-steady) state reached from |+-alpha>, |0>.

    After the fast transients and well before 1/Gamma_X, rho(t) = r_0 + c r_1 with
    c = <l_1|rho_init> / <l_1|r_1> (Macieszczak et al., PRL 116, 240404 (2016)). The steady
    state r_0 cancels in the difference, so the contrast is (c_+ - c_-) Tr(b r_1); it is
    independent of the normalization and phase of r_1 and l_1.
    """
    if result.pair is None:
        raise ValueError("result carries no eigenvectors")
    left, right = result.pair.left, result.pair.right
    overlap = np.vdot(left, right)
    c_plus, c_minus = (
        np.vdot(left, vec(codeword_density(s * params.alpha, trunc))) / overlap for s in (1, -1)
    )
    _, b = _full_grid_operators(trunc)
    b_vec = vec(_project(b, trunc).T.toarray())  # Tr(b rho) = vec(b^T) . vec(rho)
    return complex((c_plus - c_minus) * (b_vec @ right))


@dataclass(frozen=True)
class DriveResponse:
    """Gamma_X(eps_z) = rate0 + coefficient * eps_z^2 + O(eps_z^4), both rates as -lambda_1."""

    rate0: float
    coefficient: float
    residual: float


@dataclass(frozen=True)
class _SectorSetup:
    """eps_z = 0 parity blocks, the slow odd eigenpair and the drive's cross-sector blocks."""

    pair: EigenPair
    even: sp.csc_matrix
    idx_even: np.ndarray
    v_even_odd: sp.csr_matrix
    v_odd_even: sp.csr_matrix

    def project(self, even_vector: np.ndarray) -> complex:
        """<l_1| V even_vector / <l_1|r_1>."""
        p = self.pair
        return complex(np.vdot(p.left, self.v_odd_even @ even_vector) / np.vdot(p.left, p.right))


def _sector_setup(
    terms: HamiltonianTerms, dissipation: Dissipation, trunc: FockTruncation
) -> _SectorSetup:
    big_l = driven_liouvillian(terms.replace(eps_z=0.0), dissipation, trunc)
    odd, idx_odd = parity_block(big_l, trunc, odd=True)
    even, idx_even = parity_block(big_l, trunc, odd=False)
    gen = hamiltonian_generators(trunc)["eps_z"]
    drive = hamiltonian_superoperator(gen).tocsr()
    start = vec((gen + gen.T).toarray())
    pair = isolated_eigenpair(odd, start[idx_odd])
    return _SectorSetup(
        pair, even, idx_even, drive[idx_even][:, idx_odd], drive[idx_odd][:, idx_even]
    )


def drive_rate_coefficient(
    terms: HamiltonianTerms, dissipation: Dissipation, trunc: FockTruncation
) -> DriveResponse:
    """Exact second-order (linear-response) coefficient of the drive-induced rate.

    The drive superoperator V = -i[a + a^dag, .] maps the parity-odd sector to the even
    one and back, so the first-order shift vanishes and

        lambda_1(eps) = lambda_1 + eps^2 <l_1| V (lambda_1 - L_even)^{-1} V |r_1> / <l_1|r_1>.

    V r_1 is traceless (V is a commutator), so the steady state of the even sector drops out
    and L_even can be deflated as in deflate_steady_state. Cost: one LU of each parity
    block, about half the fill of the full driven Liouvillian.

    terms.eps_z is ignored (the expansion point is eps_z = 0). Every other term in
    HamiltonianTerms, and the squeezed buffer bath, conserves the memory-parity sectors.
    """
    setup = _sector_setup(terms, dissipation, trunc)
    pair = setup.pair
    shifted = _deflate_even_sector(setup.even, setup.idx_even, trunc, -dissipation.kappa_b)
    shifted = shifted - pair.value * sp.identity(shifted.shape[0], format="csc")
    x = sla.splu(shifted.tocsc(), permc_spec=LU_ORDERING).solve(setup.v_even_odd @ pair.right)
    return DriveResponse(-pair.value.real, setup.project(x).real, pair.residual)


def drive_susceptibility(
    terms: HamiltonianTerms,
    dissipation: Dissipation,
    trunc: FockTruncation,
    omegas: np.ndarray,
) -> np.ndarray:
    """c2(omega): flip rate per eps^2 for a drive eps cos(omega t) has mean c2(omega) eps^2 / 2.

    c2(omega) = -Re int_0^inf K(tau) e^{-i omega tau} d tau
              = -Re <l_1| V (lambda_1 + i omega - L_even)^{-1} V |r_1> / <l_1|r_1>,
    averaged over +-omega (one sparse LU per sign and frequency); c2(0) is the
    drive_rate_coefficient. omega is the offset of the drive from the memory frame.
    """
    setup = _sector_setup(terms, dissipation, trunc)
    pair = setup.pair
    eye = sp.identity(setup.even.shape[0], format="csc")
    rhs = setup.v_even_odd @ pair.right
    out = []
    for w in np.asarray(omegas, dtype=float):
        signs = (1,) if w == 0 else (1, -1)
        vals = [
            setup.project(
                sla.splu(
                    ((pair.value + s * 1j * w) * eye - setup.even).tocsc(), permc_spec=LU_ORDERING
                ).solve(rhs)
            )
            for s in signs
        ]
        out.append(-np.mean(vals).real)
    return np.array(out)


@dataclass(frozen=True)
class DriveKernel:
    """Second-order memory kernel K(tau) of drive-induced bit flips, on a uniform grid.

    For any drive eps_z(t) (zero outside the pulse), to second order and with no
    quasi-static assumption, the extra bit-flip probability (-lambda_1 convention) is

        p_extra = -Re int dt int_{t' < t} dt' eps(t) eps(t') K(t - t'),
        K(tau) = <l_1| V exp[(L_even - lambda_1) tau] V |r_1> / <l_1|r_1>,

    so a constant drive gives the rate c2 eps^2 with c2 = -int_0^inf K. Its Fourier
    transform c2(omega) = -Re int K e^{-i omega tau} is the flip "susceptibility" at
    drive frequency omega.
    """

    taus: np.ndarray
    values: np.ndarray
    rate0: float

    def quadratic_form(self, pieces: int, piece_duration: float) -> np.ndarray:
        """Symmetric M with p_extra = x^T M x for a piecewise-constant pulse x."""
        span = pieces * piece_duration
        if self.taus[-1] < span * (1 - 1e-12):
            raise ValueError(f"kernel grid ends at {self.taus[-1]}, pulse lasts {span}")
        k1 = cumulative_trapezoid(self.values.real, self.taus, initial=0.0)
        k2 = cumulative_trapezoid(k1, self.taus, initial=0.0)
        k2_at = np.interp(np.arange(pieces + 1) * piece_duration, self.taus, k2)
        lag = np.arange(1, pieces)
        off = k2_at[lag + 1] - 2 * k2_at[lag] + k2_at[lag - 1]
        weights = np.concatenate([[2 * k2_at[1]], off])  # weight of lag d, doubled at d = 0
        d = np.abs(np.subtract.outer(np.arange(pieces), np.arange(pieces)))
        return -0.5 * weights[d]

    def pulse_probability(self, pulse: np.ndarray, piece_duration: float) -> float:
        pulse = np.asarray(pulse, dtype=float)
        return float(pulse @ self.quadratic_form(pulse.size, piece_duration) @ pulse)


# Rows of the even-sector propagation kept in memory at once.
KERNEL_CHUNK = 256


def drive_kernel(
    terms: HamiltonianTerms, dissipation: Dissipation, trunc: FockTruncation, taus: np.ndarray
) -> DriveKernel:
    """K(tau) on a uniform grid starting at 0 (sparse expm_multiply, chunked)."""
    taus = _check_uniform(taus)
    setup = _sector_setup(terms, dissipation, trunc)
    pair = setup.pair
    generator = (setup.even - pair.value * sp.identity(setup.even.shape[0])).tocsr()
    values = _propagate_and_project(generator, setup.v_even_odd @ pair.right, setup.project, taus)
    return DriveKernel(taus, values, -pair.value.real)


def _check_uniform(taus: np.ndarray) -> np.ndarray:
    taus = np.asarray(taus, dtype=float)
    steps = np.diff(taus)
    if taus[0] != 0 or np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-9):
        raise ValueError("taus must be a uniform increasing grid starting at 0")
    return taus


def _propagate_and_project(
    generator: sp.spmatrix, u: np.ndarray, project, taus: np.ndarray
) -> np.ndarray:
    """project(exp(generator tau) u) on the uniform grid taus (chunked expm_multiply)."""
    step = taus[1] - taus[0]
    values = [project(u)]
    for begin in range(0, taus.size - 1, KERNEL_CHUNK):
        n = min(KERNEL_CHUNK, taus.size - 1 - begin)
        block = sla.expm_multiply(generator, u, start=0.0, stop=n * step, num=n + 1, endpoint=True)
        values.extend(project(row) for row in block[1:])
        u = block[-1]
    return np.array(values)


@dataclass(frozen=True)
class PulseFlip:
    """Time-domain bit-flip probability of a pulse; extra = total - rate0 * (T + settle)."""

    total: float
    extra: float
    rate0: float


@dataclass(frozen=True)
class _Metastable:
    """Undriven full Liouvillian, drive superoperator, slow pair, steady state and the
    slow amplitudes c_+- of the codewords |+-alpha>, |0>."""

    big_l: sp.csc_matrix
    drive: sp.csc_matrix
    pair: EigenPair
    steady: np.ndarray
    c_plus: complex
    c_minus: complex

    def slow_amplitude(self, rho: np.ndarray) -> complex:
        return complex(np.vdot(self.pair.left, rho) / np.vdot(self.pair.left, self.pair.right))


def _metastable(
    terms: HamiltonianTerms, dissipation: Dissipation, trunc: FockTruncation
) -> _Metastable:
    big_l = driven_liouvillian(terms.replace(eps_z=0.0), dissipation, trunc).tocsc()
    drive = hamiltonian_superoperator(hamiltonian_generators(trunc)["eps_z"]).tocsc()
    shift = -dissipation.kappa_b
    deflated = deflate_steady_state(big_l, trunc, shift)
    lu = sla.splu(deflated, permc_spec=LU_ORDERING)
    a = _project(_full_grid_operators(trunc)[0], trunc)
    pair = isolated_eigenpair(deflated, vec((a + a.T).toarray()), lu=lu)
    e0 = np.zeros(deflated.shape[0], dtype=complex)
    e0[0] = shift
    steady = lu.solve(e0)  # (L + shift e_0 1^T) r_0 = shift e_0 with Tr r_0 = 1
    norm = np.vdot(pair.left, pair.right)
    alpha = cat_amplitude(terms)
    c_plus, c_minus = (
        complex(np.vdot(pair.left, vec(codeword_density(s * alpha, trunc))) / norm) for s in (1, -1)
    )
    return _Metastable(big_l, drive, pair, steady, c_plus, c_minus)


def pulse_bit_flip_probability(
    terms: HamiltonianTerms,
    dissipation: Dissipation,
    trunc: FockTruncation,
    pulse: np.ndarray,
    piece_duration: float,
    settle: float,
) -> PulseFlip:
    """Exact (all orders, no quasi-static assumption) flip probability of a drive pulse.

    Start in the metastable state r_0 + c_+ r_1 of the undriven model reached from
    |+alpha>, |0>; evolve under L(eps_k) piece by piece (sparse expm_multiply), then idle
    for `settle` so the fast modes decay; read the slow amplitude c = <l_1|rho> / <l_1|r_1>.
    Idle evolution multiplies c by exp(lambda_1 t) exactly, so -ln(c_end / c_+) is the
    integrated rate int Gamma_X dt (-lambda_1 convention; one-way = half).
    """
    ms = _metastable(terms, dissipation, trunc)
    rho = ms.steady + ms.c_plus * ms.pair.right
    for eps in np.asarray(pulse, dtype=float):
        rho = sla.expm_multiply((ms.big_l + eps * ms.drive) * piece_duration, rho)
    rho = sla.expm_multiply(ms.big_l * settle, rho)
    total = float(-np.log((ms.slow_amplitude(rho) / ms.c_plus).real))
    rate0 = -ms.pair.value.real
    duration = len(pulse) * piece_duration + settle
    return PulseFlip(total, total - rate0 * duration, rate0)


@dataclass(frozen=True)
class ResponseKernel:
    """Exact linear response of the buffer to the readout drive, from the metastable wells.

    Delta<b>(t) = <b>_+ - <b>_- = int_0^t G(t - t') eps_z(t') dt' to first order in eps_z,
    with G(tau) = Tr[b exp(L tau) V (rho_+ - rho_-)], rho_+- = r_0 + c_+- r_1 and
    V = -i[a + a^dag, .]. Unlike the 2x2 oscillator of catqubit.optimize.LinearReadout,
    this keeps every mode of the two-mode model.
    """

    taus: np.ndarray
    values: np.ndarray

    def response_matrix(self, pieces: int, piece_duration: float, times: np.ndarray) -> np.ndarray:
        """A with Delta<b>(times) = A @ pulse for a piecewise-constant pulse."""
        times = np.asarray(times, dtype=float)
        if times.max() > self.taus[-1]:
            raise ValueError(f"response grid ends at {self.taus[-1]}, need {times.max()}")
        g1 = cumulative_trapezoid(self.values, self.taus, initial=0.0)

        def integral(tau: np.ndarray) -> np.ndarray:
            inside = np.clip(tau, 0.0, None)
            val = np.interp(inside, self.taus, g1.real) + 1j * np.interp(inside, self.taus, g1.imag)
            return np.where(tau > 0, val, 0.0)

        starts = np.arange(pieces) * piece_duration
        lag = times[:, None] - starts[None, :]
        return integral(lag) - integral(lag - piece_duration)

    def signal_form(
        self,
        pieces: int,
        piece_duration: float,
        kappa_b: float,
        efficiency: float = 1.0,
        samples_per_piece: int = 8,
    ) -> np.ndarray:
        """S with SNR^2 = x^T S x = 4 eta kappa_b int_0^T (Re Delta<b>)^2 dt (homodyne, phi = 0)."""
        n = pieces * samples_per_piece
        times = np.linspace(0.0, pieces * piece_duration, n + 1)
        weights = np.ones(n + 1)
        weights[1:-1:2], weights[2:-1:2] = 4.0, 2.0
        weights *= (times[1] - times[0]) / 3  # composite Simpson (n even)
        a = self.response_matrix(pieces, piece_duration, times).real
        return 4 * efficiency * kappa_b * (a.T * weights) @ a


def buffer_response(
    terms: HamiltonianTerms, dissipation: Dissipation, trunc: FockTruncation, taus: np.ndarray
) -> ResponseKernel:
    """G(tau) on a uniform grid from 0 (sparse expm_multiply of the undriven Liouvillian)."""
    taus = _check_uniform(taus)
    ms = _metastable(terms, dissipation, trunc)
    _, b = _full_grid_operators(trunc)
    b_vec = vec(_project(b, trunc).T.toarray())
    u = ms.drive @ ((ms.c_plus - ms.c_minus) * ms.pair.right)
    values = _propagate_and_project(ms.big_l, u, lambda rho: complex(b_vec @ rho), taus)
    return ResponseKernel(taus, values)


def cat_amplitude(terms: HamiltonianTerms) -> float:
    """Cat amplitude sqrt(eps_d / g2) of the stabilized wells."""
    return math.sqrt(terms.eps_d / terms.g2)


def _deflate_even_sector(
    even: sp.spmatrix, idx_even: np.ndarray, trunc: FockTruncation, shift: float
) -> sp.csc_matrix:
    """Brauer deflation of the even block: every diagonal |i><i| lies in the even sector."""
    n = trunc.dim
    position = {int(k): j for j, k in enumerate(idx_even)}
    columns = np.array([position[int(k)] for k in np.arange(n) * (n + 1)])
    row = position[0]
    rank_one = sp.csc_matrix(
        (np.full(n, shift, dtype=complex), (np.full(n, row), columns)), shape=even.shape
    )
    return (sp.csc_matrix(even) + rank_one).tocsc()
