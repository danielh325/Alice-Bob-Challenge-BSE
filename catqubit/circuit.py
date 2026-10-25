"""Circuit-level ATS Hamiltonian, its effective rotating-frame expansion and a Floquet check.

Model (challenge Task 2.4):

    H(t) = w_a a^dag a + w_b b^dag b - 2 E_J cos(phi_Sigma(t)) cos(phi + phi_Delta),
    phi  = phi_a (a + a^dag) + phi_b (b + b^dag),

biased at phi_Delta = pi/2 and phi_Sigma(t) = -pi/2 + sum_j eps_j cos(h_j u t + theta_j). At this
bias cos(phi + pi/2) = -sin(phi) and cos(phi_Sigma) = sin(sum_j ...), so the ATS term is odd in
the flux tones and odd in phi: there is no static nonlinearity at all.

Conventions:
- Units: time in us, every frequency and rate angular in rad/us (2 pi x MHz).
- All mode and tone frequencies are integer multiples ("harmonics") of a unit u. For the prompt
  parameters u / 2 pi = 0.8 GHz and (w_a, w_b, w_p) = (5, 6, 4) u, so the frame rotating at
  (5u, 6u) returns to itself after T = 2 pi / u = 1.25 ns and H is exactly T-periodic there.
  Optional detunings move the bare modes off the frame: w_a = h_a u + detuning_a.
- Joint Fock index is memory-major, index(n_a, n_b) = n_a * nb + n_b.
- The drive c(t) = -2 E_J cos(phi_Sigma(t)) = sum_k c_k exp(i h_k u t) multiplies the coupling
  operator C = cos(phi + phi_Delta). Interaction-picture elements C_mn rotate at (E_m - E_n) u.
- Effective Hamiltonian in the frame (Floquet-Magnus / van Vleck time averaging, Venkatraman et al.,
  PRL 129, 100601 (2022); Goldman & Dalibard, PRX 4, 031027 (2014)):
      H1 = resonant (zero-frequency) part of c(t) C,
      H2 = sum_{nu > 0} [V(nu), V(-nu)] / nu,  V_I(t) = sum_nu V(nu) exp(i nu t).
  Matrix functions of phi are taken in an enlarged Fock space and then truncated.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import reduce

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment
from scipy.special import j0, j1

from catqubit.params import G2_MHZ, to_angular

# Prompt Table (Task 2.4), ordinary frequencies in MHz.
UNIT_MHZ = 800.0
HARMONIC_A, HARMONIC_B, HARMONIC_P = 5, 6, 4
E_J_MHZ = 40_000.0
PHI_A, PHI_B = 0.06, 0.3

SIGMA_SWEET_SPOT = -math.pi / 2
DELTA_SWEET_SPOT = math.pi / 2
FUNCTION_PAD = 20  # extra Fock levels per mode for matrix functions of phi
DEFAULT_PAD = 8  # extra Fock levels per mode for intermediate states at second order
ODE_RTOL, ODE_ATOL = 1e-11, 1e-12


def eps_p_for_g2(g2: float, e_j: float, phi_a: float, phi_b: float) -> float:
    """Pump amplitude from the prompt's leading-order formula g2 = E_J eps_p phi_a^2 phi_b / 2."""
    return 2 * g2 / (e_j * phi_a**2 * phi_b)


@dataclass(frozen=True)
class ATSParams:
    """ATS circuit parameters. Frequencies are integer harmonics of `unit` (rad/us)."""

    unit: float
    harmonic_a: int
    harmonic_b: int
    harmonic_p: int
    e_j: float
    phi_a: float
    phi_b: float
    eps_p: float
    phi_delta: float = DELTA_SWEET_SPOT
    detuning_a: float = 0.0
    detuning_b: float = 0.0

    def __post_init__(self) -> None:
        if self.unit <= 0:
            raise ValueError(f"unit frequency must be positive, got {self.unit}")
        if self.e_j <= 0:
            raise ValueError(f"E_J must be positive, got {self.e_j}")
        if self.phi_a <= 0 or self.phi_b <= 0:
            raise ValueError(f"phi_a and phi_b must be positive, got {self.phi_a}, {self.phi_b}")
        if self.eps_p < 0:
            raise ValueError(f"eps_p must be non-negative, got {self.eps_p}")

    @classmethod
    def prompt(cls) -> "ATSParams":
        """Illustrative parameters of the prompt, eps_p fixed by g2 / 2 pi = 0.763 MHz."""
        e_j = to_angular(E_J_MHZ)
        return cls(
            unit=to_angular(UNIT_MHZ),
            harmonic_a=HARMONIC_A,
            harmonic_b=HARMONIC_B,
            harmonic_p=HARMONIC_P,
            e_j=e_j,
            phi_a=PHI_A,
            phi_b=PHI_B,
            eps_p=eps_p_for_g2(to_angular(G2_MHZ), e_j, PHI_A, PHI_B),
        )

    @property
    def omega_a(self) -> float:
        return self.harmonic_a * self.unit + self.detuning_a

    @property
    def omega_b(self) -> float:
        return self.harmonic_b * self.unit + self.detuning_b

    @property
    def omega_p(self) -> float:
        return self.harmonic_p * self.unit

    @property
    def period(self) -> float:
        """Period of the rotating-frame Hamiltonian, 2 pi / unit (us)."""
        return 2 * math.pi / self.unit

    @property
    def g2_formula(self) -> float:
        """Prompt's leading-order two-photon exchange rate, E_J eps_p phi_a^2 phi_b / 2."""
        return 0.5 * self.e_j * self.eps_p * self.phi_a**2 * self.phi_b

    @property
    def normal_order_factor(self) -> float:
        """exp(-(phi_a^2 + phi_b^2) / 2): vacuum renormalization of every normal-ordered term."""
        return math.exp(-(self.phi_a**2 + self.phi_b**2) / 2)

    def with_pump(self, eps_p: float) -> "ATSParams":
        return replace(self, eps_p=eps_p)


@dataclass(frozen=True)
class FluxTone:
    """Common-mode flux tone: phi_Sigma gains eps cos(harmonic * u * t + phase)."""

    amplitude: float
    harmonic: int
    phase: float = 0.0


@dataclass(frozen=True)
class DriveComponent:
    """One Fourier component c_k exp(i harmonic u t) of the ATS drive c(t)."""

    coefficient: complex
    harmonic: int


@dataclass(frozen=True)
class FockSpace:
    """Rectangular two-mode Fock truncation, memory-major ordering."""

    na: int
    nb: int

    def __post_init__(self) -> None:
        if self.na < 1 or self.nb < 1:
            raise ValueError(f"truncations must be positive, got ({self.na}, {self.nb})")

    @property
    def dim(self) -> int:
        return self.na * self.nb

    def index(self, n_a: int, n_b: int) -> int:
        return n_a * self.nb + n_b

    def occupations(self, i: int) -> tuple[int, int]:
        return divmod(int(i), self.nb)

    def padded(self, pad: int) -> "FockSpace":
        return FockSpace(self.na + pad, self.nb + pad)

    def embed_indices(self, sub: "FockSpace") -> np.ndarray:
        """Indices in this space of the states of a smaller space `sub`."""
        if sub.na > self.na or sub.nb > self.nb:
            raise ValueError("sub-space does not fit")
        n_a, n_b = np.divmod(np.arange(sub.dim), sub.nb)
        return n_a * self.nb + n_b


@dataclass(frozen=True)
class FrameHarmonics:
    """Bare frame energy of each Fock state, in integer units of u: h_a n_a + h_b n_b."""

    diagonal_energies: np.ndarray

    @property
    def differences(self) -> np.ndarray:
        return np.subtract.outer(self.diagonal_energies, self.diagonal_energies)


@dataclass(frozen=True)
class FloquetMatch:
    """Floquet quasi-energies paired with effective-Hamiltonian eigenvalues (rad/us)."""

    labels: tuple[tuple[int, int], ...]
    floquet: np.ndarray
    effective: np.ndarray
    overlaps: np.ndarray

    @property
    def min_overlap(self) -> float:
        return float(self.overlaps.min())


# --- operators ---------------------------------------------------------------------------


def _ladder(n: int) -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1, n)), 1)


def two_mode_ladders(space: FockSpace) -> tuple[np.ndarray, np.ndarray]:
    """Dense annihilation operators (a, b), memory-major."""
    a = np.kron(_ladder(space.na), np.eye(space.nb))
    b = np.kron(np.eye(space.na), _ladder(space.nb))
    return a, b


def phase_operator(ats: ATSParams, space: FockSpace) -> np.ndarray:
    a, b = two_mode_ladders(space)
    return ats.phi_a * (a + a.T) + ats.phi_b * (b + b.T)


def _truncate(matrix: np.ndarray, big: FockSpace, space: FockSpace) -> np.ndarray:
    keep = big.embed_indices(space)
    return matrix[np.ix_(keep, keep)]


def josephson_operator(ats: ATSParams, space: FockSpace, pad: int = FUNCTION_PAD) -> np.ndarray:
    """C = cos(phi + phi_Delta), built in an enlarged space and then truncated to `space`."""
    big = space.padded(pad)
    w, v = np.linalg.eigh(phase_operator(ats, big))
    full = (v * np.cos(w + ats.phi_delta)) @ v.T
    return _truncate(full, big, space)


def displacement_operator(
    space: FockSpace, beta_a: complex, beta_b: complex, pad: int = FUNCTION_PAD
) -> np.ndarray:
    """D(beta_a) D(beta_b), built in an enlarged space and then truncated."""
    big = space.padded(pad)
    a, b = two_mode_ladders(big)
    gen = beta_a * a.T - np.conj(beta_a) * a + beta_b * b.T - np.conj(beta_b) * b
    return _truncate(expm(gen), big, space)


def frame_harmonics(ats: ATSParams, space: FockSpace) -> FrameHarmonics:
    n_a, n_b = np.divmod(np.arange(space.dim), space.nb)
    return FrameHarmonics(ats.harmonic_a * n_a + ats.harmonic_b * n_b)


def _detuning_diagonal(ats: ATSParams, space: FockSpace) -> np.ndarray:
    n_a, n_b = np.divmod(np.arange(space.dim), space.nb)
    return np.diag(ats.detuning_a * n_a + ats.detuning_b * n_b).astype(complex)


# --- drive -------------------------------------------------------------------------------


def flux_drive(ats: ATSParams, tones: Sequence[FluxTone], t: float) -> float:
    """Exact scalar drive c(t) = -2 E_J cos(phi_Sigma(t)) multiplying C."""
    modulation = sum(
        tone.amplitude * math.cos(tone.harmonic * ats.unit * t + tone.phase) for tone in tones
    )
    return -2 * ats.e_j * math.cos(SIGMA_SWEET_SPOT + modulation)


def drive_components(ats: ATSParams, tones: Sequence[FluxTone]) -> tuple[DriveComponent, ...]:
    """First-harmonic Fourier components of c(t) for each tone.

    -2 E_J sin(sum_j eps_j cos x_j) has first harmonic of tone j equal to
    -2 E_J [2 J1(eps_j) prod_{k != j} J0(eps_k)] cos x_j. Intermodulation products and higher
    harmonics (order eps^3) are dropped; the exact Floquet propagator keeps them.
    """
    comps: list[DriveComponent] = []
    for j, tone in enumerate(tones):
        spectator = reduce(
            lambda acc, other: acc * j0(other.amplitude),
            (other for k, other in enumerate(tones) if k != j),
            1.0,
        )
        half_amp = -2 * ats.e_j * j1(tone.amplitude) * spectator
        for sign in (1, -1):
            comps.append(
                DriveComponent(half_amp * np.exp(1j * sign * tone.phase), sign * tone.harmonic)
            )
    return tuple(comps)


def lab_hamiltonian(
    ats: ATSParams, space: FockSpace, tones: Sequence[FluxTone], t: float
) -> np.ndarray:
    """Full lab-frame circuit Hamiltonian at time t (rad/us)."""
    n_a, n_b = np.divmod(np.arange(space.dim), space.nb)
    h0 = np.diag(ats.omega_a * n_a + ats.omega_b * n_b)
    return h0 + flux_drive(ats, tones, t) * josephson_operator(ats, space)


# --- effective Hamiltonian ---------------------------------------------------------------


def first_order_terms(
    coupling: np.ndarray, energies: np.ndarray, components: Sequence[DriveComponent]
) -> np.ndarray:
    """Resonant (time-averaged) part of sum_k c_k exp(i h_k u t) C_I(t)."""
    nu = np.subtract.outer(energies, energies)
    h = np.zeros(coupling.shape, dtype=complex)
    for comp in components:
        h += comp.coefficient * coupling * (nu + comp.harmonic == 0)
    return h


def second_order_hamiltonian(
    coupling: np.ndarray,
    energies: np.ndarray,
    components: Sequence[DriveComponent],
    unit: float,
) -> np.ndarray:
    """H2_mn = sum_k V_mk V_kn / nu1 over pairs with nu1 + nu2 = 0 and nu1 != 0.

    nu1 = (E_m - E_k + h1) u is the frequency of the left factor. This equals
    sum_{nu>0} [V(nu), V(-nu)] / nu and reproduces the two-level AC Stark shift.
    """
    nu = np.subtract.outer(energies, energies)
    h = np.zeros(coupling.shape, dtype=complex)
    for left in components:
        freq = nu + left.harmonic
        inv = np.divide(1.0, freq * unit, out=np.zeros(freq.shape), where=freq != 0)
        left_op = left.coefficient * coupling * inv
        for right in components:
            mask = nu + left.harmonic + right.harmonic == 0
            if mask.any():
                h += (left_op @ (right.coefficient * coupling)) * mask
    return h


def first_order_hamiltonian(
    ats: ATSParams,
    space: FockSpace,
    tones: Sequence[FluxTone],
    coupling: np.ndarray | None = None,
) -> np.ndarray:
    """H1 in the frame; `coupling` overrides C (e.g. a truncated Taylor series of sin phi)."""
    c_op = josephson_operator(ats, space) if coupling is None else coupling
    energies = frame_harmonics(ats, space).diagonal_energies
    return first_order_terms(c_op, energies, drive_components(ats, tones))


def effective_hamiltonian(
    ats: ATSParams,
    space: FockSpace,
    tones: Sequence[FluxTone],
    order: int = 2,
    pad: int = DEFAULT_PAD,
) -> np.ndarray:
    """Detunings + H1 (+ H2), computed with `pad` extra intermediate levels, then truncated."""
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    work = space.padded(pad)
    c_op = josephson_operator(ats, work)
    energies = frame_harmonics(ats, work).diagonal_energies
    comps = drive_components(ats, tones)
    h = _detuning_diagonal(ats, work) + first_order_terms(c_op, energies, comps)
    if order == 2:
        h = h + second_order_hamiltonian(c_op, energies, comps, ats.unit)
    return _truncate(h, work, space)


def diagonal_coefficients(h: np.ndarray, space: FockSpace) -> dict[str, float]:
    """Stark shifts, Kerrs and cross-Kerr from Fock-diagonal energies E(n_a, n_b).

    Convention: H = stark_a a^dag a + stark_b b^dag b + (kerr_a / 2) a^dag^2 a^2
    + (kerr_b / 2) b^dag^2 b^2 + chi_ab a^dag a b^dag b + const.
    """

    def e(n_a: int, n_b: int) -> float:
        i = space.index(n_a, n_b)
        return float(np.real(h[i, i]))

    return {
        "stark_a": e(1, 0) - e(0, 0),
        "stark_b": e(0, 1) - e(0, 0),
        "kerr_a": e(2, 0) - 2 * e(1, 0) + e(0, 0),
        "kerr_b": e(0, 2) - 2 * e(0, 1) + e(0, 0),
        "chi_ab": e(1, 1) - e(1, 0) - e(0, 1) + e(0, 0),
    }


def dressed_g2(h: np.ndarray, space: FockSpace, n_a: int, n_b: int) -> complex:
    """Coefficient g2(n_a, n_b) of a^2 b^dag on |n_a, n_b> (matrix element / Fock factor)."""
    element = h[space.index(n_a - 2, n_b + 1), space.index(n_a, n_b)]
    return complex(element / math.sqrt(n_a * (n_a - 1) * (n_b + 1)))


# --- Floquet check -------------------------------------------------------------------------


def pump_displacement(ats: ATSParams) -> tuple[float, float]:
    """Classical pump-induced displacement (beta_a, beta_b) at t = 0 (linear response).

    The linear part of C is -exp(-Phi^2/2) phi, so the pump drives each mode with
    F phi_k cos(w_p t)(k + k^dag), F = 4 E_J J1(eps_p) exp(-Phi^2/2), and
    beta_k(0) = -F phi_k w_k / (w_k^2 - w_p^2).
    """
    force = 4 * ats.e_j * j1(ats.eps_p) * ats.normal_order_factor
    return tuple(  # type: ignore[return-value]
        -force * ph * w / (w**2 - ats.omega_p**2)
        for ph, w in ((ats.phi_a, ats.omega_a), (ats.phi_b, ats.omega_b))
    )


def rotating_frame_floquet(
    ats: ATSParams,
    space: FockSpace,
    tones: Sequence[FluxTone],
    periods: int = 1,
    rtol: float = ODE_RTOL,
    atol: float = ODE_ATOL,
) -> np.ndarray:
    """Exact propagator over `periods` x T in the frame rotating at (h_a u, h_b u).

    The frame returns to the lab after T, so this is also the lab-frame propagator U(T).
    H_rot(t) = detunings + c(t) [C o exp(i nu u t)] with the exact c(t) (all harmonics).
    """
    c_op = josephson_operator(ats, space).astype(complex)
    nu = frame_harmonics(ats, space).differences
    h_det = _detuning_diagonal(ats, space)
    dim = space.dim

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        h = h_det + flux_drive(ats, tones, t) * c_op * np.exp(1j * ats.unit * t * nu)
        return (-1j * h @ y.reshape(dim, dim)).ravel()

    sol = solve_ivp(
        rhs,
        (0.0, periods * ats.period),
        np.eye(dim, dtype=complex).ravel(),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Floquet propagation failed: {sol.message}")
    return sol.y[:, -1].reshape(dim, dim)


def quasi_energies(u: np.ndarray, period: float) -> np.ndarray:
    """Quasi-energies -arg(lambda) / T, folded into (-pi/T, pi/T] (rad/us)."""
    return -np.angle(np.linalg.eigvals(u)) / period


def _fold(x: np.ndarray, zone: float) -> np.ndarray:
    return (x + zone / 2) % zone - zone / 2


def match_floquet_states(
    u: np.ndarray, h_eff: np.ndarray, space: FockSpace, ats: ATSParams
) -> FloquetMatch:
    """Pair Floquet modes with effective eigenstates, the latter kicked by the pump displacement.

    Floquet modes at t = 0 are D(beta(0)) |psi_eff> up to higher-order micromotion, so the
    pairing maximizes |<v_j| D(beta(0)) |w_k>|^2. Results are ordered by the dominant Fock state
    of each effective eigenvector (index 0 is vacuum-like); quasi-energies are unfolded onto the
    effective eigenvalues.
    """
    lam, v = np.linalg.eig(u)
    eps = -np.angle(lam) / ats.period
    e_eff, w = np.linalg.eigh(h_eff)
    kicked = displacement_operator(space, *pump_displacement(ats)) @ w
    overlaps = np.abs(v.conj().T @ kicked) ** 2
    rows, cols = linear_sum_assignment(-overlaps)
    floquet_of = dict(zip(cols, rows, strict=True))
    dominant = np.argmax(np.abs(w) ** 2, axis=0)
    order = np.argsort(dominant, kind="stable")
    eff = e_eff[order]
    floq_raw = np.array([eps[floquet_of[k]] for k in order])
    floq = eff + _fold(floq_raw - eff, ats.unit)
    ov = np.array([overlaps[floquet_of[k], k] for k in order])
    labels = tuple(space.occupations(dominant[k]) for k in order)
    return FloquetMatch(labels=labels, floquet=floq, effective=eff, overlaps=ov)
