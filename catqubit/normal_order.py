"""Exact normal-ordered polynomial algebra for the flux-pumped ATS circuit.

Physics
-------
H = w_a a^dag a + w_b b^dag b + V(t),  V(t) = sum_drives A_k cos(w_k t + theta_k) sin(phihat),
phihat = phi_a (a + a^dag) + phi_b (b + b^dag).
Pump:    A   = 2 E_J eps_p, w_p (nominally 2 w_a - w_b), no phase.
Readout: A_z = 2 E_J eps_z, w_z = w_a,        phase theta.

Conventions
-----------
* Operator key ``(p, q, r, s)`` is the normal-ordered monomial a^dag^p a^q b^dag^r b^s.
* Normal-ordered sine (exact, Baker-Campbell-Hausdorff):
  sin(phihat) = E * sum_{n=p+q+r+s odd} (-1)^((n-1)/2) phi_a^(p+q) phi_b^(r+s)/(p! q! r! s!)
  * a^dag^p a^q b^dag^r b^s,   E = exp(-Phi^2/2),  Phi^2 = phi_a^2 + phi_b^2.
  E is kept as an exact symbolic factor; the polynomial is truncated at degree D.
* A coefficient is a finite sum of ``Sym`` monomials times Gaussian rationals:
  (re + i im) * phi_a^i phi_b^j A^k A_z^l E^m exp(i n theta) / prod_d nu_d.
* Frequencies: ``FreqKey`` (n_a, n_b, n_p) means nu = n_a w_a + n_b w_b + n_p w_p, with w_p
  kept independent (general formulas, valid for a detuned pump) and the tone w_z = w_a folded
  into n_a. Whether nu vanishes is decided at the integer ratio w_a : w_b : w_p = 5 : 6 : 4
  (``RATIO``, u = 2 pi x 0.8 GHz). Each 1/nu is stored with the sign making nu > 0 at RATIO.
  ``Expr.reduced()`` substitutes the resonance condition w_p = 2 w_a - w_b exactly.
* Interaction picture: a^dag^p a^q b^dag^r b^s -> exp(i[(p-q) w_a + (r-s) w_b] t).
* First order: H1 = V(0) (RWA). Second order (time averaging / van Vleck):
  H2 = sum_{nu != 0} V(nu) V(-nu) / nu,  with V_I(t) = sum_nu V(nu) exp(i nu t).
  This gives the textbook AC Stark sign: a level above its partner is pushed up.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache
from types import MappingProxyType

import numpy as np

OpKey = tuple[int, int, int, int]
FreqKey = tuple[int, int, int]

RATIO: FreqKey = (5, 6, 4)  # w_a : w_b : w_p in units of u = 2 pi x 0.8 GHz
PUMP_FREQ: FreqKey = (0, 0, 1)
TONE_FREQ: FreqKey = (1, 0, 0)  # w_z = w_a
IDENTITY: OpKey = (0, 0, 0, 0)

TWO_PI = 2.0 * math.pi
MHZ = TWO_PI  # 1 MHz in rad/us
GHZ = 1000.0 * MHZ
W_A = 4.0 * GHZ
W_B = 4.8 * GHZ
E_J = 40.0 * GHZ
PHI_A = 0.06
PHI_B = 0.3
G2_TARGET = 0.763 * MHZ  # g2 := (1/2) E_J eps_p phi_a^2 phi_b


# ----------------------------------------------------------------------------- scalars


@dataclass(frozen=True)
class GaussRat:
    """Exact complex rational re + i im."""

    re: Fraction
    im: Fraction = Fraction(0)

    def __add__(self, other: GaussRat) -> GaussRat:
        return GaussRat(self.re + other.re, self.im + other.im)

    def __mul__(self, other: GaussRat) -> GaussRat:
        return GaussRat(
            self.re * other.re - self.im * other.im,
            self.re * other.im + self.im * other.re,
        )

    def times_int(self, n: int) -> GaussRat:
        return GaussRat(self.re * n, self.im * n)

    def conj(self) -> GaussRat:
        return GaussRat(self.re, -self.im)

    def is_zero(self) -> bool:
        return self.re == 0 and self.im == 0

    def to_complex(self) -> complex:
        return complex(float(self.re), float(self.im))


ONE = GaussRat(Fraction(1))
HALF = GaussRat(Fraction(1, 2))


@dataclass(frozen=True, order=True)
class Sym:
    """Symbolic monomial phi_a^i phi_b^j A^k A_z^l E^m exp(i n theta) / prod(nu_d)."""

    phi_a: int = 0
    phi_b: int = 0
    pump: int = 0
    tone: int = 0
    env: int = 0
    theta: int = 0
    denoms: tuple[FreqKey, ...] = ()

    def __mul__(self, other: Sym) -> Sym:
        return Sym(
            self.phi_a + other.phi_a,
            self.phi_b + other.phi_b,
            self.pump + other.pump,
            self.tone + other.tone,
            self.env + other.env,
            self.theta + other.theta,
            tuple(sorted(self.denoms + other.denoms)),
        )

    @property
    def phi_degree(self) -> int:
        return self.phi_a + self.phi_b

    def conj(self) -> Sym:
        return Sym(
            self.phi_a,
            self.phi_b,
            self.pump,
            self.tone,
            self.env,
            -self.theta,
            self.denoms,
        )

    def reduced(self) -> tuple[int, Sym]:
        """(sign, Sym) with w_p -> 2 w_a - w_b substituted in every denominator."""
        sign, denoms = 1, []
        for d in self.denoms:
            s, canon = canonical_denominator(reduce_frequency(d))
            sign, denoms = sign * s, [*denoms, canon]
        base = self.without_denoms()
        return sign, Sym(
            base.phi_a,
            base.phi_b,
            base.pump,
            base.tone,
            base.env,
            base.theta,
            tuple(sorted(denoms)),
        )

    def without_denoms(self) -> Sym:
        return Sym(self.phi_a, self.phi_b, self.pump, self.tone, self.env, self.theta)


NO_SYM = Sym()  # the symbolic unit 1

# ----------------------------------------------------------------------------- operators

TermKey = tuple[OpKey, Sym]


@dataclass(frozen=True, eq=False)
class Expr:
    """Immutable sum of (normal-ordered monomial x Sym x GaussRat) terms."""

    terms: Mapping[TermKey, GaussRat] = field(default_factory=lambda: MappingProxyType({}))

    @staticmethod
    def from_items(items: Iterable[tuple[TermKey, GaussRat]]) -> Expr:
        acc: dict[TermKey, GaussRat] = {}
        for key, value in items:
            acc[key] = acc[key] + value if key in acc else value
        return Expr(MappingProxyType({k: v for k, v in acc.items() if not v.is_zero()}))

    @staticmethod
    def zero() -> Expr:
        return Expr()

    @staticmethod
    def monomial(op: OpKey, sym: Sym = NO_SYM, coeff: GaussRat = ONE) -> Expr:
        return Expr.from_items([((op, sym), coeff)])

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Expr) and dict(self.terms) == dict(other.terms)

    def __hash__(self) -> int:
        return hash(frozenset(self.terms.items()))

    def __add__(self, other: Expr) -> Expr:
        return Expr.from_items([*self.terms.items(), *other.terms.items()])

    def scaled(self, coeff: GaussRat, sym: Sym = NO_SYM) -> Expr:
        return Expr.from_items(((op, s * sym), c * coeff) for (op, s), c in self.terms.items())

    def dagger(self) -> Expr:
        return Expr.from_items(
            (((q, p, s, r), sym.conj()), c.conj()) for ((p, q, r, s), sym), c in self.terms.items()
        )

    def filtered(self, keep: Callable[[OpKey, Sym], bool]) -> Expr:
        return Expr.from_items(((op, s), c) for (op, s), c in self.terms.items() if keep(op, s))

    def reduced(self) -> Expr:
        """Same operator with the resonance condition w_p = 2 w_a - w_b substituted."""
        items = []
        for (op, sym), c in self.terms.items():
            sign, red = sym.reduced()
            items.append(((op, red), c.times_int(sign)))
        return Expr.from_items(items)

    def coefficient(self, op: OpKey) -> dict[Sym, GaussRat]:
        return {s: c for (o, s), c in self.terms.items() if o == op}

    def op_keys(self) -> set[OpKey]:
        return {op for op, _ in self.terms}


@cache
def _single_mode_wick(q1: int, p2: int) -> tuple[tuple[int, int], ...]:
    """a^q1 a^dag^p2 = sum_k C(q1,k) C(p2,k) k! a^dag^(p2-k) a^(q1-k); returns (k, mult)."""
    return tuple(
        (k, math.comb(q1, k) * math.comb(p2, k) * math.factorial(k)) for k in range(min(q1, p2) + 1)
    )


@cache
def wick_product(k1: OpKey, k2: OpKey) -> tuple[tuple[OpKey, int], ...]:
    """Normal-ordered expansion of the product of two normal-ordered monomials."""
    p1, q1, r1, s1 = k1
    p2, q2, r2, s2 = k2
    out = []
    for ka, ma in _single_mode_wick(q1, p2):
        for kb, mb in _single_mode_wick(s1, r2):
            key = (p1 + p2 - ka, q1 + q2 - ka, r1 + r2 - kb, s1 + s2 - kb)
            out.append((key, ma * mb))
    return tuple(out)


def multiply(left: Expr, right: Expr, max_phi_degree: int | None = None) -> Expr:
    """Normal-ordered product left * right, dropping terms of phi degree > max_phi_degree."""
    items = []
    for (op1, s1), c1 in left.terms.items():
        for (op2, s2), c2 in right.terms.items():
            sym = s1 * s2
            if max_phi_degree is not None and sym.phi_degree > max_phi_degree:
                continue
            coeff = c1 * c2
            items.extend(((op, sym), coeff.times_int(m)) for op, m in wick_product(op1, op2))
    return Expr.from_items(items)


# ----------------------------------------------------------------------------- expansion


def _compositions(n: int) -> Iterable[OpKey]:
    for p in range(n + 1):
        for q in range(n + 1 - p):
            for r in range(n + 1 - p - q):
                yield (p, q, r, n - p - q - r)


def sin_expansion(max_degree: int) -> Expr:
    """Normal-ordered sin(phihat) truncated at operator degree <= max_degree (exact E)."""
    items = []
    for n in range(1, max_degree + 1, 2):
        sign = -1 if (n - 1) // 2 % 2 else 1
        for p, q, r, s in _compositions(n):
            denom = math.factorial(p) * math.factorial(q) * math.factorial(r) * math.factorial(s)
            sym = Sym(phi_a=p + q, phi_b=r + s, env=1)
            items.append((((p, q, r, s), sym), GaussRat(Fraction(sign, denom))))
    return Expr.from_items(items)


@dataclass(frozen=True)
class Drive:
    """A drive amplitude * cos(w t [+ theta]) multiplying sin(phihat)."""

    name: str
    freq: FreqKey
    pump_power: int
    tone_power: int
    carries_phase: bool


PUMP = Drive("pump", PUMP_FREQ, 1, 0, False)
READOUT = Drive("readout", TONE_FREQ, 0, 1, True)


def rotation(op: OpKey) -> FreqKey:
    p, q, r, s = op
    return (p - q, r - s, 0)


def _add_freq(f1: FreqKey, f2: FreqKey, sign: int = 1) -> FreqKey:
    return (f1[0] + sign * f2[0], f1[1] + sign * f2[1], f1[2] + sign * f2[2])


def reduce_frequency(freq: FreqKey) -> FreqKey:
    """Substitute w_p = 2 w_a - w_b."""
    return (freq[0] + 2 * freq[2], freq[1] - freq[2], 0)


def interaction_picture(expr: Expr) -> dict[FreqKey, Expr]:
    """Fourier components V(nu) of a static operator in the frame of w_a a^dag a + w_b b^dag b."""
    groups: dict[FreqKey, list] = {}
    for (op, s), c in expr.terms.items():
        groups.setdefault(rotation(op), []).append(((op, s), c))
    return {f: Expr.from_items(items) for f, items in groups.items()}


def frequency_components(expansion: Expr, drives: Iterable[Drive]) -> dict[FreqKey, Expr]:
    """V(nu) for V = sum_drives A cos(w t + theta) * expansion; cos = (e^{+i} + e^{-i})/2."""
    groups: dict[FreqKey, list] = {}
    rotated = interaction_picture(expansion)
    for drive in drives:
        for sign in (1, -1):
            drive_sym = Sym(
                pump=drive.pump_power,
                tone=drive.tone_power,
                theta=sign if drive.carries_phase else 0,
            )
            for f_op, part in rotated.items():
                key = _add_freq(f_op, drive.freq, sign)
                groups.setdefault(key, []).extend(part.scaled(HALF, drive_sym).terms.items())
    return {f: Expr.from_items(items) for f, items in groups.items()}


def frequency_value(freq: FreqKey, ratio: FreqKey = RATIO) -> int:
    """nu in units of u at the commensurate ratio (w_a, w_b, w_p) = ratio * u."""
    return sum(r * f for r, f in zip(ratio, freq, strict=True))


def canonical_denominator(freq: FreqKey) -> tuple[int, FreqKey]:
    """Write 1/nu as sign/|nu| so that 1/nu and 1/(-nu) cancel symbolically."""
    if frequency_value(freq) == 0:
        raise ZeroDivisionError(f"resonant denominator {freq}")
    if frequency_value(freq) > 0:
        return 1, freq
    return -1, (-freq[0], -freq[1], -freq[2])


def first_order(components: Mapping[FreqKey, Expr]) -> Expr:
    """RWA Hamiltonian: the sum of all zero-frequency components."""
    total = Expr.zero()
    for freq, part in components.items():
        if frequency_value(freq) == 0:
            total = total + part
    return total


def _half_inverse(freq: FreqKey) -> Expr:
    """(1/2) * 1/nu as a c-number Expr with a canonical-sign denominator."""
    sign, denom = canonical_denominator(freq)
    return Expr.monomial(IDENTITY, Sym(denoms=(denom,)), GaussRat(Fraction(sign, 2)))


def second_order(
    left: Mapping[FreqKey, Expr],
    right: Mapping[FreqKey, Expr],
    max_phi_degree: int | None = None,
) -> Expr:
    """Hermitian van Vleck second order, normal ordered.

    sum over pairs (nu1, nu2) with nu1 + nu2 = 0 at RATIO and nu1 != 0 of
    left(nu1) right(nu2) * (1/2) [1/nu1 - 1/nu2]. This equals left(nu1) right(-nu1) / nu1
    when nu2 = -nu1 exactly, and stays Hermitian when nu1 + nu2 vanishes only through the
    resonance condition w_p = 2 w_a - w_b (e.g. pump x tone, a^4 b^dag^2).
    """
    by_value: dict[int, list[tuple[FreqKey, Expr]]] = {}
    for freq, part in right.items():
        by_value.setdefault(frequency_value(freq), []).append((freq, part))
    items = []
    for f1, part in left.items():
        value = frequency_value(f1)
        if value == 0:
            continue
        for f2, partner in by_value.get(-value, []):
            weight = _half_inverse(f1) + _half_inverse((-f2[0], -f2[1], -f2[2]))
            product = multiply(part, partner, max_phi_degree)
            for (op, sym), c in product.terms.items():
                for (_, wsym), wc in weight.terms.items():
                    items.append(((op, sym * wsym), c * wc))
    return Expr.from_items(items)


def split_linear(
    components: Mapping[FreqKey, Expr],
) -> tuple[dict[FreqKey, Expr], dict[FreqKey, Expr]]:
    """Split each V(nu) into its linear (single a, a^dag, b, b^dag) and nonlinear parts."""
    linear = {f: e.filtered(lambda op, _s: sum(op) == 1) for f, e in components.items()}
    nonlinear = {f: e.filtered(lambda op, _s: sum(op) != 1) for f, e in components.items()}
    return linear, nonlinear


# ----------------------------------------------------------------------------- numerics


@dataclass(frozen=True)
class CircuitParams:
    """Numeric parameters (angular frequencies, e.g. rad/us)."""

    w_a: float
    w_b: float
    e_j: float
    phi_a: float
    phi_b: float
    eps_p: float
    eps_z: float = 0.0
    theta: float = 0.0
    w_p: float | None = None  # None: resonant pump 2 w_a - w_b

    @property
    def amplitude_pump(self) -> float:
        return 2.0 * self.e_j * self.eps_p

    @property
    def amplitude_tone(self) -> float:
        return 2.0 * self.e_j * self.eps_z

    @property
    def envelope(self) -> float:
        return math.exp(-(self.phi_a**2 + self.phi_b**2) / 2.0)

    @property
    def pump_frequency(self) -> float:
        return 2.0 * self.w_a - self.w_b if self.w_p is None else self.w_p

    def frequency(self, freq: FreqKey) -> float:
        return freq[0] * self.w_a + freq[1] * self.w_b + freq[2] * self.pump_frequency


def reference_parameters(eps_z: float = 0.0, theta: float = 0.0) -> CircuitParams:
    """Task 2.4 illustrative circuit in rad/us; eps_p fixed by (1/2) E_J eps_p phi_a^2 phi_b."""
    eps_p = 2.0 * G2_TARGET / (E_J * PHI_A**2 * PHI_B)
    return CircuitParams(W_A, W_B, E_J, PHI_A, PHI_B, eps_p, eps_z, theta)


def evaluate_sym(sym: Sym, params: CircuitParams) -> complex:
    value = (
        params.phi_a**sym.phi_a
        * params.phi_b**sym.phi_b
        * params.amplitude_pump**sym.pump
        * params.amplitude_tone**sym.tone
        * params.envelope**sym.env
    )
    for d in sym.denoms:
        value /= params.frequency(d)
    return value * complex(math.cos(sym.theta * params.theta), math.sin(sym.theta * params.theta))


def evaluate_coefficient(coeff: Mapping[Sym, GaussRat], params: CircuitParams) -> complex:
    return sum((c.to_complex() * evaluate_sym(s, params) for s, c in coeff.items()), 0j)


def coefficient_value(
    expr: Expr,
    op: OpKey,
    params: CircuitParams,
    phi_degrees: Iterable[int] | None = None,
) -> complex:
    """Numeric coefficient of the monomial op, optionally restricted to given phi degrees."""
    coeff = expr.coefficient(op)
    if phi_degrees is not None:
        allowed = set(phi_degrees)
        coeff = {s: c for s, c in coeff.items() if s.phi_degree in allowed}
    return evaluate_coefficient(coeff, params)


def _ladder(n: int) -> np.ndarray:
    return np.diag(np.sqrt(np.arange(1, n, dtype=float)), k=1)


def expr_to_matrix(expr: Expr, params: CircuitParams, n_a: int, n_b: int) -> np.ndarray:
    """Dense Fock matrix (a slow index); normal-ordered monomials are exact below the cutoff."""
    a, b = _ladder(n_a), _ladder(n_b)
    out = np.zeros((n_a * n_b, n_a * n_b), dtype=complex)
    for op in expr.op_keys():
        p, q, r, s = op
        value = evaluate_coefficient(expr.coefficient(op), params)
        mat_a = np.linalg.matrix_power(a.T, p) @ np.linalg.matrix_power(a, q)
        mat_b = np.linalg.matrix_power(b.T, r) @ np.linalg.matrix_power(b, s)
        out = out + value * np.kron(mat_a, mat_b)
    return out


# ----------------------------------------------------------------------------- formatting


def format_frequency(freq: FreqKey) -> str:
    """Write n_a w_a + n_b w_b + n_p w_p."""
    parts = []
    for n, name in zip(freq, ("w_a", "w_b", "w_p"), strict=True):
        if n == 0:
            continue
        mag = "" if abs(n) == 1 else f"{abs(n)} "
        parts.append(("-" if n < 0 else "+") + f" {mag}{name}")
    return " ".join(parts).removeprefix("+ ")


def _format_rational(c: GaussRat) -> str:
    if c.im == 0:
        return str(c.re)
    if c.re == 0:
        return f"{c.im}i"
    return f"({c.re}{'+' if c.im > 0 else '-'}{abs(c.im)}i)"


def format_prefactor(sym: Sym) -> str:
    factors = [
        ("A", sym.pump),
        ("A_z", sym.tone),
        ("phi_a", sym.phi_a),
        ("phi_b", sym.phi_b),
        ("E", sym.env),
    ]
    text = " ".join(name if n == 1 else f"{name}^{n}" for name, n in factors if n)
    if sym.theta:
        text += f" e^{{{sym.theta}i theta}}"
    return text or "1"


def format_coefficient(coeff: Mapping[Sym, GaussRat]) -> str:
    """Human-readable sum; E = exp(-Phi^2/2); denominators in w_a, w_b, w_p (w_z = w_a)."""
    groups: dict[Sym, list[tuple[tuple[FreqKey, ...], GaussRat]]] = {}
    for sym, c in coeff.items():
        groups.setdefault(sym.without_denoms(), []).append((sym.denoms, c))
    chunks = []
    for base in sorted(groups):
        terms = []
        for denoms, c in sorted(groups[base]):
            den = "".join(f"/({format_frequency(d)})" for d in denoms)
            terms.append(f"{_format_rational(c)}{den}")
        inner = terms[0] if len(terms) == 1 else "(" + " + ".join(terms) + ")"
        chunks.append(f"{inner} * {format_prefactor(base)}")
    return " + ".join(chunks) if chunks else "0"


# --- Fock-space all-orders cross-check (exact matrix sin(phi), no phi expansion) ---


@dataclass(frozen=True)
class FockSecondOrder:
    h2: np.ndarray
    n: int

    def element(self, bra: tuple[int, int], ket: tuple[int, int]) -> complex:
        return complex(self.h2[bra[0] * self.n + bra[1], ket[0] * self.n + ket[1]])

    def energy(self, na: int, nb: int) -> float:
        return self.element((na, nb), (na, nb)).real


def _exact_sin(params: CircuitParams, n: int) -> np.ndarray:
    lad = np.diag(np.sqrt(np.arange(1, n, dtype=float)), k=1)
    a, b = np.kron(lad, np.eye(n)), np.kron(np.eye(n), lad)
    phihat = params.phi_a * (a + a.T) + params.phi_b * (b + b.T)
    vals, vecs = np.linalg.eigh(phihat)
    return (vecs * np.sin(vals)) @ vecs.T


def _components(params: CircuitParams, n: int) -> dict[int, np.ndarray]:
    """V(nu) keyed by nu in units of u; drives: pump (4u, no phase) and tone (5u, phase theta)."""
    sin_m = _exact_sin(params, n)
    occ = np.arange(n * n)
    na, nb = occ // n, occ % n
    rot = RATIO[0] * (na[:, None] - na[None, :]) + RATIO[1] * (nb[:, None] - nb[None, :])
    drives = [(params.amplitude_pump / 2, RATIO[2], 0.0)]
    if params.amplitude_tone:
        drives.append((params.amplitude_tone / 2, RATIO[0], params.theta))
    comps: dict[int, np.ndarray] = {}
    for amp, w, phase in drives:
        for sign in (1, -1):
            total = rot + sign * w
            factor = amp * np.exp(1j * sign * phase)
            for v in np.unique(total):
                part = np.where(total == v, sin_m, 0.0) * factor
                comps[int(v)] = comps.get(int(v), 0) + part
    return comps


def fock_second_order(params: CircuitParams, n: int) -> FockSecondOrder:
    u = params.w_a / RATIO[0]
    comps = _components(params, n)
    h2 = np.zeros((n * n, n * n), dtype=complex)
    for v, mat in comps.items():
        if v == 0 or -v not in comps:
            continue
        h2 = h2 + mat @ comps[-v] / (v * u)
    return FockSecondOrder(h2, n)


def coefficients(fs: FockSecondOrder) -> dict[str, complex]:
    e = fs.energy
    return {
        "Delta_a": e(1, 0) - e(0, 0),
        "Delta_b": e(0, 1) - e(0, 0),
        "K_a": e(2, 0) - 2 * e(1, 0) + e(0, 0),
        "K_b": e(0, 2) - 2 * e(0, 1) + e(0, 0),
        "chi_ab": e(1, 1) - e(1, 0) - e(0, 1) + e(0, 0),
        "c_number": e(0, 0),
        "c_a4b2": fs.element((0, 2), (4, 0)) / math.sqrt(math.factorial(4) * 2),
        "pt_a_bdag": fs.element((0, 1), (1, 0)),
    }
