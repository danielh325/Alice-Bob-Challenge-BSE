"""Closed-form bit-flip theory for the dissipative cat (Task 2.1).

Model: the effective one-mode Lindbladian kappa2 D[a^2 - alpha^2] + kappa1 D[a], with
kappa1 = kappa_a and kappa2 = 4 g2^2 / kappa_b.

Rate conventions. Dubovitskii (PRA 111, 012617 (2025), arXiv:2407.17299) defines the
one-way jump rate Gamma_{0<->1} = -Re(lambda_z) / 2, where lambda_z is the eigenvalue of the
Bloch z component. Re<a> decays at -Re(lambda_z), so the observable bit-flip rate of this
project (catqubit.rates.bit_flip_rate) is Gamma_X = 2 Gamma_{0<->1}. Functions named
*_jump_rate use his convention; functions named *_bit_flip_rate return Gamma_X.

Formulas (checked against the arXiv text, Eqs. 6 and 23 and Appendix C):
  first order   Gamma^(1) = kappa1 alpha^2 (coth 2alpha^2 - 1) / 2  ~ kappa1 alpha^2 e^{-4 alpha^2}
  second order  Gamma^(2) = kappa1^2 alpha^2 [Shi(2a^2) + (1 - coth 2a^2) Chin(4a^2)]
                            / (2 kappa2 sinh^2 2a^2)                 ~ kappa1^2/(2 kappa2) e^{-2a^2}
with Chin(z) = Chi(z) - ln z - gamma_E (the entire part of the hyperbolic cosine integral).
Valid for kappa1 << kappa2; double precision overflows for alpha^2 above about 170.
"""

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike
from scipy.special import lambertw, shichi

from catqubit.liouvillian import FockTruncation, lindblad_superoperator, parity_block
from catqubit.params import CatParams
from catqubit.rates import slowest_eigenvalues

MIN_EXPONENT_POINTS = 3
MIN_FIT_POINTS = 2
# Memory truncation for the effective model: mean alpha^2 plus a generous Poisson tail.
EFFECTIVE_NA_OFFSET = 12
EFFECTIVE_NA_PER_ALPHA = 8
# Bracket (in alpha^2) for the root of Gamma^(1) = Gamma^(2).
CROSSOVER_BRACKET = (1e-3, 100.0)
# x e^{-x} = kappa1/kappa2 has a real solution on the lower branch only if kappa1/kappa2 < 1/e.
LAMBERT_LIMIT = 1 / math.e


def _check_positive(name: str, value: ArrayLike) -> None:
    if np.any(np.asarray(value) <= 0):
        raise ValueError(f"{name} must be positive, got {value}")


def _check_rates(kappa1: float, kappa2: float | None, alpha2: ArrayLike) -> None:
    if kappa1 < 0:
        raise ValueError(f"kappa1 must be non-negative, got {kappa1}")
    if kappa2 is not None:
        _check_positive("kappa2", kappa2)
    _check_positive("alpha2", alpha2)


def _coth_minus_one(x: ArrayLike) -> np.ndarray:
    """coth(x) - 1 = 2 / (e^{2x} - 1), without the cancellation at large x."""
    return 2.0 / np.expm1(2.0 * np.asarray(x, dtype=float))


def chin(z: ArrayLike) -> np.ndarray:
    """Chin(z) = Chi(z) - ln z - gamma_E = sum_k z^{2k} / (2k (2k)!)."""
    _, chi = shichi(z)
    return chi - np.log(z) - np.euler_gamma


def first_order_jump_rate(kappa1: float, alpha2: ArrayLike) -> np.ndarray:
    """Dubovitskii Eq. 6 (exact form, Appendix C): kappa1 alpha^2 (coth 2alpha^2 - 1) / 2."""
    _check_rates(kappa1, None, alpha2)
    alpha2 = np.asarray(alpha2, dtype=float)
    return kappa1 * alpha2 * _coth_minus_one(2 * alpha2) / 2


def second_order_jump_rate(kappa1: float, kappa2: float, alpha2: ArrayLike) -> np.ndarray:
    """Dubovitskii Eq. 23: the photon-loss path through non-computational states."""
    _check_rates(kappa1, kappa2, alpha2)
    alpha2 = np.asarray(alpha2, dtype=float)
    shi, _ = shichi(2 * alpha2)
    bracket = shi - _coth_minus_one(2 * alpha2) * chin(4 * alpha2)
    return kappa1**2 * alpha2 * bracket / (2 * kappa2 * np.sinh(2 * alpha2) ** 2)


def dubovitskii_jump_rate(kappa1: float, kappa2: float, alpha2: ArrayLike) -> np.ndarray:
    """Gamma_{0<->1} to second order in kappa1 / kappa2 (Dubovitskii's convention)."""
    return first_order_jump_rate(kappa1, alpha2) + second_order_jump_rate(kappa1, kappa2, alpha2)


def dubovitskii_bit_flip_rate(kappa1: float, kappa2: float, alpha2: ArrayLike) -> np.ndarray:
    """Observable decay rate of Re<a>: Gamma_X = 2 Gamma_{0<->1}."""
    return 2 * dubovitskii_jump_rate(kappa1, kappa2, alpha2)


def asymptotic_jump_rate(kappa1: float, kappa2: float, alpha2: ArrayLike) -> np.ndarray:
    """Large-alpha form (Dubovitskii Table I).

    kappa1 a^2 e^{-4a^2} + kappa1^2/(2 kappa2) e^{-2a^2}, in his jump-rate convention.
    """
    _check_rates(kappa1, kappa2, alpha2)
    alpha2 = np.asarray(alpha2, dtype=float)
    return kappa1 * alpha2 * np.exp(-4 * alpha2) + kappa1**2 / (2 * kappa2) * np.exp(-2 * alpha2)


def asymptotic_bit_flip_rate(kappa1: float, kappa2: float, alpha2: ArrayLike) -> np.ndarray:
    """Observable form of asymptotic_jump_rate (times 2)."""
    return 2 * asymptotic_jump_rate(kappa1, kappa2, alpha2)


def _check_weak_loss(kappa1: float, kappa2: float) -> None:
    _check_positive("kappa2", kappa2)
    if not 0 < kappa1 / kappa2 < LAMBERT_LIMIT:
        raise ValueError(f"kappa1/kappa2 must lie in (0, 1/e), got kappa1={kappa1}, {kappa2=}")


def crossover_alpha2_leading_log(kappa1: float, kappa2: float) -> float:
    """Dubovitskii's estimate alpha^2 ~ ln(kappa2/kappa1)/2 (prefactors dropped)."""
    _check_weak_loss(kappa1, kappa2)
    return math.log(kappa2 / kappa1) / 2


def crossover_alpha2_asymptotic(kappa1: float, kappa2: float) -> float:
    """alpha^2 where the two asymptotic terms are equal: x e^{-x} = kappa1/kappa2, x = 2alpha^2.

    The solution on the large-x branch is x = -W_{-1}(-kappa1/kappa2) (Lambert W)."""
    _check_weak_loss(kappa1, kappa2)
    return float(-lambertw(-kappa1 / kappa2, k=-1).real / 2)


def crossover_alpha2_exact(kappa1: float, kappa2: float) -> float:
    """alpha^2 where the exact first- and second-order rates are equal (root find)."""
    from scipy.optimize import brentq

    _check_weak_loss(kappa1, kappa2)

    def log_ratio(a2: float) -> float:
        return float(
            np.log(first_order_jump_rate(kappa1, a2))
            - np.log(second_order_jump_rate(kappa1, kappa2, a2))
        )

    return float(brentq(log_ratio, *CROSSOVER_BRACKET, xtol=1e-12))


def thermal_bit_flip_estimate(kappa1: float, n_th: float, alpha2: ArrayLike) -> np.ndarray:
    """Order-of-magnitude ESTIMATE kappa1 n_th e^{-2 alpha^2} of thermally induced bit flips.

    Gautier, Sarlette & Mirrahimi, arXiv:2112.05545 (Sec. II and Eq. B17):
    leakage at kappa_l = kappa1 n_th followed by re-convergence gives Gamma ~ kappa_l e^{-2a^2}.
    The O(1) prefactor and the factor-2 convention are not derived there; use as a scale only.
    """
    if n_th < 0:
        raise ValueError(f"n_th must be non-negative, got {n_th}")
    _check_rates(kappa1, None, alpha2)
    return kappa1 * n_th * np.exp(-2 * np.asarray(alpha2, dtype=float))


def effective_memory_dim(alpha: float) -> int:
    """A memory truncation for the one-mode model, converged to < 1e-5 for alpha <= 2.5."""
    return math.ceil(alpha**2 + EFFECTIVE_NA_PER_ALPHA * alpha + EFFECTIVE_NA_OFFSET)


def _destroy(n: int) -> sp.csr_matrix:
    return sp.diags(np.sqrt(np.arange(1, n, dtype=float)), 1, format="csr").astype(complex)


def effective_bit_flip_rate(params: CatParams, na: int, n_th: float = 0.0) -> float:
    """Gamma_X of the effective one-mode model: slowest rate of its parity-odd sector.

    L = kappa2 D[a^2 - alpha^2] + kappa1 (1 + n_th) D[a] + kappa1 n_th D[a^dag], with
    kappa2 = 4 g2^2 / kappa_b and kappa1 = kappa_a. Same convention as bit_flip_rate.
    """
    if n_th < 0:
        raise ValueError(f"n_th must be non-negative, got {n_th}")
    a = _destroy(na)
    eye = sp.identity(na, format="csr", dtype=complex)
    jumps = [math.sqrt(params.kappa2) * (a @ a - params.alpha**2 * eye)]
    if params.kappa_a > 0:
        jumps.append(math.sqrt(params.kappa_a * (1 + n_th)) * a)
    if params.kappa_a > 0 and n_th > 0:
        jumps.append(math.sqrt(params.kappa_a * n_th) * a.T.tocsr())
    big_l = lindblad_superoperator(sp.csr_matrix((na, na), dtype=complex), jumps)
    odd, _ = parity_block(big_l, FockTruncation(na=na, nb=1), odd=True)
    return float(-slowest_eigenvalues(odd, k=1)[0].real)


def _check_samples(alpha2: ArrayLike, rates: ArrayLike, minimum: int) -> tuple:
    alpha2, rates = np.asarray(alpha2, dtype=float), np.asarray(rates, dtype=float)
    if alpha2.shape != rates.shape or alpha2.size < minimum:
        word = {MIN_FIT_POINTS: "two", MIN_EXPONENT_POINTS: "three"}[minimum]
        raise ValueError(f"need at least {word} (alpha^2, rate) samples of matching shape")
    if np.any(np.diff(alpha2) <= 0):
        raise ValueError("alpha2 must be strictly increasing")
    if np.any(rates <= 0):
        raise ValueError("rates must be positive")
    return alpha2, rates


def local_exponent(alpha2: ArrayLike, rates: ArrayLike) -> np.ndarray:
    """-d ln(Gamma) / d(alpha^2) at each sample (second-order finite differences)."""
    alpha2, rates = _check_samples(alpha2, rates, MIN_EXPONENT_POINTS)
    return -np.gradient(np.log(rates), alpha2, edge_order=2)


@dataclass(frozen=True)
class ExponentFit:
    """Least-squares fit ln Gamma = log_prefactor + prefactor_power ln(alpha^2) - gamma alpha^2."""

    gamma: float
    log_prefactor: float
    prefactor_power: int
    rms_residual: float


def fit_exponent(alpha2: ArrayLike, rates: ArrayLike, prefactor_power: int = 0) -> ExponentFit:
    """Fit Gamma = A (alpha^2)^p e^{-gamma alpha^2} with p fixed.

    p = 0 is the plain form of the challenge; p = 1 is Putterman et al. (2025,
    arXiv:2409.17556), who fit T_bit-flip ~ e^{gamma |alpha|^2} / |alpha|^2.
    """
    alpha2, rates = _check_samples(alpha2, rates, MIN_FIT_POINTS)
    y = np.log(rates) - prefactor_power * np.log(alpha2)
    slope, intercept = np.polyfit(alpha2, y, 1)
    residual = y - (intercept + slope * alpha2)
    return ExponentFit(
        gamma=float(-slope),
        log_prefactor=float(intercept),
        prefactor_power=prefactor_power,
        rms_residual=float(np.sqrt(np.mean(residual**2))),
    )
