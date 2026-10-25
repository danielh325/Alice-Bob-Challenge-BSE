"""Sparse two-mode Liouvillian for exact rates (Track 2).

Conventions: column-stacking vec, so vec(A rho B) = (B^T kron A) vec(rho). The joint
Fock index runs memory-major: state (n_a, n_b) comes before (n_a, n_b + 1).
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike

from catqubit.params import CatParams

MIN_MEMORY_DIM = 3


def vec(m: ArrayLike) -> np.ndarray:
    """Column-stacked vector of a matrix."""
    return np.asarray(m).ravel(order="F")


@dataclass(frozen=True)
class FockTruncation:
    """Joint Fock truncation, optionally with a wider buffer where the memory is nearly empty.

    The buffer keeps nb levels everywhere, plus up to nb_extra more when the memory holds
    fewer than na_taper photons (tapering linearly to zero). Near the barrier between the
    two wells the memory is close to vacuum while the drive fills the buffer, so that is
    where the buffer needs room; elsewhere a few levels suffice.
    """

    na: int
    nb: int
    nb_extra: int = 0
    na_taper: float = 1.0

    def __post_init__(self) -> None:
        if self.na < MIN_MEMORY_DIM:
            raise ValueError(f"na must be at least {MIN_MEMORY_DIM}, got {self.na}")
        if self.nb < 1:
            raise ValueError(f"nb must be at least 1, got {self.nb}")
        if self.nb_extra < 0:
            raise ValueError(f"nb_extra must be non-negative, got {self.nb_extra}")
        if self.na_taper <= 0:
            raise ValueError(f"na_taper must be positive, got {self.na_taper}")

    @property
    def nb_max(self) -> int:
        return self.nb + self.nb_extra

    def buffer_levels(self, n_a: int) -> int:
        widening = max(0.0, 1.0 - n_a / self.na_taper)
        return self.nb + math.ceil(self.nb_extra * widening)

    @cached_property
    def _states(self) -> tuple[np.ndarray, np.ndarray]:
        pairs = [(n_a, n_b) for n_a in range(self.na) for n_b in range(self.buffer_levels(n_a))]
        n_a, n_b = np.array(pairs).T
        return n_a, n_b

    @property
    def memory_photons(self) -> np.ndarray:
        return self._states[0]

    @property
    def buffer_photons(self) -> np.ndarray:
        return self._states[1]

    @property
    def dim(self) -> int:
        return len(self.memory_photons)

    def kept_indices(self, full_nb: int | None = None) -> np.ndarray:
        """Positions of the kept states in the rectangular na x full_nb grid."""
        full_nb = self.nb_max if full_nb is None else full_nb
        return self.memory_photons * full_nb + self.buffer_photons


def _destroy(n: int) -> sp.csr_matrix:
    return sp.diags(np.sqrt(np.arange(1, n, dtype=float)), 1, format="csr")


def _full_grid_operators(trunc: FockTruncation) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    a = sp.kron(_destroy(trunc.na), sp.identity(trunc.nb_max), format="csr")
    b = sp.kron(sp.identity(trunc.na), _destroy(trunc.nb_max), format="csr")
    return a, b


def _project(op: sp.spmatrix, trunc: FockTruncation) -> sp.csr_matrix:
    """P op P onto the kept states. Project finished operators, never their factors:
    a product such as a^dag^2 b passes through states a tapered buffer may drop."""
    keep = trunc.kept_indices()
    return op.tocsr()[keep][:, keep].tocsr()


def two_mode_operators(trunc: FockTruncation) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Memory and buffer annihilation operators, projected onto the kept states."""
    a, b = _full_grid_operators(trunc)
    return _project(a, trunc), _project(b, trunc)


def lindblad_superoperator(h: sp.spmatrix, jump_ops: Sequence[sp.spmatrix]) -> sp.csr_matrix:
    """Matrix of rho -> -i[H, rho] + sum_k D[c_k] rho acting on vec(rho)."""
    eye = sp.identity(h.shape[0], format="csr", dtype=complex)
    big_l = -1j * (sp.kron(eye, h) - sp.kron(h.T, eye))
    for c in jump_ops:
        cdc = (c.conj().T @ c).tocsr()
        big_l = big_l + sp.kron(c.conj(), c) - 0.5 * sp.kron(eye, cdc) - 0.5 * sp.kron(cdc.T, eye)
    return big_l.tocsr()


def two_mode_hamiltonian(
    params: CatParams, trunc: FockTruncation, eps_z: float = 0.0
) -> sp.csr_matrix:
    """H_2ph + eps_z (a + a^dag), built on the full grid, then projected.

    H_2ph = g2 (a^2 b^dag + h.c.) - eps_d (b + b^dag). The readout drive eps_z (Task 2.3)
    breaks the memory-parity symmetry, so parity_block must not be used when eps_z != 0.
    """
    a, b = _full_grid_operators(trunc)
    a2 = a @ a
    h = params.g2 * (a2 @ b.T + a2.T @ b) - params.eps_d * (b + b.T) + eps_z * (a + a.T)
    return _project(h, trunc).astype(complex)


def _thermal_jumps(rate: float, op: sp.csr_matrix, op_dag: sp.csr_matrix, n_th: float) -> list:
    jumps = [math.sqrt(rate * (1 + n_th)) * op] if rate > 0 else []
    if rate > 0 and n_th > 0:
        jumps.append(math.sqrt(rate * n_th) * op_dag)
    return jumps


def two_mode_liouvillian(
    params: CatParams,
    trunc: FockTruncation,
    eps_z: float = 0.0,
    n_th_a: float = 0.0,
    n_th_b: float = 0.0,
) -> sp.csr_matrix:
    """Liouvillian of H_2ph (+ eps_z drive) with loss and thermal gain on both modes.

    Jumps: sqrt(kappa_b (1 + n_th_b)) b, sqrt(kappa_b n_th_b) b^dag, and the same for a.
    Réglade et al. (2024, Table S2) give n_th_a = 0.10, n_th_b = 0.011; the challenge sets 0.
    """
    for name, n_th in (("n_th_a", n_th_a), ("n_th_b", n_th_b)):
        if n_th < 0:
            raise ValueError(f"{name} must be non-negative, got {n_th}")
    a_full, b_full = _full_grid_operators(trunc)
    a, b = _project(a_full, trunc), _project(b_full, trunc)
    a_dag, b_dag = _project(a_full.T, trunc), _project(b_full.T, trunc)
    jump_ops = _thermal_jumps(params.kappa_b, b, b_dag, n_th_b)
    jump_ops += _thermal_jumps(params.kappa_a, a, a_dag, n_th_a)
    return lindblad_superoperator(two_mode_hamiltonian(params, trunc, eps_z), jump_ops)


def parity_block(
    big_l: sp.spmatrix, trunc: FockTruncation, odd: bool
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Restriction of L to the memory-parity odd (or even) sector, with its vec indices.

    |i><j| has memory parity (n_a(i) + n_a(j)) mod 2. Every term of the model changes the
    memory photon number of rho by an even amount on each side, or by one on both sides,
    so the two sectors never mix (a weak Z2 symmetry).
    """
    n_a = trunc.memory_photons
    sector = vec((n_a[:, None] + n_a[None, :]) % 2)
    idx = np.flatnonzero(sector == (1 if odd else 0))
    block = big_l.tocsc()[:, idx].tocsr()[idx, :].tocsc()
    return block, idx
