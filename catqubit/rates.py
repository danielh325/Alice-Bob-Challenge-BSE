"""Exact bit- and phase-flip rates from the two-mode Liouvillian (Tasks 2.1 and 2.2).

Rates are decay rates of observables: Re<a> decays at the bit-flip rate and <P> at the
phase-flip rate. The one-way jump rate |+alpha> -> |-alpha> is half the bit-flip rate.
"""

import dynamiqs as dq
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

from catqubit.liouvillian import (
    FockTruncation,
    parity_block,
    two_mode_liouvillian,
    two_mode_operators,
    vec,
)
from catqubit.params import CatParams

LU_ORDERING = "COLAMD"
# Successive inverse-iteration estimates must agree this closely before a rate is returned.
RATE_RTOL = 1e-8
MAX_INVERSE_ITERATIONS = 8
# The even sector holds the steady state (eigenvalue 0); shift slightly off it and drop it.
ZERO_MODE_SHIFT = -1e-6
ZERO_MODE_TOL = 1e-8


def slowest_eigenvalues(block: sp.spmatrix, k: int, sigma: float = 0.0) -> np.ndarray:
    """The k eigenvalues closest to sigma (shift-invert), ordered slowest decay first."""
    shifted = (block - sigma * sp.identity(block.shape[0], format="csc")).tocsc()
    lu = sla.splu(shifted, permc_spec=LU_ORDERING)
    op = sla.LinearOperator(block.shape, matvec=lu.solve, dtype=complex)
    vals = sla.eigs(block, k=k, sigma=sigma, OPinv=op, which="LM", return_eigenvectors=False)
    return vals[np.argsort(-vals.real)]


def _codeword_density(params: CatParams, trunc: FockTruncation) -> np.ndarray:
    memory = np.asarray(dq.coherent(trunc.na, params.alpha).to_jax()).ravel()
    buffer = np.zeros(trunc.nb_max)
    buffer[0] = 1.0
    psi = np.kron(memory, buffer)[trunc.kept_indices()]
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def bit_flip_rate(
    params: CatParams, trunc: FockTruncation, n_th_a: float = 0.0, n_th_b: float = 0.0
) -> float:
    """Decay rate of Re<a> from |+alpha>, i.e. the slowest rate of the parity-odd sector.

    Inverse iteration on that sector (which has no zero mode), started from the codeword
    and read out through <a + a^dag>: one LU factorization and a few solves.
    Thermal populations n_th_a, n_th_b keep the parity symmetry, so they are allowed.
    """
    big_l = two_mode_liouvillian(params, trunc, n_th_a=n_th_a, n_th_b=n_th_b)
    odd, idx = parity_block(big_l, trunc, odd=True)
    lu = sla.splu(odd, permc_spec=LU_ORDERING)
    a, _ = two_mode_operators(trunc)
    z = vec((a + a.T).toarray().T)[idx]
    x = vec(_codeword_density(params, trunc))[idx]
    previous = None
    for _ in range(MAX_INVERSE_ITERATIONS):
        x_next = lu.solve(x)
        estimate = -((z @ x) / (z @ x_next)).real
        if previous is not None and abs(estimate - previous) <= RATE_RTOL * abs(estimate):
            return float(estimate)
        previous, x = estimate, x_next / np.linalg.norm(x_next)
    raise ValueError(
        f"inverse iteration did not converge in {MAX_INVERSE_ITERATIONS} steps "
        f"(alpha={params.alpha}); the slowest odd mode may not be isolated"
    )


def phase_flip_rate(
    params: CatParams, trunc: FockTruncation, n_th_a: float = 0.0, n_th_b: float = 0.0
) -> float:
    """Decay rate of the memory parity <P>: the slowest nonzero rate of the even sector."""
    big_l = two_mode_liouvillian(params, trunc, n_th_a=n_th_a, n_th_b=n_th_b)
    even, _ = parity_block(big_l, trunc, odd=False)
    vals = slowest_eigenvalues(even, k=3, sigma=ZERO_MODE_SHIFT)
    nonzero = vals[np.abs(vals) > ZERO_MODE_TOL]
    return float(-nonzero[0].real)
