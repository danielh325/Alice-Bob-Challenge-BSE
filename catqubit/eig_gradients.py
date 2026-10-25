"""Adjoint derivative of an isolated eigenvalue, for gradient-based readout optimization.

For an isolated (non-degenerate) eigenvalue lambda of a sparse matrix M with right
eigenvector r (M r = lambda r) and left eigenvector l (l^H M = lambda l^H),

    d lambda / d theta = l^H (dM/d theta) r / (l^H r).

Both vectors come from inverse iteration on one sparse LU factorization of M: the right
vector from lu.solve(x) and the left one from lu.solve(x, trans="H"). Every control knob
of the cat-qubit Liouvillian enters it linearly, so dM/d theta is one known sparse matrix
and the full gradient costs a few sparse mat-vecs after the eigenvalue itself.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

LU_ORDERING = "COLAMD"
# Successive inverse-iteration estimates must agree this closely (relative).
EIGEN_RTOL = 1e-11
MAX_INVERSE_ITERATIONS = 60  # convergence is linear, at |lambda_1 / lambda_2| per step
# op^-1 x must be parallel to x to this relative accuracy (rejects two equally close modes).
VECTOR_RTOL = 1e-6


@dataclass(frozen=True)
class EigenPair:
    """An isolated eigenvalue with unit-norm right and left eigenvectors.

    residual and left_residual are ||M r - lambda r|| and ||M^H l - conj(lambda) l||.
    """

    value: complex
    right: np.ndarray
    left: np.ndarray
    residual: float
    left_residual: float


def _inverse_iteration(
    solve: Callable[[np.ndarray], np.ndarray], start: np.ndarray, max_iter: int
) -> tuple[complex, np.ndarray]:
    """Eigenvalue of the operator closest to 0 (as seen through solve = op^-1) and its vector."""
    x = np.asarray(start, dtype=complex)
    x = x / np.linalg.norm(x)
    previous = None
    for _ in range(max_iter):
        y = solve(x)
        y_norm = np.linalg.norm(y)
        mu = np.vdot(x, y)  # Rayleigh quotient of op^-1
        direction_settled = np.linalg.norm(y - mu * x) <= VECTOR_RTOL * y_norm
        x = y / y_norm
        if mu == 0 or not direction_settled:
            previous = None
            continue
        estimate = 1.0 / mu
        if previous is not None and abs(estimate - previous) <= EIGEN_RTOL * abs(estimate):
            return complex(estimate), x
        previous = estimate
    raise ValueError(
        f"inverse iteration did not converge in {max_iter} steps; "
        "the eigenvalue closest to the shift may not be isolated"
    )


def isolated_eigenpair(
    matrix: sp.spmatrix,
    start: np.ndarray,
    left_start: np.ndarray | None = None,
    lu: sla.SuperLU | None = None,
    max_iter: int = MAX_INVERSE_ITERATIONS,
) -> EigenPair:
    """Eigenvalue of matrix closest to 0, with its right and left eigenvectors.

    Shift the matrix beforehand if the wanted eigenvalue is not the one nearest 0 (or
    deflate a known zero mode). One LU serves both vectors; pass lu to reuse an existing one.
    """
    m = sp.csc_matrix(matrix, dtype=complex)
    lu = sla.splu(m, permc_spec=LU_ORDERING) if lu is None else lu
    value, right = _inverse_iteration(lu.solve, start, max_iter)
    left_start = start if left_start is None else left_start
    left_value, left = _inverse_iteration(lambda x: lu.solve(x, trans="H"), left_start, max_iter)
    if abs(left_value.conjugate() - value) > 1e3 * EIGEN_RTOL * abs(value):
        raise ValueError(f"left ({left_value.conjugate()}) and right ({value}) eigenvalues differ")
    residual = float(np.linalg.norm(m @ right - value * right))
    left_residual = float(np.linalg.norm(m.conj().T @ left - np.conj(value) * left))
    return EigenPair(value, right, left, residual, left_residual)


def eigenvalue_derivative(pair: EigenPair, d_matrix: sp.spmatrix) -> complex:
    """d lambda / d theta = l^H (dM/d theta) r / (l^H r) for an isolated eigenvalue."""
    overlap = np.vdot(pair.left, pair.right)
    if abs(overlap) == 0:
        raise ValueError("left and right eigenvectors are orthogonal; eigenvalue is defective")
    return complex(np.vdot(pair.left, d_matrix @ pair.right) / overlap)


def jax_rate_function(
    value_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
) -> Callable[[jax.Array], jax.Array]:
    """Expose a host-side (rate, gradient) function of a real parameter vector to JAX.

    The primal runs through jax.pure_callback (so it also works under jit), and a
    jax.custom_jvp supplies the exact adjoint gradient: jvp = gradient . tangent.
    """

    def _host(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        value, grad = value_and_grad(np.asarray(x, dtype=float))
        return np.asarray(value, dtype=x.dtype), np.asarray(grad, dtype=x.dtype)

    def _call(x: jax.Array) -> tuple[jax.Array, jax.Array]:
        shapes = (
            jax.ShapeDtypeStruct((), x.dtype),
            jax.ShapeDtypeStruct(x.shape, x.dtype),
        )
        return jax.pure_callback(_host, shapes, x, vmap_method="sequential")

    @jax.custom_jvp
    def rate(x: jax.Array) -> jax.Array:
        return _call(x)[0]

    @rate.defjvp
    def _rate_jvp(primals: tuple, tangents: tuple) -> tuple[jax.Array, jax.Array]:
        (x,), (dx,) = primals, tangents
        value, grad = _call(x)
        return value, jnp.dot(grad, dx)

    return rate
