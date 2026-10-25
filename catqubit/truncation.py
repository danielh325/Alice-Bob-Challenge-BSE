"""Hilbert-space truncation checks (challenge Section B.9)."""

import dynamiqs as dq
import jax.numpy as jnp
from dynamiqs import QArray
from jax import Array


def _check_levels(dims: tuple[int, ...], mode: int, n_levels: int) -> None:
    if not 0 <= mode < len(dims):
        raise ValueError(f"mode must index dims={dims}, got {mode}")
    if not 1 <= n_levels <= dims[mode]:
        raise ValueError(f"n_levels must be in [1, {dims[mode]}], got {n_levels}")


def top_level_population(state: QArray, dims: tuple[int, ...], mode: int, n_levels: int) -> Array:
    """Population in the highest n_levels Fock states of one mode.

    Accepts a ket or density matrix, optionally batched (e.g. result.states over
    time); the batch shape is preserved. A trustworthy simulation keeps this
    negligible at all times, not just at the end.
    """
    _check_levels(dims, mode, n_levels)
    rho = dq.ptrace(state.todm(), mode, dims)
    populations = jnp.real(jnp.diagonal(rho.to_jax(), axis1=-2, axis2=-1))
    return populations[..., -n_levels:].sum(axis=-1)


def top_level_projector(dims: tuple[int, ...], mode: int, n_levels: int) -> QArray:
    """Projector onto the highest n_levels Fock states of one mode, on the full space.

    Pass it as an exp_op to monitor truncation without saving every state.
    """
    _check_levels(dims, mode, n_levels)
    dim = dims[mode]
    local = sum(dq.fock(dim, k).todm() for k in range(dim - n_levels, dim))
    factors = [local if i == mode else dq.eye(d) for i, d in enumerate(dims)]
    return dq.tensor(*factors)
