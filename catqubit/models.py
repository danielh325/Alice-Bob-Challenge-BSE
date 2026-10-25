"""Hamiltonians and jump operators for the challenge's models (rates in rad/us)."""

from typing import NamedTuple

import dynamiqs as dq
import jax.numpy as jnp
from dynamiqs import QArray
from jax.typing import ArrayLike

from catqubit.params import CatParams

MIN_DIM = 2
# a^2 is identically zero on fewer than three Fock states.
MIN_TWO_PHOTON_DIM = 3


class Model(NamedTuple):
    """A Lindblad model ready for dq.mesolve(model.H, model.jump_ops, ...)."""

    H: QArray
    jump_ops: list[QArray]


def _check_dim(name: str, dim: int, minimum: int) -> None:
    if dim < minimum:
        raise ValueError(f"{name} truncation must be at least {minimum}, got {dim}")


def _check_rate(name: str, rate: ArrayLike) -> None:
    if jnp.any(jnp.asarray(rate) < 0):
        raise ValueError(f"{name} must be non-negative, got {rate}")


def _scaled(rate: ArrayLike, op: QArray) -> QArray:
    """sqrt(rate) * op, broadcasting a batch of rates over a leading axis."""
    root = jnp.sqrt(jnp.asarray(rate))
    return root[..., None, None] * op if root.ndim else root * op


def single_photon_loss_model(dim: int, delta: float, kappa: float) -> Model:
    """Task 1.0(a): H = delta * a^dag a, jump sqrt(kappa) * a."""
    _check_dim("memory", dim, MIN_DIM)
    _check_rate("kappa", kappa)
    return Model(H=delta * dq.number(dim), jump_ops=[_scaled(kappa, dq.destroy(dim))])


def two_photon_loss_model(dim: int, kappa2: ArrayLike) -> Model:
    """Task 1.0(b): H = 0, jump sqrt(kappa2) * a^2."""
    _check_dim("memory", dim, MIN_TWO_PHOTON_DIM)
    _check_rate("kappa2", kappa2)
    a = dq.destroy(dim)
    return Model(H=dq.zeros(dim), jump_ops=[_scaled(kappa2, a @ a)])


def effective_model(dim: int, params: CatParams, kappa2: ArrayLike | None = None) -> Model:
    """Task 1.0(c): jumps sqrt(kappa2) (a^2 - alpha^2) and sqrt(kappa_a) a, with H = 0.

    kappa2 defaults to 4 g2^2 / kappa_b; pass an array to batch over several values.
    The memory-loss jump is omitted when kappa_a == 0.
    """
    _check_dim("memory", dim, MIN_TWO_PHOTON_DIM)
    kappa2 = params.kappa2 if kappa2 is None else kappa2
    _check_rate("kappa2", kappa2)
    a = dq.destroy(dim)
    jump_ops = [_scaled(kappa2, a @ a - params.alpha**2 * dq.eye(dim))]
    if params.kappa_a > 0:
        jump_ops.append(_scaled(params.kappa_a, a))
    return Model(H=dq.zeros(dim), jump_ops=jump_ops)


def two_mode_model(na: int, nb: int, params: CatParams) -> Model:
    """Full memory + buffer model: H_2ph with jumps sqrt(kappa_b) b, sqrt(kappa_a) a."""
    _check_dim("memory", na, MIN_TWO_PHOTON_DIM)
    _check_dim("buffer", nb, MIN_DIM)
    a, b = dq.destroy(na, nb)
    H = (
        params.g2 * (a @ a @ b.dag())
        + params.g2 * (a.dag() @ a.dag() @ b)
        - params.eps_d * (b + b.dag())
    )
    jump_ops = [_scaled(params.kappa_b, b), _scaled(params.kappa_a, a)]
    return Model(H=H, jump_ops=jump_ops)
