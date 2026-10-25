"""Memory-mode observables for Task 1.3. All return real arrays and keep batch axes."""

import dynamiqs as dq
import jax.numpy as jnp
from dynamiqs import QArray
from jax import Array


def memory_state(states: QArray, na: int, nb: int) -> QArray:
    """Reduced density matrix of the memory, tracing out the buffer."""
    return dq.ptrace(states.todm(), 0, (na, nb))


def parity_expectation(state: QArray) -> Array:
    """<P> = <exp(i pi a^dag a)>: +1 for even photon numbers, -1 for odd."""
    return jnp.real(dq.expect(dq.parity(state.shape[-2]), state))


def purity(state: QArray) -> Array:
    """Tr(rho^2): 1 for a pure state, 1/2 for an equal mixture of two orthogonal states."""
    rho = state.todm().to_jax()
    return jnp.real(jnp.einsum("...ij,...ji->...", rho, rho))


def fidelity_with(state: QArray, ket: QArray) -> Array:
    """<psi|rho|psi>, the fidelity of state (ket or density matrix) with a pure ket."""
    if state.shape[-2] != ket.shape[-2]:
        raise ValueError(
            f"state dimension {state.shape[-2]} does not match ket dimension {ket.shape[-2]}"
        )
    return jnp.real(dq.expect(ket.todm(), state))


def cat_coherence(state: QArray, alpha: complex) -> Array:
    """2 Re <alpha|rho|-alpha>: the interference term between the two codewords.

    +1 for the even cat, -1 for the odd cat, and ~0 for a classical mixture of
    |+alpha> and |-alpha> or for a single codeword (up to e^{-2|alpha|^2}). This is
    what the Wigner-function fringes encode (challenge Section B.5).
    """
    dim = state.shape[-2]
    plus = dq.coherent(dim, alpha).to_jax().ravel()
    minus = dq.coherent(dim, -alpha).to_jax().ravel()
    rho = state.todm().to_jax()
    return 2 * jnp.real(jnp.einsum("i,...ij,j->...", plus.conj(), rho, minus))
