"""Adjoint derivative of an isolated Liouvillian eigenvalue (Task 2.4, plan Section 5)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from catqubit.driven_rates import (
    Dissipation,
    HamiltonianTerms,
    deflate_steady_state,
    driven_bit_flip_rate,
    driven_liouvillian,
    hamiltonian_generators,
    hamiltonian_superoperator,
)
from catqubit.eig_gradients import (
    eigenvalue_derivative,
    isolated_eigenpair,
    jax_rate_function,
)
from catqubit.liouvillian import FockTruncation, two_mode_operators, vec

TRUNC = FockTruncation(na=14, nb=4)
DEFLATION = -10.0
FD_STEP = 1e-5


def _slow_pair(terms, diss, trunc=TRUNC):
    big_l = deflate_steady_state(driven_liouvillian(terms, diss, trunc), trunc, DEFLATION)
    a, _ = two_mode_operators(trunc)
    start = vec((a + a.T).toarray())
    return isolated_eigenpair(big_l.tocsc(), start)


def test_eigenpair_on_a_small_dense_matrix():
    rng = np.random.default_rng(1)
    m = rng.normal(size=(8, 8)) + np.diag(np.arange(1.0, 9.0) * 3)
    m[0, 0] = 0.01
    pair = isolated_eigenpair(sp.csc_matrix(m), rng.normal(size=8))
    vals = np.linalg.eigvals(m)
    expected = vals[np.argmin(np.abs(vals))]
    assert pair.value == pytest.approx(expected, rel=1e-10)
    assert np.allclose(m @ pair.right, pair.value * pair.right, atol=1e-10)
    assert np.allclose(pair.left.conj() @ m, pair.value * pair.left.conj(), atol=1e-10)


def test_derivative_on_a_small_dense_matrix():
    rng = np.random.default_rng(2)
    m = rng.normal(size=(6, 6)) + np.diag(np.arange(1.0, 7.0) * 4)
    m[0, 0] = 0.05
    dm = rng.normal(size=(6, 6))
    pair = isolated_eigenpair(sp.csc_matrix(m), np.ones(6))
    h = 1e-6

    def smallest(mat):
        v = np.linalg.eigvals(mat)
        return v[np.argmin(np.abs(v))]

    fd = (smallest(m + h * dm) - smallest(m - h * dm)) / (2 * h)
    assert eigenvalue_derivative(pair, sp.csr_matrix(dm)) == pytest.approx(fd, rel=1e-6)


def test_unconverged_iteration_raises():
    m = sp.csc_matrix(np.diag([1.0, -1.0, 3.0]))  # two eigenvalues equally close to 0
    with pytest.raises(ValueError, match="did not converge"):
        isolated_eigenpair(m, np.array([1.0, 1.0, 0.0]), max_iter=4)


@pytest.mark.parametrize("knob", ["g2", "eps_z", "delta_a"])
def test_rate_derivative_matches_finite_differences(params, knob):
    p = params.with_alpha(1.0)
    diss = Dissipation.from_params(p)
    terms = HamiltonianTerms.from_params(p, eps_z=0.15, delta_a=0.05)
    pair = _slow_pair(terms, diss)
    gen = hamiltonian_generators(TRUNC)[knob]
    analytic = -eigenvalue_derivative(pair, hamiltonian_superoperator(gen)).real

    def rate(value):
        shifted = terms.replace(**{knob: value})
        return driven_bit_flip_rate(shifted, diss, TRUNC).rate

    x0 = dict(terms.coefficients())[knob]
    fd = (rate(x0 + FD_STEP) - rate(x0 - FD_STEP)) / (2 * FD_STEP)
    assert analytic == pytest.approx(fd, rel=1e-6)


def test_driven_rate_gradient_matches_the_eigenpair_derivative(params):
    p = params.with_alpha(1.0)
    diss = Dissipation.from_params(p)
    terms = HamiltonianTerms.from_params(p, eps_z=0.15)
    res = driven_bit_flip_rate(terms, diss, TRUNC, with_gradient=True)
    pair = _slow_pair(terms, diss)
    gen = hamiltonian_generators(TRUNC)["eps_z"]
    expected = -eigenvalue_derivative(pair, hamiltonian_superoperator(gen)).real
    assert res.gradient["eps_z"] == pytest.approx(expected, rel=1e-8)


def test_jax_wrapper_value_and_grad(params):
    p = params.with_alpha(1.0)
    diss = Dissipation.from_params(p)
    base = HamiltonianTerms.from_params(p)
    knobs = ("eps_z", "delta_a")

    def value_and_grad(x):
        terms = base.replace(**dict(zip(knobs, map(float, x), strict=True)))
        res = driven_bit_flip_rate(terms, diss, TRUNC, with_gradient=True)
        return res.rate, np.array([res.gradient[k] for k in knobs])

    rate_fn = jax_rate_function(value_and_grad)
    x = jnp.array([0.15, 0.05])
    val, grad = jax.value_and_grad(rate_fn)(x)
    ref_val, ref_grad = value_and_grad(np.asarray(x))
    assert float(val) == pytest.approx(ref_val, rel=1e-12)
    assert np.allclose(np.asarray(grad), ref_grad, rtol=1e-12)
    jitted = jax.jit(rate_fn)
    assert float(jitted(x)) == pytest.approx(ref_val, rel=1e-12)
